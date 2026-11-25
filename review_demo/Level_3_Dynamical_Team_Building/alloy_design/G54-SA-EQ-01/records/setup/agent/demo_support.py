"""Run observation and concise notebook displays; never supplies reference answers."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .baseline_support import HERE, REPO, RunRecord, RedactingStream, contained, save_json, sha
from .credentials import redact


def public_text(text):
    """Remove local path details from reader displays; preserve scientific content."""
    text = redact(text)
    return re.sub(r'/trace/[^\s\x22\x27<>\)\],;]+', '[local-path]', text)


def last_agent_message(history, agent_name):
    """Use the sender name: AG2 roles are relative to the history's owner."""
    for message in reversed(history):
        content = message.get('content')
        if (message.get('name') == agent_name and isinstance(content, str)
                and content.strip() and content.strip().upper() != 'TERMINATE'
                and not message.get('function_call') and not message.get('tool_calls')):
            return content
    return 'No textual answer recorded from this agent.'


def saved_data_table(value, limit=12):
    """Render small flat JSON records without changing values or agent messages."""
    rows = [value] if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        return None
    if any(isinstance(cell, (dict, list)) for row in rows for cell in row.values()):
        return None
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        return None
    def text(value):
        return public_text(str(value)).replace('|', '&#124;').replace('\n', ' ')
    lines = ['**Saved results** (notebook formatting of the agent-generated file).', '',
             '| ' + ' | '.join(text(key) for key in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(text(row.get(key, '')) for key in columns) + ' |'
              for row in rows[:limit]]
    if len(rows) > limit:
        lines += ['', f'Showing {limit} of {len(rows)} records; the linked file contains every record.']
    return '\n'.join(lines)


def price_usage(usage, pricing):
    """Provider cached-input tokens are part of input tokens, not extra tokens."""
    if not usage:
        return {"estimated_cost_usd": None, "undiscounted_cost_usd": None}
    prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    rates = pricing["per_million_tokens"]
    upper = (prompt * rates["input"] + output * rates["output"]) / 1e6
    estimate = None if cached is None else (
        (prompt-cached)*rates["input"] + cached*rates["cached_input"] + output*rates["output"]
    ) / 1e6
    return {"estimated_cost_usd": estimate, "undiscounted_cost_usd": upper}


def summarize_usage(requests):
    successful = [r for r in requests if r.get("status") == "complete"]
    counts = {key: sum((r.get("usage") or {}).get(key, 0) for r in successful)
              for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    cached = [(r.get("usage") or {}).get("prompt_tokens_details") or {} for r in successful]
    counts["provider_cached_input_tokens"] = (
        sum(d["cached_tokens"] for d in cached) if all("cached_tokens" in d for d in cached) else None)
    for key in ("estimated_cost_usd", "undiscounted_cost_usd"):
        values = [r.get(key) for r in successful]
        counts[key] = sum(values) if all(v is not None for v in values) else None
    counts.update({"completed_provider_calls": len(successful),
                   "failed_provider_calls": sum(r.get("status") == "failed" for r in requests),
                   "missing_usage_responses": sum(not r.get("usage") for r in successful)})
    return counts


class DemoRun(RunRecord):
    def __init__(self, selected, *, run_id=None, notebook, model="gpt-4o-2024-08-06", purpose="development",
                 code_timeout_seconds=60, executor_feedback_character_limit=None, api_budget_usd=None,
                 source_manifest="agent/tasks_equilibrium.json", assessment_protocol="agent/config.json"):
        if not isinstance(code_timeout_seconds, int) or code_timeout_seconds <= 0:
            raise ValueError('Execution timeout must be a positive integer in seconds')
        if executor_feedback_character_limit is not None and (
                not isinstance(executor_feedback_character_limit, int) or executor_feedback_character_limit < 1000):
            raise ValueError('Feedback limit must be at least 1000 characters or None')
        if api_budget_usd is not None and (api_budget_usd <= 0 or model != 'gpt-4o-2024-08-06'):
            raise ValueError('The estimated API budget guard supports a positive budget for the recorded GPT-4o snapshot')
        manifest_path = contained(HERE / source_manifest)
        protocol_path = contained(HERE / assessment_protocol)
        manifest = json.loads(manifest_path.read_text())
        json.loads(protocol_path.read_text())  # Fail before creating a run if the protocol is invalid.
        if run_id is None and source_manifest != "agent/tasks_equilibrium.json":
            run_id = datetime.now(timezone.utc).strftime(Path(notebook).stem + '_%Y%m%dT%H%M%SZ')
        self.current_task = None
        self._started = time.monotonic()
        super().__init__(run_id, notebook=notebook, settings={
            "model": model, "purpose": purpose, "preparation": selected["prompt"],
            "code_timeout_seconds": code_timeout_seconds,
            "executor_feedback_character_limit": executor_feedback_character_limit,
            "api_budget_usd": api_budget_usd,
            # Conservative reserve at the recorded rates for up to 128k input
            # plus 16,384 output tokens; a spending estimate, not invoice control.
            "api_request_reserve_usd": 0.50,
            "corpus": selected["corpus"], "notebook": Path(notebook).name,
            "task_outcomes": {}, "interventions": [], "api_requests": [], "llm_invocations": [],
            "tasks": [], "scientific_assessment": "pending; performed outside the agent conversation",
            "pricing": json.loads((HERE / "agent/pricing.json").read_text()),
            "source_manifest": manifest,
            "assessment_protocol": str(protocol_path.relative_to(HERE)),
        })
        if "tasks" in manifest:
            self.state.pop("shared_conversation_across_four_tasks", None)
            self.state["shared_conversation_across_tasks"] = True
            self.state["expected_task_count"] = len(manifest["tasks"])
        self.selected = selected
        self.state["active_prompt_sha256"] = sha(selected["prompt_path"])
        self.state["active_corpus"] = [
            {"path": str(p.relative_to(HERE)), "sha256": sha(p)} for p in selected["corpus_paths"]]
        self.state["input_isolation"] = "Explicit prompt/corpus inputs; executor write containment is not read isolation."
        sources = list((HERE / 'agent').glob('*.py')) + [HERE / 'agent/config.json', HERE / 'agent/pricing.json']
        sources += [selected['prompt_path'], manifest_path, *selected['corpus_paths']]
        self.state["implementation_sources"] = []
        for source in sources:
            target = contained(self.results / "setup" / source.relative_to(HERE))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            self.state["implementation_sources"].append({"path": str(source.relative_to(HERE)), "sha256": sha(source)})
        # Snapshot the actual input before the driver updates its visible outputs.
        (self.results / "notebook_input.ipynb").write_bytes(Path(notebook).read_bytes())
        self.save()

    def observe_client(self, wrapper):
        """Instance-local observation, preserving original request and cache behavior."""
        for client in wrapper._clients:
            original = client.create

            def observed(params, _original=original):
                budget = self.state.get('api_budget_usd')
                if budget is not None:
                    known = summarize_usage(self.state['api_requests'])
                    spent = known['estimated_cost_usd']
                    if spent is None:
                        spent = known['undiscounted_cost_usd']
                    if (known['failed_provider_calls'] or spent is None
                            or spent + self.state['api_request_reserve_usd'] > budget):
                        self.state['budget_stop'] = {
                            'before_task': self.current_task, 'estimated_spent_usd': spent,
                            'budget_usd': budget, 'reason': 'Insufficient reserve or unknown billing; no further provider call made'}
                        self.save()
                        raise RuntimeError('Estimated API budget guard stopped before the next provider call')
                event = {"task": self.current_task, "started_UTC": datetime.now(timezone.utc).isoformat(),
                         "status": "started", "requested_model": params.get("model")}
                self.state["api_requests"].append(event)
                self.save()
                started = time.monotonic()
                try:
                    response = _original(params)
                    usage = response.usage.model_dump() if response.usage is not None else None
                    event.update({"status": "complete", "response_id": response.id,
                                  "request_id": getattr(response, "_request_id", None),
                                  "served_model": response.model, "usage": usage,
                                  "service_tier": getattr(response, "service_tier", None),
                                  "system_fingerprint": getattr(response, "system_fingerprint", None)})
                    if response.model == self.state["pricing"]["model"] and event["service_tier"] in (None, "default"):
                        event.update(price_usage(usage, self.state["pricing"]))
                    return response
                except Exception as error:
                    event.update({"status": "failed", "error_type": type(error).__name__,
                                  "error": redact(str(error)), "billing_unknown": True})
                    # Preserve the failed request's messages without client configuration.
                    save_json(self.results / f"{self.current_task}_failed_request_messages.json", params.get("messages", []))
                    raise
                finally:
                    event["elapsed_seconds"] = time.monotonic() - started
                    self.state["usage"] = summarize_usage(self.state["api_requests"])
                    self.save()
            client.create = observed

        original_wrapper = wrapper.create

        def observed_wrapper(**config):
            before = len(self.state["api_requests"])
            event = {"task": self.current_task, "status": "started"}
            self.state["llm_invocations"].append(event)
            try:
                response = original_wrapper(**config)
                event.update({"status": "complete", "response_id": response.id,
                              "local_completion_cache_hit": len(self.state["api_requests"]) == before})
                return response
            except Exception as error:
                event.update({"status": "failed", "error_type": type(error).__name__})
                raise
            finally:
                self.save()
        wrapper.create = observed_wrapper

    @contextmanager
    def task(self, label, question):
        if any(t["task"] == label for t in self.state["tasks"]):
            self.state["interventions"].append({"task": label, "action": "manual task rerun prevented; fresh run required"})
            self.save()
            raise RuntimeError("This task already ran. Start a fresh notebook run to preserve its record.")
        self.current_task = label
        self.state["status"] = "in_progress"
        started = time.monotonic()
        entry = {"task": label, "question": question, "continued_history": bool(self.state["tasks"]),
                 "status": "running", "started_UTC": datetime.now(timezone.utc).isoformat()}
        self.state["tasks"].append(entry)
        self.save()
        try:
            with (self.results / f"{label}_trace.txt").open("w") as stream:
                sink = RedactingStream(stream)
                with redirect_stdout(sink), redirect_stderr(sink):
                    yield
            entry["status"] = "conversation_returned"
        except Exception as error:
            entry.update({"status": "failed", "error_type": type(error).__name__})
            if hasattr(self, "_agents"):
                self.checkpoint(label, *self._agents)
            self.state["status"] = "failed"
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started
            entry["usage"] = summarize_usage([r for r in self.state["api_requests"] if r["task"] == label])
            self.state["elapsed_seconds"] = time.monotonic() - self._started
            self.save_usage()
            self.save()

    def retrieval(self, query, n_results, response, document_ids):
        super().retrieval(query, n_results, response, document_ids)
        self.state["retrievals"][-1]["task"] = self.current_task
        self.save()

    def show_setup(self):
        from IPython.display import display, Markdown
        prefix = f"outputs/{self.state['run_id']}"
        display(Markdown(
            f"Model: `{self.state['model']}`; temperature: {self.state['temperature']}; "
            f"execution limit: {self.state['code_timeout_seconds']} s per script.\n\n"
            f"[Recorded setup]({prefix}/setup/) · "
            f"[System prompt]({prefix}/setup/agent/prompts/{self.selected['prompt']}.txt) · "
            f"[Run configuration and source identities]({prefix}/run.json)"))

    def show_task(self, label, assistant, proxy_user, finish=False):
        from IPython.display import display, Markdown, Image
        self.checkpoint(label, assistant, proxy_user, finish=finish)
        history = proxy_user.chat_messages.get(assistant, [])
        final = last_agent_message(history, assistant.name)
        display(Markdown(public_text(final)))
        executions = [e for e in self.state["code_executions"] if e.get("task") == label]
        failed = sum(e.get("exit_code") != 0 for e in executions)
        prefix = f"outputs/{self.state['run_id']}"
        display(Markdown(f"Executions: **{len(executions)}**; unsuccessful executions: **{failed}**. "
                         f"[Full conversation]({prefix}/{label}_conversation.json) · "
                         f"[Execution trace]({prefix}/{label}_trace.txt). "
                         ""))
        current = self.state["task_checkpoints"][-1]["artifacts"]
        previous = {Path(a["file"]).name: a["sha256"] for c in self.state["task_checkpoints"][:-1] for a in c["artifacts"]}
        for artifact in current:
            path = self.results / artifact["file"]
            if previous.get(path.name) == artifact["sha256"]:
                continue
            if path.suffix == ".png":
                display(Image(filename=str(path)))
            elif path.suffix in {".json", ".csv"}:
                display(Markdown(f"Saved data: [{path.name}]({prefix}/{artifact['file']})"))
                if path.suffix == '.json':
                    try:
                        table = saved_data_table(json.loads(path.read_text()))
                    except (ValueError, OSError):
                        table = 'The saved file could not be rendered as a table; inspect the linked file.'
                    if table:
                        display(Markdown(table))
        if finish:
            self.state["elapsed_seconds"] = time.monotonic() - self._started
            self.save()

    def show_usage(self):
        from IPython.display import display, Markdown
        lines = ["| Task | Input tokens | Cached input¹ | Output tokens | Estimated USD² |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for task in self.state["tasks"]:
            u = task["usage"]
            cost = u["estimated_cost_usd"]
            lines.append(f"| {task['task']} | {u['prompt_tokens']:,} | {u['provider_cached_input_tokens']} | "
                         f"{u['completion_tokens']:,} | {cost:.6f} |" if cost is not None else
                         f"| {task['task']} | {u['prompt_tokens']:,} | unknown | {u['completion_tokens']:,} | unknown |")
        u = summarize_usage(self.state["api_requests"])
        cost = u["estimated_cost_usd"]
        if cost is not None:
            lines.append(f"| **Total** | **{u['prompt_tokens']:,}** | **{u['provider_cached_input_tokens']:,}** | "
                         f"**{u['completion_tokens']:,}** | **{cost:.6f}** |")
        lines += ["", "¹ Cached input is included in input tokens. ² Estimated from returned API usage and "
                  "[published rates](https://developers.openai.com/api/docs/models/gpt-4o); not an invoice. "
                  "Includes the original task-1 LLM summary call. Retrieval embeddings and solver execution are local."]
        display(Markdown("\n".join(lines)))
        self.save_usage()

    def save_usage(self):
        save_json(self.results / "usage.json", {"pricing": self.state["pricing"], "total": summarize_usage(self.state["api_requests"]),
                  "tasks": [{"task": t["task"], **t.get("usage", {})} for t in self.state["tasks"]],
                  "local_completion_cache_hits": sum(e.get("local_completion_cache_hit", False)
                                                     for e in self.state["llm_invocations"]),
                  "qualification": "Returned usage only; billing for failed/unreported requests is unknown."})
