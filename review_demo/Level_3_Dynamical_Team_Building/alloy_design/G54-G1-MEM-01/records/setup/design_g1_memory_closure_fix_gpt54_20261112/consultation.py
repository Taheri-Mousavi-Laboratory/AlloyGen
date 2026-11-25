"""One required reference question per G1 assignment; reuse ordinary tool routing."""
from copy import deepcopy
from autogen import OpenAIWrapper


class CodebookConsultation:
    def begin_consultation(self):
        role = {"planner.md": "planner", "reviewer.md": "critic"}.get(self.guide)
        self._consultation_tool = ("consult_" + role + "_codebook"
                                   if role and self.guidance_variant == "rag" else None)
        self._consultation_call = None

    def _generate_oai_reply_from_client(self, llm_client, messages, cache):
        name = self._consultation_tool
        if not name:
            return super()._generate_oai_reply_from_client(llm_client, messages, cache)
        if self._consultation_call:
            # Inspect the actual provider view AFTER existing context hooks. A
            # previous assignment's retrieval cannot satisfy this assignment.
            calls = [c for m in messages for c in (m.get("tool_calls") or [])]
            replies = [r for m in messages for r in (m.get("tool_responses") or [m])]
            paired = any(c.get("id") == self._consultation_call
                         and c["function"]["name"] == name for c in calls)
            delivered = any(r.get("role") == "tool"
                            and r.get("tool_call_id") == self._consultation_call
                            and bool(r.get("content")) for r in replies)
            if not (paired and delivered):
                raise RuntimeError("The matching codebook tool reply must reach the agent before its decision")
            # A no-match/error reply is still information: normal choice resumes
            # so the agent can refine its query or report uncertainty. No loop.
            self._consultation_tool = None
            return super()._generate_oai_reply_from_client(llm_client, messages, cache)

        config = deepcopy(self.llm_config.model_dump())
        config["cache_seed"] = self.llm_config.get("cache_seed")
        offered = {t["function"]["name"] for t in config.get("tools", [])}
        if name not in offered:
            raise RuntimeError("Required reference tool is absent from this agent's tools")
        choice = {"type": "function", "function": {"name": name}}
        config.update(tool_choice=choice, parallel_tool_calls=False)
        # Installed AG2 merges config_list entries OVER call/base settings.
        # Override both levels in this temporary client, never the saved client.
        for item in config.get("config_list", []):
            item.update(tool_choice=deepcopy(choice), parallel_tool_calls=False)
        forced = OpenAIWrapper(**config)
        reply = super()._generate_oai_reply_from_client(forced, messages, cache)
        calls = reply.get("tool_calls", []) if isinstance(reply, dict) else []
        if len(calls) != 1 or calls[0]["function"]["name"] != name or not calls[0].get("id"):
            raise RuntimeError("Required consultation returned no single reference-tool call")
        self._consultation_call = calls[0]["id"]
        return reply
