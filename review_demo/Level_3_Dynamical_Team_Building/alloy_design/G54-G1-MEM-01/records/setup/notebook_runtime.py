"""Runtime and recording only. Agent construction and chat stay in the notebook."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import trial_support as support

HERE = Path(__file__).resolve().parent


def begin_run(run_id):
    support.PROTOCOL = json.loads((HERE / "groupagent_live_protocol.json").read_text())
    trial = support.Trial(run_id, "groupagent_equilibrium", "development")
    sys.path[:0] = [str(HERE / "development"), str(HERE)]
    os.environ["ALLOYGEN_EQ_OUTPUT_DIR"] = str(trial.out / "calculations")
    for name in ("notebook_runtime.py", "groupagent_live_protocol.json",
                 "team_builder_equilibrium_demo.ipynb", "execute_live_demo.py"):
        source = HERE / name
        (trial.out / "setup" / name).write_bytes(source.read_bytes())
        trial.state["source_hashes"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    trial.write()
    return trial


@contextmanager
def record_run(trial, group_agent, task):
    """Keep normal notebook stdout visible; save usage and nested conversations."""
    from GroupAgent_dev.agent_builder import AgentBuilder
    from autogen import GroupChatManager
    from autogen.agentchat import ConversableAgent

    original_build = AgentBuilder.build_from_library
    original_manager_init = GroupChatManager.__init__
    managers = []
    trial.state["task"] = task
    trial.phase = "groupagent"
    trial.write()

    def build(builder, *args, **kwargs):
        previous = trial.phase
        trial.phase = "builder"
        start = len(trial.state["api_requests"])
        try:
            return original_build(builder, *args, **kwargs)
        finally:
            for row in trial.state["api_requests"][start:]:
                for choice in row.get("response", {}).get("choices", []):
                    print("\nAgentBuilder model response:")
                    print(choice["message"].get("content") or "")
            trial.phase = previous

    def manager_init(manager, *args, **kwargs):
        original_manager_init(manager, *args, **kwargs)
        managers.append(manager)

    status = "completed_pending_scientific_review"
    error = None
    try:
        with trial.observe():
            observed_tool = ConversableAgent.execute_function

            def bounded_tool(agent, func_call, *args, **kwargs):
                calls = sum(r["request"].get("name") == "calculate_al_equilibrium"
                            for r in trial.state["tool_calls"])
                if func_call.get("name") == "calculate_al_equilibrium" and calls >= support.PROTOCOL["max_equilibrium_calls"]:
                    raise RuntimeError("Declared equilibrium-call limit reached")
                return observed_tool(agent, func_call, *args, **kwargs)

            with patch.object(AgentBuilder, "build_from_library", build), \
                 patch.object(GroupChatManager, "__init__", manager_init), \
                 patch.object(ConversableAgent, "execute_function", bounded_tool):
                yield
    except BaseException as exc:
        status, error = "failed", exc
        raise
    finally:
        trial.state["manager_conversations"] = {
            partner.name: messages for partner, messages in group_agent.assistant.chat_messages.items()}
        trial.state["nested_conversations"] = [manager.groupchat.messages for manager in managers]
        trial.state["build_history"] = group_agent.executor.build_history
        trial.state["build_times"] = group_agent.executor.build_times
        trial.state["selected_agents"] = [[a["name"] for a in cfg["agent_configs"]]
                                           for cfg in group_agent.executor.build_history.values()]
        trial.finish(status, error)
        support.save(trial.out / "selected_config.json", group_agent.executor.build_history)
