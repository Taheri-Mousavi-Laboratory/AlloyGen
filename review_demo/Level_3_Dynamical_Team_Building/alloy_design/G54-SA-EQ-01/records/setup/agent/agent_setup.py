"""Construct fresh CALPHAD agents; tasks and reference answers do not belong here."""
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
import json

from .baseline_support import HERE, PYTHON, observing_executor
from .runtime import REPO, embedding_path


def preparation():
    """The single tested preparation; no task questions or reference answers."""
    config = json.loads((HERE / 'agent/config.json').read_text())
    return {'prompt': config['prompt'], 'corpus': config['corpus'],
            'prompt_path': HERE / config['prompt_path'],
            'corpus_paths': [HERE / p for p in config['corpus_paths']]}


def create_agents(run, selected):
    """Preserve the original agents, tool signature, retrieval behavior and limits."""
    import autogen
    import chromadb
    from autogen import AssistantAgent
    from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

    settings = run.state
    assistant = AssistantAgent(
        name="Assistant", system_message=selected["prompt_path"].read_text(),
        description="Senior CALPHAD expert who can use TC-python to solve materials science problems.",
        llm_config={"timeout": settings["assistant_timeout_seconds"],
                    "cache_seed": settings["cache_seed"], "config_list": run.llm_config_list(),
                    "parallel_tool_calls": False},
    )

    def termination_msg(message):
        return isinstance(message, dict) and str(message.get("content", ""))[-9:].upper() == "TERMINATE"

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    proxy_rag = RetrieveUserProxyAgent(
        name="Proxy_w_RAG", is_termination_msg=termination_msg, human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.", max_consecutive_auto_reply=10,
        retrieve_config={"task": "code", "docs_path": [str(p) for p in selected["corpus_paths"]],
                         "chunk_token_size": 1000, "model": "gpt-4o-2024-08-06",
                         "embedding_function": SentenceTransformerEmbeddingFunction(model_name=str(embedding_path())),
                         "client": chromadb.PersistentClient(path=str(run.vector_dir) + "/"),
                         "overwrite": True, "get_or_create": False},
        code_execution_config=False,
        description="Assistant who has extra content retrieval power for solving difficult problems.",
    )
    environment = SimpleNamespace(env_dir=str(PYTHON.parent.parent), env_name=str(PYTHON.parent.parent),
                                  bin_path=str(PYTHON.parent), env_exe=str(PYTHON))
    proxy_user = autogen.UserProxyAgent(
        name="User_Proxy", is_termination_msg=termination_msg, human_input_mode="NEVER", llm_config=False,
        max_consecutive_auto_reply=settings["user_proxy_max_consecutive_auto_reply"],
        code_execution_config={"executor": observing_executor(run)(
            virtual_env_context=environment, work_dir=str(run.code_dir) + "/",
            timeout=settings["code_timeout_seconds"])},
    )

    def retrieve_content(
        message: Annotated[str, "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering."],
        n_results: Annotated[int, "number of results, choose 6"] = 6,
    ) -> str:
        proxy_rag.n_results = n_results
        response = proxy_rag.message_generator(proxy_rag, None, {"problem": message, "n_results": n_results})
        run.retrieval(message, n_results, response or message, proxy_rag._current_docs_in_context)
        return response or message

    registered = assistant.register_for_llm(
        description="retrieve content for code generation and question answering.", api_style="tool"
    )(retrieve_content)
    proxy_user.register_for_execution()(registered)

    import sys
    from autogen.tools import Tool
    sys.path.insert(0, str(REPO / 'experiments/management_alloy_design'))
    from scientific_tools import AlloyDesignTools
    science = AlloyDesignTools(run.results / 'search')
    for name in ('evaluate_coarsening', 'evaluate_phase_stability', 'evaluate_solidification'):
        method = getattr(science, name)
        tool = Tool(name=name, description=method.__doc__, func_or_tool=method)
        assistant.register_for_llm()(tool)
        tool.register_for_execution(proxy_user)
    run.state['scientific_tools'] = ['evaluate_coarsening', 'evaluate_phase_stability', 'evaluate_solidification']
    run.state['offered_tools'] = assistant.llm_config['tools']
    run.save()

    def reset():
        for agent in (assistant, proxy_rag, proxy_user):
            agent.reset()

    if hasattr(run, "observe_client"):
        run.observe_client(assistant.client)
        run._agents = (assistant, proxy_user)
    return SimpleNamespace(assistant=assistant, proxy_user=proxy_user, proxy_rag=proxy_rag,
                           reset=reset, retrieve_content=retrieve_content, science=science)
