"""Reuse the executed equilibrium/Scheil retrieval wrapper for role references.

Source: experiments/tool_using_demo_reproduction/agent_setup.py:create_agents.
No DocReaderAgent dependency, new search algorithm, or answering model.
"""
import json
import os
from pathlib import Path
from typing import Annotated

HERE = Path(__file__).resolve().parent
ROLE_NAMES = {"planner.md": "planner", "reviewer.md": "critic"}
REFERENCE_PROMPT = """Reference material for your question: {input_question}

{input_context}

Apply relevant guidance to the current task and evidence. These passages are
method references, not a verdict or a replacement for your role instructions.
Retain your required tool schema and correction/decision fields. If the material
does not resolve the question, identify the remaining uncertainty or refine the
retrieval query."""


def make_retrieval(guide, model):
    """Return the original-style retrieval function, scoped to one role/session."""
    import chromadb
    from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

    role = ROLE_NAMES[guide]
    sessions = {}

    def context():
        from .tools import _MODE
        mode = _MODE.get()
        if mode not in ('plan', 'execute', 'review'):
            raise ValueError('Reference consultation requires an active delegation mode')
        run = Path(os.environ["ALLOYGEN_G1_OUTPUT"]).resolve()
        session = os.environ["ALLOYGEN_RAG_SESSION"]
        if not run.is_relative_to(HERE.parent / "runs") or not session.replace("_", "").isalnum():
            raise ValueError("Use a contained run and an explicit retrieval session")
        folder = run / "retrieval" / session / role
        if str(folder) in sessions:
            if sessions[str(folder)][2] != mode:
                raise ValueError('A retrieval session cannot change its delegation mode')
            return sessions[str(folder)]
        folder.mkdir(parents=True, exist_ok=False)
        docs = [HERE / 'references' / role / (name + '.md') for name in ('common', mode)]
        if not all(p.is_file() for p in docs):
            raise ValueError('Common and current-mode reference documents are required')
        client = chromadb.PersistentClient(path=str(folder / "vector_db"))
        proxy_rag = RetrieveUserProxyAgent(
            name=role.capitalize() + "_Reference_Retriever",
            human_input_mode="NEVER", code_execution_config=False,
            retrieve_config={
                "task": "qa", "docs_path": [str(p) for p in docs],
                "chunk_token_size": 1000, "model": model,
                "client": client,
                # Installed AG2's VectorDB path uses db_config, not the legacy client
                # field alone. Both refer to the same demo-style Chroma client.
                "db_config": {"client": client, "path": str(folder / "vector_db")},
                "collection_name": role + "_codebook", "overwrite": True,
                "get_or_create": False, "customized_prompt": REFERENCE_PROMPT,
            },
        )

        sessions[str(folder)] = (proxy_rag, folder, mode)
        return proxy_rag, folder, mode

    def retrieve_content(
        message: Annotated[str, "A precise question about the method or interpretation needed for your current decision."],
        n_results: Annotated[int, "Number of reference passages; start with 2."] = 2,
    ) -> str:
        if not message.strip() or not 1 <= n_results <= 6:
            raise ValueError("Ask a nonempty question and request 1 to 6 passages")
        proxy_rag, folder, mode = context()
        # Same function-call -> message_generator path as both executed demos.
        proxy_rag.n_results = n_results
        response = proxy_rag.message_generator(
            proxy_rag, None, {"problem": message, "n_results": n_results})
        if not response or response.strip() == "TERMINATE":
            response = "No relevant reference material was retrieved. Refine the question or state what remains unresolved."
        response = f'Active delegation mode: {mode}.\n' + response
        records = folder / "queries.json"
        rows = json.loads(records.read_text()) if records.exists() else []
        rows.append(dict(query=message, n_results=n_results, role=role, mode=mode,
                         reference_pool=['common', mode],
                         sources=list(proxy_rag._current_docs_in_context), response=response))
        records.write_text(json.dumps(rows, indent=2) + "\n")
        return response

    retrieve_content.__name__ = "consult_" + role + "_codebook"
    retrieve_content.__doc__ = ("Search your " + role +
        " method references. Ask the information question needed for your current decision; "
        "apply the returned guidance to measured evidence. This tool neither calculates physics nor chooses an alloy.")
    return retrieve_content
