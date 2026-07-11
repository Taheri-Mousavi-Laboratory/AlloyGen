from autogen import (
    UserProxyAgent,
    ConversableAgent,
    AssistantAgent
)
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.llm_config import LLMConfig

from typing import (
    Annotated, 
    Any, 
    Literal, 
    Optional, 
    Type, 
    Union,
    List,
)

import chromadb

# helpers
# functions

def termination_msg(x):
    return isinstance(x, dict) and str(x.get("content", ""))[-9:].upper() == "TERMINATE"
    
# class DocReaderAgent(ConversableAgent):
class DocReaderAgent(AssistantAgent):
    """An agent that read customized database via files"""

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        tool_llm_config: LLMConfig | dict[str, Any] | None = None,
        # ++
        # on RAG
        docs_path: List | None = None,
        vedb_dir: str | None = None,
        rag_tool: Literal["rag", "TBA"] = "rag",
        
        **kwargs: Any,
    ) -> None:
        """Initialize the angent
        Args:
            
        """
        # 1. get the agent
        llm_config = LLMConfig.get_current_llm_config(llm_config)  # type: ignore[arg-type]
        tool_llm_config = tool_llm_config if tool_llm_config else llm_config
        
        super().__init__(llm_config=llm_config, **kwargs)

        # 2. set things up
        # need a proxy_rag_retrieve_config
        proxy_rag_retrieve_config={
            "task": "code",
            "docs_path": docs_path,
            "chunk_token_size": 1000,
            "model": tool_llm_config["config_list"][0]["model"], # tool_llm_config.config_list[0]["model"],
            # for vector-database
            "client": chromadb.PersistentClient(
                path='vector_db/', # vedb_dir
            ),
            "overwrite": True,  # set to True if you want to overwrite an existing collection
            "get_or_create": False, # True, # False,  # set to False if don't want to reuse an existing collection
        }
        
        # need an proxy to support RAG
        self.proxy_rag = RetrieveUserProxyAgent(
            name="Proxy_w_RAG",
            is_termination_msg= termination_msg,
            human_input_mode="NEVER",
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=10, # 3,
            # 
            retrieve_config=proxy_rag_retrieve_config,
            # 
            code_execution_config=False,  # we don't want to execute code in this case.
            description="Assistant who has extra content retrieval power for solving difficult problems.",
        )

        def retrieve_content(
            message: Annotated[
                str,
                "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
            ],
            n_results: Annotated[int, "number of results, choose 6"] = 6,
        ) -> str:
            
            self.proxy_rag.n_results = n_results  # Set the number of results to be retrieved.
            _context = {"problem": message, "n_results": n_results}
            ret_msg = self.proxy_rag.message_generator(
                self.proxy_rag, 
                None, 
                _context
            )
            return ret_msg or message

        if rag_tool == "rag":
            self.tool = retrieve_content 
            # Note that, above is a function, not a tool yet
        else:
            raise ValueError(f"Unsupported {rag_tool=}.")

        # =======
        self.tool_register_handle = self.register_for_llm(
            description="retrieve content for code generation and question answering.", 
            api_style="tool", # "function"
        )(self.tool)

                
        
        