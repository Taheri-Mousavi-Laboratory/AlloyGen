from autogen import ConversableAgent
from autogen.llm_config import LLMConfig

from typing import Any, Literal
# ++
from .web_search_preview import WebSearchPreviewTool

__all__ = ["WebSearchAgent"]

# for tools
inst_for_WebTool = """
Search internet for reiable information that answer the query.
Summarize what you find in clear, precise manner, include key results as well as sources.
- do not give up easily, try different ways if needed;
"""

class WebSearchAgent(ConversableAgent):
    """An agent that uses search tools from openai to interact with the web."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        web_tool_llm_config: LLMConfig | dict[str, Any] | None = None,
        web_tool: Literal[
            "browser_use", "TBA",
        ] = "browser_use",
        web_tool_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_config = LLMConfig.get_current_llm_config(llm_config)  # type: ignore[arg-type]
        web_tool_kwargs = web_tool_kwargs if web_tool_kwargs else {}
        web_tool_llm_config = web_tool_llm_config if web_tool_llm_config else llm_config
        
        # 1. for tools
        if web_tool == "WebSearchPreviewTool":
            
            self.tool = WebSearchPreviewTool(
                llm_config= web_tool_llm_config,
                instructions= inst_for_WebTool,
                user_location= None,
                text_format= None,
                **web_tool_kwargs
                # # 
                # search_context_size= 'medium',
                # user_location= None,
                # # ++
                # text_format= None,
            )
        else:
            raise ValueError(f"Unsupported {web_tool=}.")
    
        # 2. for the agent
        super().__init__(llm_config=llm_config, **kwargs)
    
        # 3. register the tool
        self.register_for_llm()(self.tool)
            
    
    
    

    