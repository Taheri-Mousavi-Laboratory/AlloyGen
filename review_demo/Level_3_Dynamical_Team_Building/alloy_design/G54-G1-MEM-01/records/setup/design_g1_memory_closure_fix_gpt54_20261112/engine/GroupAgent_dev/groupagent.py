# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import os
from typing import Any, Callable, Literal, Optional, Union

from termcolor import colored

# # --
# from .... import GroupChat, GroupChatManager, UserProxyAgent
# from ....doc_utils import export_module
# from ...conversable_agent import ConversableAgent
# ++
from autogen import GroupChat, GroupChatManager, UserProxyAgent
from autogen.doc_utils import export_module
from autogen.agentchat import ConversableAgent

from .agent_builder import AgentBuilder
from .tool_retriever import ToolBuilder, format_ag2_tool, get_full_tool_description

# 
# ++
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# for debug pins
# Debug_Level=2
from .para import Debug_Level
if Debug_Level>1:
    print (f"Debug mode: {Debug_Level}")
    
from termcolor import colored, cprint
# def color_print(lines):
#     cprint(
#         lines,
#         "red", "on_yellow"
#     )
def color_print(
    lines,
    font_color='red',
):
    cprint(
        lines,
        font_color, "on_yellow"
    )
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# @export_module("autogen.agentchat.contrib.captainagent")
@export_module("autogen.agentchat.contrib.groupagent")
class GroupAgent(ConversableAgent):
    """(In preview) Group Agent, designed to solve a modeling research tasks with an agent or a group of agents."""
    """Based on Captian Agent framework"""

    DEFAULT_NESTED_CONFIG = {
        "autobuild_init_config": {
            "config_file_or_env": "OAI_CONFIG_LIST",
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 2048},
            "code_execution_config": {
                "timeout": 300,
                "work_dir": "groupchat",
                "last_n_messages": 1,
                "use_docker": False,
            },
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": None,
        "max_turns": 5,
    }

    AUTOBUILD_TOOL = {
        "type": "function",
        "function": {
            "name": "seek_experts_help",
            "description": """Build a group of experts and let them chat with each other in a group chat.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string", "description": "[REQUIRED] Name of the group."},
                    "building_task": {
                        "type": "string",
                        "description": """Instructions that help a build manager to build a group of experts.""",
                    },
                    "execution_task": {
                        "type": "string",
                        "description": """[REQUIRED] The task that needs the experts to solve by conversation.""",
                    },
                    "execute_intent": {
                        "type": "string", "enum": ["calculate", "nominate"],
                        "description": "Only in execute mode: calculate (default) executes a ready recorded action; nominate explicitly records an eligible final choice. Missing plans never imply nomination.",
                    },
                    "mode": {
                        "type": "string", "enum": ["plan", "execute", "review"],
                        "description": "Current delegation: plan one next calculation; execute that calculation; review its saved evidence.",
                    },
                },
                "required": ["group_name", "execution_task", "mode"],
            },
        },
    }

    AUTOBUILD_SYSTEM_MESSAGE = """# Your role
You are a perfect manager of a group of advanced experts.

# How to solve the task
When a task is assigned to you, FIRST explicitly:
1. Analysis of its constraints and conditions for completion.
2. Respond with a specific plan of how to solve the task.

After that, you can solve the task in two ways:
- Delegate the resolution of tasks to other experts created by seeking a group of experts for help and derive conclusive insights from their conversation summarization.
- Analysis and solve the task with your coding and language skills.

# How to seek experts help
The tool "seek_experts_help" can build a group of experts according to the building_task and let them chat with each other in a group chat to solve the execution_task you provided.
- This tool will summarize the essence of the experts' conversation and the derived conclusions.
- You should not modify any task information from meta_user_proxy, including code blocks, but you can provide extra information.
- Within a single response, you are limited to initiating one group of experts.

## building_task
This task helps a build manager to build a group of experts for your task.
You should suggest less then five roles (including a web searcher for getting up-to-date research and debug info and a checker for verification) with the following format.

### Format
- [Detailed description for web searcher]
- [Detailed description for role 1]
- [Detailed description for role 2]
- [Detailed description for role 3]
- [Detailed description for checker]

## execution_task
This is the task that needs the experts to solve by conversation.
You should Provide the following information in markdown format.

### Format
## Task description
...
## Plan for solving the task
...
## Output format
...
## Constraints and conditions for completion
...
## [Optional] results (including code blocks) and reason from last response
...

# After seek_experts_help
You will receive a comprehensive conclusion from the conversation, including the task information, results, reason for the results, conversation contradiction or issues, and additional information.
You **must** conduct a thorough verification for the result and reason's logical compliance by leveraging the step-by-step backward reasoning with the same group of experts (with the same group name) when:
- The conversation has contradictions or issues (need double-check marked as yes), or
- The result is different from the previous results.

Note that the previous experts will forget everything after you obtain the response from them. You should provide the results (including code blocks) you collected from the previous experts' response and put it in the new execution_task.

# Some useful instructions
- You only have one tool called "seek_experts_help".
- Provide a answer yourself after "seek_experts_help".
- You should suggest python code in a python coding block (```python...```). If you need to get the value of a variable, you must use the print statement.
- When using code, you must indicate the script type in the code block.
- Do not suggest incomplete code which requires users to modify.
- Be clear about which step uses code, which step uses your language skill, and which step to build a group chat.
- If the code's result indicates there is an error, fix the error and output the whole code again.
- If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
- Include verifiable evidence in your response if possible.
- After completing all tasks and verifications, you should conclude the operation and reply "TERMINATE"
"""
    AUTOBUILD_SYSTEM_MESSAGE_ori = """# Your role
You are a perfect manager of a group of advanced experts.

# How to solve the task
When a task is assigned to you:
1. Analysis of its constraints and conditions for completion.
2. Respond with a specific plan of how to solve the task.

After that, you can solve the task in two ways:
- Delegate the resolution of tasks to other experts created by seeking a group of experts for help and derive conclusive insights from their conversation summarization.
- Analysis and solve the task with your coding and language skills.

# How to seek experts help
The tool "seek_experts_help" can build a group of experts according to the building_task and let them chat with each other in a group chat to solve the execution_task you provided.
- This tool will summarize the essence of the experts' conversation and the derived conclusions.
- You should not modify any task information from meta_user_proxy, including code blocks, but you can provide extra information.
- Within a single response, you are limited to initiating one group of experts.

## building_task
This task helps a build manager to build a group of experts for your task.
You should suggest less then three roles (including a checker for verification) with the following format.

### Format
- [Detailed description for role 1]
- [Detailed description for role 2]
- [Detailed description for checker]

## execution_task
This is the task that needs the experts to solve by conversation.
You should Provide the following information in markdown format.

### Format
## Task description
...
## Plan for solving the task
...
## Output format
...
## Constraints and conditions for completion
...
## [Optional] results (including code blocks) and reason from last response
...

# After seek_experts_help
You will receive a comprehensive conclusion from the conversation, including the task information, results, reason for the results, conversation contradiction or issues, and additional information.
You **must** conduct a thorough verification for the result and reason's logical compliance by leveraging the step-by-step backward reasoning with the same group of experts (with the same group name) when:
- The conversation has contradictions or issues (need double-check marked as yes), or
- The result is different from the previous results.

Note that the previous experts will forget everything after you obtain the response from them. You should provide the results (including code blocks) you collected from the previous experts' response and put it in the new execution_task.

# Some useful instructions
- You only have one tool called "seek_experts_help".
- Provide a answer yourself after "seek_experts_help".
- You should suggest python code in a python coding block (```python...```). If you need to get the value of a variable, you must use the print statement.
- When using code, you must indicate the script type in the code block.
- Do not suggest incomplete code which requires users to modify.
- Be clear about which step uses code, which step uses your language skill, and which step to build a group chat.
- If the code's result indicates there is an error, fix the error and output the whole code again.
- If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
- Include verifiable evidence in your response if possible.
- After completing all tasks and verifications, you should conclude the operation and reply "TERMINATE"
"""

    DEFAULT_DESCRIPTION = "A helpful AI assistant that can build a group of agents at a proper time to solve a task."

    # This is used to prompt the LLM to summarize the conversation history between CaptainAgent/GroupAgent's tool execution history
    DEFAULT_SUMMARY_PROMPT = "Read the following conversation history between an expert and a group of agent experts, summarize the conversation history. Your summarization should include the initial task, the experts' plan and the attempt, finally the results of the conversation. If the experts arrived at a conclusion, state it as it is without any modification."

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[dict[str, Any], Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = False,
        nested_config: Optional[dict[str, Any]] = None,
        agent_lib: Optional[str] = None,
        tool_lib: Optional[str] = None,
        agent_config_save_path: Optional[str] = None,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        **kwargs: Any,
    ):
        """Args:
        name (str): agent name.
        system_message (str): system message for the ChatCompletion inference.
            Please override this attribute if you want to reprogram the agent.
        llm_config (dict): llm inference configuration.
            Please refer to [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create) for available options.
        is_termination_msg (function): a function that takes a message in the form of a dictionary
            and returns a boolean value indicating if this received message is a termination message.
            The dict can contain the following keys: "content", "role", "name", "function_call".
        max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
            default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
            The limit only plays a role when human_input_mode is not "ALWAYS".
        agent_lib (str): the path or a JSON file of the agent library for retrieving the nested chat instantiated by CaptainAgent/GroupAgent.
        tool_lib (str): the path to the tool library for retrieving the tools used in the nested chat instantiated by CaptainAgent/GroupAgent.
        nested_config (dict): the configuration for the nested chat instantiated by CaptainAgent/GroupAgent.
            A full list of keys and their functionalities can be found in [docs](https://docs.ag2.ai/docs/topics/captainagent/configurations).
        agent_config_save_path (str): the path to save the generated or retrieved agent configuration.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](https://github.com/ag2ai/ag2/blob/main/autogen/agentchat/conversable_agent.py#L74).
        """
        super().__init__(
            name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )
        # ++
        color_print(f"Debug level: {Debug_Level}")
        if Debug_Level>1:
            color_print(f"Initialize the GroupAgent class instance....")
            color_print(f"Check inputs...")
            print (f"name: {name}")
            print (f"is_termination_msg: {is_termination_msg}")
            print (f"max_consecutive_auto_reply: {max_consecutive_auto_reply}")
            print (f"human_input_mode: {human_input_mode}")
            print (f"code_execution_config: {code_execution_config}")
            print (f"llm_config: {llm_config}")
            print (f"description: {description}")
            color_print(f"="*50)
            

        if system_message is None:
            system_message = self.AUTOBUILD_SYSTEM_MESSAGE
        # ++
        if Debug_Level>1:
            print (f"self.AUTOBUILD_SYSTEM_MESSAGE: {self.AUTOBUILD_SYSTEM_MESSAGE}")
            print (f"system_message: {system_message}")
            color_print(f"----------------------------------\n\n")
            
        nested_config = self._update_config(
            copy.deepcopy(self.DEFAULT_NESTED_CONFIG), # keep optional runtime hooks instance-local
            nested_config               # to be updated
        )
        # ++
        if Debug_Level>1:
            color_print (
                f"self.DEFAULT_NESTED_CONFIG: {self.DEFAULT_NESTED_CONFIG,}",
                font_color='blue'
            )
            color_print (
                f"after upate, '': {nested_config}\n\n",
                font_color='blue'
            )
            
        if nested_config["group_chat_llm_config"] is None:
            nested_config["group_chat_llm_config"] = llm_config.copy()
            
        if agent_lib:
            nested_config["autobuild_build_config"]["library_path_or_json"] = agent_lib
            # ++
            if Debug_Level>1:
                color_print (f"Get agent_lib:\n {agent_lib}\n")
        if tool_lib:
            if "autobuild_tool_config" not in nested_config:
                nested_config["autobuild_tool_config"] = {}
            nested_config["autobuild_tool_config"]["tool_root"] = tool_lib
            # ++
            if Debug_Level>1:
                color_print (f"Get tool_lib:\n {tool_lib}\n")
        # ++
        if Debug_Level>1:
            color_print (f"After update, ")
            print (f"nested_config: {nested_config}")
            print ()
            # print (f"agent_lib: {nested_config["autobuild_build_config"]["library_path_or_json"]}")
            color_print(f"----------------------------------")


        # ++
        if Debug_Level>1:
            color_print(f"Now, build top-level agents...")
            
        # self.assistant = ConversableAgent(name="CaptainAgent", system_message=system_message, llm_config=llm_config)
        manager_llm_config = copy.deepcopy(llm_config)
        if nested_config.get('sequential_delegation'):
            manager_llm_config['parallel_tool_calls'] = False
        delegation_tool = copy.deepcopy(self.AUTOBUILD_TOOL)
        if nested_config.get('controlled_review'):
            delegation_tool['function']['description'] += (
                ' The configured review stages run inside this team. Read its working '
                'findings and separate review statuses before requesting another team '
                'for specific unfinished work; do not schedule a duplicate general review.')
        self.assistant = ConversableAgent(
            name="GroupAgent", # "CaptainAgent", 
            system_message=system_message, 
            llm_config=manager_llm_config
        )
        self.assistant.update_tool_signature(
            delegation_tool,
            is_remove=False
        )
        # ++
        if Debug_Level>1:
            print (f"Use ConversableAgent to build self.assistant: {self.assistant} \n\n")

        # self.executor = CaptainUserProxyAgent(
        self.executor = GroupAgentUserProxyAgent(
            name="Team_builder", # "Expert_summoner",
            nested_config=nested_config,
            agent_config_save_path=agent_config_save_path,
            is_termination_msg=lambda x: x.get("content", "") and "terminate" in x.get("content", "").lower(),
            code_execution_config=code_execution_config,
            human_input_mode="NEVER",
        )
        memory = nested_config.get('task_memory')
        if memory is not None:
            memory.install_manager(self.assistant, self.executor)
        # ++
        if Debug_Level>1:
            # color_print (f"Use GroupAgentUserProxyAgent to build Expert_summoner: {self.executor}\n\n")
            color_print (f"Use GroupAgentUserProxyAgent to build Team_builder: {self.executor}\n\n")

        # ++
        if Debug_Level>1:
            color_print (f"now, register nested chats: from .executor to .assistant...\n\n")
        self.register_nested_chats(
            [
                {
                    "sender": self.executor,
                    "recipient": self.assistant,
                    "max_turns": nested_config["max_turns"],
                    "summary_method": nested_config.get("final_summary_method", "reflection_with_llm"),
                    "summary_args": {
                        "summary_prompt": self.DEFAULT_SUMMARY_PROMPT,
                    },
                }
            ],
            trigger=UserProxyAgent,
            position=0,
        )

    @staticmethod
    def _update_config(default_dict: dict[str, Any], update_dict: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Recursively updates the default_dict with values from update_dict."""
        if update_dict is None:
            return default_dict

        for key, value in update_dict.items():
            default_value = default_dict.get(key)
            if isinstance(default_value, dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                # default_dict[key] = CaptainAgent._update_config(default_value, value)
                default_dict[key] = GroupAgent._update_config(default_value, value)
            else:
                # Update the value or add new key
                default_dict[key] = value

        return default_dict


# class CaptainUserProxyAgent(ConversableAgent):
class GroupAgentUserProxyAgent(ConversableAgent):
    """(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."""

    CONVERSATION_REVIEW_PROMPT = """# Your task
Briefly summarize the conversation history derived from an experts' group chat by following the answer format.
If you found non-trivial errors or issues in the conversation, point it out with a detailed reason, if you think it is worth further verification, mark the "Need double-check" as "Yes"
If you find the conversation ends with TERMINATE and the task is solved, this is normal situation, you can mark the "Need double-check" as "No".

# Conversation history:
{chat_history}

# Answer format
## Task
...

## Results
...

## Reason for the results
...

## Errors or issues in the conversation
...

### Need to double-check?
[Yes or No]

## Additional information (file path, code blocks, url, etc.)
...
"""

    AUTOBUILD_TASK_DESC = """You are given: (1) a task and advises from your manager with a specific plan and (2) a general task.
Collect information from the general task, follow the suggestions from manager to solve the task.

# General Task
{general_task}

# Task and suggestions from manager
{manager_task} """

    TEAM_HANDOFF_INSTRUCTION = """Complete the assigned subtask using this team's capabilities.
The general task provides context; it is not a requirement to finish the entire
investigation within this delegation. Return completed work, relevant proposal
and calculation references, remaining work and any missing capability to the
manager. Do not repeatedly read unchanged evidence when the needed operation is
unavailable. A supported subtask result or a specific capability gap can conclude
this group after critic review. Team completion is distinct from overall completion.
"""

    MODE_INSTRUCTIONS = {
        'plan': 'Selected physics specialists first interpret method knowledge or measured comparisons and uncertainties. The planner then chooses one informative next action and calls record_design_plan. On the first plan, record the requirements map and workflow; later retain them unless an explicit change is needed. The critic checks the evidence, hypothesis and chosen measurement. Code derives fixed/varied composition metadata. Use the active batch limit. Do not calculate physics or nominate final alloys.',
        'execute': 'Execute only the assigned calculation, refinement or selection using a recorded plan. Return actual evidence and limitations; do not expand to the entire mission.',
        'review': 'Assess only the assigned saved evidence. Fresh verification is permitted when explicitly assigned. Return missing search/calculation work to the manager; do not perform a new search.',
    }

    def _shared_evidence(self, mission=False):
        provider = self._nested_config.get('shared_evidence_provider')
        if provider is None:
            return ''
        return '\n\n# Shared evidence from saved records (not a scientific verdict)\n' + json.dumps(
            provider(mode='mission') if mission else provider(), ensure_ascii=False)

    def _selection_task(self, building_task, execution_task):
        """Choose capabilities for this assignment; workers retain mission constraints."""
        if not self._nested_config.get('controlled_review'):
            memory = self._nested_config.get('task_memory')
            return building_task + (memory.context(audience='builder') if memory is not None else '')
        parts = ['# Team-building brief\n' + building_task,
                 '# Current assignment\n' + execution_task,
                 '# Delegation mode: ' + self._active_mode + '\n' + self.MODE_INSTRUCTIONS[self._active_mode]]
        provider = self._nested_config.get('shared_evidence_provider')
        evidence = provider() if provider else None
        selection_context = evidence.get('selection_context') if isinstance(evidence, dict) else None
        if selection_context is not None:
            parts.append('# Current measured status and task constraints\n'
                         + json.dumps(selection_context, ensure_ascii=False))
        parts.append('Select only capabilities needed for the current assignment. Do not select '
                     'all library members to cover future stages. In plan mode select planning '
                     'and any explicitly needed method advice; in execute mode select the '
                     'assigned calculation/selection capabilities. The configured reviewer is '
                     'required, but does not supply missing calculation tools. Full mission '
                     'constraints are delivered to the selected team. Working membership remains your choice.')
        operation_provider = self._nested_config.get('required_operation_provider')
        if operation_provider:
            parts.append('The selected team must expose this operation in the current mode: '
                         + str(self._active_operation))
        memory = self._nested_config.get('task_memory')
        if memory is not None:
            parts.append(memory.context(audience='builder'))
        return '\n\n'.join(parts)

    def _delegation_message(self, general_task, execution_task, agent_list):
        if self._nested_config.get('controlled_review'):
            context = self._nested_config.get('delegation_context', general_task)
            message = '# Assigned subtask\n' + execution_task
            message += '\n\n# Delegation mode: ' + self._active_mode + '\n' + self.MODE_INSTRUCTIONS[self._active_mode]
            scope = self._nested_config.get('delegation_scope')
            if scope:
                message += '\n\n# Mission background (not actions due now)\n' + scope
            message += '\n\n# Full task constraints (background)\n' + context
            if self._nested_config.get('review_material') and not (
                    self._active_mode == 'review' and self._nested_config.get('evidence_first_review')):
                message += ('\n\n# Material assigned for review (unverified agent claims)\n'
                            + self._nested_config['review_material'])
            if (self._nested_config.get('share_team_findings') and self._nested_config.get('include_previous_team_findings', True)
                    and getattr(self, '_last_team_result', None) and
                    getattr(self, '_last_team_scope', None) == (scope or general_task)):
                from .review_delivery import previous_team_context
                message += '\n\n# Previous team findings (claims to check, not independent validation)\n'
                message += json.dumps(previous_team_context(self._last_team_result), ensure_ascii=False)
        else:
            message = self.AUTOBUILD_TASK_DESC.format(general_task=general_task, manager_task=execution_task)
        if self._nested_config.get('team_handoff'):
            capabilities = {a.name: [t['function']['name'] for t in a.llm_config.get('tools', [])]
                            for a in agent_list if a.llm_config}
            message += '\n\n# Delegation and return\n' + self.TEAM_HANDOFF_INSTRUCTION
            message += '\nActual tool access in this team:\n' + json.dumps(capabilities)
        memory = self._nested_config.get('task_memory')
        # The closing reviewer keeps the assignment before the shared-evidence
        # section. Place memory there so existing compaction preserves it too.
        read_available = any(not a.llm_config and 'read_task_memory' in a.function_map for a in agent_list)
        return message + (memory.context(audience='team', read_available=read_available)
                          if memory is not None else '') + self._shared_evidence()

    DEFAULT_AUTO_REPLY = (
        "Use seek_experts_help to delegate required unfinished work when the available capabilities and allowances permit it. "
        "A proposed next step or missing deliverable is not itself a stopping reason. "
        "If the task is complete or further progress is blocked, state the supported result, "
        "specific blocker and remaining limitations, then append TERMINATE; do not return only the token."
    )

    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that can running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks), or the conversation history and result of a group of agents",
    }

    def __init__(
        self,
        name: str,
        nested_config: dict[str, Any],
        agent_config_save_path: str = None,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, dict[str, Any]]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[dict[str, Any], Literal[False]]] = False,
        system_message: Optional[Union[str, list]] = "",
        description: Optional[str] = None,
    ):
        """Args:
        name (str): name of the agent.
        nested_config (dict): the configuration for the nested chat instantiated by CaptainAgent/GroupAgent.
        is_termination_msg (function): a function that takes a message in the form of a dictionary
            and returns a boolean value indicating if this received message is a termination message.
            The dict can contain the following keys: "content", "role", "name", "function_call".
        max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
            default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
            The limit only plays a role when human_input_mode is not "ALWAYS".
        human_input_mode (str): whether to ask for human inputs every time a message is received.
            Possible values are "ALWAYS", "TERMINATE", "NEVER".
            (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                Under this mode, the conversation stops when the human input is "exit",
                or when is_termination_msg is True and there is no human input.
            (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                the number of auto reply reaches the max_consecutive_auto_reply.
            (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
        code_execution_config (dict or False): config for the code execution.
            To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
            - work_dir (Optional, str): The working directory for the code execution.
                If None, a default working directory will be used.
                The default working directory is the "extensions" directory under
                "path_to_autogen".
            - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                If a list or a str of image name(s) is provided, the code will be executed in a docker container
                with the first image successfully pulled.
                If False, the code will be executed in the current environment.
                We strongly recommend using docker for code execution.
            - timeout (Optional, int): The maximum execution time in seconds.
            - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
        default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
        llm_config (dict or False): llm inference configuration.
            Please refer to [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create)
            for available options.
            Default to false, which disables llm-based auto reply.
        system_message (str or List): system message for ChatCompletion inference.
            Only used when llm_config is not False. Use it to reprogram the agent.
        description (str): a short description of the agent. This description is used by other agents
            (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
        """
        description = (
            description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
        )
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
        )
        # ++
        # check https://github.com/ag2ai/ag2/blob/4fffb72065a46732c1bcf0d2819b0333f2cde138/autogen/agentchat/conversable_agent.py#L2963
        self.register_function(
            function_map={
                "seek_experts_help": lambda **args: self._run_autobuild(**args),
            }
        )
        # ++
        if Debug_Level>1:
            color_print(f"'seek_experts_help' is function ._run_autobuild(..)\n\n")
            
        self._agent_config_save_path = agent_config_save_path
        self._nested_config = nested_config.copy()
        self._code_execution_config = code_execution_config
        self.build_history = {}
        self.tool_history = {}
        self.build_times = 0
        # ++
        if Debug_Level>1:
            color_print(f"Initialize 'GroupAgentUserProxyAgent' ...")
            color_print(f"Check input parameters:")
            print (f"From parent class: ConversableAgent...")
            print (f"._oai_messages: {self._oai_messages}")
            print (f"._oai_system_message: {self._oai_system_message}")
            print (f"._description: {self._description}")
            print (f"._is_termination_msg: {self._is_termination_msg}")
            print (f".silent: {self.silent}")
            print (f".run_executor: {self.run_executor}")
            print (f".llm_config: {self.llm_config}")
            print (f".client: {self.client}")
            print (f"...")
            print (f"From son class: ")
            print (f"._agent_config_save_path: {self._agent_config_save_path}")
            print (f"._nested_config: {self._nested_config}")
            print (f"._code_execution_config: {self._code_execution_config}")
            print (f".build_history: {self.build_history}")
            print (f".tool_history: {self.tool_history}")
            print (f".build_times: {self.build_times}")
            

    def _run_autobuild(
        self, group_name: str, execution_task: str, mode: str, building_task: str = "", execute_intent: str = None
    ) -> str:
        """Scope both new and reused teams, including the registered tool executor."""
        if mode not in self.MODE_INSTRUCTIONS:
            raise ValueError('Choose delegation mode: plan, execute or review')
        provider = self._nested_config.get('required_operation_provider')
        # Validate the assignment before constructing/selecting even a reused team.
        # One explicit intent field avoids inferring nomination from candidate eligibility.
        if execute_intent is not None and (mode != 'execute' or execute_intent not in ('calculate', 'nominate')):
            raise ValueError('Use execute_intent=calculate or nominate only in execute mode')
        operation = (provider(mode, execute_intent=execute_intent) if execute_intent is not None
                     else provider(mode)) if provider else None
        previous = getattr(self, '_active_mode', None)
        previous_operation = getattr(self, '_active_operation', None)
        self._active_mode, self._active_operation = mode, operation
        memory = self._nested_config.get('task_memory')
        round_id = None
        try:
            if memory is not None:
                round_id = memory.begin_round(execution_task, dict(group_name=group_name,
                    mode=mode, building_task=building_task, execute_intent=execute_intent))
            try:
                with self._nested_config['delegation_context_manager'](mode, **({'execute_intent': execute_intent} if execute_intent is not None else {})):
                    result = self._run_autobuild_scoped(group_name, execution_task, building_task)
            except BaseException as exc:
                try:
                    if memory is not None and round_id is not None:
                        memory.finish_round(round_id, dict(error_type=type(exc).__name__,
                            note='Delegation interrupted; inspect the run trace and saved tool evidence. No completed report.'),
                            status='interrupted')
                except BaseException as archive_error:
                    exc.add_note('Working-memory archival also failed: ' + type(archive_error).__name__)
                raise
            # Archival errors after successful work are not team failures. A
            # durable report can repair its index on the next read/restart.
            if memory is not None:
                memory.finish_round(round_id, result)
                result = f'Source round: {round_id}. Update working memory with consequential findings and unresolved questions.\n' + result
            return result
        finally:
            self._active_mode, self._active_operation = previous, previous_operation

    def _run_autobuild_scoped(
        self, 
        group_name: str, 
        execution_task: str, 
        building_task: str = ""
    ) -> str:
        """Build a group of agents by AutoBuild to solve the task.
        This function requires the nested_config to contain the autobuild_init_config, autobuild_llm_config, group_chat_llm_config.
        """
        # ++
        if Debug_Level>1:
            color_print(f"Call _run_autobuild...", font_color='white')
            print (f"Check inputs")
            print (f"group_name: {group_name}")
            # print (f"execution_task:\n{execution_task}")
            # print (f"building_task:\n{building_task}")
            # print ("="*50)
            
        # print("==> Running AutoBuild...", flush=True)
        # print("\n==> Building task: ", building_task, flush=True)
        # print("\n==> Execution task: ", execution_task, flush=True)
        # ++
        print(colored("==> Now, build the expert team by running AutoBuild method...", "red"), flush=True)
        print(colored("\n==> Building task: ", "red"), building_task, flush=True)
        print(colored("\n==> Execution task: ", "red"), execution_task, flush=True)
        # ++
        if Debug_Level>1:
            color_print(f"create AgentBuilder\n")
            
            
        builder = AgentBuilder(**self._nested_config["autobuild_init_config"])
        # ++
        if Debug_Level>1:
            color_print(f"Check GroupAgentUserProxyAgent. 'autobuild_tool_config': ", font_color='green')
            # ++
            if 'autobuild_tool_config' in self._nested_config.keys():
                print (f"case 1: {self._nested_config['autobuild_tool_config']}\n")
            else:
                print (f"case 2: ._nested_config['autobuild_tool_config']: None\n")
            print (f"===")
                
        # if the group is already built, load from history
        reused_team = group_name in self.build_history
        if reused_team:
            # ++
            if Debug_Level>1:
                color_print (f"Find {group_name} in the .build_history, will load it from history")
                # 
            agent_list, agent_configs = builder.load(
                config_json=json.dumps(
                    self.build_history[group_name]
                ),
                user_proxy=self._nested_config['autobuild_build_config'].get('user_proxy'),
            )
            # ++
            if Debug_Level>1:
                color_print (f"After loading from history...")
                color_print (f"agent_list: {agent_list}")
                print (f"agent_configs: {agent_configs}\n\n\n")
            
                
            if self._nested_config.get("autobuild_tool_config", None) and agent_configs["coding"] is True:
                # tool library is enabled, reload tools and bind them to the agents
                tool_root_dir = self.tool_root_dir
                tool_builder = ToolBuilder(
                    corpus_root=tool_root_dir,
                    retriever=self._nested_config["autobuild_tool_config"].get("retriever", "all-mpnet-base-v2"),
                    type=self.tool_type,
                )
                for idx, agent in enumerate(agent_list):
                    if idx == len(self.tool_history[group_name]):
                        break
                    tool_builder.bind(agent, "\n\n".join(self.tool_history[group_name][idx]))
                agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
        else:
            if self._nested_config["autobuild_build_config"].get("library_path_or_json", None):
                # if the key exist, get ture back
                # this branch is on when given ""
                # ++
                if Debug_Level>1:
                    color_print(f"trigger 'Build from retrieval'...\n\n\n")
                    
                # Build from retrieval
                agent_list, agent_configs = builder.build_from_library(
                    self._selection_task(building_task, execution_task),
                    **self._nested_config["autobuild_build_config"]
                )
                # ++
                if Debug_Level>1:
                    color_print(f"Out of .build_from_library(...)\n")
                
                self.build_history[group_name] = agent_configs.copy()

                if self._nested_config.get("autobuild_tool_config", None) and agent_configs["coding"] is True:
                    skills = building_task.split("\n")
                    if len(skills) == 0:
                        skills = [building_task]

                    tool_type = "default"
                    if self._nested_config["autobuild_tool_config"].get("tool_root", "default") == "default":
                        print(colored("==> Retrieving tools...", "green"), flush=True)
                        cur_path = os.path.dirname(os.path.abspath(__file__))
                        tool_root_dir = os.path.join(cur_path, "tools")
                    elif isinstance(self._nested_config["autobuild_tool_config"].get("tool_root", "default"), list):
                        # We get a list, in this case, we assume it contains several tools for the agents
                        tool_root_dir = self._nested_config["autobuild_tool_config"]["tool_root"]
                        tool_type = "user_defined"
                    else:
                        tool_root_dir = self._nested_config["autobuild_tool_config"]["tool_root"]
                    self.tool_root_dir = tool_root_dir
                    self.tool_type = tool_type
                    # ++
                    if Debug_Level>1:
                        # color_print(f"get '.tool_root_dir': {self.tool_root_dir}")
                        color_print(f"get '.tool_root_dir': ")
                        print(f"{self.tool_root_dir}")
                        # color_print(f"get '.tool_type': {self.tool_type}")
                        color_print(f"get '.tool_type': ")
                        print(f"{self.tool_type}")
                        print ("\n\n")

                    # Retrieve and build tools based on the smilarities between the skills and the tool description
                    tool_builder = ToolBuilder(
                        corpus_root=tool_root_dir,
                        retriever=self._nested_config["autobuild_tool_config"].get("retriever", "all-mpnet-base-v2"),
                        type=tool_type,
                    )
                    if tool_type == "default":
                        for idx, skill in enumerate(skills):
                            tools = tool_builder.retrieve(skill)
                            docstrings = []
                            for tool in tools:
                                category, tool_name = tool.split(" ")[0], tool.split(" ")[1]
                                tool_path = os.path.join(tool_root_dir, category, f"{tool_name}.py")
                                docstring = get_full_tool_description(tool_path)
                                docstrings.append(docstring)
                            tool_builder.bind(agent_list[idx], "\n\n".join(docstrings))
                        # the last agent is the user proxy agent, we need special treatment
                        agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
                    else:
                        # a list containing all the tools that the agents share
                        docstrings = []
                        for tool in tool_root_dir:
                            docstrings.append(format_ag2_tool(tool))
                        for idx, agent in enumerate(agent_list):
                            if idx == len(agent_list) - 1:
                                break
                            tool_builder.bind(agent, "\n\n".join(docstrings))
                        agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)

                    # log tools
                    tool_history = self.tool_history.get(group_name, [])
                    tool_history.append(docstrings)
                    self.tool_history[group_name] = tool_history
            else:
                # ++
                if Debug_Level>1:
                    color_print (f"triger 'Build agents from stratch'...\n\n\n")
                # Build agents from scratch
                agent_list, agent_configs = builder.build(
                    self._selection_task(building_task, execution_task),
                    **self._nested_config["autobuild_build_config"]
                )
                # ++
                if Debug_Level>1:
                    color_print (f"Out of .build(...)\n")
                    
                self.build_history[group_name] = agent_configs.copy()

        if self._agent_config_save_path is not None:
            building_task_md5 = hashlib.md5(building_task.encode("utf-8")).hexdigest()
            with open(f"{self._agent_config_save_path}/build_history_{building_task_md5}.json", "w") as f:
                json.dump(self.build_history, f)

        self.build_times += 1
        # start nested chat
        if self._nested_config.get('controlled_review'):
            for agent in agent_list:
                if hasattr(agent, 'configure_delegation'):
                    agent.configure_delegation(self._active_mode)
        memory = self._nested_config.get('task_memory')
        if memory is not None:
            memory.install_readers(agent_list)
        capability_check = self._nested_config.get('team_capability_check')
        if capability_check:
            operation = self._active_operation
            capability_check(agent_list, operation)
        
        review = None
        chat_config = self._nested_config['group_chat_config'].copy()
        reviewer_name = self._nested_config.get('reviewer_name')
        if reviewer_name:
            from .review import ReviewTurns
            if reviewer_name not in [agent.name for agent in agent_list]:
                raise ValueError('Configured reviewer is absent from the selected team')
            if self._nested_config.get('controlled_review'):
                if self._nested_config.get('evidence_harness'):
                    from design_pilot.evidence_harness import EvidenceReviewTurns
                    review = EvidenceReviewTurns(agent_list, reviewer_name=reviewer_name,
                        executor_name=agent_list[-1].name,
                        auditor_name=self._nested_config.get('auditor_name'),
                        max_corrections=self._nested_config.get('max_review_corrections', 2),
                        evidence_view=self._nested_config.get('evidence_view', 'physics'),
                        required_indicators=self._nested_config.get('required_indicators'))
                else:
                    from .controlled_review import ControlledReviewTurns
                    review = ControlledReviewTurns(reviewer_name,
                        executor_name=agent_list[-1].name if agent_configs['coding'] else None,
                        max_corrections=self._nested_config.get('max_review_corrections', 2),
                        auditor_name=self._nested_config.get('auditor_name'),
                        closing_evidence_provider=self._nested_config.get('closing_evidence_provider'),
                        compact_closing_context=self._nested_config.get('compact_closing_context', False),
                        review_material=(self._nested_config.get('review_material')
                            if self._active_mode == 'review' and self._nested_config.get('evidence_first_review') else None),
                        delegation_mode=self._active_mode,
                        opening_review=self._nested_config.get('opening_review', True),
                        plan_readiness_provider=self._nested_config.get('plan_readiness_provider'))
            else:
                review = ReviewTurns(reviewer_name)
            chat_config['speaker_selection_method'] = review
        nested_group_chat = GroupChat(
            agents=agent_list,
            messages=[],
            allow_repeat_speaker=agent_list[:-1] if agent_configs["coding"] is True else agent_list,
            **chat_config,
        )
        manager = GroupChatManager(
            groupchat=nested_group_chat,
            llm_config=self._nested_config["group_chat_llm_config"],
            **({'is_termination_msg': lambda message: False} if review else {}),
        )
        key = list(self.chat_messages.keys())[0]
        general_task = self.chat_messages[key][0]["content"]
        try:
            initiator = agent_list[0]
            if review and self._nested_config.get('controlled_review'):
                # A host assignment is external input, not a specialist's own reply.
                initiator = (nested_group_chat.agent_by_name(review.executor_name) if review.executor_name
                    else UserProxyAgent(name='Task_briefing', llm_config=False, code_execution_config=False,
                        human_input_mode='NEVER', is_termination_msg=lambda message: False))
            initiator.initiate_chat(
                manager,
                message=self._delegation_message(general_task, execution_task, agent_list)
            )
        finally:
            if review and self._nested_config.get('controlled_review'):
                review.restore()
                self.review_history = getattr(self, 'review_history', []) + [
                    {'group_name': group_name, 'mode': self._active_mode, 'events': review.events, 'stop_reason': review.stop_reason,
                     'corrections': review.corrections, 'audit_reports': review.audit_reports,
                     'closing_evidence': getattr(review, 'evidence_packets', []),
                     'scientific_review_status': review.scientific_review_status(),
                     'scientific_review': review.final_review}]
        if review and self._nested_config.get('controlled_review') and self._nested_config.get('share_team_findings'):
            excluded = {review.reviewer_name, review.auditor_name, review.executor_name}
            from .review_delivery import worker_contributions, tool_activity
            worker_findings = worker_contributions(nested_group_chat.messages, excluded)
            self._last_team_scope = self._nested_config.get('delegation_scope') or general_task
            self._last_team_result = {
                'group_name': group_name, 'mode': self._active_mode, 'assigned_subtask': execution_task,
                'worker_findings': worker_findings, 'audit_reports': list(review.audit_reports),
                'tool_activity': tool_activity(nested_group_chat.messages, excluded),
                'scientific_review_status': review.scientific_review_status(),
                'scientific_review': review.final_review, 'controller_status': review.stop_reason}
            if self._active_mode == 'review' and review.evidence_packets:
                self._last_team_result['comparison_fingerprint'] = review.evidence_packets[-1].get('comparison', {}).get('fingerprint')
                self._last_team_result['review_fingerprint'] = review.evidence_packets[-1].get('review_fingerprint')
            if self._nested_config.get('evidence_harness'):
                self._last_team_result['validated_contributions'] = review.progress()
        if self._nested_config.get('team_return_mode') == 'verbatim':
            # No additional LLM verdict or rewriting between the team and manager.
            from .review_delivery import worker_contributions, tool_activity
            excluded = {review.reviewer_name, review.auditor_name, review.executor_name} if review else set()
            findings = worker_contributions(nested_group_chat.messages, excluded)
            result = {'group_name': group_name, 'assigned_subtask': execution_task,
                      'mode': self._active_mode,
                      'reused_roster': reused_team,
                      'actual_capabilities': {a.name: [t['function']['name'] for t in a.llm_config.get('tools', [])]
                                              for a in agent_list if a.llm_config},
                      'worker_findings': findings,
                      'tool_activity': tool_activity(nested_group_chat.messages, excluded),
                      'review': review.findings() if review else None,
                      'status_meaning': 'Review returned is procedural, not scientific or overall completion. '
                          'A reused name preserves its roster; use a new team name for missing capabilities.'}
            return '# Team return (verbatim contributions)\n' + json.dumps(result, ensure_ascii=False) + self._shared_evidence(mission=True)
        chat_history = []
        key = list(agent_list[0].chat_messages.keys())[0]
        chat_messages = agent_list[0].chat_messages[key]
        for item in chat_messages:
            chat_history.append(item)

        # Review the group chat history
        summary_model = builder.builder_model
        summarized_history = (
            summary_model.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.CONVERSATION_REVIEW_PROMPT.format(chat_history=chat_history),
                    }
                ]
            )
            .choices[0]
            .message.content
        )

        response = f"# Response from seek_agent_help: \n{summarized_history}"
        if review:
            # Preserve the review verbatim so the summary cannot erase findings.
            heading = 'Team review findings' if self._nested_config.get('controlled_review') else 'Scientific critic findings'
            response += '\n\n# ' + heading + '\n' + review.findings()
        # Factual records survive the language-model summary unchanged.
        response += self._shared_evidence(mission=True)
        return response
