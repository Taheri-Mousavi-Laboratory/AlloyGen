# Single-agent alloy-design comparison

[Open the demonstration](demo.ipynb) · [Multi-agent demonstration](../G54-G1-MEM-01/demo.ipynb)

**G54-SA-EQ-01**, executed 20 November 2026, is the supporting comparison using the equilibrium demo's single reasoning assistant with retrieval and execution proxies. It uses the same design requirements and three physics calculation families. It has no team builder, external task memory or specialist planner/critic tools.

## View the results

Read the saved notebook on GitHub, or run its viewing cells in this folder with Python/Jupyter/IPython. They only read saved files and do not call a model or Thermo-Calc. The [full recorded execution notebook](records/recorded_notebook.ipynb), [transcript](records/transcript.txt), [conversation](records/conversation.json) and [generated scripts](generated_code/) retain the original process and errors.

## Setup and code version

- Model: `gpt-5.4-2026-03-05`; [protocol](records/setup/protocol.json) and [recorded environment](records/run.json).
- Task, instructions and knowledge: [task](records/setup/task.md), [system prompt](records/setup/agent/prompts/alloy_design.txt), [RAG corpus](records/setup/agent/knowledge/).
- Single-agent setup: [captured agent source](records/setup/agent/). This is the recorded comparison configuration, not the earlier single-agent variant equipped with additional organizational tools.
- Physics: [captured method source/settings](records/search/setup/), method `design_indicators_v1_20261107`, TC-Python `2025.2.30` in the scientific worker. The orchestrator environment separately records TC-Python `2024.1.30332.76`; that is not the version used by the isolated scientific worker. Licensed software/database files are not distributed.

Source snapshots preserve workspace-specific assumptions and are not a portable fresh-run installation. The root `alloygen/` installation does not supply this setup automatically. Public path/credential redactions and source hashes are recorded in [the publication manifest](../../../PUBLIC_DEMO_MANIFEST.json). No model or solver calls were made to prepare the public copy.

## Outcome

The conversation ended but the design task remained incomplete. All 35 scientific invocations completed; generated-script parsing errors prevented recognition of a feasible candidate. No candidate was nominated or freshly verified. The author-read feasible composition and its limitations are explicitly separated from the agent's final claims in the notebook. This supports a case-specific comparison of task completion, not a general single-agent failure claim.
