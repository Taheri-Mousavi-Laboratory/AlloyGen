# Multi-agent alloy design

[Open the demonstration](demo.ipynb) · [All demonstrations](../../../../README.md)

**G54-G1-MEM-01** is the recorded 12 November 2026 investigation, run `alloy_design_G1_memory_fresh_gpt54_20261112_01`. The notebook walks through the task, team memberships, measured refinement, Ni comparison, critic intervention, final shortlist and verification.

## View the results

Read the saved notebook outputs on GitHub. For a local replay of the viewing cells, use Python with Jupyter/IPython and start in this folder. Those cells only read saved JSON/CSV records; they do not use an API, launch Thermo-Calc or reconstruct a new experiment.

The [full recorded notebook](records/recorded_notebook.ipynb), [console transcript](records/trace.txt), [structured run](records/run.json) and [individual round reports](records/rounds/) retain the scientific process, including errors. Large raw records can be downloaded if GitHub does not preview them. Author reading notes and comparisons are explicitly labeled in the front notebook.

## Setup and code version

- Model: `gpt-5.4-2026-03-05`; generation settings and limits: [saved protocol](records/setup/design_g1_memory_closure_fix_gpt54_20261112/protocol.json).
- Instructions and expertise: [task](records/setup/design_g1_memory_closure_fix_gpt54_20261112/task.md), [agent catalog](records/setup/design_g1_memory_closure_fix_gpt54_20261112/agents.json), [codebooks](records/setup/design_g1_memory_closure_fix_gpt54_20261112/codebooks/) and [retrieval references](records/setup/design_g1_memory_closure_fix_gpt54_20261112/references/).
- Engine: [the captured GroupAgent_dev source](records/setup/design_g1_memory_closure_fix_gpt54_20261112/engine/GroupAgent_dev/), including team building, review and working memory. This is the run-associated development snapshot, not the older repository-root `alloygen/` package.
- Experiment tools and coordination: [selected runtime source](records/setup/design_g1_memory_closure_fix_gpt54_20261112/). Internal diagnostic scripts and unrelated experiments are excluded.
- Physics implementation and method settings: [captured scientific source](records/search/setup/). Method `design_indicators_v1_20261107`, TC-Python `2025.2.30`; the database is identified by its saved SHA-256 fingerprint. The licensed database and Thermo-Calc distribution are not included.

The snapshots document the implementation used for the recorded run. They retain original workspace assumptions and are **not a standalone fresh-run installation**: original environment/caches, database, credentials, support modules and preservation manifests would need to be configured. Do not use the archived execution notebook as the viewing entry point. The [publication manifest](../../../PUBLIC_DEMO_MANIFEST.json) records source/export hashes and any publication-only redactions. No model or solver was rerun for this package.

## Result

Preferred C005: Al–1Ni–0.4Er–0.8Zr at.%, J≈0.278045. Balanced alternative C001: Al–1.5Ni–0.4Er–0.4Zr at.%, J≈0.305270. Both meet the 660°C phase constraint; C005 has a substantial FR/HCS penalty versus reference. The notebook contains every indicator and the author-recalculated paper-design comparison.

The investigation made 20 assignments with eight distinct memberships; preferred/reference verification matched 38/38 scalar values. Search coverage, cost and an incorrect final primary-L1₂ direction label remain limitations. This is a single demonstration, not a repeated-trial success-rate claim.
