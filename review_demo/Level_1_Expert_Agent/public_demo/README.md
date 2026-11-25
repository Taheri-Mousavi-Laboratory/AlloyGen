# CALPHAD agent demonstrations

- [Single equilibrium](single_equilibrium.ipynb): Gibbs energy, M23C6 volume fraction and sweep, sampled M7C3 distribution.
- [Scheil solidification](scheil.ipynb): cumulative latent heat and temperature at 80% solid.

Saved outputs can be inspected without running the notebooks. Each contains one recorded conversation, the original questions, responses, plots and links to generated code/data. Brief notes identify units and sampling assumptions.

## Run locally

1. Use Python 3.12 and a working Thermo-Calc 2024a installation, license and FEDEMO database. Install TC-Python **2024.1.30332.76** from that installation. Configure `TC24A_HOME` and licensing for your installation.
2. Install the dependencies in that environment:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env.local`, enter this project's key and restrict permissions on Linux/macOS:

   ```bash
   cp .env.example .env.local
   chmod 600 .env.local
   ```

   Only this folder's private, Git-ignored `.env.local` is read. No key is bundled.
4. Open the notebook with this folder as the working directory, choose the configured Python kernel and run in order from a fresh kernel. Generated Python executes locally in that environment. Fresh results go to a new `outputs/<run-id>/` folder.

Use a short installation path for Thermo-Calc 2024a state files. If necessary, set `ALLOYGEN_RUNTIME_ROOT` to a shorter writable directory before starting the kernel; caches use its `.tc/` subfolder. A startup check rejects insufficient path headroom. Saved examples are never reused as fresh model responses.

## Shared preparation

Both notebooks use [the same setup](agent/agent_setup.py): `gpt-4o-2024-08-06`, temperature 0.3, the [system prompt](agent/prompts/minimal_v5.txt) and 25 retrieval files in [config.json](agent/config.json). These examples and API/notation notes are separate from FEDEMO. Assessment data are excluded from retrieval inputs.

The local embedding model is `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. First execution downloads that revision if absent. Alternatively, set `ALLOYGEN_EMBEDDING_CACHE` to an existing `models--sentence-transformers--all-MiniLM-L6-v2` cache directory containing that revision. Offline execution requires the local model.

Limits: 900 seconds per generated script, 1,200 seconds per assistant request, ten automatic proxy replies, 24,000 characters of execution feedback. Full logs and usage are retained. A new execution uses the API and may produce different results. Cost displays use the recorded [pricing assumptions](agent/pricing.json), not an invoice.

## Saved executions

Sources: `equilibrium_repeat_20261028_01` and `scheil_repeat_v5_20261029_01`. Configuration records accompany the [equilibrium outputs](outputs/single_equilibrium/) and [Scheil outputs](outputs/scheil/). Setup cells were adapted for this package after execution. Original task calls and scientific outputs remain intact, with installation paths redacted. Rerunning does not reproduce the saved conversation deterministically.

## Earlier demonstrations and controls

The main repository README links to the two recorded demonstrations above. The original public notebooks remain available at their established paths:

- [Earlier single-equilibrium demonstration](../Demo_1_ExpertAgent_Solving_SingleEquilibrium.ipynb)
- [Earlier Scheil demonstration](../Demo_2_ExpertAgent_Solving_ScheilSolidification.ipynb)
- [Original single-equilibrium control](../Control_Case_for_demo_1.ipynb)
- [Original Scheil control](../Control_Case_for_demo_2.ipynb)

These earlier notebooks and controls are separate recorded configurations; their outputs have not been replaced. Repetition campaigns are outside this demo bundle.

## Code associated with these records

Use the local `agent/` preparation and this folder's dependency instructions for these revised demos. They are independent of the repository-root `alloygen/` implementation used by the older examples. The recorded source/run identities are listed above; [the publication manifest](../../PUBLIC_DEMO_MANIFEST.json) records source and exported-file hashes. Setup portability changes do not represent new scientific executions.
