# AlloyGen: Physics-grounded self-adaptive agents for autonomous alloy-design workflows


Bo Ni<sup>1,&#42;</sup>, S. Mohadeseh Taheri-Mousavi<sup>1,2,&#42;</sup>

<sup>1</sup> Carnegie Mellon University

<sup>&#42;</sup> Corresponding authors

<sup>2</sup> Lead contact: smtaherimousavi@cmu.edu


AlloyGen is the agentic workflow component of the broader AlloyGPT package. It is a physics-grounded, self-adaptive multi-agent framework for autonomous alloy-design workflows in additive manufacturing. Additive manufacturing expands the alloy-design space by coupling composition, processing, printability, thermodynamics and performance, but practical design still requires continuous reasoning over incomplete knowledge, evolving objectives, tool selection, simulation errors and intermediate evidence.

AlloyGen addresses this missing reasoning layer by combining large-language-model agents with curated materials knowledge, executable CALPHAD thermodynamic simulations, critique and dynamic team organization. The framework demonstrates three levels of adaptivity. At the individual-agent level, a domain-expert CALPHAD agent learns specialized tool use through retrieval, execution feedback and self-correction. At the team level, role-specialized agents collaborate through dynamic interaction patterns to transform an underspecified Al-Si printability prompt into explicit criteria, executable simulations and traceable screening decisions. At the organizational level, a team-builder agent assembles task-specific expertise and tools for open-ended alloy-research tasks, demonstrated on refractory-alloy research synthesis.

AlloyGen treats alloy design not only as a prediction or optimization problem, but as an adaptive scientific workflow that organizes knowledge, physics-based tools, simulations, critique and team structure as evidence accumulates.

![AlloyGen graphical abstract](./assets/TOC.png)

## Review demos

The review demos are organized according to the three adaptivity levels evaluated in the manuscript:

- **Level 1: domain-expert agent**
  - [Demo 1: single-equilibrium CALPHAD problems](./review_demo/Level_1_Expert_Agent/Demo_1_ExpertAgent_Solving_SingleEquilibrium.ipynb)
  - [Demo 2: Scheil solidification problems](./review_demo/Level_1_Expert_Agent/Demo_2_ExpertAgent_Solving_ScheilSolidification.ipynb)
  - [Control case for Demo 1](./review_demo/Level_1_Expert_Agent/Control_Case_for_demo_1.ipynb)
  - [Control case for Demo 2](./review_demo/Level_1_Expert_Agent/Control_Case_for_demo_2.ipynb)
- **Level 2: collaborative agent team**
  - [Demo 3: Al-Si alloy design for additive-manufacturing printability](./review_demo/Level_2_Collaborative_Team/Demo_3_CollaborativeTeam_Design_for_Printability.ipynb)
- **Level 3: dynamic team building**
  - [Demo 4: task-specific team building for AM alloy research synthesis](./review_demo/Level_3_Dynamical_Team_Building/Demo_4_Dynamical_Team_Building_For_Alloy_Research.ipynb)

These notebooks are provided for the peer-review process as records of the demonstration workflows. Running them locally may require adjusting paths and runtime configurations. They also require access to the relevant LLM API credentials, and demos involving CALPHAD calculations require a working Thermo-Calc/TC-Python installation and license.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Taheri-Mousavi-Laboratory/AlloyGen.git
cd AlloyGen
conda env create --prefix=./AlloyGen_env --file=AlloyGen_env_environment.yml
conda activate ./AlloyGen_env
pip install -e .
```

The environment file is exported from the development environment used for this work. Thermo-Calc/TC-Python functionality may require a valid local Thermo-Calc installation and license.

## Reference

Citation information will be added after publication.

## License

This repository is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](./LICENSE).
