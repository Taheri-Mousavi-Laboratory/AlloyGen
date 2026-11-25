"""Shared setup for the two CALPHAD demonstration notebooks."""
from pathlib import Path
from .agent_setup import preparation, create_agents
from .demo_support import DemoRun

def start_demo(kind, notebook):
    if kind not in {'equilibrium', 'scheil'}:
        raise ValueError('Choose equilibrium or scheil')
    selected = preparation()
    run = DemoRun(selected, notebook=Path(notebook),
                  code_timeout_seconds=900, executor_feedback_character_limit=24000,
                  source_manifest=f'agent/tasks_{kind}.json',
                  assessment_protocol='agent/config.json', purpose='demonstration')
    agents = create_agents(run, selected)
    return run, agents
