"""Small role/mode lookup, called by the existing team-stage controller."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def decision_section(guide, mode, stage):
    role = {'planner.md': ('planner', 'work'), 'reviewer.md': ('critic', 'closing')}.get(guide)
    if role is None or stage != role[1]:
        return None
    if mode not in ('plan', 'execute', 'review'):
        raise ValueError('Unknown delegation mode')
    path = HERE / 'codebooks' / (role[0] + '.md')
    sections = {}
    for section in path.read_text().split('\n## ')[1:]:
        name, text = section.split('\n', 1)
        sections[name.strip()] = text.strip()
    text = sections['common'] + '\n\n' + sections[mode]
    return dict(section=f'{role[0]}/{mode}', text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest())
