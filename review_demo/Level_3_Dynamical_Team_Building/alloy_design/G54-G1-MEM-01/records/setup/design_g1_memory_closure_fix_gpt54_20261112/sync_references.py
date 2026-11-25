"""Keep the original-style RAG documents identical to their source codebooks."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def expected_documents():
    for role in ('planner', 'critic'):
        text = (HERE / 'codebooks' / (role + '.md')).read_text()
        sections = dict(part.split('\n', 1) for part in text.split('\n## ')[1:])
        if set(sections) != {'common', 'plan', 'execute', 'review'}:
            raise ValueError('Expected four role/mode sections: ' + role)
        for section, body in sections.items():
            path = HERE / 'references' / role / (section + '.md')
            yield path, f'# {role.capitalize()} guidance: {section}\n\n{body.strip()}\n'


def check_references():
    expected = dict(expected_documents())
    actual = set((HERE / 'references').glob('*/*.md'))
    if actual != set(expected) or any(p.read_text() != s for p, s in expected.items()):
        raise RuntimeError('Retrieval text is stale. Run sync_references.py --write before freezing the setup.')
    return len(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        for path, text in expected_documents():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print('Current retrieval documents:', check_references())
