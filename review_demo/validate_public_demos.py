"""Read-only checks for the selected demo package. No model or solver calls.

Run from any directory: python review_demo/validate_public_demos.py
Only standard-library dependencies are needed. Findings report locations, never
credential values. This is a pattern scan, not a guarantee against every secret.
"""
from pathlib import Path
from urllib.parse import unquote
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'review_demo/PUBLIC_DEMO_MANIFEST.json'
L1 = ROOT/'review_demo/Level_1_Expert_Agent/public_demo'
DESIGN = ROOT/'review_demo/Level_3_Dynamical_Team_Building/alloy_design'
TEXT = {'.py','.md','.txt','.json','.jsonl','.ipynb','.csv','.log','.example','.yaml','.yml','.toml',''}
PATTERNS = {
    'API key': r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}',
    'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',
    'AWS access key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'URL credentials': r'://[^\s/:@"\x27<>]+:[^\s/@"\x27<>]+@',
    'bearer credential': r'(?i)\bBearer\s+[A-Za-z0-9_.=-]{20,}',
}

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def markdown_links(text):
    # Ignore fenced examples, which may contain paths supplied to agents.
    text=re.sub(r'```[^\n]*\n[\s\S]*?```','',text)
    for match in re.finditer(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text):
        yield match.group(1).strip('<>')
    for match in re.finditer(r'(?:href|src)=["\x27]([^"\x27]+)',text):
        yield match.group(1)

def notebook_text(nb):
    for cell in nb['cells']:
        if cell['cell_type']=='markdown':
            yield ''.join(cell['source'])
        for output in cell.get('outputs',[]):
            for mime,value in output.get('data',{}).items():
                if mime in {'text/markdown','text/html'}:
                    yield ''.join(value) if isinstance(value,list) else value

def main():
    m=json.loads(MANIFEST.read_text())
    errors=[]
    count={'source_exports':len(m['source_files']), 'existing_files_preserved':0,
           'links_checked':0,'notebooks_checked':0,'credential_scan_files':0}
    selected={ROOT/name for name in m['source_files']}
    selected.update(ROOT/name for name in m['new_reading_files_sha256'])
    selected.update({ROOT/'README.md',MANIFEST,Path(__file__).resolve()})
    for name,row in m['source_files'].items():
        p=ROOT/name
        if not p.is_file() or hash_file(p)!=row['export_sha256']:
            errors.append(f'Export hash mismatch: {name}')
    for name,value in m['new_reading_files_sha256'].items():
        p=ROOT/name
        if not p.is_file() or hash_file(p)!=value:
            errors.append(f'Reading-file hash mismatch: {name}')
    for name,value in m['preserved_public_files_sha256'].items():
        p=ROOT/name
        if not p.is_file() or hash_file(p)!=value:
            errors.append(f'Existing public file changed: {name}')
        else:
            count['existing_files_preserved']+=1
        selected.add(p)
    for p in sorted(selected):
        if not p.is_file():
            continue
        rel=p.relative_to(ROOT).as_posix()
        if p.stat().st_size>50*1024*1024:
            errors.append(f'File above 50 MiB: {rel}')
        if p.name in {'.env','.env.local'} or p.suffix in {'.pem','.key','.p12'}:
            errors.append(f'Credential/private file selected: {rel}')
        if any(part in {'__pycache__','.ipynb_checkpoints','repeatability_20261125'} for part in p.parts):
            errors.append(f'Excluded runtime/campaign material selected: {rel}')
        if p.suffix not in TEXT:
            continue
        text=p.read_text(encoding='utf-8')
        count['credential_scan_files']+=1
        for kind,pattern in PATTERNS.items():
            if re.search(pattern,text):
                errors.append(f'Potential {kind}: {rel} (value withheld)')
        # Detect hard-coded secret literals in addition to vendor-specific prefixes.
        for match in re.finditer(r'''(?ix)["']?(?:api[_-]?key|password|access[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["']([^"'\n]{8,})["']''',text):
            value=match.group(1)
            lower=value.lower()
            placeholders=['your_','your-','placeholder','example','redacted','dummy','not-a-real','test-','fake-','<','[','$','os.getenv','os.environ','getenv(']
            if not any(marker in lower for marker in placeholders) and re.fullmatch(r'[A-Za-z0-9_./+=-]{16,}',value):
                errors.append(f'Potential hard-coded credential: {rel} (value withheld)')
        if p.suffix=='.ipynb':
            nb=json.loads(text)
            count['notebooks_checked']+=1
            if p.name=='demo.ipynb' and p.parent.name in {'G54-G1-MEM-01','G54-SA-EQ-01'}:
                if nb['metadata'].get('alloygen_public_demo',{}).get('mode')!='saved_results_viewer':
                    errors.append(f'Missing viewer declaration: {rel}')
                for cell in nb['cells']:
                    if cell['cell_type']=='code':
                        if cell.get('execution_count') is None:
                            errors.append(f'Unexecuted viewing cell: {rel}')
                        if any(o['output_type']=='error' for o in cell['outputs']):
                            errors.append(f'Viewing-cell error: {rel}')
    # Active public navigation only. Historical strings in transcripts/source remain
    # archival locators, not current Markdown links supplied to readers.
    pages=[ROOT/'README.md',*L1.glob('*.md'),*L1.glob('*.ipynb'),DESIGN/'README.md']
    for case in ['G54-G1-MEM-01','G54-SA-EQ-01']:
        d=DESIGN/case
        pages += [d/'README.md',d/'demo.ipynb',d/'records/README.md',d/'records/recorded_notebook.ipynb']
        pages += list((d/'records/rounds').glob('*.md'))
    for p in pages:
        text=p.read_text()
        pieces=notebook_text(json.loads(text)) if p.suffix=='.ipynb' else [text]
        for piece in pieces:
            for target in markdown_links(piece):
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):
                    continue
                path=unquote(target.split('#')[0].split('?')[0])
                if not path:
                    continue
                count['links_checked']+=1
                resolved=(p.parent/path).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    errors.append(f'Broken local link: {p.relative_to(ROOT)} -> {path}')
    # Verify logs and evidence really are eligible for an explicit git add.
    rels=[p.relative_to(ROOT).as_posix() for p in selected if p.is_file()]
    result=subprocess.run(['git','check-ignore','--no-index','--stdin'],cwd=ROOT,
                          input='\n'.join(rels)+'\n',capture_output=True,text=True)
    if result.returncode not in (0,1):
        errors.append('Could not check Git ignore rules')
    else:
        for name in result.stdout.splitlines():
            errors.append(f'Selected publication file ignored by Git: {name}')
    print(json.dumps({'status':'pass' if not errors else 'fail','checks':count,
                      'new_model_or_solver_calls':0,'failures':errors},indent=2))
    return 1 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
