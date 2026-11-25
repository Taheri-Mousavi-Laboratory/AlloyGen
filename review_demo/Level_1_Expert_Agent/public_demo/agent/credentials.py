"""Load only this demonstration folder's private API key."""
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent.parent
_secret=None

def load_project_key():
    global _secret
    path=HERE/'.env.local'
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Create .env.local from .env.example and set the project key')
    if os.name=='posix' and path.stat().st_mode & 0o077:
        raise RuntimeError('Restrict .env.local permissions with chmod 600 .env.local')
    if shutil_which_git():
        tracked=subprocess.run(['git','ls-files','--error-unmatch',str(path)],cwd=HERE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if tracked.returncode==0:raise RuntimeError('The private key file must not be tracked by Git')
    entries=[]
    for line in path.read_text().splitlines():
        name,sep,value=line.strip().partition('=')
        if sep and name.strip()=='OPENAI_API_KEY':entries.append(value.strip().strip('\"\''))
    if len(entries)!=1 or not entries[0] or any(c.isspace() for c in entries[0]):
        raise RuntimeError('Expected one nonempty OPENAI_API_KEY in .env.local')
    _secret=entries[0]
    return _secret

def shutil_which_git():
    from shutil import which
    return which('git')

def redact(text):
    text=str(text)
    if _secret:text=text.replace(_secret,'[REDACTED]')
    return re.sub(r'\bsk-[A-Za-z0-9_.*-]{6,}','[REDACTED]',text)

def sanitized(value):
    if isinstance(value,dict):return {k:'[REDACTED]' if k.lower() in {'api_key','authorization','openai_api_key'} else sanitized(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [sanitized(v) for v in value]
    return redact(value) if isinstance(value,str) else value
