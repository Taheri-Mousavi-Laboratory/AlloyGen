"""Runtime, saved evidence and usage accounting for small library tests."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]
PROTOCOL = json.loads((HERE / "test_protocol.json").read_text())
sys.path.insert(0, str(REPO / "experiments/tool_using_demo_reproduction"))
from credentials import load_project_key, sanitized, redact


def save(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(REPO):
        raise ValueError("Output must remain inside the revision repository")
    path.write_text(json.dumps(sanitized(value), indent=2, allow_nan=False) + "\n")


def setup(run_id):
    if os.environ.get("ALLOYGEN_CONTAINED") != "1":
        raise RuntimeError("Use run_contained.py")
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Invalid run ID")
    runtime = HERE / ".runtime" / run_id
    runtime.mkdir(parents=True, exist_ok=False)
    variables = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "IPYTHONDIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR", "TORCH_HOME", "NUMBA_CACHE_DIR", "TIKTOKEN_CACHE_DIR", "HF_HOME", "HF_MODULES_CACHE")
    for name in variables:
        folder = runtime / name.lower()
        folder.mkdir(exist_ok=True)
        os.environ[name] = str(folder)
    for name in ("OPENAI_API_KEY", "OAI_CONFIG_LIST", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        os.environ.pop(name, None)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", ANONYMIZED_TELEMETRY="False", TOKENIZERS_PARALLELISM="false", AUTOGEN_USE_DOCKER="False")
    os.chdir(runtime)
    return runtime


def embedding_path():
    path = Path('[LOCAL_PATH]') / PROTOCOL['embedding_snapshot']
    if not (path / 'config.json').is_file():
        raise RuntimeError('The cached original embedding model is unavailable')
    return str(path)


class Trial:
    def __init__(self, run_id, mode, source):
        self.runtime = setup(run_id)
        self.out = HERE / 'runs' / run_id
        self.out.mkdir(parents=True, exist_ok=False)
        self.phase = 'preflight'
        self.state = dict(run_id=run_id, mode=mode, status='prepared', started_UTC=datetime.now(timezone.utc).isoformat(), api_requests=[], tool_calls=[], protocol=PROTOCOL, source=source)
        self.started = time.monotonic()
        self.key = None
        sources = list((HERE / source).rglob('*.py')) + list((HERE / source).glob('*.json'))
        sources += [HERE / name for name in ('test_protocol.json','trial_support.py','run_trial.py')]
        sources += list((HERE/'specialists').glob('*.py')) + list((HERE/'library').glob('*.json')) + list((HERE/'library').glob('*.md'))
        self.state['source_hashes'] = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        calculator=EXPERIMENT/'scientific_calculator.py'
        self.state['scientific_calculator_sha256']=hashlib.sha256(calculator.read_bytes()).hexdigest()
        for p in sources:
            target = self.out / 'setup' / p.relative_to(HERE)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
        (self.out/'setup/scientific_calculator_snapshot.py').write_bytes(calculator.read_bytes())
        self.write()

    def config(self, offline=False):
        self.key = 'offline-placeholder-not-a-real-key' if offline else load_project_key()
        cfg = dict(model=PROTOCOL['model'], api_key=self.key)
        self.config_env = 'ALLOYGEN_LIBRARY_TEST_CONFIG'
        os.environ[self.config_env] = json.dumps([cfg])
        return dict(config_list=[cfg], temperature=PROTOCOL['temperature'], cache_seed=None, max_tokens=PROTOCOL['max_tokens'], timeout=180)

    def write(self):
        save(self.out/'run.json',self.state)

    @contextmanager
    def observe(self):
        from openai.resources.chat.completions import Completions
        from openai.resources.responses import Responses
        from autogen import ConversableAgent
        original_chat, original_responses = Completions.create, Responses.create
        original_tool = ConversableAgent.execute_function
        trial = self
        def observed(original, endpoint):
            def call(client, *args, **kwargs):
                requests = trial.state['api_requests']
                if len(requests) >= PROTOCOL['max_provider_requests']:
                    raise RuntimeError('Declared provider-request limit reached')
                if trial.phase == 'builder' and sum(r['phase']=='builder' for r in requests) >= PROTOCOL['max_builder_requests']:
                    raise RuntimeError('Declared builder-request limit reached')
                row = dict(endpoint=endpoint, phase=trial.phase, status='started', model=kwargs.get('model'), request=deepcopy(kwargs))
                requests.append(row); trial.write()
                began = time.monotonic()
                try:
                    response = original(client,*args,**kwargs)
                    raw = response.model_dump()
                    row.update(status='complete',response=raw,usage=raw.get('usage'),served_model=raw.get('model'))
                    usage = raw.get('usage') or {}
                    inp = usage.get('prompt_tokens',usage.get('input_tokens',0))
                    out = usage.get('completion_tokens',usage.get('output_tokens',0))
                    cached = (usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}).get('cached_tokens',0)
                    search_calls = sum(v.get('type')=='web_search_call' for v in raw.get('output',[]))
                    row.update(input_tokens=inp,output_tokens=out,cached_input_tokens=cached,web_search_calls=search_calls)
                    rates = PROTOCOL['cost_rates_per_million'].get(row['served_model'])
                    if rates and usage:
                        row['estimated_cost_usd'] = ((inp-cached)*rates['input']+cached*rates['cached_input']+out*rates['output'])/1e6 + search_calls*PROTOCOL['web_search_preview_per_call_usd']
                    return response
                except Exception as exc:
                    row.update(status='failed',error=redact(str(exc)),error_type=type(exc).__name__,billing_unknown=True)
                    raise
                finally:
                    row['seconds']=time.monotonic()-began
                    trial.write()
            return call
        def tool(agent, func_call, *args, **kwargs):
            if len(trial.state['tool_calls']) >= PROTOCOL['max_tool_calls']:
                raise RuntimeError('Declared tool-call limit reached')
            row=dict(executor=agent.name,request=deepcopy(func_call),status='started')
            trial.state['tool_calls'].append(row);trial.write()
            start=time.monotonic()
            try:
                ok, result=original_tool(agent,func_call,*args,**kwargs)
                row.update(status='complete' if ok else 'failed',result=result)
                return ok,result
            finally:
                row['seconds']=time.monotonic()-start;trial.write()
        with patch.object(Completions,'create',observed(original_chat,'chat.completions')), patch.object(Responses,'create',observed(original_responses,'responses')), patch.object(ConversableAgent,'execute_function',tool):
            yield

    def finish(self, status, error=None):
        os.environ.pop('ALLOYGEN_LIBRARY_TEST_CONFIG',None)
        self.state.update(status=status,seconds=time.monotonic()-self.started,finished_UTC=datetime.now(timezone.utc).isoformat())
        if error:
            self.state['error']=redact(str(error))
        rows=self.state['api_requests']
        usage={name:sum(r.get(name,0) for r in rows) for name in ('input_tokens','output_tokens','cached_input_tokens','web_search_calls')}
        usage.update(api_requests=len(rows),failed_requests=sum(r['status']=='failed' for r in rows),known_estimated_cost_usd=sum(r.get('estimated_cost_usd',0) for r in rows),unpriced_requests=sum('estimated_cost_usd' not in r for r in rows))
        self.state['usage']=usage
        self.state['inputs_unchanged']=all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h for p,h in self.state['source_hashes'].items())
        self.write()
        save(self.out/'usage.json',usage)


def write_notebook(trial):
    import nbformat as nbf
    state=trial.state
    title={'web':'Building a literature-search specialist','direct_equilibrium':'Equilibrium specialist with a calculation tool','builder_equilibrium':'Selecting an equilibrium specialist from an agent library'}[state['mode']]
    nb=nbf.v4.new_notebook()
    nb['cells']=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+state['task']),
      nbf.v4.new_markdown_cell('> **Notes**\n> - The model is `'+PROTOCOL['model']+'`. A new run may produce different interactions.\n> - The specialist instructions and tools are loaded from reusable Python modules and library entries.\n> - The saved conversation and tool results below document this execution.'),
      nbf.v4.new_code_cell('from run_trial import run_case\n\n# Choose a new run identifier before executing again.\nresult = run_case('+repr(state['mode'])+', '+repr(state['run_id'])+')')]
    messages=state.get('messages',[])
    output='Selected agents: '+', '.join(state.get('selected_agents',[]))+'\n\n'
    for m in messages:
        output+='['+m.get('name',m.get('role','message'))+']\n'+str(m.get('content') or '')+'\n'
        if m.get('tool_calls'):
            output+=json.dumps(m['tool_calls'],indent=2)+'\n'
        if m.get('tool_responses'):
            output+=json.dumps(m['tool_responses'],indent=2)+'\n'
        output+='\n'
    output=redact(output).replace(str(REPO),'[repository]')
    nb['cells'][-1]['execution_count']=1
    nb['cells'][-1]['outputs']=[nbf.v4.new_output('stream',name='stdout',text=output)]
    nb['cells'] += [nbf.v4.new_markdown_cell('**Execution record:** '+state['status']+'.\n\n'+str(state.get('assessment',''))),nbf.v4.new_markdown_cell('**API usage:** '+str(state['usage']['input_tokens'])+' input tokens and '+str(state['usage']['output_tokens'])+' output tokens. Estimated known cost: $'+format(state['usage']['known_estimated_cost_usd'],'.6f')+'. Estimates use returned usage and [published pricing](https://developers.openai.com/api/docs/pricing); unpriced requests: '+str(state['usage']['unpriced_requests'])+'.')]
    nb['metadata']['kernelspec']={'name':'12_llm_modeling','display_name':'Python (12_LLM_Modeling)','language':'python'}
    nb['metadata']['alloygen']={'kind':'rendered saved-run transcript; execution occurs in the contained Python runner','run_id':state['run_id']}
    nbf.write(nb,trial.out/'demo.ipynb')
