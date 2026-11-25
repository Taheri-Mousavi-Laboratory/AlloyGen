"""Recording and local setup for the copied equilibrium demo; no agent controller."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent.parent
REPO = next(p for p in HERE.parents if (p / 'revision/API_ACCOUNTING_PERIOD.json').exists())
os.environ['ALLOYGEN_REPOSITORY'] = str(REPO)
os.environ.setdefault('ALLOYGEN_KEY_FILE', str(REPO / 'experiments/tool_using_demo_reproduction/.env.local'))
os.environ['ALLOYGEN_RUNTIME_ROOT'] = str(HERE)
sys.dont_write_bytecode = True

from .baseline_support import save_json, sha
from .demo_support import DemoRun, summarize_usage
from .credentials import redact

PROTOCOL = json.loads((HERE / 'protocol.json').read_text())

def verify_inputs(protected=False):
    manifest = json.loads((HERE / 'provenance/source_manifest.json').read_text())
    for item in manifest['copied']:
        if sha(HERE / item['copy']) != item['sha256']:
            raise RuntimeError('Preserved source changed: ' + item['copy'])
    for name, digest in manifest['physics'].items():
        if sha(REPO / name) != digest:
            raise RuntimeError('Scientific code differs from MEM-01: ' + name)
    frozen = HERE / 'PREPARED_INPUTS.json'
    if frozen.exists():
        for name, digest in json.loads(frozen.read_text()).items():
            if sha(HERE / name) != digest:
                raise RuntimeError('Prepared input changed: ' + name)
    if protected:
        for name, digest in json.loads((HERE / 'checks/protected_before.json').read_text()).items():
            if sha(REPO / name) != digest:
                raise RuntimeError('Existing result changed: ' + name)
    return {'physics_files': len(manifest['physics']), 'preserved_sources': len(manifest['copied'])}

def embedding_path():
    path = Path(os.environ.get('ALLOYGEN_EMBEDDING_SNAPSHOT',
        '[LOCAL_PATH]'
        'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/'
        'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'))
    if not path.is_dir():
        raise FileNotFoundError('Set ALLOYGEN_EMBEDDING_SNAPSHOT to the installed pinned MiniLM snapshot.')
    return path

class Tee:
    def __init__(self, screen, log):
        self.screen, self.log = screen, log
    def write(self, text):
        text = redact(text)
        self.log.write(text)
        self.log.flush()
        return self.screen.write(text)
    def flush(self):
        self.log.flush()
        self.screen.flush()

class AlloyRun(DemoRun):
    def __init__(self, selected, *, run_id, notebook, offline=False):
        verify_inputs()
        # DemoRun/RunRecord are copied from the equilibrium demo. Offline mode
        # only supplies a dummy client credential; checks prohibit SDK requests.
        self.offline = offline
        from . import baseline_support
        original_loader = baseline_support.load_project_key
        if offline:
            baseline_support.load_project_key = lambda: 'offline-placeholder'
        try:
            super().__init__(selected, run_id=run_id, notebook=notebook,
                model=PROTOCOL['model'], purpose='offline_check' if offline else 'single_agent_comparison',
                code_timeout_seconds=1800, executor_feedback_character_limit=None,
                source_manifest='agent/task_manifest.json', assessment_protocol='protocol.json')
        finally:
            baseline_support.load_project_key = original_loader
        config = json.loads((HERE / 'agent/config.json').read_text())
        self.state.update({k: config[k] for k in ('temperature', 'cache_seed',
            'assistant_timeout_seconds', 'user_proxy_max_consecutive_auto_reply')})
        self.state.update(offline=offline, comparison='G54-G1-MEM-01',
            accounting_period=json.loads((REPO / 'revision/API_ACCOUNTING_PERIOD.json').read_text())['accounting_period'],
            scientific_tools=[], protocol=PROTOCOL, status='prepared')
        for path in [HERE / 'task.md', HERE / 'protocol.json', HERE / 'PREPARED_INPUTS.json']:
            if path.exists():
                (self.results / 'setup' / path.name).write_bytes(path.read_bytes())
        # The same class used by the team is made available to ordinary Python.
        # No new scientific function or fresh=True option is introduced.
        module = '''from pathlib import Path
import os, sys
REPO = Path(os.environ['ALLOYGEN_REPOSITORY'])
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / 'experiments/management_alloy_design'))
from scientific_tools import AlloyDesignTools
OUTPUT = Path(os.environ['ALLOYGEN_SINGLE_OUTPUT'])
tools = AlloyDesignTools(OUTPUT / 'search')
'''
        (self.code_dir / 'science_api.py').write_text(module)
        os.environ['ALLOYGEN_SINGLE_OUTPUT'] = str(self.results)
        self.save()

    def observe_client(self, wrapper):
        super().observe_client(wrapper)
        # Record actual prompts without changing them; use the common API ceiling.
        for client in wrapper._clients:
            client._oai_client.max_retries = 0
            previous = client.create
            def observed(params, _previous=previous):
                if self.offline:
                    raise AssertionError('Provider calls prohibited in offline preparation')
                if len(self.state['api_requests']) >= PROTOCOL['max_provider_requests']:
                    raise RuntimeError('Declared provider-request allowance reached')
                index = len(self.state['api_requests']) + 1
                save_json(self.results / 'requests' / f'{index:04d}.json',
                          {k: v for k, v in params.items() if k not in ('api_key', 'authorization')})
                return _previous(params)
            client.create = observed

    @contextmanager
    def task(self, label, question):
        if self.state['tasks']:
            raise RuntimeError('Use a new run ID; retain the previous conversation unchanged.')
        self.current_task = label
        entry = {'task': label, 'question': question, 'status': 'running',
                 'started_UTC': datetime.now(timezone.utc).isoformat()}
        self.state['tasks'].append(entry)
        self.state['status'] = 'running'
        self.save()
        began = time.monotonic()
        try:
            with (self.results / 'transcript.txt').open('w') as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    yield
            entry['status'] = 'conversation_returned'
            self.state['status'] = 'returned_pending_scientific_assessment'
        except BaseException as error:
            entry.update(status='interrupted', error=redact(str(error)))
            self.state['status'] = 'interrupted'
            raise
        finally:
            if hasattr(self, '_agents'):
                assistant, proxy = self._agents
                save_json(self.results / 'conversation.json', proxy.chat_messages.get(assistant, []))
            entry.update(elapsed_seconds=time.monotonic() - began,
                         usage=summarize_usage(self.state['api_requests']))
            self.state['finished_UTC'] = datetime.now(timezone.utc).isoformat()
            self.save_usage()
            self.save()
