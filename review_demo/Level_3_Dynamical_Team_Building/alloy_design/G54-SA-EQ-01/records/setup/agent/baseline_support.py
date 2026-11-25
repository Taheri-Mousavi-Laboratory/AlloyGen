"""Filesystem/configuration and observation helpers shared by baseline and demos."""
from datetime import datetime, timezone
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .credentials import load_project_key, redact, sanitized

HERE = Path(__file__).resolve().parent.parent
REPO = Path(os.environ.get('ALLOYGEN_RUNTIME_ROOT', HERE)).resolve()
PYTHON = Path(sys.executable)
MODEL = 'gpt-4o-2024-08-06'

def contained(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) and path != root for root in (HERE, REPO)):
        raise ValueError('Write path must stay within the demo or configured runtime root')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = contained(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = contained(path.with_suffix(path.suffix+'.tmp'))
    temp.write_text(json.dumps(sanitized(value), indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def configure_paths(runtime):
    """Keep caches local; use the recorded embedding revision and installed TC."""
    runtime = contained(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    variables = {
        'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp', 'XDG_CACHE_HOME':'cache',
        'XDG_CONFIG_HOME':'config', 'XDG_DATA_HOME':'data', 'XDG_STATE_HOME':'state',
        'IPYTHONDIR':'ipython', 'JUPYTER_CONFIG_DIR':'jupyter/config',
        'JUPYTER_DATA_DIR':'jupyter/data', 'JUPYTER_RUNTIME_DIR':'jupyter/runtime',
        'MPLCONFIGDIR':'matplotlib', 'TORCH_HOME':'torch', 'NUMBA_CACHE_DIR':'numba',
        'TRITON_CACHE_DIR':'triton', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TIKTOKEN_CACHE_DIR':'tiktoken',
    }
    for variable, relative in variables.items():
        path=runtime/relative;path.mkdir(parents=True,exist_ok=True)
        os.environ[variable]=str(path)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', ANONYMIZED_TELEMETRY='False',
                      PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode=True
    # The original local MiniLM retrieval model is supplied as a read-only snapshot.
    token_cache = Path(os.environ['ALLOYGEN_REPOSITORY']) / 'experiments/tool_using_demo_reproduction/runtime/equilibrium_repeat_20261028_01/tiktoken'
    for name in ('9b5ad71b2ce5302211f9c61530b329a4922fc6a4', 'fb374d419588a4632f3f557e76b4b70aebbca790'):
        shutil.copyfile(token_cache / name, Path(os.environ['TIKTOKEN_CACHE_DIR']) / name)
    os.environ.pop('OPENAI_API_KEY',None)
    # TC24A_HOME and licensing are supplied by the user's Thermo-Calc installation.



class RedactingStream:
    def __init__(self, stream):
        self.stream = stream
    def write(self, text):
        return self.stream.write(redact(text))
    def flush(self):
        return self.stream.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


class RunRecord:
    def __init__(self, run_id=None, *, notebook=None, settings=None):
        run_id = run_id or datetime.now(timezone.utc).strftime('single_equilibrium_%Y%m%dT%H%M%SZ')
        if not run_id.replace('_','').replace('-','').isalnum():
            raise ValueError('Invalid run ID')
        self.runtime = contained(HERE/'runtime'/run_id)
        self.results = contained(HERE/'outputs'/run_id)
        self.results.mkdir(parents=True, exist_ok=False)
        self.runtime.mkdir(parents=True, exist_ok=False)
        configure_paths(self.runtime)
        self.vector_dir = self.runtime/'vector_db'
        self.vector_dir.mkdir()
        os.chdir(self.runtime)  # AG2's default .cache and tmp/db now stay in this run.
        key = "offline-placeholder" if (settings or {}).get("offline") else load_project_key()
        sys.stdout = RedactingStream(sys.stdout)
        sys.stderr = RedactingStream(sys.stderr)
        self.state = {
            'run_id': run_id, 'status': 'initialized', 'started_UTC': datetime.now(timezone.utc).isoformat(),
            'model': MODEL, 'original_model': 'gpt-4o-2024-05-13', 'temperature': 0.3,
            'cache_seed': 42, 'fresh_run_directory': True, 'shared_conversation_across_four_tasks': True,
            'assistant_timeout_seconds': 1200, 'code_timeout_seconds': 60,
            'user_proxy_max_consecutive_auto_reply': 10,
            'python_executable': sys.executable, 'python_version': sys.version,
            'packages': {p: metadata.version(p) for p in ['ag2','openai','tc-python','chromadb','sentence-transformers','numpy','scipy','matplotlib']},
            'source_manifest': json.loads((HERE/'agent/config.json').read_text()),
            'embedding_snapshot': json.loads((HERE/'agent/config.json').read_text())['embedding_snapshot'],
            'filesystem_namespace': os.environ.get('ALLOYGEN_CONTAINED') == '1',
            'solver_temp_directory': os.environ['TMPDIR'],
            'credential_source': 'experiment-local .env.local; value never exported',
            'retrievals': [], 'code_executions': [], 'task_checkpoints': [],
            'support_sha256': sha(Path(__file__)),
            'notebook_sha256': sha(notebook or HERE/'single_equilibrium.ipynb'),
        }
        if settings:
            self.state.update(copy.deepcopy(settings))
        self.code_dir = self.runtime / "code"
        self.code_dir.mkdir()
        # Scientific workers retain their own existing short-path runtime.
        # Explicit key only; do not load old OAI_CONFIG_LIST or inherited credentials.
        self._config_list = [{'model': self.state['model'], 'api_key': key, 'base_url': 'https://api.openai.com/v1',
                              'max_completion_tokens': 12000, 'price': [0.0025, 0.015]}]
        self.save()

    def llm_config_list(self):
        return self._config_list

    def save(self):
        save_json(self.results/'run.json', self.state)

    def retrieval(self, query, n_results, response, document_ids):
        self.state['retrievals'].append({'query': query, 'n_results': n_results, 'response': response,
                                        'document_ids': list(document_ids), 'UTC': datetime.now(timezone.utc).isoformat()})
        self.save()

    def checkpoint(self, label, assistant, proxy_user, finish=False):
        history = list(proxy_user.chat_messages.get(assistant, []))
        save_json(self.results/f'{label}_conversation.json', history)
        artifacts = []
        for source in sorted(self.code_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() not in {'.py','.json','.csv','.png','.txt','.log'}:
                continue
            data = source.read_bytes()
            # Preserve image bytes; redact credentials from any text artifact.
            if source.suffix.lower() != '.png':
                data = redact(data.decode('utf-8')).encode()
            dest = contained(self.results/'artifacts'/label/source.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            artifacts.append({'file': str(dest.relative_to(self.results)), 'sha256': sha(dest)})
        entry = {'task': label, 'messages': len(history), 'artifacts': artifacts,
                 'actual_usage': copy.deepcopy(assistant.client.actual_usage_summary),
                 'total_usage_including_cache': copy.deepcopy(assistant.client.total_usage_summary)}
        self.state['task_checkpoints'].append(entry)
        self.state['status'] = 'conversation_complete' if finish else 'in_progress'
        if finish:
            self.state['finished_UTC'] = datetime.now(timezone.utc).isoformat()
        self.save()
        print(label, ':', len(history), 'messages;', len(artifacts), 'saved files')


def observing_executor(record):
    """Record full results; optionally bound verbose feedback without editing code."""
    from autogen.coding import LocalCommandLineCodeExecutor
    class ObservingExecutor(LocalCommandLineCodeExecutor):
        def execute_code_blocks(self, code_blocks):
            event = {'blocks': [{'language': b.language, 'code': b.code} for b in code_blocks],
                     'started_UTC': datetime.now(timezone.utc).isoformat()}
            if hasattr(record, 'current_task'):
                event['task'] = record.current_task
            record.state['code_executions'].append(event)
            record.save()
            start = time.monotonic()
            try:
                result = super().execute_code_blocks(code_blocks)
                event.update({'exit_code': result.exit_code, 'output': result.output,
                              'code_file': result.code_file})
                limit = record.state.get('executor_feedback_character_limit')
                event['feedback_abbreviated'] = bool(limit and len(result.output) > limit)
                if event['feedback_abbreviated']:
                    log = contained(record.code_dir / f'execution_output_{len(record.state["code_executions"]):03d}.log')
                    log.write_text(redact(result.output))
                    head = limit // 4
                    tail = limit - head
                    feedback = (result.output[:head] +
                                f'\n[Execution output abbreviated: {len(result.output)} characters total. '
                                f'Full output saved as {log.name} in the working directory. '
                                'Beginning and end follow; intermediate output is omitted.]\n' +
                                result.output[-tail:])
                    event.update(feedback_output=feedback, full_output_file=log.name)
                    result = result.model_copy(update={'output': feedback})
                return result
            finally:
                event['elapsed_seconds'] = time.monotonic()-start
                record.save()
    return ObservingExecutor
