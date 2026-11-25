"""Contained recording and frozen inputs; construction/chat stay visible in notebook."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(HERE/'engine'), str(BASE), str(BASE.parent)]
import trial_support as support
from notebook_runtime import record_run as _record_run


@contextmanager
def record_run(trial, group_agent, task):
    """Retain optional memory state even when a nested conversation is interrupted."""
    error = None
    try:
        with _record_run(trial, group_agent, task):
            yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        memory = group_agent.executor._nested_config.get('task_memory')
        if memory is not None:
            try:
                trial.state['task_memory'] = memory.snapshot()
                trial.write()
            except BaseException as snapshot_error:
                if error is None:
                    raise
                error.add_note('Working-memory snapshot also failed: ' + type(snapshot_error).__name__)


@contextmanager
def retrieval_scope(mode, execute_intent=None):
    """Use the existing delegation boundary to isolate role reference queries."""
    from .tools import delegation_scope, folder
    previous = os.environ.get('ALLOYGEN_RAG_SESSION')
    try:
        with delegation_scope(mode, execute_intent=execute_intent):
            counts = json.loads((folder()/'delegation_limits.json').read_text())
            serial = sum(counts[k] for k in ('plan', 'execute', 'review'))
            os.environ['ALLOYGEN_RAG_SESSION'] = f'delegation_{serial:03d}'
            yield
    finally:
        if previous is None:
            os.environ.pop('ALLOYGEN_RAG_SESSION', None)
        else:
            os.environ['ALLOYGEN_RAG_SESSION'] = previous


def preservation():
    manifest = json.loads((HERE/'checks/protected_inputs.json').read_text())
    changed = [p for p,h in manifest.items()
               if not (support.REPO/p).is_file()
               or hashlib.sha256((support.REPO/p).read_bytes()).hexdigest()!=h]
    if changed:
        raise AssertionError('Protected input changed: ' + repr(changed))
    return len(manifest)


def begin_run(run_id):
    from .sync_references import check_references
    check_references()
    support.PROTOCOL = json.loads((HERE/'protocol.json').read_text())
    from .provider_guard import RecordedTrial, ACCOUNTING_PERIOD
    trial = RecordedTrial(run_id, 'guided_fixed_task_G1', HERE.name)
    trial.state['accounting_period'] = ACCOUNTING_PERIOD
    # Reuse the demonstration's contained cache setup and MiniLM embedding cache.
    from baseline_support import configure_paths
    configure_paths(trial.runtime)
    # Reuse the demo's tokenizer resources without a first-use network download.
    cached = support.REPO / 'experiments/tool_using_demo_reproduction/runtime/equilibrium_repeat_20261028_01/tiktoken'
    for name in ('9b5ad71b2ce5302211f9c61530b329a4922fc6a4', 'fb374d419588a4632f3f557e76b4b70aebbca790'):
        shutil.copy2(cached / name, Path(os.environ['TIKTOKEN_CACHE_DIR']) / name)
    os.environ['ALLOYGEN_G1_OUTPUT'] = str(trial.out)
    for source in [*HERE.glob('*.md'), *HERE.glob('guides/*.md'),
                   *HERE.glob('baseline_guides/*.md'), *HERE.glob('codebooks/*.md'),
                   *HERE.glob('references/*/*.md'),
                   *HERE.glob('cases/*.json'), BASE/'notebook_runtime.py']:
        relative = source.relative_to(BASE)
        target = trial.out/'setup'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        trial.state['source_hashes'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    trial.write()
    return trial


def manager_message():
    catalog = json.loads((HERE/'agents.json').read_text())
    from .tools import render_text
    return render_text((HERE/'manager.md').read_text() + '\n\n' + (HERE/'codebooks/manager.md').read_text()) + '\nAvailable expertise:\n' + '\n'.join(
        f"- {a['name']}: {a['description']}" for a in catalog)


def final_answer(sender, recipient, summary_args):
    from .checkpoint import FullDesignEvidence
    prose = (recipient.last_message(sender) or {}).get('content') or ''
    packet = FullDesignEvidence().shared(mode='mission')
    return (str(prose or '') + '\n\nFactual comparison from saved tool records (host-generated):\n'
        + packet['comparison']['table'] + '\n\nMechanical evidence status (not scientific approval):\n'
        + json.dumps(packet['completion']))


def initial_assignment(task):
    """Use existing measured readiness at startup; manager chooses the next job."""
    from .checkpoint import FullDesignEvidence
    packet = FullDesignEvidence().shared(mode='mission')
    summary = {k: packet[k] for k in ('current_action', 'selection_context', 'completion')}
    return task + '\n\nCurrent investigation readiness (measured state, not a recommendation):\n' + json.dumps(summary)
