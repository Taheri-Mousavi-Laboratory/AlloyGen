"""Short repository-local execution paths, independent of descriptive run IDs."""
import hashlib
import os
from pathlib import Path

from .baseline_support import REPO, contained, save_json, sha

# The local 2024a diagnostic truncated an absolute state-file path at 256 bytes.
# Reserve eight bytes, plus the observed system/cache collision suffix below.
MAX_TC_PATH_BYTES = 248
CACHE_RELATIVE_BYTES = 48
TC_STATE_SUFFIX = ('System-' + '0' * 32 + '/-clash-' + '0' * 36 + '/System.poly3')


def path_bytes(path):
    return len(os.fsencode(str(path)))


def check_cache_path(cache_path):
    """Check the actual resolved cache location with the longest observed suffix."""
    cache_path = contained(cache_path)
    longest = cache_path / TC_STATE_SUFFIX
    length = path_bytes(longest)
    if length > MAX_TC_PATH_BYTES:
        raise ValueError(
            f'Thermo-Calc cache path exceeds the runtime budget: {length} bytes '
            f'with state-file suffix; maximum {MAX_TC_PATH_BYTES}. Use a shorter directory.')
    return length


def proposed_layout(run_id):
    """Fail before execution if this checkout leaves insufficient path headroom."""
    token = hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:12]
    root = contained(REPO / '.tc' / token)
    layout = {name: root / short for name, short in
              [('code', 'c'), ('temporary', 't'), ('java_home', 'j')]}
    check_cache_path(layout['code'] / ('x' * CACHE_RELATIVE_BYTES))
    check_cache_path(layout['temporary'])
    return {'root': root, **layout}


def install_short_runtime(run):
    """Apply before creating agents; prompts, scientific inputs and tools stay intact."""
    if run.state.get('api_requests') or run.state.get('code_executions'):
        raise RuntimeError('Runtime paths must be selected before any model or code execution')
    layout = proposed_layout(run.state['run_id'])
    layout['root'].parent.mkdir(parents=True, exist_ok=True)
    layout['root'].mkdir(exist_ok=False)  # Never reuse a previous trial/cache.
    for name in ['code', 'temporary', 'java_home']:
        layout[name].mkdir()
    save_json(layout['root'] / 'owner.json', {'run_id': run.state['run_id']})
    run.code_dir = layout['code']
    for name in ['TMPDIR', 'TMP', 'TEMP']:
        os.environ[name] = str(layout['temporary'])
    os.environ['JAVA_TOOL_OPTIONS'] = ' '.join([
        f'-Djava.io.tmpdir={layout["temporary"]}',
        f'-Duser.home={layout["java_home"]}', '-XX:-UsePerfData',
        f'-XX:ErrorFile={layout["root"] / "hs_err_pid%p.log"}',
        f'-XX:HeapDumpPath={layout["root"]}',
    ])
    run.state['solver_temp_directory'] = str(layout['temporary'])
    run.state['runtime_path_policy'] = {
        'version': 'short-tc-runtime-v1',
        'directories_relative_to_repository': {k: str(v.relative_to(REPO)) for k,v in layout.items()},
        'executor_directory_bytes': path_bytes(run.code_dir),
        'reserved_cache_relative_bytes': CACHE_RELATIVE_BYTES,
        'maximum_tc_path_bytes': MAX_TC_PATH_BYTES,
        'reserved_state_path_bytes': check_cache_path(run.code_dir / ('x' * CACHE_RELATIVE_BYTES)),
        'scope': 'Executor working and solver temporary directories only; no generated code, prompt, composition, result or solver-setting modification.',
    }
    source = Path(__file__)
    destination = run.results / 'setup' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    run.state.setdefault('implementation_sources', []).append(
        {'path': source.name, 'sha256': sha(source)})
    run.save()
    return layout


class ShortRuntimeMixin:
    """Compose with DemoRun or an existing budgeted run without changing its logic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'runtime_path_policy' not in self.state:
            install_short_runtime(self)
