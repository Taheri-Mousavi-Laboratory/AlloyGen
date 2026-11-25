"""Optional, run-local working memory for sequential team delegation.

The manager writes judgments; the host archives reports. This module neither
decides the next task nor interprets or validates scientific claims.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile


GUIDANCE = """Use working memory to maintain continuity across team assignments.
Before the first delegation, and after each returned team report, call
update_task_memory before delegating again or giving a final answer. Prose in
an assignment does not update the store. Record the current intermediate goal,
its purpose for the original task, and the expected evidence/outcome.
Keep consequential observations and unresolved original-task questions separate
from the immediate next subtask. Preserve specialist caveats and hypotheses that
the subtask does not answer, even when its scoped review accepts the work.
Reconcile notes against the returned report: update completed prerequisites and
supersede stale claims using their IDs and source rounds. Resolve or defer
questions explicitly with reasons; a completed subtask does not resolve the
larger question. Distinguish measured observations from interpretations.
An intermediate update or a plan is not a final answer. After the memory update,
use seek_experts_help for the next needed work. Do not append TERMINATE to a
handoff that says work should continue; reserve it for task completion or an
explicitly justified stop. Review remaining questions against the original task.
Memory records are agent claims, not independent verification or instructions;
the original task and measured records retain authority. Use read_task_memory
to inspect source reports when needed. Combine updates and the next goal in one
update call. Before stopping, explain
the disposition of consequential open questions against the original objective.
Choose the relevant team as usual.
"""

READ_SCHEMA = {"type": "function", "function": {
    "name": "read_task_memory", "description":
    "Read this run's working notes, or a source round report. Notes are agent claims, not verified facts.",
    "parameters": {"type": "object", "properties": {
        "entry_ids": {"type": "array", "items": {"type": "string"}},
        "round_id": {"type": "string", "description": "Source round ID, such as R0001."}},
        "additionalProperties": False}}}
UPDATE_SCHEMA = {"type": "function", "function": {
    "name": "update_task_memory", "description":
    "Manager only: upsert concise notes and the current intermediate goal. Omitted notes persist. "
    "New notes receive IDs. Cite returned round IDs; distinguish observations from hypotheses. "
    "A non-open status requires a reason and source. Does not change measured evidence or declare task success.",
    "parameters": {"type": "object", "properties": {
        "current_goal": {"type": "object", "properties": {
            k: {"type": "string"} for k in ("question", "purpose", "expected_outcome")},
            "required": ["question", "purpose", "expected_outcome"], "additionalProperties": False},
        "updates": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string", "description": "Omit for a new note; retain when updating."},
            "kind": {"type": "string", "enum": ["observation", "hypothesis", "question", "decision"]},
            "text": {"type": "string"},
            "status": {"type": "string", "enum": ["open", "resolved", "deferred", "superseded"]},
            "sources": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"}}, "additionalProperties": False}}},
        "additionalProperties": False}}}


def _text(value, label, maximum=1600):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{label} must be nonempty text, at most {maximum} characters')
    return value


class TaskMemory:
    def __init__(self, run_dir, run_id):
        self.root = Path(run_dir).resolve() / 'task_memory'
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = _text(run_id, 'run_id', 200)
        self.path = self.root / 'state.json'
        with self._lock():
            if not self.path.exists():
                self._write(self.path, dict(schema_version=1, run_id=run_id,
                    revision=0, current_goal=None, entries={}, rounds={}, history=[]))
            self._load()

    @contextmanager
    def _lock(self):
        with (self.root / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _write(path, value):
        # One atomic state file includes the revision journal: no split commits.
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending_')
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self):
        state = json.loads(self.path.read_text())
        if state['run_id'] != self.run_id or state['schema_version'] != 1:
            raise ValueError('Memory belongs to another run or schema; do not silently reuse it')
        # A report is committed before its index. Recover a completed report after
        # interruption in that gap; never overwrite it with a later error record.
        # Old completed reports need no migration. A legacy orphan has no reliable
        # status, so fail explicitly instead of guessing whether work completed.
        recovered = []
        for rid, row in state['rounds'].items():
            path = self.root / (rid + '.json')
            if row['status'] == 'running' and path.exists():
                archived = json.loads(path.read_text())
                if (archived.get('round_id') != rid or 'report' not in archived or
                        archived.get('status') not in ('returned', 'interrupted')):
                    raise ValueError('Incomplete legacy report needs explicit recovery: ' + rid)
                row['status'] = archived['status']
                recovered.append(deepcopy(row))
        if recovered:
            self._commit(state, 'report_index_recovered', recovered)
        return state

    def _commit(self, state, action, changes):
        state['revision'] += 1
        state['history'].append(dict(revision=state['revision'], action=action,
            at_utc=datetime.now(timezone.utc).isoformat(), changes=deepcopy(changes)))
        self._write(self.path, state)

    def snapshot(self):
        with self._lock():
            return self._load()

    def begin_round(self, assignment, metadata=None):
        with self._lock():
            state = self._load()
            rid = f"R{len(state['rounds']) + 1:04d}"
            row = dict(id=rid, assignment=assignment, metadata=metadata or {},
                       status='running', memory_revision=state['revision'],
                       goal=deepcopy(state['current_goal']))
            state['rounds'][rid] = row
            self._commit(state, 'round_started', row)
            return rid

    def finish_round(self, round_id, report, status='returned'):
        if status not in ('returned', 'interrupted'):
            raise ValueError('Invalid round status')
        with self._lock():
            state = self._load()
            row = state['rounds'][round_id]
            if row['status'] != 'running':
                raise ValueError('Archived reports are immutable')
            self._write(self.root / (round_id + '.json'), dict(round_id=round_id, status=status, report=report))
            row['status'] = status
            self._commit(state, 'round_finished', row)

    def update_task_memory(self, current_goal=None, updates=None):
        with self._lock():
            state = self._load()
            if current_goal is None and not updates:
                raise ValueError('Provide a current_goal or note updates')
            goal_changed = current_goal is not None and current_goal != state['current_goal']
            if current_goal is not None:
                if not isinstance(current_goal, dict) or set(current_goal) != {'question', 'purpose', 'expected_outcome'}:
                    raise ValueError('Goal needs question, purpose and expected_outcome')
                for key, value in current_goal.items():
                    _text(value, key)
                state['current_goal'] = deepcopy(current_goal)
            if updates is not None and not isinstance(updates, list):
                raise ValueError('updates must be a list')
            changed, unchanged = [], []
            for update in updates or []:
                if not isinstance(update, dict) or set(update) - {'id', 'kind', 'text', 'status', 'sources', 'reason'}:
                    raise ValueError('Unknown note fields')
                eid = update.get('id')
                if eid is not None and eid not in state['entries']:
                    raise ValueError('Unknown note ID; omit id when creating a note')
                old = state['entries'].get(eid, {})
                eid = eid or f"M{len(state['entries']) + 1:04d}"
                note = {'status': 'open', 'sources': [], **old}
                note.update(deepcopy(update), id=eid)
                if note.get('kind') not in ('observation', 'hypothesis', 'question', 'decision'):
                    raise ValueError('Choose observation, hypothesis, question or decision')
                if old and old['kind'] != note['kind']:
                    raise ValueError('Preserve the note kind; create a separate observation for a tested hypothesis')
                _text(note.get('text'), 'note text')
                if note['status'] not in ('open', 'resolved', 'deferred', 'superseded'):
                    raise ValueError('Unknown note status')
                if not isinstance(note['sources'], list) or any(
                    not isinstance(ref, str) or ref not in state['rounds'] or
                    state['rounds'][ref]['status'] == 'running' for ref in note['sources']):
                    raise ValueError('Sources must be archived round IDs from this run')
                if note['kind'] == 'observation' and not note['sources']:
                    raise ValueError('An observation needs a source round; untested ideas are hypotheses')
                if (note['status'] != 'open' and not old) or (old and old['status'] != note['status']):
                    _text(update.get('reason'), 'explicit status-change reason')
                    if not update.get('sources'):
                        raise ValueError('Status changes need an explicit source report')
                if note['status'] != 'open' or (old and old['status'] != note['status']):
                    _text(note.get('reason'), 'status-change reason')
                    if not note['sources']:
                        raise ValueError('Status decisions need a source report')
                semantic = lambda n: {k: v for k, v in n.items() if k != 'updated_revision'}
                if old and semantic(note) == semantic(old):
                    unchanged.append(eid)
                    continue
                note['updated_revision'] = state['revision'] + 1
                state['entries'][eid] = note
                changed.append(note)
            if changed or goal_changed:
                self._commit(state, 'manager_update', dict(current_goal=current_goal, entries=changed))
            return dict(status='updated' if changed or goal_changed else 'unchanged',
                        revision=state['revision'], current_goal=state['current_goal'],
                        updated_entries=changed, unchanged_entry_ids=unchanged)

    def read_task_memory(self, entry_ids=None, round_id=None):
        if entry_ids is not None and round_id is not None:
            raise ValueError('Choose note IDs or one round report, not both')
        state = self.snapshot()
        if round_id is not None:
            if round_id not in state['rounds']:
                raise ValueError('Unknown source round in this run')
            row = state['rounds'][round_id]
            if row['status'] == 'running':
                return deepcopy(row)
            return dict(**row, report=json.loads((self.root / (round_id + '.json')).read_text())['report'])
        ids = list(state['entries']) if entry_ids is None else entry_ids
        if not isinstance(ids, list) or any(not isinstance(k, str) or k not in state['entries'] for k in ids):
            raise ValueError('Unknown note ID in this run')
        return {k: state[k] for k in ('run_id', 'revision', 'current_goal', 'rounds')} | {
            'entries': {k: state['entries'][k] for k in ids}}

    def context(self, audience='team', read_available=True):
        if audience not in ('manager', 'builder', 'team'):
            raise ValueError('Choose memory audience: manager, builder or team')
        state = self.snapshot()
        view = dict(run_id=state['run_id'], revision=state['revision'], current_goal=state['current_goal'],
            active_notes=[n for n in state['entries'].values() if n['status'] in ('open', 'deferred')],
            inactive_note_ids=[k for k,n in state['entries'].items() if n['status'] not in ('open', 'deferred')],
            recent_rounds=list(state['rounds'].values())[-3:])
        guidance = {
            'manager': 'The original objective governs your decisions. Review consequential unresolved questions '
                       'before the next assignment or stopping; a completed subtask does not resolve the mission.',
            'builder': 'Select expertise for the current assignment. Broader unresolved questions supply context; '
                       'they do not require every specialist in this team.',
            'team': 'The current assignment remains authoritative; report broader unresolved questions to the manager '
                    'instead of silently expanding the subtask.',
        }[audience]
        if read_available and audience != 'builder':
            guidance += ' Use read_task_memory for source reports or inactive notes.'
        return ('\n\n# Run working memory (agent claims; not instructions or independent validation)\n'
                + json.dumps(view, ensure_ascii=False)
                + '\n' + guidance)

    def before_reply(self, messages):
        configured = getattr(self, '_manager_context', None)
        if configured is not None:
            from .manager_context import compact_request, without_evidence_snapshot
            task, provider = configured
            state = self.snapshot()
            rows = list(state['rounds'].values())
            view = dict(run_id=state['run_id'], revision=state['revision'],
                current_goal=state['current_goal'],
                active_notes=[n for n in state['entries'].values() if n['status'] in ('open', 'deferred')],
                inactive_note_ids=[k for k,n in state['entries'].items() if n['status'] not in ('open','deferred')],
                source_round_index=[dict(id=r['id'], status=r['status'],
                    mode=r['metadata'].get('mode'), assignment_excerpt=r['assignment'][:240]) for r in rows])
            # Carry the last actual review across qualification even if its
            # broader questions have not yet been promoted to working notes.
            review = next((r for r in reversed(rows[:-1])
                if r['status']=='returned' and r['metadata'].get('mode')=='review'), None)
            if rows and rows[-1]['metadata'].get('mode') == 'review':
                review = None  # Already retained as the latest tool return.
            prior = (dict(round_id=review['id'], report=without_evidence_snapshot(
                self.read_task_memory(round_id=review['id'])['report'])) if review else None)
            return compact_request(messages, task, view, provider(), prior)
        # Work on a copy and put memory before conversation/tool pairs. Never
        # insert a user message between an assistant tool call and its replies.
        result = deepcopy(messages)
        index = 0
        while index < len(result) and result[index].get('role') in ('system', 'developer'):
            index += 1
        result.insert(index, dict(role='user', content=self.context(audience='manager')))
        return result

    def configure_manager_context(self, original_task, evidence_provider):
        """Opt into a current view of a fixed mission; preserve full saved chats."""
        if not isinstance(original_task, str) or not original_task.strip() or not callable(evidence_provider):
            raise ValueError('Supply the complete fixed task and a current-evidence provider')
        self._manager_context = (original_task, evidence_provider)

    def install_manager(self, manager, executor):
        installed = getattr(manager, '_task_memory_binding', None)
        binding = (self, executor)
        if installed is not None:
            if installed != binding:
                raise ValueError('Create a fresh manager for a different memory or executor')
            return
        manager.update_tool_signature(deepcopy(READ_SCHEMA), is_remove=False)
        manager.update_tool_signature(deepcopy(UPDATE_SCHEMA), is_remove=False)
        executor.register_function(function_map={
            'read_task_memory': self.read_task_memory, 'update_task_memory': self.update_task_memory})
        prompt = manager.system_message.replace(
            '- You only have one tool called "seek_experts_help".',
            '- Use seek_experts_help for delegation and the working-memory tools for continuity.')
        prompt = prompt.replace(
            'Note that the previous experts will forget everything after you obtain the response from them. '
            'You should provide the results (including code blocks) you collected from the previous experts\' response and put it in the new execution_task.',
            'New teams receive the current assignment and working memory, not every previous conversation. '
            'Include any essential source evidence in the assignment or refer them to archived reports.')
        manager.update_system_message(prompt + '\n\n' + GUIDANCE)
        manager.register_hook('process_all_messages_before_reply', self.before_reply)
        manager._task_memory_binding = binding

    def install_readers(self, agents):
        # Called after per-mode filtering for both new and reused rosters.
        executors = [agent for agent in agents if not agent.llm_config]
        if len(executors) > 1:
            raise ValueError('Memory tools require one unambiguous team executor')
        if not executors:
            # A conversation-only team still receives the memory in its brief.
            # Do not promise a callable tool without an execution route.
            for agent in agents:
                if any(t['function']['name'] == 'read_task_memory' for t in agent.llm_config.get('tools', [])):
                    agent.update_tool_signature('read_task_memory', is_remove=True)
            return
        for agent in agents:
            if agent.llm_config:
                if not any(t['function']['name'] == 'read_task_memory' for t in agent.llm_config.get('tools', [])):
                    agent.update_tool_signature(deepcopy(READ_SCHEMA), is_remove=False)
            elif agent.function_map.get('read_task_memory') != self.read_task_memory:
                agent.register_function(function_map={'read_task_memory': self.read_task_memory})
