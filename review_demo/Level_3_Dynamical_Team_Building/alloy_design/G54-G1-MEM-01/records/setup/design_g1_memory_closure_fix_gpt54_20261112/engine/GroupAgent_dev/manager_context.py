"""Compact input for an opted-in fixed mission; raw chats/archives stay intact."""
from copy import deepcopy
import json

EVIDENCE_MARKER = '\n\n# Shared evidence from saved records (not a scientific verdict)\n'


def without_evidence_snapshot(text):
    return text.split(EVIDENCE_MARKER, 1)[0] if isinstance(text, str) else text


def latest_delegation_start(messages):
    """Return the assistant-call boundary of the last completed delegation."""
    pending, latest = {}, None
    for i, message in enumerate(messages):
        for call in message.get('tool_calls') or []:
            pending[call['id']] = (i, call['function']['name'])
        replies = message.get('tool_responses') or ([message] if message.get('tool_call_id') else [])
        for reply in replies:
            previous = pending.get(reply['tool_call_id'])
            if previous and previous[1] == 'seek_experts_help':
                latest = previous[0]
    return latest


def compact_request(messages, task, memory_view, evidence, previous_review=None):
    """Keep current tool exchanges paired; replace historical evidence copies.

    The supplied task is the complete, fixed original mission. This opt-in view
    is not intended to infer additional human instructions from old chat text.
    Source reports remain available through read_task_memory.
    """
    start = latest_delegation_start(messages)
    if start is None:
        start = next((i for i, m in enumerate(messages)
                      if m.get('role') == 'assistant'), len(messages))
    tail = deepcopy(messages[start:])
    delegation_ids = {c['id'] for m in tail for c in m.get('tool_calls') or []
                      if c['function']['name'] == 'seek_experts_help'}
    for message in tail:
        replies = message.get('tool_responses') or ([message] if message.get('tool_call_id') else [])
        for reply in replies:
            if reply.get('tool_call_id') in delegation_ids:
                reply['content'] = without_evidence_snapshot(reply.get('content'))
        if message.get('tool_responses'):
            message['content'] = '\n\n'.join(str(r.get('content') or '') for r in replies)
    prefix = [deepcopy(m) for m in messages[:start] if m.get('role') in ('system', 'developer')]
    prefix.append(dict(role='user', content=task))
    view = dict(working_memory=memory_view, current_measured_evidence=evidence)
    if previous_review:
        view['previous_review_at_its_source_round'] = previous_review
    prefix.append(dict(role='user', content=(
        '# Current investigation context\n' + json.dumps(view, ensure_ascii=False)
        + '\nNotes and previous reviews are agent claims, not independent validation. '
        'Any startup readiness in the original task and historical review statements refer '
        'to their source time; current measured records '
        'determine calculation availability. The latest returned assignment follows. '
        'A local review decision closes only that assignment. Reconcile the notes, then '
        'delegate remaining work or justify whole-mission closure. Older reports are '
        'retrievable by round ID using read_task_memory.')))
    return prefix + tail
