"""Small delivery/execution helpers; no scientific selection or verdict."""
import ast
from copy import deepcopy
import json
import re


def termination_requested(message):
    return not message.get('tool_calls') and review_decision(str(message.get('content') or '')) == 'TERMINATE'


def review_decision(text):
    """Read an explicit current decision line or the legacy final-paragraph form.

    Exclude fenced/quoted material and reject conflicting explicit decisions.
    A current Decision: line may have explanatory prose after it. This extracts
    the agent's declared control decision; it does not judge scientific validity.
    """
    if text.count('```') % 2 or text.count('~~~') % 2:
        return None
    visible = re.sub(r'```[\s\S]*?```|~~~[\s\S]*?~~~', '', text)
    visible = '\n'.join(line for line in visible.splitlines() if not line.lstrip().startswith('>'))
    plain = re.sub(r'\*\*|__|`', '', visible)
    if any(len(set(re.findall(r'\b(?:TERMINATE|REVISE)\b', line))) > 1
           for line in plain.splitlines() if re.match(r'^\s*(?:Decision|Verdict|Action)\s*:', line)):
        return None
    explicit = set(re.findall(r'(?m)^\s*(?:Decision|Verdict|Action)\s*:\s*(TERMINATE|REVISE)(?:[.!:\s-]|$)', plain))
    terminal = _terminal_decision(text)
    decisions = explicit | ({terminal} if terminal else set())
    return next(iter(decisions)) if len(decisions) == 1 else None


def _terminal_decision(text):
    """Legacy form retained for existing transcripts without a decision label."""
    paragraph = re.split(r'\n\s*\n', text.strip())[-1]
    if not paragraph or '```' in paragraph or any(line.lstrip().startswith('>') for line in paragraph.splitlines()):
        return None
    plain = re.sub(r'\*\*|__|`', '', paragraph).strip()
    tokens = set(re.findall(r'\b(?:TERMINATE|REVISE)\b', plain))
    if len(tokens) != 1:
        return None
    start = re.match(r'(?:(?:Action|Verdict|Decision)\s*:\s*)?(TERMINATE|REVISE)(?:[.!:]|\s|$)', plain)
    end = re.search(r'(?:^|[.!?]\s+|\n)(TERMINATE|REVISE)[.!]?\s*$', plain)
    return (start or end).group(1) if start or end else None


def substantive_review(text):
    plain = re.sub(r'\*\*|__|`', '', text).strip()
    return bool(plain) and re.fullmatch(r'(?:(?:Action|Verdict|Decision)\s*:\s*)?(?:TERMINATE|REVISE)[.!]?', plain) is None


def correction_fields(text):
    """Read explicit routing labels, including Markdown headings; ignore examples."""
    if text.count('```') % 2 or text.count('~~~') % 2:
        return {}
    visible = re.sub(r'```[\s\S]*?```|~~~[\s\S]*?~~~', '', text)
    visible = '\n'.join(line for line in visible.splitlines() if not line.lstrip().startswith('>'))
    visible = re.sub(r'\*\*|__|`', '', visible)
    pairs = re.findall(
        r'(?mi)^\s*(?:#{1,6}\s+)?(?:[-*+]\s+)?Correction (kind|owner|operation):\s*([A-Za-z_][A-Za-z_0-9]*)\s*$',
        visible)
    fields = {}
    for key, value in pairs:
        fields.setdefault(key.lower(), set()).add(value)
    return fields


def delegation_handoff(content):
    """Decode a tool's explicit prerequisite failure, not an agent's prose."""
    try:
        result = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        try:
            result = ast.literal_eval(content)
        except (ValueError, SyntaxError, TypeError):
            return None
    if not isinstance(result, dict) or result.get('status') != 'delegation_required':
        return None
    step = result.get('next_delegation')
    if (not isinstance(step, dict) or step.get('mode') not in ('plan', 'execute', 'review')
            or not all(isinstance(step.get(k), str) and step[k] for k in ('owner', 'operation'))):
        return None
    return dict(next_delegation=step, reason=result.get('reason', 'Required operation is outside this delegation.'))


def previous_team_context(result):
    """Deliver findings/proposals without recycling a completed critic verdict.

    The full return remains in manager messages and the archived run. Explicit
    corrections, incomplete/unknown reviews and current-cycle feedback are not
    discarded. Support both live-host and imported verbatim-return field names.
    """
    view = deepcopy(result)
    history, feedback = [], []
    for field in ('scientific_review', 'review'):
        raw = view.pop(field, None)
        if not isinstance(raw, str) or not raw.strip():
            continue
        status = result.get('scientific_review_status')
        body = raw
        wrapper = re.search(r'Scientific critic review \(([^)]+)\):\n', raw)
        if wrapper:
            status = status or wrapper.group(1)
            body = raw[wrapper.end():].rsplit('\nController status:', 1)[0]
        decision = review_decision(body)
        completed = (decision == 'TERMINATE' and substantive_review(body)
                     and status in (None, 'completed'))
        history.append(dict(field=field, status=status or 'unknown',
                            completed_verdict_omitted=completed))
        if not completed:
            feedback.append(dict(field=field, text=raw,
                note='Historical unresolved feedback; retain its correction or limitation in the new assignment.'))
    if history:
        view['previous_review_record'] = dict(entries=history,
            note='Historical procedural status only. Full reviews remain in the manager return/archive; assess current evidence and worker claims anew.')
    if feedback:
        view['unresolved_review_feedback'] = feedback
    return view


def tool_activity(messages, excluded=()):
    """Keep exact call arguments and matched outcomes, including content-null calls.

    Successful responses stay in the evidence ledger; errors are returned here.
    A returned response is not a scientific-validity claim. Unanswered calls are
    explicitly unknown, and mode-blocked calculations are unexecuted proposals.
    """
    replies = {}
    for message in messages:
        for reply in message.get('tool_responses') or ([message] if message.get('tool_call_id') else []):
            replies[reply['tool_call_id']] = reply.get('content')
    activity = []
    for index, message in enumerate(messages[1:], 1):
        if message.get('name') in excluded:
            continue
        for call in message.get('tool_calls') or []:
            function = call['function']
            entry = dict(caller=message.get('name'), message_index=index,
                         tool_call_id=call['id'], tool=function['name'],
                         arguments=function['arguments'])
            content = replies.get(call['id'])
            if call['id'] not in replies:
                entry['outcome'] = 'response_missing_execution_unknown'
            elif isinstance(content, str) and content.startswith('Error:'):
                blocked = content.startswith('Error: Scientific calculation is unavailable in ')
                entry.update(outcome='unexecuted_proposal' if blocked else 'reported_error', error=content)
            else:
                entry['outcome'] = 'response_returned_not_scientific_approval'
                try:
                    result = ast.literal_eval(content) if isinstance(content, str) else content
                except (ValueError, SyntaxError):
                    result = None
                if isinstance(result, dict):
                    entry['result_summary'] = {k: result[k] for k in ('status', 'batch_id', 'plan_version') if k in result}
                handoff = delegation_handoff(content)
                if handoff:
                    entry.update(outcome='delegation_required_no_calculation', **handoff)
            activity.append(entry)
    return activity


def worker_contributions(messages, excluded=()):
    """Retain ordered prose, including tool-bearing turns and earlier findings."""
    findings = {}
    for index, item in enumerate(messages[1:], 1):
        name, content = item.get('name'), item.get('content')
        if (name and name not in excluded and isinstance(content, str)
                and content.strip() and content.strip() != 'TERMINATE'
                and item.get('role') not in ('tool', 'function')):
            findings.setdefault(name, []).append({'message_index': index, 'content': content})
    return findings


def install_duplicate_guard(executor, names):
    """Suppress identical selected tool calls only within ONE assistant response.

    Every call ID gets a response. Later identical requests still execute, so
    deliberate verification remains possible. Uses AG2's normal sync dispatch.
    """
    executor.duplicate_events = []

    def guarded_reply(recipient, messages=None, sender=None, config=None):
        messages = messages if messages is not None else recipient.chat_messages[sender]
        calls = messages[-1].get('tool_calls') or []
        keys = []
        for call in calls:
            function = call.get('function', {})
            try:
                key = (function['name'], json.dumps(json.loads(function['arguments']), sort_keys=True))
            except (KeyError, ValueError, TypeError):
                key = None  # Let normal tool error handling explain malformed input.
            keys.append(key if function.get('name') in names else None)
        selected = [key for key in keys if key is not None]
        if len(selected) == len(set(selected)):
            return False, None
        seen, responses = {}, []
        for call, key in zip(calls, keys):
            identifier = call['id']
            if key is not None and key in seen:
                event = {'suppressed_call_id': identifier, 'original_call_id': seen[key],
                         'tool': call['function']['name']}
                recipient.duplicate_events.append(event)
                content = json.dumps(dict(event, status='duplicate_not_executed',
                    instruction='Use the original call response. No second computation occurred.'))
            else:
                _, result = recipient.execute_function(call['function'], call_id=identifier)
                content = result.get('content') or ''
                if key is not None:
                    seen[key] = identifier
            responses.append({'role': 'tool', 'tool_call_id': identifier, 'content': content})
        return True, {'role': 'tool', 'tool_responses': responses,
                      'content': '\n\n'.join(str(r['content']) for r in responses)}

    executor.register_reply([object, None], guarded_reply, position=0)
    return guarded_reply


class CompletionGuard:
    """Give the manager one actionable pending-work response before accepting stop."""
    def __init__(self, evidence_provider, max_repairs=1):
        self.evidence_provider = evidence_provider
        self.max_repairs = max_repairs
        self.repairs = 0
        self.events = []

    def reply(self, recipient, messages=None, sender=None, config=None):
        messages = messages if messages is not None else recipient.chat_messages[sender]
        message = messages[-1]
        if not termination_requested(message):
            return False, None
        completion = self.evidence_provider()['completion']
        pending = completion['pending']
        action = 'accept_mechanical_completion' if not pending else (
            'return_pending_work' if self.repairs < self.max_repairs else 'stop_incomplete')
        self.events.append({'action': action, 'pending': list(pending)})
        if action != 'return_pending_work':
            return False, None
        self.repairs += 1
        return True, ('Required work remains according to the saved evidence:\n'
            + json.dumps(pending) + '\nIdentify the current unresolved question and delegate work '
            'that can resolve it within the existing mission and allowances. Distinguish a report '
            'correction, a design refinement, missing exact-composition calculations, and an unrecorded '
            'nomination. For closure, record the planner choice in execute mode before critic verification '
            'in review mode. Use the next_delegation in the saved prerequisite feedback. Reuse a '
            'capable team; rebuilding is optional. A missing result alone is unfinished work, not '
            'a reason it cannot be obtained. If further work is not justified or possible, explain '
            'why and label the result incomplete. This is a mechanical completion check, not a '
            'scientific verdict. One completion reminder remains; continue the existing '
            'investigation rather than restarting it.')
