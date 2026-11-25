"""Thin scientific interfaces and a run-local evidence ledger; no alloy selection."""
import json
import hashlib
import math
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from scientific_tools import AlloyDesignTools
from scientific_tools.tools import candidate, KINDS, REPO
from .qualification import result_qualification, usable_result

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
_MODE = ContextVar('alloy_design_delegation_mode', default=None)
_PLAN_START = ContextVar('alloy_design_plan_at_delegation_start', default=None)


def render_text(text):
    return text.replace('{{MAX_BATCH_SIZE}}', str(PROTOCOL['max_batch_size']))

# Expose the same field names/types checked by record_design_plan to the model.
NEXT_ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'question': {'type': 'string'},
        'tool': {'type': 'string', 'enum': ['evaluate_coarsening',
            'evaluate_phase_stability', 'evaluate_solidification']},
        'compositions': {'type': 'array', 'minItems': 1, 'maxItems': PROTOCOL['max_batch_size'],
            'description': 'Candidate at.%; Al balance. Supply this OR candidate_ids.',
            'items': {'type': 'object', 'properties': {
                k: {'type': 'number', 'minimum': 0, 'maximum': maximum}
                for k, maximum in [('Ni', 4), ('Er', 2), ('Zr', 2)]},
                'required': ['Ni', 'Er', 'Zr'], 'additionalProperties': False}},
        'candidate_ids': {'type': 'array', 'minItems': 1, 'maxItems': PROTOCOL['max_batch_size'],
            'items': {'type': 'string'}, 'description': 'Existing IDs instead of compositions.'},
        'decision': {'type': 'string'},
        'comparison_ids': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['question', 'tool', 'decision'],
    'additionalProperties': False,
}


@contextmanager
def delegation_scope(mode, execute_intent=None):
    """Host-owned scope, reset even if a builder or tool fails."""
    if mode not in ('plan', 'execute', 'review'):
        raise ValueError('Choose delegation mode: plan, execute or review')
    from .checkpoint import enter_delegation
    enter_delegation(mode)
    start = len(load()['plan'])
    if mode == 'execute' and not start:
        raise ValueError('No accepted plan exists. Delegate Alloy_Design_Planner in plan mode '
                         'to repair record_design_plan before building an execution team.')
    if mode == 'execute':
        from .checkpoint import required_operation, guard_action
        operation=required_operation(mode, execute_intent=execute_intent)
        if operation.startswith('evaluate_'):
            guard_action(load())
    token = _MODE.set(mode)
    plan_token = _PLAN_START.set(start)
    try:
        yield
    finally:
        _PLAN_START.reset(plan_token)
        _MODE.reset(token)


def require_scope(operation, modes):
    if _MODE.get() not in modes:
        raise ValueError(f'{operation} is unavailable in {_MODE.get() or "unassigned"} mode; '
                         f'delegate it in {" or ".join(modes)} mode')


def folder():
    path = Path(os.environ['ALLOYGEN_G1_OUTPUT']).resolve()
    if not path.is_relative_to(HERE.parent / 'runs'):
        raise ValueError('Use a new run folder under agent_development/runs')
    path.mkdir(parents=True, exist_ok=True)
    return path


def load():
    path = folder() / 'evidence.json'
    return json.loads(path.read_text()) if path.exists() else dict(
        candidates={}, reference={}, calls=[], plan=[], recommendations=[], verification=None)


def save(state):
    (folder() / 'evidence.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')


def identity(comp):
    return json.dumps(candidate(comp), sort_keys=True)


def checked(compositions):
    if not isinstance(compositions, list) or len(compositions) > PROTOCOL['max_batch_size']:
        raise ValueError(f"Supply at most {PROTOCOL['max_batch_size']} candidate dictionaries")
    comps = [candidate(c) for c in compositions]
    for comp in comps:
        if not (comp['Ni'] <= 4 and comp['Er'] <= 2 and comp['Zr'] <= 2):
            raise ValueError('Outside task domain: Ni 0–4, Er/Zr 0–2 at.%')
    if len({identity(c) for c in comps}) != len(comps):
        raise ValueError('Remove duplicate compositions within this batch')
    return comps


def fr_diagnostics(row):
    """Describe the saved FR interval, without changing the metric."""
    for record in row.get('records', []):
        path = (REPO / record).resolve()
        if not path.is_relative_to(folder()):
            continue
        raw = json.loads(path.read_text())
        curve = raw.get('curves', {}).get('solid_mole_fraction')
        if not curve:
            continue
        points = sorted(zip(curve['temperature_K'], curve['values']), reverse=True)
        start = min(points, key=lambda pair: abs(pair[1] - .02))
        return dict(start_temperature_C=start[0]-273.15,
                    start_solid_mole_fraction=start[1],
                    terminal_temperature_C=points[-1][0]-273.15)
    return None


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('records', 'composition_percent', 'basis')}
    if 'Al3Zr_boundary' in row:
        result['qualification'] = result_qualification('phase_stability', row)
    return result


def evaluate(kind, compositions, reason):
    require_scope('Scientific calculation', ('execute',))
    if not load()['plan']:
        raise ValueError('Record a concrete first-action plan in plan mode before calculations')
    comps = checked(compositions)
    from .checkpoint import guard_action
    guard_action(load(), kind, comps)
    if not reason.strip():
        raise ValueError('State what decision this batch supports')
    state = load()
    existing = {identity(v['composition_at_percent']): k for k, v in state['candidates'].items()}
    new = {identity(c) for c in comps} - set(existing)
    if len(existing) + len(new) > PROTOCOL['max_distinct_candidates']:
        raise ValueError('Distinct candidate allowance exhausted; use evaluated evidence')
    limit = {'phase_stability': 'max_candidate_phase_profiles',
             'solidification': 'max_candidate_scheil'}.get(kind)
    attempted = {identity(c) for call in state['calls'] if call['kind'] == kind
                 for c in call['compositions']}
    if limit and len(attempted | {identity(c) for c in comps}) > PROTOCOL[limit]:
        raise ValueError(f'{kind} candidate allowance exceeded; reduce this batch')
    ids = []
    for comp in comps:
        key = identity(comp)
        if key not in existing:
            cid = f'C{len(state["candidates"])+1:03d}'
            state['candidates'][cid] = dict(composition_at_percent=comp, methods={})
            existing[key] = cid
        ids.append(existing[key])
    call = dict(id=f'B{len(state["calls"])+1:03d}', kind=kind, compositions=comps,
                candidate_ids=ids, new_candidate_ids=[cid for cid, comp in zip(ids, comps)
                    if identity(comp) in new], reason=reason, status='started')
    state['calls'].append(call)
    save(state)  # Decision is recorded before calculation, including failed attempts.
    response = getattr(AlloyDesignTools(folder() / 'search'), 'evaluate_' + kind)(comps)
    response_path = folder() / (call['id'] + '.json')
    response_path.write_text(json.dumps(response, indent=2, allow_nan=False) + '\n')
    call.update(status=response['status'], response_file=response_path.name,
                usage=response.get('usage'), seconds=response.get('seconds'),
                fingerprint=response.get('method_fingerprint'))
    rows = [('reference', response.get('reference'))] + list(zip(ids, response.get('results', [])))
    output = dict(batch_id=call['id'], reason=reason, status=response['status'],
                  new_candidate_ids=call['new_candidate_ids'], results={})
    for cid, row in rows:
        if row is None:
            continue
        row = dict(row, evidence_id=f'{call["id"]}:{cid}:{kind}')
        if kind == 'solidification':
            row['FR_interval'] = fr_diagnostics(row)
        if cid == 'reference':
            state['reference'][kind] = row
        else:
            state['candidates'][cid]['methods'][kind] = row
        output['results'][cid] = dict(compact(row), composition_percent=row['composition_percent'], basis=row['basis'])
    save(state)
    output['allowances'] = allowances(state)
    return output


def evaluate_coarsening(compositions: list[dict[str, float]], reason: str) -> dict:
    """Calculate J at 250°C for up to {{MAX_BATCH_SIZE}} Ni/Er/Zr at.% candidates; Al balance. State the sampling reason. Includes native reference and evidence IDs. Empty list evaluates reference only."""
    return evaluate('coarsening', compositions, reason)


def evaluate_phase_stability(compositions: list[dict[str, float]], reason: str) -> dict:
    """Calculate equilibrium L12/Al3Zr at 250/660°C and Al3Zr boundary for a targeted shortlist. Inputs Ni/Er/Zr at.%. State sampling reason. Includes native reference and liquid flag."""
    return evaluate('phase_stability', compositions, reason)


def evaluate_solidification(compositions: list[dict[str, float]], reason: str) -> dict:
    """Calculate terminal Scheil phases, total-solid FR, interpolated CSC and HCS=CSC*FR/100. Inputs Ni/Er/Zr at.%. State sampling reason. Includes reference and interval diagnostics."""
    return evaluate('solidification', compositions, reason)


def allowances(state):
    return dict(distinct_candidates_used=len(state['candidates']),
                distinct_candidates_limit=PROTOCOL['max_distinct_candidates'],
                **{kind: dict(used=len({identity(c) for call in state['calls'] if call['kind'] == kind
                                       for c in call['compositions']}), limit=PROTOCOL[limit])
                   for kind, limit in [('phase_stability', 'max_candidate_phase_profiles'),
                                       ('solidification', 'max_candidate_scheil')]})


def read_design_results(candidate_ids: list[str]) -> dict:
    """Read exact-composition results, latest plan/nomination and remaining allowances. Empty list returns all. Equilibrium and Scheil quantities remain in separate labeled groups; no selection is automated."""
    state = load()
    if any(cid not in state['candidates'] for cid in candidate_ids):
        raise ValueError('Unknown candidate ID')
    ids = candidate_ids or list(state['candidates'])
    return dict(reference={kind: dict(compact(row), composition_percent=row['composition_percent'], basis=row['basis'])
                           for kind, row in state['reference'].items()},
                candidates={cid: dict(composition_at_percent=state['candidates'][cid]['composition_at_percent'],
                    **{kind: compact(row) for kind, row in state['candidates'][cid]['methods'].items()}) for cid in ids},
                latest_plan=state['plan'][-1] if state['plan'] else None,
                next_action_execution=next_action_execution(state),
                latest_recommendation=latest_recommendation(state),
                nomination_readiness=nomination_readiness(state),
                latest_nomination_attempt=state['recommendations'][-1] if state['recommendations'] else None,
                verification=state['verification'], allowances=allowances(state))


def read_design_evidence(evidence_id: str) -> dict:
    """Read a saved evidence ID with compact physical diagnostics: equilibrium phase compositions/Er-Zr inventories, J misfit/resistivity components, or phases at the FR cutoff. Uses this run's records only; no solver calls or storage search."""
    from .diagnostics import describe
    state = load()
    rows = list(state['reference'].values()) + [r for c in state['candidates'].values() for r in c['methods'].values()]
    for row in rows:
        if row['evidence_id'] == evidence_id:
            return dict(row, diagnostics=describe(row, folder()))
    # Earlier evidence remains available even if superseded by a repeat.
    for call in state['calls']:
        if evidence_id.startswith(call['id'] + ':') and call.get('response_file'):
            response = json.loads((folder() / call['response_file']).read_text())
            rows = dict(zip(call['candidate_ids'], response.get('results', [])))
            rows['reference'] = response.get('reference')
            for cid, row in rows.items():
                if row and evidence_id == f'{call["id"]}:{cid}:{call["kind"]}':
                    return dict(row, evidence_id=evidence_id, diagnostics=describe(row, folder()))
    raise ValueError('Unknown evidence ID; only this run is accessible')


def record_design_plan(next_action: dict, requirements_map: str = "", workflow_mermaid: str = "") -> dict:
    """Record one numerical action: question, tool, compositions OR candidate_ids, and decision rationale. First call also requires requirements_map and workflow_mermaid; later actions retain them. Optional comparison_ids identify prior baselines. Code derives fixed/varied elements within the submitted batch and outputs; compare historical IDs separately. Call this tool to record the action; JSON in prose does not record it. Recording does not establish scientific usefulness."""
    require_scope('Record plan', ('plan',))
    try:
        result = _record_design_plan(next_action, requirements_map, workflow_mermaid)
    except ValueError as error:
        summary = dict(status='rejected', reason=str(error))
        if isinstance(next_action, dict):
            summary.update(tool=next_action.get('tool'),
                candidate_ids=next_action.get('candidate_ids'),
                compositions=next_action.get('compositions'))
        (folder()/'planning_feedback.json').write_text(json.dumps(summary, indent=2)+'\n')
        raise
    (folder()/'planning_feedback.json').write_text(json.dumps(dict(
        status='recorded', plan_version=result['plan_version']), indent=2)+'\n')
    return result


def _record_design_plan(next_action, requirements_map="", workflow_mermaid=""):
    state = load()
    if not isinstance(next_action, dict):
        raise ValueError('next_action must be an object matching its tool schema')
    required = ('question', 'tool', 'decision')
    issues = ['missing next_action.' + k for k in required if k not in next_action]
    if issues:
        raise ValueError('Plan not recorded: ' + '; '.join(issues))
    if any(not isinstance(next_action[k], str) or not next_action[k].strip() for k in required):
        raise ValueError('State a nonempty question, tool and decision rationale')
    output_fields = {
        'evaluate_coarsening': ['J'],
        'evaluate_phase_stability': ['L12_250_mol_percent', 'L12_660_mol_percent',
            'Al3Zr_250_mol_percent', 'Al3Zr_660_mol_percent', 'Al3Zr_boundary'],
        'evaluate_solidification': ['terminal_solid_mol_percent', 'ternary_mol_percent',
            'L12_mol_percent', 'Al3Ni_mol_percent', 'Al3Zr_mol_percent', 'FR_K', 'CSC', 'HCS'],
    }
    if next_action['tool'] not in output_fields:
        raise ValueError('Choose one available numerical tool. Read existing results separately.')
    # Old payloads may contain redundant metadata. Always recompute it from
    # the submitted recipes; no sample is added, removed or changed here.
    allowed = set(NEXT_ACTION_SCHEMA['properties']) | {'fixed_inputs', 'varied_inputs', 'expected_outputs'}
    if set(next_action) - allowed:
        raise ValueError('Unknown next_action fields: ' + ', '.join(sorted(set(next_action)-allowed)))
    comps, ids = next_action.get('compositions', []), next_action.get('candidate_ids', [])
    if not isinstance(comps, list) or not isinstance(ids, list) or bool(comps) == bool(ids):
        raise ValueError('Supply either a list of concrete compositions or existing candidate_ids')
    if not 1 <= len(comps or ids) <= PROTOCOL['max_batch_size']:
        raise ValueError(f"Each action permits a batch of 1 to {PROTOCOL['max_batch_size']} compositions. Choose that many; no samples were selected automatically.")
    if ids:
        if any(not isinstance(cid, str) or cid not in state['candidates'] for cid in ids):
            raise ValueError('Next-action candidate IDs must exist in this run')
        comps = [state['candidates'][cid]['composition_at_percent'] for cid in ids]
    comps = checked(comps)
    comparisons = next_action.get('comparison_ids', [])
    if not isinstance(comparisons, list) or any(not isinstance(cid,str) or cid not in state['candidates'] for cid in comparisons):
        raise ValueError('comparison_ids must name existing candidates')
    varied = [k for k in ('Ni', 'Er', 'Zr') if len({c[k] for c in comps}) > 1]
    action = {k:v for k,v in next_action.items() if k in NEXT_ACTION_SCHEMA['properties']}
    action.update(fixed_inputs={k:comps[0][k] for k in ('Ni','Er','Zr') if k not in varied},
                  varied_inputs=varied, expected_outputs=output_fields[action['tool']])
    previous = state['plan'][-1] if state['plan'] else {}
    requirements_map=requirements_map or previous.get('requirements_map','')
    workflow_mermaid=workflow_mermaid or previous.get('workflow_mermaid','')
    if not all(isinstance(v,str) and v.strip() for v in (requirements_map,workflow_mermaid)):
        raise ValueError('First plan requires a requirements_map and workflow_mermaid. Later actions retain the recorded workflow.')
    entry = dict(requirements_map=requirements_map,
                 workflow_mermaid=workflow_mermaid,
                 assignments=action['decision'], next_action=action, after_batches=len(state['calls']))
    from .checkpoint import validate_proposal
    validate_proposal(state, entry)
    state['plan'].append(entry)
    save(state)
    return dict(status='recorded_new_action_not_scientifically_approved', plan_version=len(state['plan']),
                next_action_execution=next_action_execution(state), next_action=action,
                metadata_origin='fixed/varied fields describe the submitted batch only; historical contrasts require comparison of the named candidate rows',
                after_batches=entry['after_batches'])


def next_action_execution(state):
    """Factual input/result coverage, not a choice of the next alloy or tool."""
    if not state['plan']:
        return dict(status='no_plan')
    action = state['plan'][-1].get('next_action')
    if not action:
        return dict(status='no_concrete_next_action')
    if action['tool'] == 'read_design_results':
        return dict(status='read_requested', candidate_ids=action.get('candidate_ids', []),
                    note='Reading saved results is not a new scientific evaluation.')
    kind = action['tool'].removeprefix('evaluate_')
    comps = action.get('compositions') or [state['candidates'][cid]['composition_at_percent']
                                          for cid in action.get('candidate_ids', [])]
    known = {identity(v['composition_at_percent']): (cid, v)
             for cid, v in state['candidates'].items()}
    coverage = []
    for comp in comps:
        cid, entry = known.get(identity(comp), (None, {}))
        row = entry.get('methods', {}).get(kind, {})
        batches = [c['id'] for c in state['calls'] if c['kind'] == kind
                   and any(identity(comp) == identity(p) for p in c['compositions'])]
        coverage.append(dict(composition_at_percent=comp, candidate_id=cid,
                             result_status=row.get('status', 'not_available'),
                             qualification=result_qualification(kind, row),
                             evidence_id=row.get('evidence_id'), attempted_batches=batches))
    reference = state['reference'].get(kind, {})
    complete = bool(coverage) and all(c['qualification']['usable'] for c in coverage) and usable_result(kind, reference)
    attempted = any(c['attempted_batches'] or c['evidence_id'] for c in coverage)
    return dict(status='already_evaluated' if complete else 'incomplete_results' if attempted else 'not_evaluated',
                tool=action['tool'], candidates=coverage, reference_status=reference.get('status', 'not_available'),
                note='Exact-composition evidence coverage only. Read existing results before scheduling; '
                     'state a reason for intentional repeats. No proposal is executed automatically.')


def recommendation_eligibility(state, cid):
    """Only declared coverage/status and the task's explicit hard phase limit."""
    methods = state['candidates'].get(cid, {}).get('methods', {})
    issues = [f'{kind}: complete candidate and reference results required (a documented phase scan nondetection is qualified with its limits)'
              for kind in KINDS if not usable_result(kind, methods.get(kind, {}))
              or not usable_result(kind, state['reference'].get(kind, {}))]
    phase = methods.get('phase_stability', {}).get('L12_660_mol_percent')
    if isinstance(phase, bool) or not isinstance(phase, (float, int)) or not math.isfinite(phase):
        issues.append('660°C L12 result unavailable')
    elif phase > PROTOCOL['phase_limit_mol_percent']:
        issues.append(f'660°C L12={phase:g} mol% exceeds {PROTOCOL["phase_limit_mol_percent"]:g} mol%')
    return dict(eligible=not issues, issues=issues, L12_660_mol_percent=phase,
                qualifications={kind:result_qualification(kind, methods.get(kind, {})) for kind in KINDS},
                limit_mol_percent=PROTOCOL['phase_limit_mol_percent'])


def latest_recommendation(state):
    return next((r for r in reversed(state['recommendations']) if r.get('eligible')
                 and all(recommendation_eligibility(state, cid)['eligible'] for cid in r['candidate_ids'])), None)


def verification_input_fingerprint(state, candidate_id):
    """Bind a repeat to measured inputs, independent of archive location/labels."""
    def scientific(value):
        if isinstance(value, dict):
            return {k: scientific(v) for k, v in value.items() if k not in ('records', 'evidence_id')}
        if isinstance(value, list):
            return [scientific(v) for v in value]
        return value
    item = state['candidates'][candidate_id]
    payload = dict(composition=item['composition_at_percent'], methods=item['methods'], reference=state['reference'])
    return hashlib.sha256(json.dumps(scientific(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()


def verification_matches(state, candidate_id):
    entry = state.get('verification') or {}
    return (entry.get('candidate_id') == candidate_id and entry.get('status') == 'matched'
            and entry.get('input_fingerprint') == verification_input_fingerprint(state, candidate_id))


def nomination_readiness(state):
    """Explain existing eligibility checks; never choose samples or relax constraints."""
    checks = {cid: recommendation_eligibility(state, cid) for cid in state['candidates']}
    eligible = [cid for cid, check in checks.items() if check['eligible']]
    next_delegation = None
    decision_options = []
    nomination = latest_recommendation(state)
    if eligible:
        message = ('Eligible for nomination by coverage and the hard phase limit: '
                   + ', '.join(eligible) + '. The planner must still justify the scientific choice and tradeoffs.')
        if nomination is None:
            decision_options = [
                dict(mode='review', purpose='Assess the remaining tradeoff before first nomination; no new calculation or nomination is due.'),
                dict(mode='plan', owner='Alloy_Design_Planner', operation='record_design_plan',
                     purpose='Record the informative experiment justified by the review.'),
                dict(mode='execute', execute_intent='nominate', owner='Alloy_Design_Planner', operation='record_recommendation',
                     purpose='Record the choice when the stopping rationale is supported.')]
            message += (' Eligibility is permission, not a stopping decision. Complete any pending recorded calculation first. Before first nomination, '
                        'obtain one focused review with the planner and relevant physics expertise: '
                        'address outstanding specialist suggestions and recommend further investigation '
                        'or justify stopping. If that review has already answered the question, act on '
                        'its conclusion rather than repeat it. Review mode does not require a numerical '
                        'plan, nomination or fresh verification for this assignment. A justified experiment '
                        'uses plan then execute; a supported stopping decision uses planner execute-mode '
                        'record_recommendation, then critic review-mode verify_candidate. These are choices '
                        'for the manager, not an automatically selected action.')
        elif state['verification'] is None or state['verification'].get('status') in ('started', 'interrupted'):
            next_delegation = dict(mode='review', owner='Scientific_Critic', operation='verify_candidate')
            message = ('Nomination is recorded. Delegate Scientific_Critic in review mode to call '
                       'verify_candidate for the recorded preferred ID, then assess the scientific interpretation. '
                       'An interrupted repeat continues the same attempt using completed families.')
        else:
            message = ('Nomination and a verification attempt are recorded. Review its actual status, '
                       'differences and scientific interpretation; do not assume a successful match or rerun it automatically.')
    elif not checks:
        message = 'No candidate results yet. Plan the initial search and calculate the required evidence.'
    else:
        reasons = '; '.join(cid + ': ' + ', '.join(check['issues']) for cid, check in checks.items())
        message = ('No eligible candidate. ' + reasons + '. Return to plan mode to address the measured constraint '
                   'or missing evidence while allowances remain. A hard violation requires refinement, not another '
                   'nomination or reviewer approval. If progress is blocked, report the scientific reason.')
    if not eligible:
        next_delegation = dict(mode='plan', owner='Alloy_Design_Planner', operation='record_design_plan')
    from .investigation import investigation_facts
    branch_facts = investigation_facts(state)['Ni_evidence_by_candidate_branch']
    conditional = {cid: dict(branch_Ni_support=branch_facts[cid],
        missing_if_preferred=([] if branch_facts[cid]['has_controlled_Ni_comparison'] else
            ['R3: a controlled Ni comparison is still needed on this Er/Zr branch if retained as preferred.']))
        for cid in eligible}
    pending_for_preferred = (conditional[nomination['candidate_ids'][0]]['missing_if_preferred']
                             if nomination else [])
    if pending_for_preferred:
        message = ('The recorded preferred choice has required evidence pending: '
                   + ' '.join(pending_for_preferred)
                   + ' If retaining this choice, plan and complete that comparison before final verification. '
                     'Alternatively, amend the recommendation with a supported scientific rationale. '
                     'Recording a limitation does not satisfy the required comparison; verification does not replace it.')
        next_delegation = dict(mode='plan', owner='Alloy_Design_Planner', operation='record_design_plan')
        decision_options = [next_delegation, dict(mode='execute', execute_intent='nominate',
                            owner='Alloy_Design_Planner',operation='record_recommendation',
                            purpose='Amend the preferred choice if the measured tradeoffs justify doing so.')]
    return dict(eligible_ids=eligible, message=message, next_delegation=next_delegation,
                decision_options=decision_options, candidate_requirements=conditional,
                pending_for_preferred=pending_for_preferred,
                eligibility_scope='Calculation coverage and phase feasibility; candidate_requirements exposes conditional mission work, not a ranking.')


def record_recommendation(candidate_ids: list[str], rationale: str) -> dict:
    """Record or amend a nomination: preferred ID first, optional alternative, full tradeoff rationale. Use execute mode with execute_intent=nominate, including a same-ID rationale correction after verification. Appends history; does not approve claims or recalculate. Unchanged candidate evidence retains verification. Changing the preferred ID or measured inputs makes an existing verification stale; the one-repeat allowance is not reset. Qualification and hard phase limit are checked."""
    require_scope('Final nomination', ('execute',))
    state = load()
    if not 1 <= len(candidate_ids) <= 2 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Supply one preferred ID and optionally one distinct alternative')
    if any(cid not in state['candidates'] for cid in candidate_ids):
        raise ValueError('Unknown candidate ID')
    if not rationale.strip():
        raise ValueError('Provide the scientific rationale')
    # Only validate explicit, structured citations; prose truth remains the critic's job.
    cited = set(re.findall(r'\bB\d{3}:(?:C\d{3}|reference):(?:coarsening|phase_stability|solidification)\b', rationale))
    for evidence_id in sorted(cited):
        try:
            read_design_evidence(evidence_id)
        except ValueError as exc:
            raise ValueError('Nomination not recorded: unknown evidence citation ' + evidence_id
                             + '. Read the exact saved result and correct the citation.') from exc
    checks = {cid: recommendation_eligibility(state, cid) for cid in candidate_ids}
    eligible = all(c['eligible'] for c in checks.values())
    entry = dict(candidate_ids=candidate_ids, rationale=rationale, eligible=eligible, checks=checks)
    state['recommendations'].append(entry)
    save(state)
    readiness = nomination_readiness(state)
    return dict(status='recorded_not_scientifically_approved' if eligible else 'ineligible_nomination',
                instruction=readiness['message'],pending_for_preferred=readiness['pending_for_preferred'],
                next_delegation=readiness['next_delegation'],
                **entry)


def numbers(value, prefix=''):
    result = {}
    for k, v in value.items():
        if k in ('composition_percent', 'records', 'FR_interval'):
            continue
        name = prefix + k
        if isinstance(v, dict):
            result.update(numbers(v, name + '.'))
        elif isinstance(v, (float, int)) and not isinstance(v, bool):
            result[name] = v
    return result


def verify_candidate(candidate_id: str) -> dict:
    """Freshly repeat all three scientific functions for the nominated preferred candidate and included reference. Returns scalar differences and statuses, not scientific approval. One fresh verification per run; resume an interrupted attempt for unchanged inputs, without repeating completed families."""
    require_scope('Fresh final verification', ('review',))
    state = load()
    nomination = latest_recommendation(state)
    if nomination is None:
        readiness = nomination_readiness(state)
        # A requested verification has a concrete prerequisite. Keep this
        # recovery route distinct from optional pre-nomination decisions.
        prerequisite = (dict(mode='execute', execute_intent='nominate', owner='Alloy_Design_Planner', operation='record_recommendation')
                        if readiness['eligible_ids'] else readiness['next_delegation'])
        return dict(status='delegation_required',
                    reason='Fresh verification did not run: no eligible nomination is recorded. ' + readiness['message'],
                    next_delegation=prerequisite)
    if nomination['candidate_ids'][0] != candidate_id:
        raise ValueError('Verify the recorded preferred ID ' + nomination['candidate_ids'][0]
                         + '; do not substitute another candidate during verification')
    readiness = nomination_readiness(state)
    if readiness['pending_for_preferred']:
        return dict(status='delegation_required',
                    reason='Fresh verification did not run. ' + readiness['message'],
                    pending_for_preferred=readiness['pending_for_preferred'],
                    next_delegation=readiness['next_delegation'])
    if not recommendation_eligibility(state, candidate_id)['eligible']:
        raise ValueError('Preferred candidate no longer has complete, eligible evidence')
    original = state['candidates'][candidate_id]
    fingerprint = verification_input_fingerprint(state, candidate_id)
    entry = state['verification']
    if entry is not None:
        if entry.get('candidate_id') != candidate_id or entry.get('input_fingerprint') != fingerprint:
            return dict(status='verification_stale', verification=entry,
                reason='The saved repeat belongs to different or unbound measured inputs. The one-final-verification allowance is already used; do not claim a current match or silently start another repeat.')
        if entry['status'] not in ('started', 'interrupted'):
            return dict(status='already_requested', verification=entry)
    else:
        entry = dict(candidate_id=candidate_id, input_fingerprint=fingerprint,
                     status='started', comparisons=[], responses=[])
        state['verification'] = entry
    entry['status'] = 'started'
    entry.pop('error_type', None)
    save(state)
    completed = {r['kind'] for r in entry['responses']}
    response_count, comparison_count = len(entry['responses']), len(entry['comparisons'])
    try:
        fresh = AlloyDesignTools(folder() / 'fresh_verification')
        for kind in KINDS:
            if kind in completed:
                continue
            response_count, comparison_count = len(entry['responses']), len(entry['comparisons'])
            response = getattr(fresh, 'evaluate_' + kind)([original['composition_at_percent']])
            path = folder() / ('verification_' + kind + '.json')
            path.write_text(json.dumps(response, indent=2, allow_nan=False)+'\n')
            entry['responses'].append(dict(kind=kind, status=response['status'], record=path.name,
                                           usage=response.get('usage'), fingerprint=response.get('method_fingerprint')))
            for cid, old, new in [(candidate_id, original['methods'][kind], (response.get('results') or [{}])[0]),
                                  ('reference', state['reference'][kind], response.get('reference', {}))]:
                oldn, newn = numbers(old), numbers(new)
                diffs = []
                for name in sorted(oldn.keys() | newn.keys()):
                    tolerance = 5. if 'temperature' in name and 'boundary' in name else 1e-5
                    a, b = oldn.get(name), newn.get(name)
                    diff = abs(a-b) if a is not None and b is not None else None
                    diffs.append(dict(metric=name, original=a, fresh=b, absolute_difference=diff,
                                      tolerance=tolerance, passed=diff is not None and diff <= tolerance))
                oldq, newq = result_qualification(kind, old), result_qualification(kind, new)
                boundary_same = True
                if kind == 'phase_stability':
                    a, b = old.get('Al3Zr_boundary', {}), new.get('Al3Zr_boundary', {})
                    boundary_same = (a.get('status') == b.get('status') and
                                     a.get('liquid_at_upper_boundary') == b.get('liquid_at_upper_boundary'))
                    if oldq.get('scan') or newq.get('scan'):
                        # Source paths differ in a fresh calculation; the reported
                        # scan scope must agree, not merely the absence of a number.
                        scope_keys = ('phase','samples','lower_C','upper_C','maximum_sample_gap_K','phase_tolerance')
                        boundary_same &= all(oldq.get('scan',{}).get(k) == newq.get('scan',{}).get(k) for k in scope_keys)
                passed = bool(diffs) and oldq['usable'] and newq['usable'] and boundary_same and all(v['passed'] for v in diffs)
                entry['comparisons'].append(dict(candidate_id=cid, kind=kind, passed=passed, differences=diffs,
                    original_qualification=oldq, fresh_qualification=newq, boundary_assessment_matches=boundary_same))
            save(state)
    except BaseException:
        # Keep only complete families; a partial family must not become a cache hit.
        del entry['responses'][response_count:]
        del entry['comparisons'][comparison_count:]
        entry['status'] = 'interrupted'
        import sys
        entry['error_type'] = type(sys.exception()).__name__
        try:
            save(state)
        except Exception:
            pass  # Preserve the original solver/serialization exception.
        raise
    entry['status'] = 'matched' if len(entry['comparisons']) == 2 * len(KINDS) and all(c['passed'] for c in entry['comparisons']) else 'mismatch_or_incomplete'
    save(state)
    return entry


def handoff(mode=None):
    result = read_design_results([])
    # Repeat differences are available to the reviewer tool; shared context keeps a concise status.
    if result['verification']:
        v = result['verification']
        result['verification'] = {k: v[k] for k in ('candidate_id', 'status')}
    mode = mode or _MODE.get() or 'mission'
    pending = []
    if result['latest_plan'] is None:
        pending.append('Planner must record requirement mapping and workflow')
    start = _PLAN_START.get()
    if mode == 'plan' and start is not None:
        updated = len(load()['plan']) > start
        result['planning_update'] = dict(plan_version_at_entry=start,
                                        recorded_in_this_delegation=updated)
        if not updated and result['latest_plan'] is not None:
            pending.append('This planning delegation has not recorded a next action; the saved plan predates it')
        if result['next_action_execution']['status'] == 'already_evaluated':
            pending.append('The next batch already has complete results. Read them or record a concrete changed batch; explain any intentional repeat')
    if mode == 'mission':
        if result['latest_recommendation'] is None:
            pending.append(result['nomination_readiness']['message'])
        elif not verification_matches(load(), result['latest_recommendation']['candidate_ids'][0]):
            pending.append('Preferred candidate and reference need matched fresh verification')
    obligations = {
        'plan': 'Record requirement mapping, workflow and one concrete next action. Future calculated tables and final nomination are not due in this delegation.',
        'execute': 'Return the calculations or selection requested by this assignment with evidence and limitations. Unassigned final deliverables are not due.',
        'review': 'Assess the assigned evidence. A pre-nomination review recommends refinement or justifies stopping; no nomination or fresh verification is due. Freshly verify only when final verification is assigned. Return next work to the manager.',
        'mission': 'Complete the fixed task and its final scientific review.',
    }
    result['completion'] = dict(mode=mode, pending=pending, assignment_check=obligations[mode],
                                note='Mechanical coverage only, not scientific acceptance')
    return result


def mission_coverage(state):
    """Factual task coverage; no ranking, phase desirability or scientific verdict."""
    groups = {}
    for cid, item in state['candidates'].items():
        if item['methods'].get('solidification', {}).get('status') == 'complete':
            comp = item['composition_at_percent']
            groups.setdefault((comp['Er'], comp['Zr']), []).append((cid, comp['Ni']))
    controlled = [dict(Er=k[0], Zr=k[1], candidate_ids=[x[0] for x in rows],
                       Ni_levels=sorted({x[1] for x in rows}))
                  for k, rows in groups.items() if len({x[1] for x in rows}) >= 2]
    nomination = latest_recommendation(state)
    ids = nomination['candidate_ids'] if nomination else []
    preferred_comparison = None
    if ids:
        comp = state['candidates'][ids[0]]['composition_at_percent']
        preferred_comparison = next((row for row in controlled
            if row['Er'] == comp['Er'] and row['Zr'] == comp['Zr']), None)
    verification = state.get('verification') or {}
    missing = []
    if not state['plan'] or not all(state['plan'][-1].get(k) for k in ('requirements_map','workflow_mermaid')):
        missing.append('R6: record the requirements-to-workflow plan')
    if not controlled:
        missing.append('R3: calculate a controlled Ni comparison at fixed Er/Zr, not an isolated composition')
    elif ids and preferred_comparison is None:
        missing.append('R3: the nominated preferred candidate ' + ids[0]
            + ' lacks a controlled Ni comparison on its retained Er/Zr branch; '
            'comparisons on another branch do not satisfy this requirement')
    if not ids:
        missing.append('R5: nomination pending; use nomination_readiness for measured blockers and the next kind of work')
    if ids and not verification_matches(state, ids[0]):
        missing.append('R5: freshly verify the nominated preferred candidate and reference')
    return dict(controlled_Ni_comparisons=controlled, nominated_ids=ids,
                preferred_branch_Ni_comparison=preferred_comparison,
                nomination_checks={cid: recommendation_eligibility(state, cid) for cid in ids},
                missing=missing, meaning='Calculation/record coverage only. Scientific review still assesses R1–R6, useful sampling, priorities and claims.')


def closing_evidence(messages):
    return handoff()
