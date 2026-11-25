"""Full-task coordination using existing evidence, actions and review machinery."""
import json
from pathlib import Path
import math
from . import tools
from .comparison import DesignEvidence

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
REPO=BASE.parents[2]

def phase_inventory(candidates):
    """Constraint availability is distinct from completeness of a boundary scan."""
    rows={cid:c['methods'].get('phase_stability',{}) for cid,c in candidates.items()}
    known={cid:r for cid,r in rows.items() if r.get('status') in ('complete','partial')
           and isinstance(r.get('L12_660_mol_percent'),(float,int))
           and not isinstance(r.get('L12_660_mol_percent'),bool)
           and math.isfinite(r['L12_660_mol_percent'])}
    return dict(passing=[cid for cid,r in known.items() if r['L12_660_mol_percent']<=tools.PROTOCOL['phase_limit_mol_percent']],
        failing=[cid for cid,r in known.items() if r['L12_660_mol_percent']>tools.PROTOCOL['phase_limit_mol_percent']],
        unchecked=[cid for cid in rows if cid not in known],
        profile_qualifications={cid:tools.result_qualification('phase_stability', r)
                                for cid,r in rows.items() if r},
        meaning='Passing/failing refers only to the measured 660 C point; full-profile coverage and nomination eligibility are separate.')

def validate_proposal(state,entry):
    """Reject unavailable, completed or over-budget actions before storing a plan."""
    a=entry['next_action'];kind=a['tool'].removeprefix('evaluate_')
    if kind not in ('coarsening','phase_stability','solidification'):
        raise ValueError('Record a numerical next_action. Read existing evidence directly if needed before choosing it.')
    comps=a.get('compositions') or [state['candidates'][cid]['composition_at_percent'] for cid in a.get('candidate_ids',[])]
    if not 1<=len(comps)<=tools.PROTOCOL['max_batch_size']:
        raise ValueError(f"Each action permits a batch of 1 to {tools.PROTOCOL['max_batch_size']} compositions.")
    preview=dict(state,plan=[*state['plan'],entry])
    covered=tools.next_action_execution(preview)
    if covered['status']=='already_evaluated':
        facts=[]
        for c in covered['candidates']:
            row=state['candidates'][c['candidate_id']]['methods'][kind]
            facts.append(dict(candidate=c['candidate_id'],source=row['evidence_id'],
                **{k:row[k] for k in ('J','L12_660_mol_percent','FR_K','CSC','HCS') if k in row}))
        raise ValueError('Plan not recorded: these exact calculations are complete. '+json.dumps(facts)+
            '. Use this evidence to choose a changed numerical action; do not execute the completed batch.')
    existing={tools.identity(c['composition_at_percent']) for c in state['candidates'].values()}
    if len(existing | {tools.identity(c) for c in comps})>tools.PROTOCOL['max_distinct_candidates']:
        raise ValueError('Plan exceeds remaining distinct-composition allowance.')
    limit={'phase_stability':'max_candidate_phase_profiles','solidification':'max_candidate_scheil'}.get(kind)
    attempted={tools.identity(c) for call in state['calls'] if call['kind']==kind for c in call['compositions']}
    if limit and len(attempted | {tools.identity(c) for c in comps})>tools.PROTOCOL[limit]:
        raise ValueError('Plan exceeds remaining '+kind+' allowance.')

def current_action_state(state):
    if not state['plan']:
        return dict(status='needs_plan',reason='Record the initial workflow and numerical action.')
    if state['plan'][-1]['after_batches'] != len(state['calls']):
        return dict(status='action_attempted',coverage=tools.next_action_execution(state),
                    reason='Interpret the recorded results, then plan further measurements or nominate a fully qualified candidate.')
    try:
        validate_proposal(dict(state,plan=state['plan'][:-1]),state['plan'][-1])
    except ValueError as error:
        return dict(status='needs_plan',reason=str(error))
    return dict(status='ready',next_action=state['plan'][-1]['next_action'],plan_version=len(state['plan']))


def plan_readiness():
    state=tools.load(); start=tools._PLAN_START.get()
    return dict(ready=current_action_state(state)['status']=='ready' and start is not None and len(state['plan'])>start,
                plan_version=len(state['plan']),reason='Planning requires a new recorded numerical action in this delegation.')


def enter_delegation(mode):
    path=tools.folder()/'delegation_limits.json'
    counts=json.loads(path.read_text()) if path.exists() else dict(plan=0,execute=0,review=0,batch_count=0,plans_since_batch=0)
    if sum(counts[k] for k in ('plan','execute','review'))>=tools.PROTOCOL['max_delegations']:
        raise ValueError('Full-run delegation limit reached. Report the actual incomplete work and stop.')
    batches=len(tools.load()['calls'])
    if counts['batch_count']!=batches:
        counts.update(batch_count=batches,plans_since_batch=0)
    if mode=='plan':
        if counts['plans_since_batch']>=2:
            raise ValueError('Two planning delegations without a new batch; report the blocker rather than restarting the search.')
        counts['plans_since_batch']+=1
    counts[mode]+=1
    path.write_text(json.dumps(counts,indent=2)+'\n')


def guard_action(state,kind=None,compositions=None):
    action=current_action_state(state)
    if action['status']!='ready':
        raise ValueError('A new recorded action is needed before this numerical batch. '+action.get('reason',''))
    a=action['next_action']
    expected=a.get('compositions') or [state['candidates'][cid]['composition_at_percent'] for cid in a['candidate_ids']]
    if kind is not None and (a['tool']!='evaluate_'+kind or
            {tools.identity(c) for c in compositions}!={tools.identity(c) for c in expected}):
        raise ValueError('Calculation differs from the recorded tool/compositions. Execute that action or return to planning.')


def required_operation(mode, execute_intent=None):
    if execute_intent not in (None, 'calculate', 'nominate'):
        raise ValueError('execute_intent must be calculate or nominate')
    if mode != 'execute' and execute_intent is not None:
        raise ValueError('execute_intent applies only to execute mode')
    state=tools.load()
    if mode=='plan':return 'record_design_plan'
    if mode=='execute':
        action=current_action_state(state)
        if execute_intent == 'nominate':
            if action['status']=='ready':
                raise ValueError('A recorded numerical action is pending. Execute it or explicitly revise the plan before nomination.')
            if any(tools.recommendation_eligibility(state,cid)['eligible'] for cid in state['candidates']):
                return 'record_recommendation'
            raise ValueError('Nomination unavailable. ' + tools.nomination_readiness(state)['message'])
        if action['status']=='ready':return action['next_action']['tool']
        raise ValueError('No ready numerical action. Return to plan mode to record the intended calculation; '
                         'candidate eligibility does not imply nomination. For an intended eligible nomination use '
                         'execute_intent=nominate. ' + tools.nomination_readiness(state)['message'])
    if mode=='review' and tools.latest_recommendation(state) and (state['verification'] is None
            or state['verification'].get('status') in ('started', 'interrupted')):
        return 'verify_candidate'
    return None

def validate_team(agents, operation):
    """Verify selected LLM tool access; do not silently add or replace agents."""
    offered={t['function']['name'] for a in agents for t in (a.llm_config or {}).get('tools',[])}
    if operation and operation not in offered:
        raise ValueError(f'Selected team lacks {operation}. Rebuild a team with that operation; no working chat started.')
    if operation == 'record_design_plan' and not any(getattr(a, 'planning_expertise', None) for a in agents):
        raise ValueError('Planning needs a physics specialist to interpret the current evidence before the planner selects samples. Select the expertise relevant to the unresolved objective or constraint.')

class FullDesignEvidence(DesignEvidence):
    """Reuse full-design evidence and completion; add facts for team selection."""
    def shared(self,mode=None):
        packet=super().shared(mode=mode);state=tools.load();phase=phase_inventory(state['candidates'])
        coverage={method:{'complete':sum(c['methods'].get(method,{}).get('status')=='complete' for c in state['candidates'].values()),
            'partial':sum(c['methods'].get(method,{}).get('status')=='partial' for c in state['candidates'].values())}
            for method in ('coarsening','phase_stability','solidification')}
        for method,counts in coverage.items():
            counts['qualified']=sum(tools.usable_result(method,c['methods'].get(method,{})) for c in state['candidates'].values())
            counts['without_usable_record']=len(state['candidates'])-counts['qualified']
        action=current_action_state(state)
        packet.update(current_action=action,phase_constraint_status=phase)
        packet['investigation_progress']=dict(
            sampled_levels={element:sorted({c['composition_at_percent'][element]
                for c in state['candidates'].values()}) for element in ('Ni','Er','Zr')},
            last_planned_question=dict(
                text=state['plan'][-1]['next_action']['question'] if state['plan'] else None,
                execution_status=action['status'],
                meaning='Historical question recorded before its calculation; current_action and saved results describe its execution.'),
            latest_batch={k:state['calls'][-1].get(k) for k in ('id','kind','candidate_ids','status')} if state['calls'] else None,
            missing_or_unqualified={cid:[kind for kind in tools.KINDS
                if not tools.usable_result(kind,c['methods'].get(kind,{}))]
                for cid,c in state['candidates'].items()},
            meaning='Measured coverage, not a prescribed next action. Review the current question, result, unresolved issue and next deliverable; the same capable team may continue.')
        from .investigation import investigation_facts
        packet['investigation_facts'] = investigation_facts(state)
        packet['selection_context']=dict(primary_objective=dict(quantity='J',temperature_C=250,direction='minimize',tie_absolute=tools.PROTOCOL['J_tie_absolute']),
            phase_constraint=dict(quantity='L12_660_mol_percent',maximum=tools.PROTOCOL['phase_limit_mol_percent'],
                passing=len(phase['passing']),failing=len(phase['failing']),unchecked=len(phase['unchecked'])),
            candidate_result_coverage=coverage,current_action=action,allowances=tools.allowances(state),
            investigation_progress=packet['investigation_progress'],
            investigation_facts=packet['investigation_facts'],
            nomination_readiness=tools.nomination_readiness(state),
            outstanding_requirements=packet['mission_coverage']['missing'],
            note='Task requirements and measured availability, not a recommended team, search direction or candidate.')
        path=tools.folder()/'delegation_limits.json'
        counts=json.loads(path.read_text()) if path.exists() else {}
        total=sum(counts.get(k,0) for k in ('plan','execute','review'))
        blocked=(counts.get('batch_count')==len(state['calls']) and counts.get('plans_since_batch',0)>=2
                 and action['status']=='needs_plan')
        packet['stopping']=dict(stop_now=total>=tools.PROTOCOL['max_delegations'] or blocked,
            delegations=counts,max_total_delegations=tools.PROTOCOL['max_delegations'])
        return packet

    def manager_view(self):
        """Current factual packet once; full shared packets remain in archives."""
        packet = self.shared(mode='mission')
        for key in ('current_action', 'allowances', 'investigation_progress', 'investigation_facts'):
            packet['selection_context'].pop(key, None)
        return packet

from GroupAgent_dev.review_delivery import CompletionGuard, termination_requested

class BoundedCompletionGuard(CompletionGuard):
    def reply(self,recipient,messages=None,sender=None,config=None):
        messages=messages if messages is not None else recipient.chat_messages[sender]
        packet=self.evidence_provider()
        if termination_requested(messages[-1]) and packet['stopping']['stop_now']:
            self.events.append(dict(action='stop_bounded_incomplete',pending=packet['completion']['pending']))
            return False,None
        return super().reply(recipient,messages,sender,config)
