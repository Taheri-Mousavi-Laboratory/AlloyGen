"""Opt-in staged coordination; experts still choose tools and scientific judgments."""
from .review import ReviewTurns
import json
from copy import deepcopy
import re
from .review_delivery import (worker_contributions, tool_activity, review_decision,
                              substantive_review, correction_fields, delegation_handoff)


class ControlledReviewTurns(ReviewTurns):
    INSTRUCTIONS = {
        'opening': 'Opening readiness check only: use the supplied assignment, capability list and evidence packet to check that the team can start. Return a brief scope/capability check and yield to the worker. Do not audit every historical result or reread supplied quantities. Use a read tool only if a named missing fact or specific discrepancy prevents this readiness decision; state that gap before the call and read only what resolves it. Requested new results are work to do, not a stopping reason. Numerical and scientific assessment belongs to closing review, after the assigned work.',
        'work': 'Complete your current assigned deliverable and address any critic correction. Use tools when this scope permits and requires them. If blocked, identify the failed operation or missing capability. Give a substantive report with evidence, decisions and limitations. A report without a tool request hands your contribution to the next specialist or closing reviewer; it is not approval of the task.',
        'closing': 'Closing evidence review. Read current result records and check the assigned deliverables, numerical claims, IDs and limitations. If feasible assigned work is missing or unattempted, request a concrete correction and end with REVISE. A local correction must be possible with the owner\'s tools in the current mode. Otherwise use Correction kind: handoff and identify the needed operation/mode for the manager. Otherwise state the evidence-supported outcome and limitations, then TERMINATE. Honest incompletion alone does not justify stopping feasible work. Do not return only a termination token.',
    }

    def __init__(self, reviewer_name, executor_name=None, max_corrections=2, auditor_name=None,
                 closing_evidence_provider=None, compact_closing_context=False, delegation_mode='execute',
                 review_material=None, plan_readiness_provider=None, opening_review=True):
        super().__init__(reviewer_name)
        self.executor_name = executor_name
        self.delegation_mode = delegation_mode
        self.opening_review = opening_review
        self.worker_reminders = set()
        self.incomplete_workers = set()
        self.auditor_name = auditor_name
        self.audit_reports = []
        self.audit_reminders = 0
        self.instructions = dict(self.INSTRUCTIONS)
        self.failed_plan_submissions = 0
        self.plan_readiness_provider = plan_readiness_provider
        self.plan_submission_reminders = 0
        self.plan_version_before_advice_correction = None
        if auditor_name:
            self.instructions.update(
                work=self.INSTRUCTIONS['work'].replace('Address any critic correction.', 'Address any factual-audit or scientific-review correction in its stated scope. Evidence_Auditor checks facts and exports after work; Scientific_Critic interprets their physical meaning.'),
                audit='Evidence audit after specialist work. Use audit_design_evidence. Cover its quantity table with values, units, numerical differences and source IDs. Explicitly identify specialist numerical claims contradicted by these facts and any export discrepancies. End with REVISE only for a named claim or export correction possible within the current assignment. Unavailable calculations outside this assignment are limitations: report them and pass the factual audit with TERMINATE to scientific interpretation. Do not judge scientific desirability or manage allowances.',
                closing='Scientific interpretation after the evidence audit. Use the compact indicator/comparison/meaning/source-ID table in your guide for every audited indicator, followed by conflicting comparisons, case-specific missing-data limits and a scoped verdict. Use the audited values and IDs; do not repeat arithmetic, file checks or allowance accounting. If a material scientific claim or comparison needs correction, request that correction and end with REVISE. Otherwise give a scoped verdict and remaining scientific limits, then TERMINATE. A factual audit alone is not scientific acceptance.')
        if delegation_mode == 'plan':
            self.instructions['work'] = ('Planning contribution only. Selected physics specialists first interpret saved measurements; the planner then chooses and records the next experiment. '
                'Calculation verbs in the manager text describe future execution. Do not call calculation or nomination tools. '
                'Each specialist supplies its own analysis, not a recorded plan. Only the planner records the action after considering those analyses.')
            self.instructions['closing'] = ('Closing adaptive-decision review. Check the premise against current measurements, whether the recorded numerical action obtains missing information, and why that information matters for the current objective/constraint. '
                'The declared plan mode controls this delegation even when the manager combines planning and execution in its wording. '
                'Use REVISE for a missing or invalid plan item that can be repaired in plan mode. '
                'Do not request calculation or final-result tables in this mode. If the plan is ready, give a supported TERMINATE verdict '
                'and identify execute mode as the next delegation. Future uncomputed results are not a planning defect.')
        self.max_corrections = max_corrections
        self.corrections = 0
        self.stage = 'opening'
        self.pending_caller = None
        self.pending_tool_calls = []
        self.last_failed_tool_request = None
        self.last_worker = None
        self.pending_workers = []
        self.worker_index = 0
        self.original_prompts = {}
        self.executor_predicates = {}
        self.events = []
        self.stop_reason = None
        self.closure_reminders = 0
        self.review_text_by_cycle = {}
        self.final_decision = None
        self.closing_evidence_provider = closing_evidence_provider
        self.evidence_packets = []
        self._evidence_cycle = None
        self._review_messages = []
        self.compact_closing_context = compact_closing_context
        self._closing_hook = None
        self._closing_agent = None
        self._compact_tail_start = None
        self._compact_base = None
        self.review_material = review_material
        self.opening_prose = []
        self.context_hooks = {}
        self.worker_names = set()

    def _initial_analysis(self, agent):
        return (self.delegation_mode == 'plan' and self.stage == 'work'
                and self.corrections == 0 and bool(getattr(agent, 'planning_expertise', None)))

    def _discussion_messages(self, agent, messages):
        """Independent initial advice; retain evidence, tool pairs and the full saved transcript."""
        messages = self._without_readiness(messages)
        if not self._initial_analysis(agent):
            return messages
        output = []
        for message in messages:
            peer = message.get('name') in self.worker_names and message.get('name') != agent.name
            if peer and message.get('role') not in ('tool', 'function') and not message.get('tool_responses'):
                if message.get('tool_calls') or message.get('function_call'):
                    message = dict(message, content=None)
                else:
                    continue
            output.append(message)
        return output

    def _without_readiness(self, messages):
        """Remove opening commentary from provider context, preserving the transcript and tool pairs."""
        if self.stage == 'opening' or not self.opening_prose:
            return messages
        remaining = list(self.opening_prose)
        output = []
        for message in messages:
            text = message.get('content')
            if message.get('name') == self.reviewer_name and text in remaining:
                remaining.remove(text)
                if message.get('tool_calls'):
                    message = dict(message, content=None)  # Keep call IDs and their responses intact.
                else:
                    continue
            output.append(message)
        return output

    def _closing_messages(self, messages):
        if self.stage != 'closing':
            return messages
        if self._compact_tail_start is None:
            self._compact_tail_start = len(messages)
            # The fresh factual packet is already in the system message. Keep the
            # assignment and ALL worker prose, not repeated raw solver output.
            assignment = self._review_messages[0].get('content') or ''
            assignment = assignment.split('\n\n# Shared evidence from saved records', 1)[0]
            body = {'assignment': assignment,
                    'worker_contributions_unverified': worker_contributions(self._review_messages,
                        {self.reviewer_name, self.auditor_name, self.executor_name}),
                    'tool_activity': tool_activity(self._review_messages,
                        {self.reviewer_name, self.auditor_name, self.executor_name}),
                    'prior_review_feedback': [m['content'] for m in self._without_readiness(self._review_messages)[1:]
                        if m.get('name') == self.reviewer_name and m.get('content') and not m.get('tool_calls')]}
            claims = (self.evidence_packets[-1].get('recorded_claims_unverified')
                      if self.evidence_packets else None)
            if claims is not None:
                body['recorded_claims_unverified'] = claims
                # Full calls remain in the transcript. Avoid repeating the same
                # stored rationale inside call arguments as if it were evidence.
                for call in body['tool_activity']:
                    if call['tool'] == 'record_recommendation':
                        try:
                            args = json.loads(call['arguments'])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(args, dict) and 'rationale' in args:
                            args['rationale'] = '[Unverified recorded rationale is supplied separately; original call retained in transcript.]'
                        call['arguments'] = json.dumps(args)
            self._compact_base = {'role': 'user', 'content': json.dumps(body, ensure_ascii=False)}
        # Refresh changed claims without resetting the tail of reviewer tool pairs.
        body = json.loads(self._compact_base['content'])
        claims = (self.evidence_packets[-1].get('recorded_claims_unverified')
                  if self.evidence_packets else None)
        if claims is None:
            body.pop('recorded_claims_unverified', None)
        else:
            body['recorded_claims_unverified'] = claims
        self._compact_base = dict(self._compact_base, content=json.dumps(body, ensure_ascii=False))
        # Keep subsequent critic tool calls/results intact and paired. This hook
        # changes only the provider view, never the saved group/notebook transcript.
        return [self._compact_base, *messages[self._compact_tail_start:]]

    def _select(self, agent):
        if self.stage == 'closing' and not self._plan_recorded():
            return self._stop(None, 'Planning incomplete: no recorded current action; scientific closing review cannot declare it ready')
        original = self.original_prompts.setdefault(agent, agent.system_message)
        if agent not in self.context_hooks:
            def hook(messages, target=agent):
                return self._discussion_messages(target, messages)
            agent.register_hook('process_all_messages_before_reply', hook)
            self.context_hooks[agent] = hook
        instruction = self.instructions[self.stage]
        mode_instruction = {
            'plan': 'Interpretation precedes sample selection. Physics specialists identify measured comparisons, the unresolved objective/constraint and uncertainty. The planner uses those findings to record one next_action. Establish the workflow when absent and preserve it during subsequent numerical refinement. Do not execute science in plan mode.',
            'execute': 'Current assignment is execution or selection: complete the assigned work with the recorded plan and existing tool evidence.',
            'review': 'Current assignment is review: assess saved evidence and perform fresh final verification only if assigned. Return missing search work to the manager.',
        }[self.delegation_mode]
        if self.stage == 'opening':
            instruction += '\nThe delegation mode is ' + self.delegation_mode + '. It defines the workers\' assigned work; your current action is only the readiness check above.'
        else:
            instruction += '\n' + mode_instruction
        deliverable = (agent.current_deliverable(self.delegation_mode)
            if self.stage == 'work' and hasattr(agent, 'current_deliverable')
            else getattr(agent, 'work_deliverable', '') if self.delegation_mode == 'execute' else '')
        if self.stage == 'work' and deliverable:
            instruction += '\n\nYour contribution in this turn:\n' + deliverable
            instruction += '\nApply this capability only to the current assigned subtask. Other available calculations remain future work unless explicitly assigned.'
        if self._initial_analysis(agent):
            instruction += ('\nThis is your initial independent analysis. Use the assigned question and measured evidence; '
                            'other specialists\' recommendations are withheld for this stage. The planner will receive all analyses. '
                            'Shared discussion is available for subsequent corrections.')
        if self.stage == 'work' and (self.corrections, agent.name) in self.worker_reminders:
            instruction += '\nYour previous reply was empty or only a control token. Give the assigned deliverable or a specific blocker now.'
        if self.stage == 'work' and self.plan_submission_reminders and agent.name == 'Alloy_Design_Planner':
            instruction += '\nYour preceding prose did not record an action. Call record_design_plan now with the question, numerical tool, selected compositions OR IDs, and decision rationale. This is the one submission reminder; another prose-only reply will end this delegation as incomplete.'
        if self.stage == 'closing' and self.incomplete_workers:
            instruction += '\nWorkers without substantive contributions: ' + ', '.join(sorted(self.incomplete_workers))
        if self.stage == 'closing':
            instruction += '\nAfter your substantive scoped verdict or concrete corrections, write a separate final line exactly: Decision: TERMINATE or Decision: REVISE.'
        if self.stage == 'closing' and self.closure_reminders:
            instruction += '\nThe previous response did not provide a parseable decision. Include that separate final line now.'
        if self.stage == 'closing' and self.closing_evidence_provider:
            # Reviewer tools may update evidence within the same correction cycle.
            # Refresh before the next model reply; archive only changed snapshots.
            packet = self.closing_evidence_provider(self._review_messages)
            new_cycle = self._evidence_cycle != self.corrections
            if new_cycle or not self.evidence_packets or packet != self.evidence_packets[-1]:
                self.evidence_packets.append(deepcopy(packet))
                self._evidence_cycle = self.corrections
                if new_cycle:
                    self._compact_tail_start = None
                print('\nClosing review evidence:\n' + json.dumps(packet, indent=2), flush=True)
            factual = {k: v for k, v in self.evidence_packets[-1].items()
                       if k != 'recorded_claims_unverified'}
            instruction += ('\n\n# Current measured evidence and procedural status (not scientific approval)\n'
                            + json.dumps(factual))
            if self.evidence_packets[-1].get('recorded_claims_unverified'):
                instruction += ('\nThe separately supplied recorded rationale is the artifact to audit, not a source of measured facts. '
                    'Resolve disagreements against the exact result rows/read tools. A corrected summary does not repair an incorrect stored rationale: '
                    'request its owner to call record_recommendation in execute mode, then review that updated record. '
                    'Fresh numerical agreement alone does not validate the prose.')
                if not self.compact_closing_context:
                    instruction += ('\n\n# Recorded agent claims, unverified\n'
                                    + json.dumps(self.evidence_packets[-1]['recorded_claims_unverified']))
        if self.stage == 'closing' and self.compact_closing_context:
            if self._closing_hook is None:
                self._closing_hook = self._closing_messages
                self._closing_agent = agent
                agent.register_hook('process_all_messages_before_reply', self._closing_hook)
            if self.delegation_mode != 'plan':
                instruction += ('\nAssess the supplied unverified conclusion explicitly: which claims survive '
                'and which need correction? Review the assigned scope only. Where evaluated, compare J with its tie tolerance, equilibrium L12 at both temperatures, '
                'and EACH of FR, CSC and HCS against the reference and comparison cases when those quantities are in the assignment. Distinguish equilibrium phases at each temperature from terminal Scheil phases. '
                'Explain conflicting indicators using their underlying intervals. State the chosen '
                'tradeoff and preference departures; matching repeat calculations do not validate a design choice.')
        if self.stage == 'closing' and self.delegation_mode == 'plan':
            instruction += '\nFinal scope reminder: assess only the recorded plan. Hand execution to the manager; do not ask these workers to execute it.'
        if self.review_material and self.stage == 'work':
            instruction += ('\nEstablish the assigned row-by-row facts from the saved comparison first: '
                'values, differences, constraint eligibility, practical ties, bases, liquid flags and sources. '
                'Then explain the physical tradeoffs. The prior draft will be supplied separately to closing review.')
        if self.review_material and self.stage == 'closing':
            instruction += ('\nCheck the current worker report and the earlier draft independently against the factual table. '
                'Give a short claim-check table: claim and candidate, determining value/limit or source, '
                'supported/contradicted/missing, required correction. Then explain the physical tradeoffs and give your scoped verdict. '
                'Do not treat prior approval as evidence.\n\n# Earlier draft for checking (unverified)\n' + self.review_material)
        # Optional capability-owned procedure; existing restoration removes it
        # on reused teams. No extra turn or scientific routing.
        card = (agent.decision_codebook(self.delegation_mode, self.stage)
                if hasattr(agent, 'decision_codebook') else None)
        if card:
            instruction += '\n\n# Decision codebook: ' + card['section'] + '\n' + card['text']
        agent.update_system_message(original + '\n\n# Current team stage\n' + instruction)
        event = {'stage': self.stage, 'speaker': agent.name}
        if card:
            event['codebook'] = {k: card[k] for k in ('section', 'sha256')}
        self.events.append(event)
        print(f'\nTeam stage: {self.stage}; speaker: {agent.name}', flush=True)
        return agent

    def _stop(self, text, reason):
        # An audit-stage stop is not a response from the scientific reviewer.
        self.final_review = text if self.stage == 'closing' else None
        self.final_decision = review_decision(text) if self.final_review else None
        self.stop_reason = reason
        self.events.append({'stage': 'returned', 'reason': reason})
        return None

    def _plan_recorded(self):
        if self.delegation_mode != 'plan' or self.plan_readiness_provider is None:
            return True
        readiness = self.plan_readiness_provider()
        return readiness['ready'] and (self.plan_version_before_advice_correction is None
            or readiness['plan_version'] > self.plan_version_before_advice_correction)

    def __call__(self, last_speaker, groupchat):
        self._review_messages = groupchat.messages
        reviewer = groupchat.agent_by_name(self.reviewer_name)
        if reviewer is None:
            raise ValueError('Required reviewer is missing from the team')
        auditor = groupchat.agent_by_name(self.auditor_name) if self.auditor_name else None
        if self.auditor_name and (auditor is None or self.auditor_name in (self.reviewer_name, self.executor_name)):
            raise ValueError('A distinct configured evidence auditor must be present')
        workers = [a for a in groupchat.agents if a.name not in (self.reviewer_name, self.executor_name, self.auditor_name)]
        self.worker_names = {a.name for a in workers}
        if self.delegation_mode == 'plan':
            # Preserve the builder's specialist order; the planner integrates their evidence last.
            workers.sort(key=lambda a: a.name == 'Alloy_Design_Planner')
        message = groupchat.messages[-1]
        if len(groupchat.messages) == 1:
            self.pending_workers = list(workers)
            if workers and not self.opening_review:
                self.initial_review_done = True
                self.stage = 'work'
                self.last_worker = workers[0]
                return self._select(self.last_worker)
            self.stage = 'opening' if workers else 'audit' if auditor else 'closing'
            return self._select(auditor if self.stage == 'audit' else reviewer)
        if self.stage == 'opening' and last_speaker.name == self.reviewer_name and message.get('content'):
            self.opening_prose.append(message['content'])
        calls = message.get('tool_calls') or []
        if calls:
            names = [c['function']['name'] for c in calls]
            executors = [a for a in groupchat.agents if a.can_execute_function(names)]
            if len(executors) != 1 or executors[0].name != self.executor_name:
                raise ValueError('Expected the registered executor for requested tools')
            executor = executors[0]
            self.executor_predicates.setdefault(executor, executor._is_termination_msg)
            executor._is_termination_msg = lambda m: False
            self.pending_caller = last_speaker
            self.pending_tool_calls = calls
            self.events.append({'stage': 'tool', 'speaker': executor.name, 'caller': last_speaker.name, 'tools': names})
            return executor
        if last_speaker.name == self.executor_name:
            if self.pending_caller is None:
                raise ValueError('Executor replied without a pending tool request')
            caller, self.pending_caller = self.pending_caller, None
            replies = {r.get('tool_call_id'): r.get('content')
                       for r in message.get('tool_responses', [])}
            for call in self.pending_tool_calls:
                handoff = delegation_handoff(replies.get(call['id']))
                if handoff:
                    self.events.append(dict(stage='delegation_required', caller=caller.name,
                                            tool=call['function']['name'], **handoff))
                    return self._stop(None, 'Tool prerequisite requires manager handoff: '
                                      + json.dumps(handoff, ensure_ascii=False))
            failures = []
            for call in self.pending_tool_calls:
                error = replies.get(call['id'])
                if not isinstance(error, str) or not error.startswith('Error:'):
                    failures = []
                    break
                function = call['function']
                try:
                    arguments = json.loads(function['arguments'])
                except (ValueError, TypeError):
                    arguments = function['arguments']
                failures.append((function['name'], arguments, error))
            signature = json.dumps([caller.name, failures], sort_keys=True) if failures else None
            self.failed_plan_submissions += sum(f[0] == 'record_design_plan' for f in failures)
            if self.failed_plan_submissions >= 3:
                return self._stop(None, 'Three rejected planning submissions; bounded delegation stopped without a valid plan')
            if signature and signature == self.last_failed_tool_request:
                self.events.append({'stage': 'repeated_tool_failure', 'caller': caller.name,
                                    'failures': failures})
                return self._stop(None, 'Same tool arguments failed twice without progress; return to manager: '
                                  + failures[0][0] + ': ' + failures[0][2])
            self.last_failed_tool_request = signature
            return self._select(caller)
        text = str(message.get('content') or '').strip()
        if self.stage == 'opening':
            self.initial_review_done = True
            self.stage = 'work'
            self.last_worker = workers[0]
            return self._select(self.last_worker)
        if self.stage == 'work':
            self.last_worker = last_speaker
            if last_speaker.name == 'Alloy_Design_Planner' and not self._plan_recorded():
                if not self.plan_submission_reminders:
                    self.plan_submission_reminders += 1
                    self.events.append(dict(stage='missing_plan_submission', speaker=last_speaker.name,
                        evidence=self.plan_readiness_provider()))
                    return self._select(last_speaker)
                return self._stop(None, 'Planning incomplete after one submission reminder; no current action was recorded')
            substantive = text and text.strip('*` \n\t').upper() not in ('TERMINATE', 'REVISE')
            key = (self.corrections, last_speaker.name)
            if not substantive and key not in self.worker_reminders:
                self.worker_reminders.add(key)
                return self._select(last_speaker)
            if not substantive:
                self.incomplete_workers.add(last_speaker.name)
                self.events.append({'stage': 'incomplete_contribution', 'speaker': last_speaker.name})
            else:
                self.incomplete_workers.discard(last_speaker.name)
            self.pending_workers.remove(last_speaker)
            if not self.pending_workers:
                self.stage = 'audit' if auditor else 'closing'
                return self._select(auditor if auditor else reviewer)
            return self._select(self.pending_workers[0])
        if self.stage == 'closing' and not self._plan_recorded():
            return self._stop(None, 'Planning incomplete: critic approval cannot override a missing recorded action')
        # A formatting-only follow-up belongs to the substantive review from
        # this stage and correction cycle, never an earlier cycle's verdict.
        decision = review_decision(text)
        key = (self.stage, self.corrections)
        if substantive_review(text):
            self.review_text_by_cycle[key] = text
        elif decision and key in self.review_text_by_cycle:
            text = self.review_text_by_cycle[key] + '\n\nDecision: ' + decision
        if decision == 'REVISE' and substantive_review(text):
            fields = correction_fields(text)
            kinds = {k.lower() for k in fields.get('kind', set())}
            if len(kinds) > 1:
                return self._stop(text, 'Conflicting correction kinds; return to manager for a scoped assignment')
            if 'handoff' in kinds:
                return self._stop(text, 'Correction requires another delegation; return to manager with the requested operation/mode')
            if self.delegation_mode != 'plan' and 'plan' in kinds:
                return self._stop(text, 'New design decision requested; return to planner through manager')
            if self.stage == 'audit':
                self.audit_reports.append(text)
            if not workers:
                return self._stop(text, 'Review requires a working capability outside this review-only team')
            owners = fields.get('owner', set())
            named = [a for a in workers if a.name in owners]
            if not owners:
                return self._stop(text, 'No explicit correction owner; return to manager rather than repeat every worker')
            if owners - {a.name for a in workers}:
                return self._stop(text, 'Correction names a capability outside this team; return to manager')
            operations = fields.get('operation', set())
            if 'report' in kinds and not operations:
                return self._stop(text, 'Report correction needs an explicit operation: report_prose for this worker response, or record_recommendation in planner execute/nominate mode for the stored nomination. Return to manager for a scoped assignment.')
            offered = {t['function']['name'] for a in named
                       for t in (a.llm_config or {}).get('tools', [])}
            offered.add('report_prose')
            if operations - offered:
                return self._stop(text, 'Correction operation unavailable to its owner in '
                                  + self.delegation_mode + ' mode: '
                                  + ', '.join(sorted(operations - offered)) + '. Return to manager.')
            if self.corrections >= self.max_corrections:
                return self._stop(text, 'Correction limit reached; unresolved review returned to manager')
            self.corrections += 1
            self.audit_reminders = 0
            self.closure_reminders = 0
            self.stage = 'work'
            self.pending_workers = named
            if self.delegation_mode == 'plan' and any(a.name != 'Alloy_Design_Planner' for a in self.pending_workers):
                planner = next((a for a in workers if a.name == 'Alloy_Design_Planner'), None)
                if planner and planner not in self.pending_workers:
                    self.pending_workers.append(planner)
                self.pending_workers.sort(key=lambda a: a.name == 'Alloy_Design_Planner')
                if self.plan_readiness_provider:
                    self.plan_version_before_advice_correction = self.plan_readiness_provider()['plan_version']
            self.events.append({'stage': 'correction_targets', 'workers': [a.name for a in self.pending_workers]})
            return self._select(self.pending_workers[0])
        if self.stage == 'audit':
            if decision == 'TERMINATE' and substantive_review(text):
                self.audit_reports.append(text)
                self.stage = 'closing'
                return self._select(reviewer)
            if self.audit_reminders:
                return self._stop(None, 'Factual audit missing or bare after format reminder')
            self.audit_reminders += 1
            return self._select(auditor)
        if decision == 'TERMINATE' and substantive_review(text):
            return self._stop(text, 'Substantive closing review returned; scientific outcome requires its evidence')
        if self.closure_reminders:
            return self._stop(None, 'Closing verdict missing or bare after format reminder')
        self.closure_reminders += 1
        return self._select(reviewer)

    def restore(self):
        if self._closing_hook is not None:
            self._closing_agent.hook_lists['process_all_messages_before_reply'].remove(self._closing_hook)
            self._closing_hook = None
        for agent, hook in self.context_hooks.items():
            agent.hook_lists['process_all_messages_before_reply'].remove(hook)
        self.context_hooks.clear()
        for agent, prompt in self.original_prompts.items():
            agent.update_system_message(prompt)
        for executor, predicate in self.executor_predicates.items():
            executor._is_termination_msg = predicate

    def scientific_review_status(self):
        if self.final_review is None:
            return 'not_completed'
        if self.final_decision == 'REVISE':
            return 'correction_requested'
        return 'incomplete_contributions' if self.incomplete_workers else 'completed'

    def findings(self):
        audit = ('Evidence auditor reports:\n' + '\n\n'.join(self.audit_reports) + '\n\n') if self.audit_reports else ''
        return (audit + 'Scientific critic review (' + self.scientific_review_status() + '):\n' +
                super().findings() + '\nController status: ' + (self.stop_reason or 'Closing review not reached'))
