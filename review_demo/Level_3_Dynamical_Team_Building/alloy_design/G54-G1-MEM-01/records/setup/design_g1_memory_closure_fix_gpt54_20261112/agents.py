"""Five focused roles; all knowledge is explicit and answer-free."""
from pathlib import Path
from copy import deepcopy
from autogen import ConversableAgent
from autogen.tools import Tool
from . import tools
from .consultation import CodebookConsultation

HERE = Path(__file__).resolve().parent


class Specialist(CodebookConsultation, ConversableAgent):
    guide = ''
    planning_expertise = None
    functions = ()
    group_discussion_instruction = ('Use the assigned scope and measured evidence to form your own judgment. '
        'Follow the current team-stage instruction. When discussing others\' conclusions, evaluate their evidence '
        'and explain agreement or disagreement; prior agreement is not evidence.')
    group_completion_instruction = ('Complete the delegated scope, report evidence and limitations, '
        'then hand off with a substantive report. The critic gives a substantive scoped verdict or a concrete REVISE request.')

    def __init__(self, *, llm_config, guidance_variant='rag', **kwargs):
        if guidance_variant not in ('baseline', 'codebook', 'rag'):
            raise ValueError('Choose baseline, codebook or rag guidance')
        self.guidance_variant = guidance_variant
        kwargs.setdefault('human_input_mode', 'NEVER')
        kwargs.setdefault('is_termination_msg', lambda message: False)
        files = ['common.md', 'physics.md', self.guide]
        if self.guide == 'planner.md' and guidance_variant == 'baseline':
            files.append('planner_procedure_G1.md')
        guide_root = HERE / ('baseline_guides' if guidance_variant == 'baseline' else 'guides')
        message = tools.render_text('\n\n'.join((guide_root/name).read_text() for name in files))
        if self.guide in ('coarsening.md', 'phase.md', 'solidification.md'):
            message += '\n\n' + tools.render_text((HERE / 'codebooks' / self.guide).read_text())
        role = {'planner.md': 'planner', 'reviewer.md': 'critic'}.get(self.guide)
        if guidance_variant == 'rag' and role:
            message = message.replace('Apply the current decision-codebook section to the supplied evidence.',
                'Consult your codebook and apply relevant guidance to the supplied evidence.')
            message = message.replace('Apply the current decision-codebook section independently to consequential claims.',
                'Consult your codebook and independently apply relevant guidance to consequential claims.')
            message += ('\n\nBefore committing a consequential experiment or scientific verdict, '
                'identify the methodological question and call consult_' + role + '_codebook. '
                'Check any candidate IDs and composition premises against saved evidence first. '
                'Formulate the question yourself; ask what needs to be determined rather than ask for confirmation. '
                'Use the returned references to explain briefly the implication for the current evidence, '
                'then submit the action or verdict using your existing required fields. '
                'If needed, refine the question. Reference guidance does not replace those required fields.')
            message += ('\nG1 consultation is required at the start of each assignment. '
                'Ask your own focused question first, receive the reference-tool reply, '
                'then make the decision. Reference lookup is allowed in every delegation mode; '
                'it is not a new physics calculation. If retrieval fails or is irrelevant, '
                'refine the query or state the remaining uncertainty.')
        super().__init__(llm_config=llm_config, system_message=message, **kwargs)
        self.bound_tools = []
        functions = [*self.functions, tools.read_design_results, tools.read_design_evidence]
        if guidance_variant == 'rag' and role:
            from .retrieval import make_retrieval
            functions.append(make_retrieval(self.guide, tools.PROTOCOL.get('retrieval_tokenizer_model', tools.PROTOCOL['model'])))
        for function in functions:
            tool = Tool(name=function.__name__, description=tools.render_text(function.__doc__), func_or_tool=function)
            self.register_for_llm()(tool)
            self.bound_tools.append(tool)
        for schema in list(self.llm_config.get('tools', [])):
            if schema['function']['name'] == 'record_design_plan':
                schema = deepcopy(schema)
                schema['function']['parameters']['properties']['next_action'] = deepcopy(tools.NEXT_ACTION_SCHEMA)
                self.update_tool_signature(schema, is_remove=False)
            elif schema['function']['name'].startswith('evaluate_'):
                schema = deepcopy(schema)
                schema['function']['parameters']['properties']['compositions']['maxItems'] = tools.PROTOCOL['max_batch_size']
                self.update_tool_signature(schema, is_remove=False)
        self._all_tool_schemas = deepcopy(self.llm_config.get('tools', []))
        self.begin_consultation()
        if tools._MODE.get():
            self.configure_delegation(tools._MODE.get())

    def configure_delegation(self, mode):
        """Offer only actions legal in this mode, including on reused agents."""
        legal = {
            'plan': {'record_design_plan', 'read_design_results', 'read_design_evidence'},
            'execute': {'evaluate_coarsening', 'evaluate_phase_stability', 'evaluate_solidification',
                        'record_recommendation', 'read_design_results', 'read_design_evidence'},
            'review': {'verify_candidate', 'read_design_results', 'read_design_evidence'},
        }[mode]
        legal |= {'consult_planner_codebook', 'consult_critic_codebook'}
        current = {t['function']['name'] for t in self.llm_config.get('tools', [])}
        for schema in self._all_tool_schemas:
            name = schema['function']['name']
            if name in current and name not in legal:
                self.update_tool_signature(name, is_remove=True)
            elif name not in current and name in legal:
                self.update_tool_signature(deepcopy(schema), is_remove=False)
        self.begin_consultation()

    def current_deliverable(self, mode):
        return self.mode_deliverables[mode]

    def decision_codebook(self, mode, stage):
        """Read only this role/mode's procedure; never select a scientific action."""
        if self.guidance_variant != 'codebook':
            return None
        from .codebook import decision_section
        return decision_section(self.guide, mode, stage)

    def register_tools_for_execution(self, executor):
        for tool in self.bound_tools:
            if tool.name not in executor.function_map:
                tool.register_for_execution(executor)


class Planner(Specialist):
    guide = 'planner.md'
    functions = (tools.record_design_plan, tools.record_recommendation)
    work_deliverable = 'Translate the requirements into a workflow, choose informative next experiments and make the final evidence-supported nomination. Own sample selection.'
    mode_deliverables = {
        'plan': 'Use the selected specialist analyses to record one next_action. On the first plan also record requirements_map and workflow_mermaid; with no measurements, justify the initial design using task and method knowledge. On later plans connect measured comparisons to the next hypothesis and calculation. Own exact samples; JSON in prose is not a submission.',
        'execute': 'When selection is assigned, integrate specialist findings and record the justified nomination after exact-composition coverage. Do not run physics.',
        'review': 'For a pre-nomination assignment, recommend further investigation or justify stopping from specialist evidence; no numerical plan or nomination is due. Otherwise explain the assigned design choice and tradeoffs. Do not repeat the full numerical table.'}


class Coarsening(Specialist):
    planning_expertise = 'coarsening'
    guide = 'coarsening.md'
    functions = (tools.evaluate_coarsening,)
    work_deliverable = 'If the recorded action assigns coarsening, execute and interpret that J comparison. If another calculation is assigned, give only a relevant handoff; do not launch a J calculation. Identify valid low-J regions and ties with exact composition IDs and reference. Give the planner a bounded refinement or next-capability recommendation.'
    mode_deliverables = {
        'plan': 'Apply the planning-analysis method in your common guide to measured J. Identify validity, practical ties and the coarsening tradeoff. Leave exact samples to the planner.',
        'execute': work_deliverable,
        'review': 'Check J validity, comparison to the reference and practical ties. Add a distinct J conclusion; do not reproduce a phase/Scheil report.'}


class PhaseStability(Specialist):
    planning_expertise = 'phase_stability'
    guide = 'phase.md'
    functions = (tools.evaluate_phase_stability,)
    work_deliverable = 'If the recorded action assigns phase stability, execute and interpret that phase work. If another calculation is assigned, give only a relevant handoff; do not launch phase confirmation. State phase feasibility at each exact composition and the reason for any proposed refinement. Do not equate equilibrium and Scheil phase amounts.'
    mode_deliverables = {
        'plan': 'Build the matched-pair table specified in your phase guide from the supplied measurements, then explain the service-phase gain, high-temperature constraint margin and competing-phase costs. Distinguish known phase amounts from unavailable boundary results. Leave exact samples to the planner.',
        'execute': work_deliverable,
        'review': 'Assess phase qualification and useful improvement opportunities, including unused constraint margin and Er/Zr tradeoffs. Recommend the scientific question for another measurement when consequential; a complete profile is not a stopping reason. Use existing J as context.'}


class Solidification(Specialist):
    planning_expertise = 'solidification'
    guide = 'solidification.md'
    functions = (tools.evaluate_solidification,)
    work_deliverable = 'If the recorded action assigns solidification, execute that comparison and interpret all terminal phases plus FR/CSC/HCS with the same reference. If another calculation is assigned, give only a relevant handoff; do not launch Scheil. State favorable and unfavorable comparisons, using actual intervals when explaining them.'
    mode_deliverables = {
        'plan': 'Apply the planning-analysis method in your common guide to measured solidification evidence, if relevant to the assigned question. Explain reservoir/cracking tradeoffs and the limits of what Scheil can answer. Leave exact samples to the planner.',
        'execute': work_deliverable,
        'review': 'Assess downstream costs of the current composition branch, including Er/Zr as well as Ni. Check reservoir, competing phases and FR/CSC/HCS. If Ni variants share a large penalty, explain whether upstream Er/Zr evidence merits reconsideration. Read the FR-start phase diagnostic when its origin matters. Suggest the unresolved measurement, leaving exact samples to the planner.'}


class ScientificCritic(Specialist):
    guide = 'reviewer.md'
    functions = (tools.verify_candidate,)
