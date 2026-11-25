"""Optional reviewer turns around an otherwise freely coordinated group chat."""


class ReviewTurns:
    def __init__(self, reviewer_name):
        self.reviewer_name = reviewer_name
        self.initial_review_done = False
        self.final_review_pending = False
        self.final_review = None

    def __call__(self, last_speaker, groupchat):
        reviewer = groupchat.agent_by_name(self.reviewer_name)
        if reviewer is None:
            raise ValueError('Required reviewer is missing from the team')
        message = groupchat.messages[-1]
        calls = message.get('tool_calls') or []
        if calls:
            names = [c['function']['name'] for c in calls]
            executors = [a for a in groupchat.agents if a.can_execute_function(names)]
            if len(executors) != 1:
                raise ValueError('Expected one registered executor for requested tools')
            return executors[0]
        # The first message is the delegated task, even if the initiating agent
        # happens to be the reviewer. Tool results are not a completed review.
        is_review = last_speaker == reviewer and len(groupchat.messages) > 1
        text = str(message.get('content') or '').strip()
        if not self.initial_review_done:
            if not is_review:
                return reviewer
            self.initial_review_done = True
            return 'auto'
        if self.final_review_pending:
            if not is_review:
                return reviewer
            self.final_review = text
            self.final_review_pending = False
            if text.endswith('TERMINATE'):
                return None
            return 'auto'
        if text.endswith('TERMINATE'):
            # A provisional conclusion must receive a subsequent reviewer turn.
            self.final_review = None
            self.final_review_pending = True
            return reviewer
        return 'auto'

    def findings(self):
        if self.final_review is None or not self.final_review.removesuffix('TERMINATE').strip():
            return 'Final critic review was not completed. Treat the team outcome as incomplete.'
        return self.final_review
