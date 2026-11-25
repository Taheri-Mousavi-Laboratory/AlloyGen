"""One capacity description for library selection and manager preparation."""


def capacity_message(max_agents, required_agents):
    required = list(dict.fromkeys(required_agents or []))
    if not isinstance(max_agents, int) or isinstance(max_agents, bool) or max_agents < 1:
        raise ValueError('max_agents must be a positive integer')
    if len(required) > max_agents:
        raise ValueError('Required membership alone exceeds max_agents')
    available = max_agents - len(required)
    return (
        f'Team capacity: {max_agents} total expert slots minus {len(required)} '
        f'required members leaves {available} additional working slots. '
        'Required members count once even if explicitly selected. '
        'The tool executor does not consume an expert slot. '
        'Use only the working expertise needed for this assignment; a review-only '
        'assignment needs no additional worker. If essential capabilities cannot '
        'fit, the manager must split the work into scoped sequential assignments; '
        'do not silently drop essential expertise or a required reviewer.'
    )
