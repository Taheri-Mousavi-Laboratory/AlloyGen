"""Facts about sampling and branch coverage; no scientific verdict or selection."""

def investigation_facts(state):
    seen_ids, seen_pairs, phase_seen = set(), set(), False
    batches = []
    for call in state['calls']:
        new_ids = [cid for cid in call['candidate_ids'] if cid not in seen_ids]
        new_pairs = []
        for cid, comp in zip(call['candidate_ids'], call['compositions']):
            pair = (comp['Er'], comp['Zr'])
            if pair not in seen_pairs:
                new_pairs.append(dict(candidate_id=cid, Er=pair[0], Zr=pair[1]))
            seen_pairs.add(pair)
        batches.append(dict(batch_id=call['id'], kind=call['kind'], status=call['status'],
                            new_candidate_ids=new_ids, new_Er_Zr_pairs=new_pairs,
                            follows_phase_calculation=phase_seen))
        seen_ids.update(call['candidate_ids'])
        phase_seen |= call['kind'] == 'phase_stability'
    groups = {}
    for cid,c in state['candidates'].items():
        if c['methods'].get('solidification',{}).get('status') == 'complete':
            comp=c['composition_at_percent']
            groups.setdefault((comp['Er'],comp['Zr']),[]).append((cid,comp['Ni']))
    support = {}
    for cid,c in state['candidates'].items():
        comp=c['composition_at_percent'];rows=groups.get((comp['Er'],comp['Zr']),[])
        support[cid]=dict(Er=comp['Er'],Zr=comp['Zr'],
                         measured_Ni_levels=sorted({row[1] for row in rows}),
                         solidification_ids=[row[0] for row in rows],
                         has_controlled_Ni_comparison=len({row[1] for row in rows})>=2)
    return dict(batches=batches, Ni_evidence_by_candidate_branch=support,
                meaning='Sampling facts only. A new pair is not proof of useful refinement; a controlled Ni comparison is not proof of a plateau or optimum. Review its question, evidence, calculation status and decision consequence.')
