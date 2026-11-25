"""Compact factual comparisons from saved evidence; no solver or alloy selection."""
import hashlib
import json
from . import tools
from .qualification import result_qualification, valid_phase_point


# These describe the declared design indicators, not preferred compositions.
INDICATORS = {
    'J': dict(method='coarsening', unit='dimensionless', purpose='coarsening at 250 C; lower preferred'),
    'L12_250_mol_percent': dict(method='phase_stability', unit='mol%', purpose='service-temperature strengthening phase; higher preferred'),
    'L12_660_mol_percent': dict(method='phase_stability', unit='mol%', purpose='high-temperature phase constraint; compare with the declared maximum'),
    'Al3Zr_250_mol_percent': dict(method='phase_stability', unit='mol%', purpose='competing phase at 250 C; lower preferred'),
    'Al3Zr_660_mol_percent': dict(method='phase_stability', unit='mol%', purpose='competing phase at 660 C; lower preferred'),
    'Al3Zr_boundary': dict(method='phase_stability', unit='degC', purpose='upper competing-phase disappearance; higher persistence is not favorable under this objective; liquid involvement must be reported'),
    'ternary_mol_percent': dict(method='solidification', unit='mol%', purpose='terminal Scheil ternary reservoir; seek useful gain/plateau, not unlimited maximization'),
    'L12_mol_percent': dict(method='solidification', unit='mol%', purpose='terminal Scheil primary L12; avoid unnecessary primary phase; not aged precipitate amount'),
    'Al3Ni_mol_percent': dict(method='solidification', unit='mol%', purpose='terminal Scheil competing Al3Ni; limit unnecessary amount'),
    'Al3Zr_mol_percent': dict(method='solidification', unit='mol%', purpose='terminal Scheil competing Al3Zr; lower preferred with coarsening tradeoff'),
    'FR_K': dict(method='solidification', unit='K', purpose='total-solid approximately 2%-to-termination interval; lower preferred'),
    'CSC': dict(method='solidification', unit='dimensionless', purpose='interpolated heat-coordinate interval ratio; lower preferred'),
    'HCS': dict(method='solidification', unit='scaled', purpose='CSC times total-solid FR / 100; lower preferred; no experimental acceptance threshold'),
    'terminal_solid_mol_percent': dict(method='solidification', unit='mol%', purpose='terminal accumulated solid amount; calculation state, not an optimization objective'),
    'terminal_temperature_C': dict(method='solidification', unit='degC', purpose='actual Scheil termination temperature; calculation state, not an optimization objective'),
}


def comparison_packet(state, candidate_ids, baseline_ids, protocol):
    """Carry all named comparators and their exact sources without interpreting a winner."""
    ids = list(dict.fromkeys([*baseline_ids, *candidate_ids]))
    unknown = set(ids) - set(state['candidates'])
    if unknown:
        raise ValueError('Unknown comparison IDs: ' + ', '.join(sorted(unknown)))

    def measurements(methods):
        output = {}
        for key, meta in INDICATORS.items():
            record = methods.get(meta['method'], {})
            value = record.get(key)
            boundary = value if isinstance(value, dict) else {}
            if key == 'Al3Zr_boundary':
                value = boundary.get('upper_temperature_C')
            item = dict(value=value, result_status=record.get('status', 'not_available'),
                        evidence_id=record.get('evidence_id'))
            if key == 'Al3Zr_boundary':
                item.update(liquid_at_upper_boundary=boundary.get('liquid_at_upper_boundary'),
                            boundary_status=boundary.get('status', 'not_available'))
            output[key] = item
        return output

    reference = state['reference']
    ref_identity = next(iter(reference.values()), {})
    rows = [dict(id='reference', role='reference', composition_percent=ref_identity.get('composition_percent'),
                 basis=ref_identity.get('basis'), values=measurements(reference),
                 phase_qualification=result_qualification('phase_stability', reference.get('phase_stability', {})))]
    for cid in ids:
        candidate = state['candidates'][cid]
        rows.append(dict(id=cid, role='baseline' if cid in baseline_ids else 'candidate',
                         composition_percent=candidate['composition_at_percent'], basis='atomic_percent',
                         values=measurements(candidate['methods']),
                         phase_qualification=result_qualification('phase_stability', candidate['methods'].get('phase_stability', {}))))
    for row in rows:
        for key, item in row['values'].items():
            value, ref = item['value'], rows[0]['values'][key]['value']
            item['minus_reference'] = value-ref if value is not None and ref is not None else None
        phase = row['values']['L12_660_mol_percent']
        row['meets_660_limit'] = (phase['value'] <= protocol['phase_limit_mol_percent']
            if valid_phase_point(phase['value']) else None)
        j = row['values']['J']['value']
        row['J_tied_with_baseline'] = {}
        for base in rows:
            if base['role'] == 'baseline':
                bj = base['values']['J']['value']
                row['J_tied_with_baseline'][base['id']] = abs(j-bj) < protocol['J_tie_absolute'] if j is not None and bj is not None else None
    for row in rows:
        j = row['values']['J']['value']
        row['J_ties_with'] = [other['id'] for other in rows if other['id'] != row['id']
            and j is not None and other['values']['J']['value'] is not None
            and abs(j - other['values']['J']['value']) < protocol['J_tie_absolute']]
    packet = dict(indicators=INDICATORS, phase_limit_mol_percent=protocol['phase_limit_mol_percent'],
                  J_tie_absolute=protocol['J_tie_absolute'], rows=rows,
                  scope='Reference, named baseline and candidate rows are distinct populations. Never extend a new-candidates-only conclusion to the baseline.',
                  meaning='Saved facts and declared indicator purposes, not a ranking, recommendation or scientific approval. Missing values remain unavailable.')
    packet['fingerprint'] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    return packet


def render_comparison(packet):
    """One readable table view of the existing packet; no new physical inference."""
    def show(value):
        return 'unavailable' if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)
    rows = packet['rows']
    lines = [packet['scope'], packet['meaning'],
        f"L12_660 limit: <= {packet['phase_limit_mol_percent']} mol%; absolute J tie tolerance: {packet['J_tie_absolute']}.",
        '\n| Case | Role | Composition basis | Composition (%) |', '|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['role']} | {row['basis']} | {show(row['composition_percent'])} |")
    lines += ['\nEach numerical cell is value; delta from reference. Values retain saved precision.',
        '| Quantity (unit) | ' + ' | '.join(r['id'] for r in rows) + ' |',
        '|---|' + '---|' * len(rows)]
    for key, meta in packet['indicators'].items():
        cells = [show(r['values'][key]['value']) + '; delta ' + show(r['values'][key]['minus_reference']) for r in rows]
        lines.append(f"| {key} ({meta['unit']}) | " + ' | '.join(cells) + ' |')
    lines += ['\n| Case | Passes 660 C point constraint | J ties with | Liquid at upper boundary | Boundary status |', '|---|---|---|---|---|']
    for row in rows:
        boundary = row['values']['Al3Zr_boundary']
        lines.append('| ' + ' | '.join([row['id'], show(row['meets_660_limit']), show(row['J_ties_with']),
            show(boundary['liquid_at_upper_boundary']), show(boundary['boundary_status'])]) + ' |')
    lines.append('\nExact sources and result status:')
    for row in rows:
        qualification = row['phase_qualification']
        if row['values']['L12_660_mol_percent']['evidence_id']:
            lines.append(f"- {row['id']}: full phase qualification usable={qualification['usable']}; "
                         + json.dumps({k: qualification[k] for k in ('required_points_complete',
                             'available_point_quantities', 'missing_or_invalid_point_quantities', 'boundary_status')}, ensure_ascii=False))
        if qualification.get('scan'):
            lines.append(f"- {row['id']}: phase qualification {qualification['interpretation']}; "
                         + json.dumps(qualification['scan'], ensure_ascii=False))
        groups = {}
        for key, item in row['values'].items():
            groups.setdefault((item['evidence_id'], item['result_status']), []).append(key)
        for (source, status), quantities in groups.items():
            lines.append(f"- {row['id']}: {', '.join(quantities)} — source {show(source)}; status {status}.")
    lines.append('\nIndicator purposes:')
    for key, meta in packet['indicators'].items():
        lines.append(f"- {key}: {meta['purpose']}.")
    return '\n'.join(lines)


class DesignEvidence:
    """Current decision context and final nominees, without choosing candidates."""
    def comparison(self, state, mode=None):
        mode = mode or tools._MODE.get() or 'mission'
        nomination = tools.latest_recommendation(state)
        action = state['plan'][-1]['next_action'] if state['plan'] else {}
        if mode in ('review', 'mission'):
            # A whole-investigation review must see earlier comparators too,
            # even when the latest action evaluated a different composition.
            # The ledger is bounded by the existing 32-candidate allowance.
            ids = list(state['candidates'])
        elif nomination:
            ids = nomination['candidate_ids']
        else:
            current = state['calls'][-1]['candidate_ids'] if state['calls'] else []
            planned = list(action.get('candidate_ids', []))
            recipes = {tools.identity(c) for c in action.get('compositions', [])}
            planned += [cid for cid,c in state['candidates'].items()
                        if tools.identity(c['composition_at_percent']) in recipes]
            ids = list(dict.fromkeys([*current, *planned, *action.get('comparison_ids', [])]))
        if mode == 'execute':
            # A nomination may explain historical controlled contrasts, not only
            # the latest shortlist. Coverage supplies IDs, never a ranking.
            for contrast in tools.mission_coverage(state)['controlled_Ni_comparisons']:
                ids = list(dict.fromkeys([*ids, *contrast['candidate_ids']]))
        return comparison_packet(state, ids, [], tools.PROTOCOL)

    def shared(self, mode=None):
        state = tools.load()
        shared = tools.handoff(mode=mode)
        coverage = tools.mission_coverage(state)
        if (mode or tools._MODE.get() or 'mission') == 'mission':
            shared['completion']['pending'] = list(dict.fromkeys([*shared['completion']['pending'], *coverage['missing']]))
        overview = []
        for cid, item in state['candidates'].items():
            methods = item['methods']
            overview.append(dict(id=cid, composition_at_percent=item['composition_at_percent'],
                J=methods.get('coarsening', {}).get('J'),
                L12_250_mol_percent=methods.get('phase_stability', {}).get('L12_250_mol_percent'),
                L12_660_mol_percent=methods.get('phase_stability', {}).get('L12_660_mol_percent'),
                methods={k:v['status'] for k,v in methods.items()}))
        comparison = self.comparison(state, mode=mode)
        diagnostics = {}
        for row in comparison['rows']:
            methods = state['reference'] if row['id'] == 'reference' else state['candidates'][row['id']]['methods']
            s = methods.get('solidification', {})
            if s:
                diagnostics[row['id']] = {k:s.get(k) for k in ('FR_interval', 'CSC_intervals', 'evidence_id')}
        nomination = shared['latest_recommendation']
        attempt = shared['latest_nomination_attempt']
        claims = dict(meaning='Recorded agent claims to check against measured records; not factual evidence or scientific approval.',
                      current_nomination=nomination)
        if attempt != nomination:
            claims['latest_ineligible_attempt'] = attempt
        review_fingerprint = hashlib.sha256(json.dumps(dict(
            comparison=comparison['fingerprint'],
            verification={k: (state.get('verification') or {}).get(k)
                          for k in ('candidate_id', 'status', 'input_fingerprint')},
            nomination=None if nomination is None else {
                k: nomination[k] for k in ('candidate_ids', 'rationale')}), sort_keys=True).encode()).hexdigest()
        return dict(completion=shared['completion'], mission_coverage=coverage,
            latest_plan=shared['latest_plan'], next_action_execution=shared['next_action_execution'],
            planning_update=shared.get('planning_update'), allowances=shared['allowances'],
            recorded_claims_unverified=claims, review_fingerprint=review_fingerprint, verification=shared['verification'],
            candidate_overview=overview, comparison=dict(fingerprint=comparison['fingerprint'],
                candidate_ids=[row['id'] for row in comparison['rows'] if row['id'] != 'reference'],
                table=render_comparison(comparison) if state['reference'] else 'No scientific results yet.'),
            solidification_diagnostics=diagnostics)

    def closing(self, messages):
        # The closing critic audits all claims, including during execute mode.
        # This changes evidence visibility, never the delegation's permissions.
        return self.shared(mode='review')

    def completion(self, host):
        packet = self.shared(mode='mission')
        last = getattr(host, '_last_team_result', {}) or {}
        if not packet['completion']['pending'] and not (
            last.get('mode') == 'review' and last.get('scientific_review_status') == 'completed'
            and last.get('review_fingerprint') == packet['review_fingerprint']):
            packet['completion']['pending'].append('Review the current nomination, full evidence table and actual recommendation after fresh verification.')
        return packet
