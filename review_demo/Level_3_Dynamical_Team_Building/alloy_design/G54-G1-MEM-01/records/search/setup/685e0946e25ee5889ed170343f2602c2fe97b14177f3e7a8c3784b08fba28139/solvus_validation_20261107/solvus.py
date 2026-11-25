"""Find a phase's upper disappearance boundary from equilibrium evaluations.

No Thermo-Calc import, file writes, reference answers or agent dependencies.
The callback returns {status: 'complete', phases: [{name, mole_fraction}, ...]}.
An absent phase is zero only after a successful equilibrium calculation.
"""
import math


def fraction(record, phase):
    if record.get('status') != 'complete':
        raise ValueError('Cannot infer phase absence from an incomplete calculation')
    phases = record['phases']
    values = [p['mole_fraction'] for p in phases]
    if not values or any(not math.isfinite(x) or x < -1e-10 for x in values):
        raise ValueError('Invalid equilibrium phase fractions')
    if abs(sum(values) - 1) > 1e-7:
        raise ValueError('Equilibrium phase fractions do not sum to one')
    return sum(p['mole_fraction'] for p in phases if p['name'].split('#')[0] == phase)


def find_disappearance(evaluate, phase='AL3M_D023', lower_K=523.15,
                       upper_K=1673.15, step_K=100., precision_K=5.,
                       phase_tolerance=1e-8):
    """Scan the full interval and refine all sampled disappearance brackets.

    Return the highest-temperature boundary, explicitly retaining any lower
    disappearance intervals. This is an upper stability boundary, not necessarily
    a solid-state solvus. A coarse scan cannot exclude unsampled narrow phase
    pockets. Exceptions propagate; the caller preserves failures. Units: kelvin.
    """
    settings = [lower_K, upper_K, step_K, precision_K, phase_tolerance]
    if not all(math.isfinite(x) for x in settings):
        raise ValueError('Settings must be finite')
    if not 0 < lower_K < upper_K or not 0 < precision_K <= step_K or not 0 < phase_tolerance < 1:
        raise ValueError('Invalid scan settings')
    samples = {}

    def at(t):
        if t not in samples:
            r = evaluate(t)
            samples[t] = {'temperature_K': t, 'target_mole_fraction': fraction(r, phase),
                          'liquid_mole_fraction': fraction(r, 'LIQUID')}
        return samples[t]['target_mole_fraction'] > phase_tolerance

    grid = [lower_K + i * step_K for i in range(math.ceil((upper_K-lower_K)/step_K))]
    grid.append(upper_K)
    present = [at(t) for t in grid]
    brackets = [(a, b) for a, b, pa, pb in zip(grid, grid[1:], present, present[1:]) if pa and not pb]
    base = dict(phase=phase, phase_tolerance=phase_tolerance, precision_K=precision_K,
                sampled_brackets_K=brackets, temperature_K=None)
    if present[-1]:
        return dict(base, status='present_at_upper_bound', samples=list(samples.values()))
    if not any(present):
        return dict(base, status='not_detected_on_scan', samples=list(samples.values()))
    refined = []
    for lo, hi in brackets:
        while hi - lo > precision_K:
            mid = (lo+hi)/2
            if at(mid):
                lo = mid
            else:
                hi = mid
        refined.append({'bracket_K':[lo,hi], 'temperature_K':(lo+hi)/2,
                        'liquid_at_boundary':any(samples[t]['liquid_mole_fraction']>phase_tolerance for t in (lo,hi))})
    lo, hi = refined[-1]['bracket_K']
    base.update(temperature_K=(lo+hi)/2)
    return dict(base, status='bracketed', bracket_K=[lo, hi],
                selection='highest sampled disappearance boundary',
                disappearance_intervals=refined, multiple_intervals=len(refined)>1,
                below=samples[lo], above=samples[hi],
                liquid_at_boundary=any(samples[t]['liquid_mole_fraction'] > phase_tolerance for t in (lo,hi)),
                samples=sorted(samples.values(), key=lambda x:x['temperature_K']))
