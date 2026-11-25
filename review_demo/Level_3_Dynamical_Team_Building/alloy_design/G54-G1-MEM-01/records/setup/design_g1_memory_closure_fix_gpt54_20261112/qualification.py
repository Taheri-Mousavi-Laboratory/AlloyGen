"""Interpret saved calculation evidence without changing results or choosing alloys."""
import json
import math
from pathlib import Path

from scientific_tools.tools import REPO

PHASE_POINTS = ('L12_250_mol_percent', 'L12_660_mol_percent',
                'Al3Zr_250_mol_percent', 'Al3Zr_660_mol_percent')
NONDETECTION_ISSUE = 'Al3Zr upper boundary unavailable: not_detected_on_scan'


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_phase_point(value):
    return finite_number(value) and 0 <= value <= 100


def scan_summary(row):
    """Read the worker's saved scan; no solver, inferred boundary or phase absence."""
    for record in row.get('records', []):
        path = (REPO / record).resolve()
        if not path.is_relative_to(REPO) or not path.name.startswith('boundary_'):
            continue
        try:
            scan = json.loads(path.read_text())
            samples = scan.get('samples', [])
            tolerance = scan.get('phase_tolerance')
            if (scan.get('status') != 'not_detected_on_scan'
                    or scan.get('phase') != 'AL3M_D023'
                    or scan.get('temperature_K') is not None
                    or not finite_number(tolerance) or not 0 < tolerance < 1
                    or len(samples) < 2):
                continue
            temperatures = [s['temperature_K'] for s in samples]
            amounts = [s['target_mole_fraction'] for s in samples]
            if (not all(finite_number(t) and t > 0 for t in temperatures)
                    or len(set(temperatures)) != len(temperatures)
                    or not all(finite_number(x) and 0 <= x <= tolerance for x in amounts)):
                continue
            ordered = sorted(temperatures)
            return dict(source=record, phase=scan.get('phase'), samples=len(samples),
                        lower_C=ordered[0]-273.15, upper_C=ordered[-1]-273.15,
                        maximum_sample_gap_K=max(b-a for a,b in zip(ordered,ordered[1:])),
                        phase_tolerance=tolerance,
                        limitation='Not detected at sampled temperatures; an unsampled narrow phase pocket is not excluded. No solvus temperature is assigned.')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def result_qualification(kind, row):
    """Coverage readiness, separate from raw status, hard phase limit and design merit."""
    status = row.get('status', 'not_available')
    if kind != 'phase_stability':
        return dict(usable=status == 'complete', interpretation=status)
    available = {k: row[k] for k in PHASE_POINTS if valid_phase_point(row.get(k))}
    missing = [k for k in PHASE_POINTS if k not in available]
    points_valid = not missing
    coverage = dict(required_points_complete=points_valid,
                    available_point_quantities=available,
                    missing_or_invalid_point_quantities=missing,
                    boundary_status=(row.get('Al3Zr_boundary') or {}).get('status', 'not_available'))
    if status == 'complete' and points_valid:
        return dict(usable=True, interpretation='complete', **coverage)
    boundary = row.get('Al3Zr_boundary') or {}
    # Only the known nondetection case is qualified. Other partial/failure
    # reasons, missing point results and unbounded persistence stay unresolved.
    nondetection = (status == 'partial' and points_valid
        and row.get('issues') == [NONDETECTION_ISSUE]
        and boundary.get('status') == 'not_detected_on_scan'
        and boundary.get('upper_temperature_C') is None
        and not boundary.get('disappearance_intervals'))
    scan = scan_summary(row) if nondetection else None
    if scan:
        return dict(usable=True, interpretation='qualified_with_scan_limit',
                    scan=scan, **coverage)
    return dict(usable=False, interpretation=status, **coverage,
                limitation='Full phase qualification is unresolved; available individual points remain measured evidence. Missing points and a failed boundary are unknown, not zero.')


def usable_result(kind, row):
    return result_qualification(kind, row)['usable']
