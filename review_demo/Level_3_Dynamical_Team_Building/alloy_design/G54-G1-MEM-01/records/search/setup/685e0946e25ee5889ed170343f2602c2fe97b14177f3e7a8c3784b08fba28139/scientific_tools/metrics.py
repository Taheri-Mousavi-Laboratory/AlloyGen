"""Internal postprocessing. Only one selected convention is exposed per indicator."""

import math
import numpy as np
from scientific_calculator import coarsening
from ni_comparison.historical_functions import CSC_4, FR_cutoff


def equilibrium_checks(record):
    phases = record["phases"]
    numbers = [p["mole_fraction"] for p in phases]
    numbers += [v for p in phases for v in p["composition_mole_fraction"].values()]
    if not phases or any(not math.isfinite(v) or v < -1e-10 for v in numbers):
        raise ValueError("Invalid equilibrium fractions")
    errors = {
        e: sum(p["mole_fraction"] * p["composition_mole_fraction"][e] for p in phases)
        - v
        for e, v in record["bulk_mole_fractions"].items()
    }
    checks = dict(
        phase_sum_error=abs(sum(p["mole_fraction"] for p in phases) - 1),
        max_component_balance_error=max(map(abs, errors.values())),
        max_phase_composition_sum_error=max(
            abs(sum(p["composition_mole_fraction"].values()) - 1) for p in phases
        ),
        input_condition_error=record["input_condition_error"],
    )
    if max(checks.values()) > 1e-7:
        raise ValueError(f"Equilibrium conservation/input check failed: {checks}")
    return checks


def phase_percent(record, phase):
    if record.get("status") != "complete":
        raise ValueError("An incomplete calculation cannot establish phase absence")
    return 100 * sum(
        p["mole_fraction"] for p in record["phases"] if p["name"].split("#")[0] == phase
    )


def coarsening_metric(record, parameters, reference_m):
    if (
        record.get("status") != "complete"
        or abs(record["temperature_K"] - parameters["temperature_K"]) > 1e-8
    ):
        raise ValueError(
            "Coarsening parameters apply only to complete 250 C equilibria"
        )
    d = coarsening(record["phases"], parameters)
    if d["status"] != "computed_with_historical_parameters":
        return dict(status="unavailable", J=None, reason=d["reason"])
    if not math.isfinite(reference_m) or reference_m <= 0:
        raise ValueError("Invalid coarsening reference")
    return dict(status="complete", J=d["m_raw"] / reference_m)


def solidification_metrics(record):
    """Sort matched histories by cooling temperature before calling source formulas.

    Unavailable CSC is null rather than a historical fallback 0 or 1. Phase
    quantities/FR can still be reported when the CSC interval is undefined.
    """
    if record.get("status") != "complete":
        raise ValueError("Incomplete Scheil trajectory")
    curves = record["curves"]

    def series(key):
        c = curves[key]
        if len(c["temperature_K"]) != len(c["values"]):
            raise ValueError(f"Mismatched curve lengths: {key}")
        points = sorted(zip(c["temperature_K"], c["values"]), key=lambda p: -p[0])
        if not points or not all(math.isfinite(v) for p in points for v in p):
            raise ValueError(f"Invalid curve: {key}")
        return np.asarray(points, dtype=float).T

    t, ns = series("solid_mole_fraction")
    th, nh = series("heat_per_mole")
    if len(t) < 2 or len(t) != len(th) or not np.allclose(t, th, atol=1e-8, rtol=0):
        raise ValueError("Solid and heat histories are not aligned")
    if (
        np.any(np.diff(t) >= 0)
        or np.any(np.diff(ns) < -1e-9)
        or np.any(ns < -1e-9)
        or np.any(ns > 1 + 1e-6)
    ):
        raise ValueError("Invalid cooling/total-solid history")
    terminal = {}
    total = np.zeros(len(t))
    for name in curves:
        if not name.startswith("phase_"):
            continue
        tp, fp = series(name)
        if (
            len(tp) != len(t)
            or not np.allclose(tp, t, atol=1e-8, rtol=0)
            or np.any(fp < -1e-9)
        ):
            raise ValueError(
                "Phase histories are not aligned or contain invalid amounts"
            )
        total += fp
        base = name[6:].split("#")[0]
        terminal[base] = terminal.get(base, 0.0) + float(fp[-1]) * 100
    balance = float(np.max(np.abs(total - ns)))
    if not terminal or balance > 1e-6:
        raise ValueError("Accumulated phase sums do not match total solid")
    output = dict(
        status="complete",
        terminal_temperature_C=float(t[-1] - 273.15),
        terminal_solid_mol_percent=float(100 * ns[-1]),
        ternary_mol_percent=terminal.get("AL23NI6M4", 0.0),
        L12_mol_percent=terminal.get("AL3M_L12", 0.0),
        Al3Ni_mol_percent=terminal.get("AL3NI1", 0.0),
        Al3Zr_mol_percent=terminal.get("AL3M_D023", 0.0),
        FR_K=None,
        CSC=None,
        HCS=None,
        max_phase_sum_error=balance,
    )
    issues = []
    # A prematurely terminated history cannot establish the chosen FR endpoint.
    if ns[0] > 0.02 or ns[-1] < 0.995 - 1e-8:
        issues.append(
            "Incomplete total-solid interval for FR to the prescribed Scheil termination"
        )
    else:
        output["FR_K"] = float(FR_cutoff(ns, t, 0.02))
    if (
        ns[0] > 0.40
        or ns[-1] < 0.98
        or nh[0] <= nh[-1]
        or np.any(np.diff(nh) > 1e-6)
        or np.any(np.diff(ns) <= 0)
    ):
        issues.append(
            "CSC needs covered thresholds, increasing solid fraction and a usable heat history"
        )
    else:
        csc, _, tv, tr, t1, t2, t3 = CSC_4(ns, nh, dx=0.001)
        if (
            not all(math.isfinite(float(v)) for v in (csc, tv, tr, t1, t2, t3))
            or tr <= 0
            or tv < 0
        ):
            issues.append(
                "CSC time interval is undefined; source fallback values are not physical results"
            )
        else:
            output.update(
                CSC=float(csc),
                CSC_intervals=dict(relaxation=float(tr), vulnerable=float(tv)),
            )
    if output["FR_K"] is not None and output["CSC"] is not None:
        output["HCS"] = output["CSC"] * output["FR_K"] / 100
    if issues:
        output.update(status="partial", issues=issues)
    return output
