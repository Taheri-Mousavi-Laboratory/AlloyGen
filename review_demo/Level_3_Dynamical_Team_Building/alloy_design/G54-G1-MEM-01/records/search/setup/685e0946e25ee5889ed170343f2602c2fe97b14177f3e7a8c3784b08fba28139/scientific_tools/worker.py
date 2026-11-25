"""Contained scientific worker for the three alloy-design tools. No model calls."""

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from ni_comparison.run_ni_comparison import configure, clean_error
from scientific_calculator import coarsening
from solvus_validation_20261107.solvus import find_disappearance
from scientific_tools.metrics import (
    equilibrium_checks,
    phase_percent,
    coarsening_metric,
    solidification_metrics,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Engine:
    def __init__(self, folder, settings, stack):
        self.folder = folder
        self.output = folder.parents[1]
        self.settings = settings
        self.stack = stack
        self.db = ROOT / "author_reference/private_assets/Al_jan05.TDB"
        if digest(self.db) != settings["database_sha256"]:
            raise RuntimeError("Unexpected thermodynamic database")
        inputs = [
            HERE / n for n in ("settings.json", "metrics.py", "worker.py", "tools.py")
        ]
        inputs += [
            ROOT / "scientific_calculator.py",
            ROOT / "solvus_validation_20261107/solvus.py",
            ROOT / "ni_comparison/historical_functions.py",
            ROOT / "ni_comparison/run_ni_comparison.py",
        ]
        self.hashes = {str(p.relative_to(REPO)): digest(p) for p in inputs}
        self.fingerprint = hashlib.sha256(
            json.dumps(self.hashes, sort_keys=True).encode()
        ).hexdigest()
        snapshot = self.output / "setup" / self.fingerprint
        if not snapshot.exists():
            snapshot.mkdir(parents=True)
            for p in inputs:
                dest = snapshot / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
            save(snapshot / "manifest.json", self.hashes)
        self.cache = self.output / "cache" / self.fingerprint
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "equilibrium_calls": 0,
            "scheil_calls": 0,
            "cache_hits": 0,
            "paid_model_calls": 0,
        }
        self.used = set()
        self.system = None
        self.runtime = configure("scientific_tools_" + folder.name)
        if importlib.metadata.version("tc-python") != settings["tc_python_version"]:
            raise RuntimeError("Unexpected TC-Python version")

    def get_system(self):
        if self.system is None:
            from tc_python import TCPython, LoggingPolicy

            session = self.stack.enter_context(
                TCPython(logging_policy=LoggingPolicy.NONE)
            )
            session.set_cache_folder(str(self.runtime / "c"), precision_for_floats=15)
            self.system = session.select_user_database_and_elements(
                str(self.db), self.settings["elements"]
            ).get_system()
        return self.system

    def raw(self, kind, composition, basis, temperature=None):
        identity = dict(
            kind=kind,
            composition_percent=composition,
            basis=basis,
            temperature_K=temperature,
            method_fingerprint=self.fingerprint,
        )
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".json")
        self.used.add(str(path.relative_to(REPO)))
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity or record["status"] != "complete":
                raise ValueError("Invalid saved calculation identity/status")
            self.stats["cache_hits"] += 1
            return record
        record = dict(
            identity=identity, status="started", record=str(path.relative_to(REPO))
        )
        began = time.monotonic()
        try:
            from tc_python import (
                ThermodynamicQuantity as TQ,
                CompositionUnit,
                ScheilQuantity as SQ,
                ScheilOptions,
            )

            system = self.get_system()
            elements = self.settings["elements"]
            fractions = {e: v / 100 for e, v in composition.items()}
            if kind == "equilibrium":
                factory = (
                    TQ.mass_fraction_of_a_component
                    if basis == "mass_percent"
                    else TQ.mole_fraction_of_a_component
                )
                calc = (
                    system.with_single_equilibrium_calculation()
                    .set_condition("T", temperature)
                    .set_condition("P", self.settings["pressure_Pa"])
                )
                for e, v in fractions.items():
                    calc.set_condition(factory(e), v)
                self.stats["equilibrium_calls"] += 1
                save(self.folder / "progress.json", self.stats)
                answer = calc.calculate_with_state(
                    timeout_in_minutes=self.settings["equilibrium_timeout_minutes"]
                )
                record.update(
                    temperature_K=temperature,
                    bulk_mole_fractions={
                        e: float(
                            answer.get_value_of(TQ.mole_fraction_of_a_component(e))
                        )
                        for e in elements
                    },
                    input_condition_error=max(
                        abs(float(answer.get_value_of(factory(e))) - v)
                        for e, v in fractions.items()
                    ),
                    phases=[
                        dict(
                            name=p,
                            mole_fraction=float(
                                answer.get_value_of(TQ.mole_fraction_of_a_phase(p))
                            ),
                            composition_mole_fraction={
                                e: float(
                                    answer.get_value_of(
                                        TQ.composition_of_phase_as_mole_fraction(p, e)
                                    )
                                )
                                for e in elements
                            },
                        )
                        for p in answer.get_stable_phases()
                    ],
                )
                record["checks"] = equilibrium_checks(record)
            else:
                s = self.settings["scheil"]
                unit = (
                    CompositionUnit.MASS_FRACTION
                    if basis == "mass_percent"
                    else CompositionUnit.MOLE_FRACTION
                )
                calc = (
                    system.with_scheil_calculation()
                    .set_composition_unit(unit)
                    .set_start_temperature(s["start_temperature_K"])
                )
                for e, v in fractions.items():
                    calc.set_composition(e, v)
                calc.with_options(
                    ScheilOptions()
                    .set_temperature_step(s["temperature_step_K"])
                    .terminate_on_fraction_of_liquid_phase(
                        s["termination_liquid_fraction"]
                    )
                )
                self.stats["scheil_calls"] += 1
                save(self.folder / "progress.json", self.stats)
                answer = calc.calculate(timeout_in_minutes=s["timeout_minutes"])
                quantities = {
                    "solid_mole_fraction": SQ.mole_fraction_of_all_solid_phases(),
                    "heat_per_mole": SQ.heat_per_mole(),
                }
                quantities.update(
                    {
                        "phase_" + p: SQ.mole_fraction_of_a_solid_phase(p)
                        for p in answer.get_stable_phases()
                        if p.split("#")[0] != "LIQUID"
                    }
                )
                record["curves"] = {}
                for name, q in quantities.items():
                    t, y = answer.get_values_of(SQ.temperature(), q)
                    record["curves"][name] = dict(
                        temperature_K=list(map(float, t)), values=list(map(float, y))
                    )
            record.update(status="complete", seconds=time.monotonic() - began)
            save(path, record)
        except Exception as exc:
            record.update(
                status="failed",
                error=clean_error(exc),
                seconds=time.monotonic() - began,
            )
            failed = self.folder / ("failed_" + key + ".json")
            record["record"] = str(failed.relative_to(REPO))
            save(failed, record)
            self.used.discard(str(path.relative_to(REPO)))
            self.used.add(record["record"])
            raise
        return record

    def phase_stability(self, composition, basis):
        row = dict(status="complete")
        errors = []
        records = []
        for t, label in zip(self.settings["temperatures_K"], ("250", "660")):
            try:
                record = self.raw("equilibrium", composition, basis, t)
                records.append(record["record"])
                row["L12_" + label + "_mol_percent"] = phase_percent(record, "AL3M_L12")
                row["Al3Zr_" + label + "_mol_percent"] = phase_percent(
                    record, "AL3M_D023"
                )
            except Exception as exc:
                row["L12_" + label + "_mol_percent"] = None
                row["Al3Zr_" + label + "_mol_percent"] = None
                errors.append(clean_error(exc))
        try:
            scan_records = []

            def evaluate(t):
                record = self.raw("equilibrium", composition, basis, t)
                scan_records.append(record["record"])
                return record

            result = find_disappearance(evaluate, **self.settings["solvus"])
            result["source_records"] = sorted(set(scan_records))
            boundary_path = self.folder / (
                "boundary_"
                + hashlib.sha256(
                    json.dumps([composition, basis], sort_keys=True).encode()
                ).hexdigest()[:12]
                + ".json"
            )
            save(boundary_path, result)
            records.append(str(boundary_path.relative_to(REPO)))
            row["Al3Zr_boundary"] = dict(
                status=result["status"],
                upper_temperature_C=result["temperature_K"] - 273.15
                if result["temperature_K"] is not None
                else None,
                precision_K=self.settings["solvus"]["precision_K"],
                liquid_at_upper_boundary=result.get("liquid_at_boundary"),
                disappearance_intervals=[
                    dict(
                        lower_C=i["bracket_K"][0] - 273.15,
                        upper_C=i["bracket_K"][1] - 273.15,
                        temperature_C=i["temperature_K"] - 273.15,
                        liquid_present=i["liquid_at_boundary"],
                    )
                    for i in result.get("disappearance_intervals", [])
                ],
            )
            if result["status"] != "bracketed":
                errors.append("Al3Zr upper boundary unavailable: " + result["status"])
        except Exception as exc:
            row["Al3Zr_boundary"] = dict(status="failed", upper_temperature_C=None)
            errors.append(clean_error(exc))
        row["records"] = sorted(set(records))
        if errors:
            row.update(status="partial", issues=errors)
        return row

    def evaluate(self, kind, composition, basis, reference_m=None):
        row = dict(composition_percent=composition, basis=basis)
        try:
            if kind == "coarsening":
                r = self.raw(
                    "equilibrium",
                    composition,
                    basis,
                    self.settings["temperatures_K"][0],
                )
                row.update(
                    coarsening_metric(r, self.settings["parameters"], reference_m),
                    records=[r["record"]],
                )
            elif kind == "phase_stability":
                row.update(self.phase_stability(composition, basis))
            else:
                r = self.raw("scheil", composition, basis)
                row.update(solidification_metrics(r), records=[r["record"]])
        except Exception as exc:
            row.update(status="failed", error=clean_error(exc))
        return row


def main():
    folder = Path(sys.argv[1]).resolve()
    if os.environ.get("ALLOYGEN_CONTAINED") != "1" or not folder.is_relative_to(ROOT):
        raise RuntimeError("Use the contained project launcher")
    settings = json.loads((HERE / "settings.json").read_text())
    request = json.loads((folder / "request.json").read_text())
    from scientific_tools.tools import KINDS, candidate

    if request["kind"] not in KINDS:
        raise ValueError("Unknown tool")
    for c in request["compositions"]:
        candidate(c)
    # These source functions implement exactly the selected single convention.
    if (
        settings["CSC"]
        != {"solid_thresholds": [0.4, 0.9, 0.98], "interpolation_step": 0.001}
        or settings["FR_solid_cutoff"] != 0.02
        or settings["HCS_divisor"] != 100
    ):
        raise ValueError("Method constants differ from the validated formulas")
    output = dict(
        tool="evaluate_" + request["kind"],
        status="started",
        record=str((folder / "result.json").relative_to(REPO)),
        started_UTC=datetime.now(timezone.utc).isoformat(),
    )
    began = time.monotonic()
    engine = None
    # Serialize calls sharing one output folder so duplicate concurrent requests reuse work.
    with (folder.parents[1] / ".worker.lock").open("a") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            engine = Engine(folder, settings, stack)
            m = None
            reference = settings["reference_mass_percent"]
            if request["kind"] == "coarsening":
                r = engine.raw(
                    "equilibrium",
                    reference,
                    "mass_percent",
                    settings["temperatures_K"][0],
                )
                d = coarsening(r["phases"], settings["parameters"])
                if d["status"] != "computed_with_historical_parameters":
                    raise ValueError("Reference coarsening metric unavailable")
                m = d["m_raw"]
            output["reference"] = engine.evaluate(
                request["kind"], reference, "mass_percent", m
            )
            output["results"] = [
                engine.evaluate(request["kind"], c, "atomic_percent", m)
                for c in request["compositions"]
            ]
            output["status"] = (
                "complete"
                if all(
                    r["status"] == "complete"
                    for r in [output["reference"], *output["results"]]
                )
                else "partial"
            )
            units = {
                "coarsening": dict(J="m/m_reference; lower is better"),
                "phase_stability": dict(
                    phase_amounts="mol% of total system", temperatures="degC"
                ),
                "solidification": dict(
                    phase_amounts="mol% of total system",
                    temperatures="degC",
                    FR_K="K",
                    CSC="dimensionless heat-derived time ratio",
                    HCS="CSC * FR_K / 100; source scaling",
                ),
            }
            output["units"] = dict(
                candidate_compositions="at.%; Al balance",
                reference_composition="wt.%; Al balance",
                **units[request["kind"]],
            )
            save(
                folder / "provenance.json",
                dict(source_records=sorted(engine.used), source_sha256=engine.hashes),
            )
            output.update(
                method_version=settings["method_version"],
                method_fingerprint=engine.fingerprint,
                usage=engine.stats,
                provenance=str((folder / "provenance.json").relative_to(REPO)),
            )
            output["inputs_unchanged"] = (
                all(digest(REPO / p) == h for p, h in engine.hashes.items())
                and digest(engine.db) == settings["database_sha256"]
            )
        except Exception as exc:
            output.update(status="failed", error=clean_error(exc))
            if engine is not None:
                output["usage"] = engine.stats
        finally:
            output["seconds"] = time.monotonic() - began
            save(folder / "result.json", output)
    print(output["tool"], output["status"], output.get("usage", {}), flush=True)


if __name__ == "__main__":
    main()
