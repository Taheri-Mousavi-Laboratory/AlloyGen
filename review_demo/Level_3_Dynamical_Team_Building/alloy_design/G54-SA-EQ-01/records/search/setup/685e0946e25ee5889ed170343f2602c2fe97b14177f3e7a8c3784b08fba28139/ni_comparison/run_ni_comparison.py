"""Selective Ni comparison with the accepted TC-Python 2025b setup. No LLM calls.

Run through the existing run_contained.py launcher; specify a new run ID.
All source data and installed environments are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]
SOURCE = EXPERIMENT / "author_reference/sources"
SDK = EXPERIMENT / ".runtime/tc_python_2025b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    assert path.resolve().is_relative_to(REPO)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def clean_error(exc):
    message = f"{type(exc).__name__}: {exc}"
    for key in ["LSHOST", "LSFORCEHOST"]:
        if os.environ.get(key):
            message = message.replace(os.environ[key], "[license host redacted]")
    return message


def configure(run_id):
    if os.environ.get("ALLOYGEN_CONTAINED") != "1":
        raise RuntimeError("Use the repository contained launcher")
    if not os.environ.get("TC25B_HOME") or not Path(os.environ["TC25B_HOME"]).is_dir():
        raise RuntimeError("Configured Thermo-Calc 2025b installation is unavailable")
    runtime = REPO / ".tc" / ("m" + hashlib.sha256(run_id.encode()).hexdigest()[:7])
    runtime.mkdir(parents=True, exist_ok=False)
    for key, folder in {"TMPDIR": "t", "TMP": "t", "TEMP": "t", "XDG_CACHE_HOME": "xc",
                        "XDG_CONFIG_HOME": "xf", "XDG_DATA_HOME": "xd", "XDG_STATE_HOME": "xs"}.items():
        target = runtime / folder
        target.mkdir(exist_ok=True)
        os.environ[key] = str(target)
    (runtime / "j").mkdir()
    os.environ["JAVA_TOOL_OPTIONS"] = " ".join([
        f"-Djava.io.tmpdir={runtime / 't'}", f"-Duser.home={runtime / 'j'}",
        "-XX:-UsePerfData", f"-XX:ErrorFile={runtime / 'hs_err_pid%p.log'}",
        f"-XX:HeapDumpPath={runtime}",
    ])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(SDK))
    os.chdir(runtime)
    return runtime


def coarsening(phases, parameters):
    matrix = [p for p in phases if p["name"].split("#")[0] == "FCC_A1" and p["mole_fraction"] > 1e-12]
    precip = [p for p in phases if p["name"].split("#")[0] == "AL3M_L12" and p["mole_fraction"] > 1e-12]
    if len(matrix) != 1 or len(precip) != 1:
        return {"status": "undefined", "reason": "Need one populated FCC matrix and one populated L12 composition set"}
    a, p = matrix[0]["composition_mole_fraction"], precip[0]["composition_mole_fraction"]
    solutes = ["Ni", "Er", "Zr", "Y", "Yb"]
    terms = {e: (p[e] - a[e]) ** 2 / parameters["Diffusivity"][e] for e in solutes}
    resistance = sum(terms.values())
    lattice = parameters["latticep"]
    a_matrix = sum(a[e] * lattice[e] for e in solutes) + (1 - sum(a[e] for e in solutes)) * lattice["Al"]
    a_precip = sum(p[e] * lattice[e] for e in solutes) + (1 - sum(p[e] for e in solutes)) * lattice["Al"]
    epsilon = 100 * abs(1 - a_precip / a_matrix)
    if resistance <= 0 or not all(math.isfinite(v) for v in [resistance, epsilon, a_matrix, a_precip]):
        return {"status": "undefined", "reason": "Nonpositive resistivity or nonfinite input"}
    return {"status": "computed_with_historical_parameters", "R_code": resistance,
            "resistivity_terms": terms, "a_matrix_angstrom": a_matrix,
            "a_L12_angstrom": a_precip, "misfit_percent": epsilon,
            "m_raw": epsilon / resistance, "m_historical_scaled": epsilon / (resistance * 1e-23),
            "L12_mole_fraction": precip[0]["mole_fraction"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        parser.error("Use letters, digits and underscores for run ID")
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    parameters = json.loads((SOURCE / "local_descriptor_parameters.json").read_text())
    database = EXPERIMENT / "author_reference/private_assets/Al_jan05.TDB"
    if sha(database) != protocol["database_sha256"]:
        raise RuntimeError("Database differs from the inspected source")
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_ni_comparison_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "protocol.json").write_bytes(protocol_path.read_bytes())
    save(out / "parameters.json", parameters)
    report = {"run_id": args.run_id, "started_UTC": datetime.now(timezone.utc).isoformat(),
              "status": "prepared", "python": sys.version, "protocol_sha256": sha(protocol_path),
              "script_sha256": sha(Path(__file__)), "database_sha256": sha(database),
              "paid_model_calls": 0, "evaluations": []}
    save(out / "run.json", report)
    try:
        runtime = configure(args.run_id)
        import tc_python
        from tc_python import TCPython, LoggingPolicy, ThermodynamicQuantity as TQ
        from tc_python import ScheilQuantity as SQ, CompositionUnit, ScheilOptions
        version = importlib.metadata.version("tc-python")
        if version != protocol["tc_python_version"]:
            raise RuntimeError(f"Expected 2025b API; loaded {version}")
        report.update(tc_python_version=version, api_path=tc_python.__file__,
                      thermocalc_home=os.environ["TC25B_HOME"], runtime=str(runtime.relative_to(REPO)), status="running")
        save(out / "run.json", report)
        elements = ["Al", "Ni", "Er", "Zr", "Y", "Yb"]
        with TCPython(logging_policy=LoggingPolicy.NONE) as session:
            session.set_cache_folder(str(runtime / "c"), precision_for_floats=15)
            system = session.select_user_database_and_elements(str(database), elements).get_system()
            report["available_phases"] = system.get_phases_in_system()
            for case in protocol["cases"]:
                for temperature in case["equilibrium_temperatures_K"]:
                    label = f"{case['id']}_eq_{temperature:g}K"
                    began = time.monotonic()
                    record = {"id": label, "kind": "equilibrium", "case": case["id"], "temperature_K": temperature,
                              "input_basis": case["basis"], "input_solute_fractions": case["composition"]}
                    print("Starting", label, flush=True)
                    try:
                        calc = system.with_single_equilibrium_calculation().set_condition("T", temperature).set_condition("P", protocol["pressure_Pa"])
                        factory = TQ.mass_fraction_of_a_component if case["basis"] == "mass_fraction" else TQ.mole_fraction_of_a_component
                        for element, value in case["composition"].items():
                            calc.set_condition(factory(element), value)
                        result = calc.calculate()
                        phases = []
                        for name in result.get_stable_phases():
                            phase = {"name": name, "mole_fraction": float(result.get_value_of(TQ.mole_fraction_of_a_phase(name))),
                                     "composition_mole_fraction": {e: float(result.get_value_of(TQ.composition_of_phase_as_mole_fraction(name, e))) for e in elements}}
                            phases.append(phase)
                        record.update(status="complete", phases=phases)
                        if temperature == protocol["coarsening_temperature_K"]:
                            record["coarsening"] = coarsening(phases, parameters)
                    except Exception as exc:
                        record.update(status="failed", error=clean_error(exc))
                    record["seconds"] = time.monotonic() - began
                    save(out / (label + ".json"), record)
                    report["evaluations"].append(record)
                    save(out / "run.json", report)
                    print(label, record["status"], flush=True)
                    if record["status"] == "failed":
                        raise RuntimeError(f"Reference calculation failed: {label}; stop before further solver work")
            # Collect raw histories; do not silently substitute mole fractions for volume fractions.
            for case in protocol["cases"]:
                if not case["scheil"]:
                    continue
                label = case["id"] + "_scheil"
                print("Starting", label, flush=True)
                began = time.monotonic()
                record = {"id": label, "kind": "scheil", "case": case["id"], "input_basis": case["basis"],
                          "input_solute_fractions": case["composition"], "curves": {}, "quantity_errors": {}}
                try:
                    sp = protocol["scheil"]
                    unit = CompositionUnit.MASS_FRACTION if case["basis"] == "mass_fraction" else CompositionUnit.MOLE_FRACTION
                    calc = system.with_scheil_calculation().set_composition_unit(unit).set_start_temperature(sp["start_temperature_K"])
                    for element, value in case["composition"].items():
                        calc.set_composition(element, value)
                    calc.with_options(ScheilOptions().set_temperature_step(sp["temperature_step_K"]).terminate_on_fraction_of_liquid_phase(sp["termination_liquid_fraction"]))
                    result = calc.calculate(timeout_in_minutes=sp["timeout_minutes_per_calculation"])
                    factories = {"solid_mole_fraction": SQ.mole_fraction_of_all_solid_phases,
                                 "heat_per_mole": SQ.heat_per_mole, "latent_heat_per_mole": SQ.latent_heat_per_mole}
                    for key, factory in factories.items():
                        try:
                            t, y = result.get_values_of(SQ.temperature(), factory())
                            record["curves"][key] = {"temperature_K": list(map(float, t)), "values": list(map(float, y))}
                        except Exception as exc:
                            record["quantity_errors"][key] = clean_error(exc)
                    record["stable_phases"] = result.get_stable_phases()
                    for phase in record["stable_phases"]:
                        if phase.split("#")[0] == "LIQUID":
                            continue
                        try:
                            t, y = result.get_values_of(SQ.temperature(), SQ.mole_fraction_of_a_solid_phase(phase))
                            record["curves"]["phase_" + phase] = {"temperature_K": list(map(float, t)), "values": list(map(float, y))}
                        except Exception as exc:
                            record["quantity_errors"]["phase_" + phase] = clean_error(exc)
                    record["status"] = "complete"
                except Exception as exc:
                    record.update(status="failed", error=clean_error(exc))
                record["seconds"] = time.monotonic() - began
                save(out / (label + ".json"), record)
                report["evaluations"].append({k: v for k, v in record.items() if k != "curves"})
                save(out / "run.json", report)
                print(label, record["status"], flush=True)
                if record["status"] == "failed":
                    raise RuntimeError(f"Reference calculation failed: {label}; stop before further solver work")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=clean_error(exc))
        print(report["error"], flush=True)
    finally:
        report["finished_UTC"] = datetime.now(timezone.utc).isoformat()
        report["database_unchanged"] = sha(database) == protocol["database_sha256"]
        save(out / "run.json", report)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
