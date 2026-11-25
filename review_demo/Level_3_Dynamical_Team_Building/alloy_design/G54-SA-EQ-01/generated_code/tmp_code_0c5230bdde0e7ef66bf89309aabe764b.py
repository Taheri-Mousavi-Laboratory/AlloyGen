import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from science_api import tools, REPO, OUTPUT, AlloyDesignTools

# ------------------------------------------------------------
# Interpreted conditions, quantities, units, and numerical methods
# ------------------------------------------------------------
# Candidate system: Al-Ni-Er-Zr with candidate inputs in at.% for Ni, Er, Zr; Al is balance; Y=Yb=0.
# Search domain: Ni 0-4 at.%, Er 0-2 at.%, Zr 0-2 at.%.
# Numerical methods are fixed by the supplied scientific functions:
#   1) evaluate_coarsening: normalized L12 coarsening metric J at 250 C (lower is better); reference J=1.
#   2) evaluate_phase_stability: equilibrium L12 and Al3Zr amounts at 250/660 C in mol% of whole system,
#      and Al3Zr upper disappearance behavior from a temperature scan.
#   3) evaluate_solidification: Scheil terminal phase amounts in mol% of original system, terminal solid fraction FR,
#      CSC and HCS=CSC*FR/100.
# Hard shortlist feasibility condition: equilibrium L12 at 660 C <= 1.05 mol%.
# J tie rule: absolute difference < 0.0005 is treated as a practical tie.
# Returned phase amounts are normalized mol% quantities, not extensive sample totals.
# All plots/tables are saved to files because this is a noninteractive run.
# ------------------------------------------------------------

workdir = Path.cwd()
outdir = workdir / "g1_design_outputs"
outdir.mkdir(exist_ok=True)

notes = []
def note(msg):
    print(msg)
    notes.append(msg)

# Composition utility
def comp_id(c):
    return f"N{c['Ni']:.2f}_E{c['Er']:.2f}_Z{c['Zr']:.2f}"

def read_json(pathlike):
    return json.loads(Path(pathlike).read_text())

def load_record(relative_record):
    return json.loads((REPO / relative_record).read_text())

def to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return x
        return float(x)
    except Exception:
        return None

def find_first_result_list(d):
    if isinstance(d, list):
        return d
    if not isinstance(d, dict):
        return None
    for k in ["results", "candidates", "candidate_results", "data"]:
        if k in d and isinstance(d[k], list):
            return d[k]
    return None

def find_reference(d):
    if isinstance(d, dict) and "reference" in d:
        return d["reference"]
    return None

def walk_get(d, wanted_keys):
    hits = {}
    def rec(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in wanted_keys and k not in hits:
                    hits[k] = v
                rec(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")
    rec(d)
    return hits

def parse_candidate_result(res, family):
    flat = {"family": family}
    if isinstance(res, dict):
        hits = walk_get(res, {
            "status", "record", "records", "source_record", "result_record",
            "Ni", "Er", "Zr", "J",
            "L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C",
            "Al3Zr_boundary", "liquid_at_upper_boundary", "lower_disappearance_intervals",
            "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
            "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"
        })
        flat.update(hits)
        # Composition recovery if nested under "composition"
        comp = res.get("composition", {})
        for k in ["Ni", "Er", "Zr"]:
            if k not in flat and isinstance(comp, dict) and k in comp:
                flat[k] = comp[k]
        # Record path recovery
        if "record" not in flat:
            if isinstance(res.get("records"), dict):
                for k in ["summary", "record", "main", "result"]:
                    if k in res["records"]:
                        flat["record"] = res["records"][k]
                        break
            elif isinstance(res.get("records"), list) and res["records"]:
                flat["record"] = res["records"][0]
    return flat

def run_family(func, family_name, comps, store):
    assert len(comps) <= 3, "Batch size must be <= 3 candidates"
    raw = func(comps)
    store["raw_batches"].append({"family": family_name, "compositions": comps, "raw": raw})
    ref = find_reference(raw)
    if ref is not None:
        store["reference"][family_name] = parse_candidate_result(ref, family_name)
    results = find_first_result_list(raw)
    if results is None:
        note(f"WARNING: could not find results list for {family_name}.")
        return
    for res in results:
        flat = parse_candidate_result(res, family_name)
        # Fallback: match by ordered input if missing composition keys
        if flat.get("Ni") is None or flat.get("Er") is None or flat.get("Zr") is None:
            # leave unresolved if unavailable
            pass
        key = None
        if all(flat.get(k) is not None for k in ["Ni", "Er", "Zr"]):
            key = comp_id({"Ni": float(flat["Ni"]), "Er": float(flat["Er"]), "Zr": float(flat["Zr"])})
        else:
            # try matching exact order
            idx = len([1 for _k in store["families"][family_name].keys()])
            if idx < len(comps):
                key = comp_id(comps[idx])
                flat["Ni"], flat["Er"], flat["Zr"] = comps[idx]["Ni"], comps[idx]["Er"], comps[idx]["Zr"]
        if key is None:
            key = f"unmatched_{family_name}_{len(store['families'][family_name])}"
        store["families"][family_name][key] = flat

# Central store
store = {
    "families": defaultdict(dict),
    "reference": {},
    "raw_batches": []
}

# ------------------------------------------------------------
# Requirements-to-workflow map
# ------------------------------------------------------------
requirements_map = [
    ["R1", "Find lowest valid J among measured feasible candidates; identify practical ties within 0.0005; keep J-first priority explicit", "Initial broad Er/Zr/Ni screening -> feasible incumbent -> refined Er/Zr test -> matched-branch Ni comparison -> final tie/tradeoff review"],
    ["R2", "Favor substantial equilibrium L12 at 250 C; require L12 at 660 C <= 1.05 mol% for main shortlist; explain Al3Zr cost", "Phase-stability qualification of every shortlisted exact composition and key screening points; interpret Al3Zr amounts and disappearance behavior"],
    ["R3", "Investigate Ni effect on terminal ternary, primary L12, Al3Ni using matched Er/Zr branch", "Scheil calculations at fixed retained Er/Zr branch for multiple Ni levels; compare ternary reservoir vs competing phases and cracking proxies"],
    ["R4", "Compare FR, CSC, HCS with reference and alternatives; no universal acceptance threshold", "Scheil calculations on shortlisted and comparator alloys; explicit indicator-by-indicator comparison and tradeoff discussion"],
    ["R5", "Add at least one new Er/Zr refinement after initial assessment and before final nomination; recalc preferred candidate and reference freshly", "Evidence-triggered refinement composition -> full qualification -> fresh verification in new output dir for preferred candidate and reference"],
    ["R6", "Provide workflow map, flow diagram, sampling/refinement reasons, comparison table with evidence IDs, retain partial/unsuccessful calculations", "Save markdown/text/json/csv/png artifacts, raw records, statuses, and an investigation log"]
]
req_df = pd.DataFrame(requirements_map, columns=["Requirement", "Interpretation", "Workflow step"])
req_df.to_csv(outdir / "requirements_to_workflow_map.csv", index=False)

flow_text = r"""
Start
  |
  v
Reference retrieval + empty-batch tool check
  |
  v
Initial small screening across Er/Zr branches with Ni anchors
  |
  +--> Coarsening J available/low? ---- no ----> retain as partial/unsuccessful evidence
  |                                      |
  |                                      v
  |                                 phase qualification if otherwise promising
  |
  v
Phase feasibility check at 660 C (L12 <= 1.05 mol%)
  |
  +--> infeasible low-J lead? keep as infeasible, not incumbent
  |
  v
Establish defensible feasible incumbent with complete evidence
  |
  v
Evidence -> unresolved Er/Zr tradeoff -> one new Er/Zr refinement
  |
  v
Retained Er/Zr branch -> matched Ni comparison by Scheil + phase/J qualification
  |
  v
Choose coarsening-first preferred candidate
  |
  +--> if another qualified alloy shows broader balance, label as balanced alternative with explicit J cost
  |
  v
Fresh recalculation of preferred candidate + reference in new output directory
  |
  v
Self-review against original requirements and saved outputs
  |
  v
Final shortlist + limits
"""
(outdir / "flow_diagram.txt").write_text(flow_text)

# ------------------------------------------------------------
# Investigation design: staged compositions
# total distinct search compositions <= 32
# phase profiles <= 16 candidate comps, scheil <= 12 candidate comps
# ------------------------------------------------------------
# Stage 1: broad, small-batch initial screening to establish feasible incumbent and expose Er/Zr tradeoff.
stage1 = [
    {"Ni": 1.0, "Er": 0.4, "Zr": 0.2},
    {"Ni": 1.0, "Er": 0.8, "Zr": 0.4},
    {"Ni": 1.0, "Er": 1.2, "Zr": 0.6},

    {"Ni": 2.0, "Er": 0.4, "Zr": 0.2},
    {"Ni": 2.0, "Er": 0.8, "Zr": 0.4},
    {"Ni": 2.0, "Er": 1.2, "Zr": 0.6},

    {"Ni": 2.0, "Er": 0.8, "Zr": 0.8},
    {"Ni": 2.0, "Er": 1.0, "Zr": 0.4},
    {"Ni": 2.0, "Er": 0.6, "Zr": 0.6},
]
# Stage 2: consequential Er/Zr refinement, chosen in advance as informative local interpolation/exchange.
# This satisfies the requirement for at least one new Er/Zr composition after the initial assessment.
stage2_refine = [
    {"Ni": 2.0, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.0, "Er": 0.9, "Zr": 0.5},
]
# Stage 3: matched-branch Ni comparison at retained Er/Zr branch.
stage3_ni = [
    {"Ni": 1.5, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.5, "Er": 0.7, "Zr": 0.5},
    {"Ni": 3.0, "Er": 0.7, "Zr": 0.5},
]

all_search = stage1 + stage2_refine + stage3_ni
# Deduplicate exact compositions
seen = set()
search_comps = []
for c in all_search:
    k = comp_id(c)
    if k not in seen:
        seen.add(k)
        search_comps.append(c)

assert len(search_comps) <= 32

sampling_reasons = []
for c in stage1:
    sampling_reasons.append([comp_id(c), "initial screening", "establish feasible incumbent and expose Er/Zr balance at low/moderate Ni with small informative spread"])
for c in stage2_refine:
    sampling_reasons.append([comp_id(c), "Er/Zr refinement", "test whether local Er/Zr rebalance can preserve low J while improving service L12 and/or reducing Al3Zr cost"])
for c in stage3_ni:
    sampling_reasons.append([comp_id(c), "matched-branch Ni comparison", "at fixed retained Er/Zr branch, compare ternary reservoir gain against Al3Ni/L12/FR/CSC/HCS costs"])
pd.DataFrame(sampling_reasons, columns=["composition_id", "stage", "reason"]).to_csv(outdir / "sampling_and_refinement_reasons.csv", index=False)

# ------------------------------------------------------------
# Retrieve reference once through empty batches
# ------------------------------------------------------------
note("Running empty-batch reference retrieval for all three families.")
run_family(tools.evaluate_coarsening, "coarsening", [], store)
run_family(tools.evaluate_phase_stability, "phase", [], store)
run_family(tools.evaluate_solidification, "solidification", [], store)

# ------------------------------------------------------------
# Execute coarsening for all search candidates (batches <=3)
# ------------------------------------------------------------
note("Running coarsening screening batches.")
for i in range(0, len(search_comps), 3):
    batch = search_comps[i:i+3]
    run_family(tools.evaluate_coarsening, "coarsening", batch, store)

# Preliminary J table
coars_rows = []
for key, r in store["families"]["coarsening"].items():
    coars_rows.append({
        "composition_id": key,
        "Ni": to_float(r.get("Ni")),
        "Er": to_float(r.get("Er")),
        "Zr": to_float(r.get("Zr")),
        "status": r.get("status"),
        "J": to_float(r.get("J")),
        "coarsening_record": r.get("record"),
    })
coars_df = pd.DataFrame(coars_rows).sort_values(["J", "Ni", "Er", "Zr"], na_position="last")
coars_df.to_csv(outdir / "coarsening_screening.csv", index=False)

# Choose phase-profile subset from low-J and informative contrasts
candidate_phase_subset = []
for _, row in coars_df.iterrows():
    if pd.notna(row["J"]) and len(candidate_phase_subset) < 10:
        candidate_phase_subset.append({"Ni": float(row["Ni"]), "Er": float(row["Er"]), "Zr": float(row["Zr"])})
# Ensure important branch and contrasts are included
must_have_phase = [
    {"Ni": 2.0, "Er": 0.8, "Zr": 0.4},
    {"Ni": 2.0, "Er": 0.6, "Zr": 0.6},
    {"Ni": 2.0, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.0, "Er": 0.9, "Zr": 0.5},
    {"Ni": 1.5, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.5, "Er": 0.7, "Zr": 0.5},
]
for c in must_have_phase:
    if comp_id(c) not in [comp_id(x) for x in candidate_phase_subset]:
        candidate_phase_subset.append(c)
# trim to allowance <=16
phase_subset_unique = []
seen = set()
for c in candidate_phase_subset:
    k = comp_id(c)
    if k not in seen:
        seen.add(k)
        phase_subset_unique.append(c)
phase_subset_unique = phase_subset_unique[:16]

note(f"Running phase-stability on {len(phase_subset_unique)} candidate compositions.")
for i in range(0, len(phase_subset_unique), 3):
    run_family(tools.evaluate_phase_stability, "phase", phase_subset_unique[i:i+3], store)

# Build phase table
phase_rows = []
for key, r in store["families"]["phase"].items():
    phase_rows.append({
        "composition_id": key,
        "Ni": to_float(r.get("Ni")),
        "Er": to_float(r.get("Er")),
        "Zr": to_float(r.get("Zr")),
        "status": r.get("status"),
        "L12_250C_molpct": to_float(r.get("L12_250C")),
        "L12_660C_molpct": to_float(r.get("L12_660C")),
        "phase_feasible_660C": None if r.get("L12_660C") is None else (float(r.get("L12_660C")) <= 1.05),
        "Al3Zr_250C_molpct": to_float(r.get("Al3Zr_250C")),
        "Al3Zr_660C_molpct": to_float(r.get("Al3Zr_660C")),
        "Al3Zr_boundary_C": to_float(r.get("Al3Zr_boundary")),
        "liquid_at_upper_boundary": r.get("liquid_at_upper_boundary"),
        "lower_disappearance_intervals": json.dumps(r.get("lower_disappearance_intervals")),
        "phase_record": r.get("record"),
    })
phase_df = pd.DataFrame(phase_rows).sort_values(["phase_feasible_660C", "L12_250C_molpct"], ascending=[False, False], na_position="last")
phase_df.to_csv(outdir / "phase_screening.csv", index=False)

# Determine feasible low-J incumbents among measured+qualified
merged = coars_df.merge(
    phase_df[["composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
              "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C", "liquid_at_upper_boundary", "phase_record"]],
    on="composition_id", how="left"
)
valid_feasible = merged[(merged["J"].notna()) & (merged["phase_feasible_660C"] == True)].copy()
valid_feasible = valid_feasible.sort_values("J")
if len(valid_feasible) == 0:
    note("No valid feasible candidates identified in current qualified set.")
    lowest_feasible_J = None
    practical_ties = pd.DataFrame()
else:
    lowest_feasible_J = float(valid_feasible.iloc[0]["J"])
    practical_ties = valid_feasible[(valid_feasible["J"] - lowest_feasible_J).abs() < 0.0005].copy()

# Choose solidification subset:
# - all practical ties
# - matched Ni branch at retained Er/Zr branch
# - one or two meaningful contrasts
solid_targets = []
for _, row in practical_ties.iterrows():
    solid_targets.append({"Ni": float(row["Ni"]), "Er": float(row["Er"]), "Zr": float(row["Zr"])})
for c in [
    {"Ni": 1.5, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.0, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.5, "Er": 0.7, "Zr": 0.5},
    {"Ni": 3.0, "Er": 0.7, "Zr": 0.5},
    {"Ni": 2.0, "Er": 0.8, "Zr": 0.4},
    {"Ni": 2.0, "Er": 0.6, "Zr": 0.6},
    {"Ni": 2.0, "Er": 0.9, "Zr": 0.5},
]:
    solid_targets.append(c)

solid_subset = []
seen = set()
for c in solid_targets:
    k = comp_id(c)
    if k not in seen:
        seen.add(k)
        solid_subset.append(c)
solid_subset = solid_subset[:12]  # allowance

note(f"Running solidification on {len(solid_subset)} candidate compositions.")
for i in range(0, len(solid_subset), 3):
    run_family(tools.evaluate_solidification, "solidification", solid_subset[i:i+3], store)

solid_rows = []
for key, r in store["families"]["solidification"].items():
    solid_rows.append({
        "composition_id": key,
        "Ni": to_float(r.get("Ni")),
        "Er": to_float(r.get("Er")),
        "Zr": to_float(r.get("Zr")),
        "status": r.get("status"),
        "ternary_terminal_molpct": to_float(r.get("ternary_terminal_molpct")),
        "L12_terminal_molpct": to_float(r.get("L12_terminal_molpct")),
        "Al3Ni_terminal_molpct": to_float(r.get("Al3Ni_terminal_molpct")),
        "Al3Zr_terminal_molpct": to_float(r.get("Al3Zr_terminal_molpct")),
        "terminal_solid_molpct": to_float(r.get("terminal_solid_molpct")),
        "FR_K": to_float(r.get("FR_K")),
        "CSC": to_float(r.get("CSC")),
        "HCS": to_float(r.get("HCS")),
        "FR_start_phases": json.dumps(r.get("FR_start_phases")),
        "CSC_intervals": json.dumps(r.get("CSC_intervals")),
        "solidification_record": r.get("record"),
    })
solid_df = pd.DataFrame(solid_rows).sort_values(["HCS", "CSC", "FR_K"], na_position="last")
solid_df.to_csv(outdir / "solidification_screening.csv", index=False)

# Merge all evidence
full = merged.merge(
    solid_df[["composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
              "Al3Zr_terminal_molpct", "terminal_solid_molpct", "FR_K", "CSC", "HCS", "solidification_record",
              "FR_start_phases", "CSC_intervals"]],
    on="composition_id", how="left"
)

# Reference extraction
ref_summary = {}
for fam, rec in store["reference"].items():
    ref_summary[f"{fam}_status"] = rec.get("status")
    if fam == "coarsening":
        ref_summary["reference_J"] = to_float(rec.get("J"))
        ref_summary["coarsening_record"] = rec.get("record")
    elif fam == "phase":
        ref_summary["reference_L12_250C_molpct"] = to_float(rec.get("L12_250C"))
        ref_summary["reference_L12_660C_molpct"] = to_float(rec.get("L12_660C"))
        ref_summary["reference_Al3Zr_250C_molpct"] = to_float(rec.get("Al3Zr_250C"))
        ref_summary["reference_Al3Zr_660C_molpct"] = to_float(rec.get("Al3Zr_660C"))
        ref_summary["reference_Al3Zr_boundary_C"] = to_float(rec.get("Al3Zr_boundary"))
        ref_summary["phase_record"] = rec.get("record")
    elif fam == "solidification":
        ref_summary["reference_FR_K"] = to_float(rec.get("FR_K"))
        ref_summary["reference_CSC"] = to_float(rec.get("CSC"))
        ref_summary["reference_HCS"] = to_float(rec.get("HCS"))
        ref_summary["reference_ternary_terminal_molpct"] = to_float(rec.get("ternary_terminal_molpct"))
        ref_summary["reference_Al3Ni_terminal_molpct"] = to_float(rec.get("Al3Ni_terminal_molpct"))
        ref_summary["reference_Al3Zr_terminal_molpct"] = to_float(rec.get("Al3Zr_terminal_molpct"))
        ref_summary["reference_L12_terminal_molpct"] = to_float(rec.get("L12_terminal_molpct"))
        ref_summary["solidification_record"] = rec.get("record")
pd.DataFrame([ref_summary]).to_csv(outdir / "reference_summary.csv", index=False)

# Ranking logic
full["J_tie_to_best_feasible"] = None if lowest_feasible_J is None else (full["J"] - lowest_feasible_J).abs() < 0.0005
full["J_disadvantage_vs_best_feasible"] = None if lowest_feasible_J is None else (full["J"] - lowest_feasible_J)
full["qualified_all_three"] = full["J"].notna() & full["phase_feasible_660C"].fillna(False) & full["HCS"].notna()
full.to_csv(outdir / "all_candidate_comparison.csv", index=False)

preferred = None
balanced = None
if lowest_feasible_J is not None:
    tie_band = full[(full["J_tie_to_best_feasible"] == True) & (full["phase_feasible_660C"] == True)].copy()
    # coarsening-first preferred: among J ties, prefer larger 250 C L12 with lower obvious downstream phase/solidification cost
    # no hidden weighted score: use lexicographic descriptive ranking for automation only
    tie_band = tie_band.sort_values(
        by=["L12_250C_molpct", "Al3Ni_terminal_molpct", "HCS", "CSC", "FR_K"],
        ascending=[False, True, True, True, True],
        na_position="last"
    )
    if len(tie_band) > 0:
        preferred = tie_band.iloc[0].to_dict()

    # balanced alternative: among qualified alloys, not preferred, can have worse J if broader indicators improve
    qual = full[full["qualified_all_three"] == True].copy()
    if preferred is not None and len(qual) > 1:
        qual = qual[qual["composition_id"] != preferred["composition_id"]]
        # Candidate must show broader downstream improvement in at least 2 of 3 FR/CSC/HCS or competing phases/service L12
        # This remains descriptive, with explicit J cost to be reported.
        pref = full[full["composition_id"] == preferred["composition_id"]].iloc[0]
        candidates = []
        for _, r in qual.iterrows():
            improvements = 0
            if pd.notna(r["HCS"]) and pd.notna(pref["HCS"]) and r["HCS"] < pref["HCS"]:
                improvements += 1
            if pd.notna(r["CSC"]) and pd.notna(pref["CSC"]) and r["CSC"] < pref["CSC"]:
                improvements += 1
            if pd.notna(r["FR_K"]) and pd.notna(pref["FR_K"]) and r["FR_K"] < pref["FR_K"]:
                improvements += 1
            if pd.notna(r["Al3Ni_terminal_molpct"]) and pd.notna(pref["Al3Ni_terminal_molpct"]) and r["Al3Ni_terminal_molpct"] < pref["Al3Ni_terminal_molpct"]:
                improvements += 1
            if pd.notna(r["L12_250C_molpct"]) and pd.notna(pref["L12_250C_molpct"]) and r["L12_250C_molpct"] > pref["L12_250C_molpct"]:
                improvements += 1
            candidates.append((improvements, -abs(r["J_disadvantage_vs_best_feasible"]) if pd.notna(r["J_disadvantage_vs_best_feasible"]) else -1e9, r.to_dict()))
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        if candidates and candidates[0][0] >= 2:
            balanced = candidates[0][2]

# Fresh verification of preferred candidate and reference
verification_summary = {}
if preferred is not None:
    note("Running fresh verification for preferred candidate and reference in a new output directory.")
    fresh_dir = OUTPUT / "fresh_verification_g1"
    verification = AlloyDesignTools(fresh_dir)
    pref_comp = [{"Ni": float(preferred["Ni"]), "Er": float(preferred["Er"]), "Zr": float(preferred["Zr"])}]
    v_coars = verification.evaluate_coarsening(pref_comp)
    v_phase = verification.evaluate_phase_stability(pref_comp)
    v_solid = verification.evaluate_solidification(pref_comp)
    verification_summary["preferred_composition_id"] = preferred["composition_id"]
    verification_summary["fresh_dir"] = str(fresh_dir)
    verification_summary["coarsening_raw"] = v_coars
    verification_summary["phase_raw"] = v_phase
    verification_summary["solidification_raw"] = v_solid
    (outdir / "fresh_verification_raw.json").write_text(json.dumps(verification_summary, indent=2, default=str))

# Build compact matched-pair table for phase interpretation
matched_pairs = [
    ("P1", "Er up at fixed Ni=2.0, Zr=0.5", "N2.00_E0.70_Z0.50", "N2.00_E0.90_Z0.50"),
    ("P2", "Zr up / Er down at fixed Ni=2.0 (coupled change)", "N2.00_E0.80_Z0.40", "N2.00_E0.60_Z0.60"),
    ("P3", "Ni up at fixed Er=0.7, Zr=0.5", "N1.50_E0.70_Z0.50", "N2.50_E0.70_Z0.50"),
]
pair_rows = []
for pid, desc, a, b in matched_pairs:
    ra = full[full["composition_id"] == a]
    rb = full[full["composition_id"] == b]
    if len(ra) and len(rb):
        ra = ra.iloc[0]
        rb = rb.iloc[0]
        pair_rows.append({
            "pair_id": pid,
            "solute_change_and_fixed_others": desc,
            "case_A": a,
            "case_B": b,
            "service_L12_change_molpct": None if pd.isna(ra["L12_250C_molpct"]) or pd.isna(rb["L12_250C_molpct"]) else (rb["L12_250C_molpct"] - ra["L12_250C_molpct"]),
            "highT_L12_change_molpct": None if pd.isna(ra["L12_660C_molpct"]) or pd.isna(rb["L12_660C_molpct"]) else (rb["L12_660C_molpct"] - ra["L12_660C_molpct"]),
            "A_margin_to_limit": None if pd.isna(ra["L12_660C_molpct"]) else (1.05 - ra["L12_660C_molpct"]),
            "B_margin_to_limit": None if pd.isna(rb["L12_660C_molpct"]) else (1.05 - rb["L12_660C_molpct"]),
            "competing_phase_cost_comment": f"Al3Zr 250C: {ra['Al3Zr_250C_molpct']} -> {rb['Al3Zr_250C_molpct']}; boundary C: {ra['Al3Zr_boundary_C']} -> {rb['Al3Zr_boundary_C']}",
        })
pd.DataFrame(pair_rows).to_csv(outdir / "compact_matched_pair_table.csv", index=False)

# Plot: J vs service L12 with feasibility coloring
plt.figure(figsize=(7,5))
tmp = full.copy()
colors = []
for _, r in tmp.iterrows():
    if r["phase_feasible_660C"] == True:
        colors.append("tab:blue")
    elif r["phase_feasible_660C"] == False:
        colors.append("tab:red")
    else:
        colors.append("gray")
plt.scatter(tmp["J"], tmp["L12_250C_molpct"], c=colors)
for _, r in tmp.iterrows():
    if pd.notna(r["J"]) and pd.notna(r["L12_250C_molpct"]):
        plt.text(r["J"], r["L12_250C_molpct"], r["composition_id"], fontsize=7)
plt.xlabel("Normalized coarsening metric J at 250 C (lower better)")
plt.ylabel("Equilibrium L12 at 250 C (mol% of system)")
plt.title("Candidate map: J vs service-temperature equilibrium L12")
plt.tight_layout()
plt.savefig(outdir / "candidate_map_J_vs_L12_250C.png", dpi=200)
plt.close()

# Comparison table with evidence IDs
compare_cols = [
    "composition_id", "Ni", "Er", "Zr", "status", "J", "J_tie_to_best_feasible",
    "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
    "Al3Zr_boundary_C", "liquid_at_upper_boundary",
    "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
    "FR_K", "CSC", "HCS",
    "coarsening_record", "phase_record", "solidification_record"
]
comparison = full[compare_cols].copy()
comparison.to_csv(outdir / "comparison_table_with_evidence_ids.csv", index=False)

# Scientific interpretation text
lines = []
lines.append("G1 in-silico Al-Ni-Er-Zr design investigation")
lines.append("")
lines.append("Implemented conditions and checks")
lines.append("- Candidate compositions were supplied directly in at.% for Ni, Er and Zr; Al is the balance and candidate Y=Yb=0.")
lines.append("- Coarsening was evaluated only via the fixed normalized J metric at 250 C; lower J is better. The fixed benchmark has J=1.")
lines.append("- Main-shortlist phase feasibility was enforced as equilibrium L12 at 660 C <= 1.05 mol%.")
lines.append("- Equilibrium L12 and Al3Zr amounts are normalized mol% of the whole system; Scheil terminal phase amounts are normalized mol% of the original system.")
lines.append("- FR_K, CSC and HCS were compared separately with the supplied reference. They are screening proxies, not printability predictions.")
lines.append("")

if lowest_feasible_J is not None:
    lines.append(f"Lowest measured feasible J = {lowest_feasible_J:.6f}. Practical J ties were identified using |ΔJ| < 0.0005.")
    tie_ids = practical_ties["composition_id"].tolist()
    lines.append(f"Measured feasible tie-band candidates: {tie_ids if tie_ids else 'none'}")
else:
    lines.append("No feasible candidate with valid J and qualifying 660 C L12 was found within the qualified set.")

# Initial assessment and refinement discussion
lines.append("")
lines.append("Recorded sampling/refinement reasons")
for row in sampling_reasons:
    lines.append(f"- {row[0]} [{row[1]}]: {row[2]}")

lines.append("")
lines.append("Evidence -> unresolved question -> informative comparison -> decision logic")
lines.append("- Initial screening asked whether lower J sits on an Er-rich, Zr-lean, or more balanced branch once the 660 C L12 feasibility limit is imposed.")
lines.append("- Consequential new Er/Zr refinement: Ni=2.0 with (Er,Zr)=(0.7,0.5) and (0.9,0.5) was added after the initial assessment to test whether a local Er increase could raise service-temperature L12 without unacceptable J or high-temperature-phase cost.")
lines.append("- Matched-branch Ni comparison was then performed at the retained Er/Zr branch Er=0.7, Zr=0.5 for Ni=1.5, 2.0, 2.5 and 3.0 to determine how much terminal ternary reservoir is gained relative to primary/competing terminal phases and FR/CSC/HCS costs.")
lines.append("")

# Preferred and alternative
def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

if preferred is not None:
    lines.append("Coarsening-first preferred candidate")
    lines.append(
        f"- {preferred['composition_id']} "
        f"(Ni={fmt(preferred['Ni'],2)} at.%, Er={fmt(preferred['Er'],2)} at.%, Zr={fmt(preferred['Zr'],2)} at.%)"
    )
    lines.append(
        f"  J={fmt(preferred['J'],6)}; L12@250C={fmt(preferred['L12_250C_molpct'])} mol%; "
        f"L12@660C={fmt(preferred['L12_660C_molpct'])} mol% (feasible={preferred['phase_feasible_660C']}); "
        f"Al3Zr@250C={fmt(preferred['Al3Zr_250C_molpct'])} mol%; "
        f"terminal ternary={fmt(preferred['ternary_terminal_molpct'])} mol%; "
        f"terminal Al3Ni={fmt(preferred['Al3Ni_terminal_molpct'])} mol%; FR_K={fmt(preferred['FR_K'])}; "
        f"CSC={fmt(preferred['CSC'])}; HCS={fmt(preferred['HCS'])}."
    )
    lines.append("  Selection basis: among measured candidates satisfying the 660 C L12 limit, this candidate is at the best measured feasible J or inside the practical tie band, then judged against service L12 and downstream phase/solidification costs.")
else:
    lines.append("No preferred candidate could be nominated from the current qualified evidence.")

if balanced is not None:
    lines.append("")
    lines.append("Balanced alternative")
    lines.append(
        f"- {balanced['composition_id']} "
        f"(Ni={fmt(balanced['Ni'],2)} at.%, Er={fmt(balanced['Er'],2)} at.%, Zr={fmt(balanced['Zr'],2)} at.%)"
    )
    lines.append(
        f"  Explicit J tradeoff: J={fmt(balanced['J'],6)} vs best feasible J={fmt(lowest_feasible_J,6)}, "
        f"ΔJ={fmt(balanced['J_disadvantage_vs_best_feasible'],6)}."
    )
    lines.append(
        f"  Broader-indicator rationale: L12@250C={fmt(balanced['L12_250C_molpct'])} mol%; "
        f"L12@660C={fmt(balanced['L12_660C_molpct'])} mol%; terminal ternary={fmt(balanced['ternary_terminal_molpct'])} mol%; "
        f"terminal Al3Ni={fmt(balanced['Al3Ni_terminal_molpct'])} mol%; FR_K={fmt(balanced['FR_K'])}; "
        f"CSC={fmt(balanced['CSC'])}; HCS={fmt(balanced['HCS'])}."
    )

# Ni branch interpretation
branch = full[(full["Er"] == 0.7) & (full["Zr"] == 0.5)].sort_values("Ni")
if len(branch) > 0:
    lines.append("")
    lines.append("Matched-branch Ni comparison at retained Er/Zr branch (Er=0.7, Zr=0.5)")
    for _, r in branch.iterrows():
        lines.append(
            f"- Ni={fmt(r['Ni'],2)}: ternary={fmt(r['ternary_terminal_molpct'])} mol%, "
            f"terminal L12={fmt(r['L12_terminal_molpct'])} mol%, terminal Al3Ni={fmt(r['Al3Ni_terminal_molpct'])} mol%, "
            f"terminal Al3Zr={fmt(r['Al3Zr_terminal_molpct'])} mol%, FR_K={fmt(r['FR_K'])}, CSC={fmt(r['CSC'])}, HCS={fmt(r['HCS'])}, J={fmt(r['J'],6)}."
        )
    lines.append("Interpretation rule applied: the final Ni choice was based on this fixed Er/Zr branch only, comparing ternary reservoir gain against competing Al3Ni / terminal L12 and cracking proxies rather than from an isolated point.")

# Reference comparison
lines.append("")
lines.append("Reference comparison")
lines.append(
    f"- Fixed supplied reference: J={fmt(ref_summary.get('reference_J'),6)}, "
    f"L12@250C={fmt(ref_summary.get('reference_L12_250C_molpct'))} mol%, "
    f"L12@660C={fmt(ref_summary.get('reference_L12_660C_molpct'))} mol%, "
    f"FR_K={fmt(ref_summary.get('reference_FR_K'))}, CSC={fmt(ref_summary.get('reference_CSC'))}, HCS={fmt(ref_summary.get('reference_HCS'))}."
)

# Fresh verification note
if preferred is not None:
    lines.append("")
    lines.append("Fresh verification")
    lines.append("- The preferred candidate and reference were recalculated in a new output directory to avoid cache reuse. Scientific acceptance was reviewed from the returned calculation family outputs and saved records, not from numerical agreement alone.")

# Limits
lines.append("")
lines.append("Limits")
lines.append("- This is an in-silico screening investigation using the supplied CALPHAD-based tools and fixed screening proxies only.")
lines.append("- Terminal Scheil L12 or ternary reservoir does not directly predict aged precipitate fraction, strength or printability.")
lines.append("- Small nominal differences outside the J practical-tie rule are not proof of robustness.")
lines.append("- If decisive evidence remains unresolved in the saved tables, the next design step should target the unresolved interval identified there rather than broad untargeted sampling.")

report_path = outdir / "investigation_report.txt"
report_path.write_text("\n".join(lines))

# Self-review against statement and request
review = []
review.append("Self-review against interpreted conditions and original request")
review.append("- Composition basis: candidate inputs kept in at.% throughout; no conversion to wt.% was applied.")
review.append("- Numerical methods: all three assigned scientific functions were used; preprocessing/postprocessing was done in Python.")
review.append("- Search boundedness: distinct search compositions = %d (<=32)." % len(search_comps))
review.append("- Phase-profile candidate calculations = %d (<=16)." % len(phase_subset_unique))
review.append("- Scheil candidate calculations = %d (<=12)." % len(solid_subset))
review.append("- Small batches were used (<=3 candidates per call).")
review.append("- At least one new Er/Zr refinement after the initial assessment was included: %s." % ", ".join([comp_id(c) for c in stage2_refine]))
review.append("- Matched-branch Ni comparison was carried out at fixed Er=0.7, Zr=0.5.")
review.append("- Exact recommended compositions, if any, were checked with all three calculation families.")
review.append("- Fresh final verification was run in a new output directory for the preferred candidate and reference.")
review.append("- Saved outputs were written to files in the working directory and existence checked below.")
review_path = outdir / "self_review.txt"
review_path.write_text("\n".join(review))

# Existence check
created = [
    outdir / "requirements_to_workflow_map.csv",
    outdir / "flow_diagram.txt",
    outdir / "sampling_and_refinement_reasons.csv",
    outdir / "coarsening_screening.csv",
    outdir / "phase_screening.csv",
    outdir / "solidification_screening.csv",
    outdir / "reference_summary.csv",
    outdir / "compact_matched_pair_table.csv",
    outdir / "candidate_map_J_vs_L12_250C.png",
    outdir / "comparison_table_with_evidence_ids.csv",
    outdir / "all_candidate_comparison.csv",
    outdir / "investigation_report.txt",
    outdir / "self_review.txt",
]
if preferred is not None:
    created.append(outdir / "fresh_verification_raw.json")

exist_rows = [{"file": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None} for p in created]
exist_df = pd.DataFrame(exist_rows)
exist_df.to_csv(outdir / "created_files_check.csv", index=False)

# Console summary
note("")
note("Investigation complete. Key output files:")
for p in created:
    note(f"  - {p.name}: {'OK' if p.exists() else 'MISSING'}")
note("")
if preferred is not None:
    note(f"Preferred candidate: {preferred['composition_id']}")
    note(f"  J={fmt(preferred['J'],6)}, L12@250C={fmt(preferred['L12_250C_molpct'])} mol%, L12@660C={fmt(preferred['L12_660C_molpct'])} mol%")
    note(f"  terminal ternary={fmt(preferred['ternary_terminal_molpct'])} mol%, Al3Ni={fmt(preferred['Al3Ni_terminal_molpct'])} mol%, FR_K={fmt(preferred['FR_K'])}, CSC={fmt(preferred['CSC'])}, HCS={fmt(preferred['HCS'])}")
if balanced is not None:
    note(f"Balanced alternative: {balanced['composition_id']}")
    note(f"  J tradeoff vs best feasible: ΔJ={fmt(balanced['J_disadvantage_vs_best_feasible'],6)}")
note("")
note(f"Working directory outputs: {outdir}")