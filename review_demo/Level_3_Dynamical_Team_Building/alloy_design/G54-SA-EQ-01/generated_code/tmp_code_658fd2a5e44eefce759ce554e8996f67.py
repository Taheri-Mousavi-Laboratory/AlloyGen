import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

from science_api import tools, REPO, OUTPUT, AlloyDesignTools

# ------------------------------------------------------------
# Interpreted conditions, basis, outputs, methods for this follow-up
# ------------------------------------------------------------
# Goal now: audit the previously saved outputs, determine whether "no feasible candidate"
# was due to true 660 C L12 infeasibility or due to field/record parsing, then run a small,
# evidence-driven follow-up within remaining allowances.
#
# Candidate basis: Ni, Er, Zr in at.% directly; Al balance; Y=Yb=0.
# Hard feasibility for main shortlist: equilibrium L12 at 660 C <= 1.05 mol%.
# J metric: normalized 250 C coarsening J, lower better, valid values only.
# Practical ties: |ΔJ| < 0.0005.
# Phase amounts: normalized mol% of whole system.
# Scheil terminal phases: normalized mol% of original system.
# FR_K, CSC, HCS compared separately; lower preferred.
#
# Numerical methods remain the fixed supplied functions:
# evaluate_coarsening, evaluate_phase_stability, evaluate_solidification.
#
# This script:
# 1) reads saved outputs from the previous successful run,
# 2) robustly inspects raw scientific records if needed,
# 3) selects at most 6 additional evidence-driven compositions,
# 4) recalculates only missing/critical evidence in small batches,
# 5) produces an updated shortlist and evidence table.
# ------------------------------------------------------------

workdir = Path.cwd()
prev_outdir = workdir / "g1_design_outputs"
follow_outdir = workdir / "g1_followup_outputs"
follow_outdir.mkdir(exist_ok=True)

def note(msg):
    print(msg)

def read_csv(name):
    p = prev_outdir / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def comp_id(c):
    return f"N{c['Ni']:.2f}_E{c['Er']:.2f}_Z{c['Zr']:.2f}"

def to_float(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None

def load_record(relpath):
    if relpath is None or (isinstance(relpath, float) and pd.isna(relpath)):
        return None
    p = REPO / str(relpath)
    if not p.exists():
        return None
    return json.loads(p.read_text())

def find_value(obj, keynames):
    found = []
    def rec(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keynames:
                    found.append((path + "." + k, v))
                rec(v, path + "." + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, path + f"[{i}]")
    rec(obj)
    return found

def extract_phase_from_record(rec):
    if rec is None:
        return {}
    keysets = {
        "L12_250C": ["L12_250C", "l12_250C", "L12_250", "l12_250", "eq_L12_250C_molpct"],
        "L12_660C": ["L12_660C", "l12_660C", "L12_660", "l12_660", "eq_L12_660C_molpct"],
        "Al3Zr_250C": ["Al3Zr_250C", "al3zr_250C", "Al3Zr_250", "eq_Al3Zr_250C_molpct"],
        "Al3Zr_660C": ["Al3Zr_660C", "al3zr_660C", "Al3Zr_660", "eq_Al3Zr_660C_molpct"],
        "Al3Zr_boundary": ["Al3Zr_boundary", "al3zr_boundary", "upper_boundary_C", "upper_disappearance_temperature_C"],
        "liquid_at_upper_boundary": ["liquid_at_upper_boundary"],
        "lower_disappearance_intervals": ["lower_disappearance_intervals"],
    }
    out = {}
    for std, ks in keysets.items():
        vals = find_value(rec, ks)
        if vals:
            out[std] = vals[0][1]
            out[std + "_path"] = vals[0][0]
    return out

def extract_solid_from_record(rec):
    if rec is None:
        return {}
    keysets = {
        "ternary_terminal_molpct": ["ternary_terminal_molpct", "Al23Ni6M4_terminal_molpct", "terminal_ternary_molpct"],
        "L12_terminal_molpct": ["L12_terminal_molpct", "terminal_L12_molpct"],
        "Al3Ni_terminal_molpct": ["Al3Ni_terminal_molpct", "terminal_Al3Ni_molpct"],
        "Al3Zr_terminal_molpct": ["Al3Zr_terminal_molpct", "terminal_Al3Zr_molpct"],
        "FR_K": ["FR_K"],
        "CSC": ["CSC"],
        "HCS": ["HCS"],
        "terminal_solid_molpct": ["terminal_solid_molpct", "total_solid_terminal_molpct"],
        "FR_start_phases": ["FR_start_phases"],
        "CSC_intervals": ["CSC_intervals"],
    }
    out = {}
    for std, ks in keysets.items():
        vals = find_value(rec, ks)
        if vals:
            out[std] = vals[0][1]
            out[std + "_path"] = vals[0][0]
    return out

def batch_run(func, compositions):
    out = []
    for i in range(0, len(compositions), 3):
        out.append(func(compositions[i:i+3]))
    return out

def parse_results(raw, family):
    rows = []
    if not isinstance(raw, dict):
        return rows
    results = raw.get("results", [])
    for r in results:
        row = {"family": family}
        comp = r.get("composition", {})
        for k in ["Ni", "Er", "Zr"]:
            if k in comp:
                row[k] = comp[k]
            elif k in r:
                row[k] = r[k]
        row["status"] = r.get("status")
        for k in ["J", "L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C", "Al3Zr_boundary",
                  "liquid_at_upper_boundary", "lower_disappearance_intervals",
                  "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
                  "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
                  "FR_start_phases", "CSC_intervals"]:
            if k in r:
                row[k] = r[k]
        if "records" in r and isinstance(r["records"], dict):
            row["record"] = r["records"].get("summary") or r["records"].get("record") or next(iter(r["records"].values()), None)
        elif "record" in r:
            row["record"] = r["record"]
        rows.append(row)
    return rows

# ------------------------------------------------------------
# Step 1: read previous outputs
# ------------------------------------------------------------
coars = read_csv("coarsening_screening.csv")
phase = read_csv("phase_screening.csv")
solid = read_csv("solidification_screening.csv")
all_prev = read_csv("all_candidate_comparison.csv")
ref = read_csv("reference_summary.csv")

note("Loaded previous outputs.")
note(f"  coarsening rows: {len(coars)}")
note(f"  phase rows: {len(phase)}")
note(f"  solidification rows: {len(solid)}")
note(f"  merged rows: {len(all_prev)}")

# ------------------------------------------------------------
# Step 2: audit whether phase fields were parsed correctly
# ------------------------------------------------------------
audit_rows = []
for _, row in phase.iterrows():
    rec = load_record(row.get("phase_record"))
    extracted = extract_phase_from_record(rec)
    audit_rows.append({
        "composition_id": row["composition_id"],
        "csv_L12_250C": row.get("L12_250C_molpct"),
        "csv_L12_660C": row.get("L12_660C_molpct"),
        "rec_L12_250C": extracted.get("L12_250C"),
        "rec_L12_660C": extracted.get("L12_660C"),
        "rec_Al3Zr_250C": extracted.get("Al3Zr_250C"),
        "rec_Al3Zr_660C": extracted.get("Al3Zr_660C"),
        "rec_Al3Zr_boundary": extracted.get("Al3Zr_boundary"),
        "rec_liquid_at_upper_boundary": extracted.get("liquid_at_upper_boundary"),
        "status": row.get("status"),
        "phase_record": row.get("phase_record")
    })
audit_phase = pd.DataFrame(audit_rows)
audit_phase.to_csv(follow_outdir / "phase_record_audit.csv", index=False)

# Prefer record-derived values when CSV fields are missing
phase_fixed = phase.copy()
for idx, r in phase_fixed.iterrows():
    if pd.isna(r.get("L12_250C_molpct")) or pd.isna(r.get("L12_660C_molpct")):
        rec = load_record(r.get("phase_record"))
        ex = extract_phase_from_record(rec)
        for src, dst in [
            ("L12_250C", "L12_250C_molpct"),
            ("L12_660C", "L12_660C_molpct"),
            ("Al3Zr_250C", "Al3Zr_250C_molpct"),
            ("Al3Zr_660C", "Al3Zr_660C_molpct"),
            ("Al3Zr_boundary", "Al3Zr_boundary_C"),
            ("liquid_at_upper_boundary", "liquid_at_upper_boundary"),
            ("lower_disappearance_intervals", "lower_disappearance_intervals"),
        ]:
            if src in ex:
                phase_fixed.at[idx, dst] = ex[src]

# Recompute feasibility
phase_fixed["phase_feasible_660C"] = phase_fixed["L12_660C_molpct"].apply(lambda x: None if pd.isna(x) else (float(x) <= 1.05))
phase_fixed.to_csv(follow_outdir / "phase_screening_fixed.csv", index=False)

# Audit solidification records similarly for missing fields
audit_solid_rows = []
for _, row in solid.iterrows():
    rec = load_record(row.get("solidification_record"))
    extracted = extract_solid_from_record(rec)
    audit_solid_rows.append({
        "composition_id": row["composition_id"],
        "csv_FR_K": row.get("FR_K"),
        "csv_CSC": row.get("CSC"),
        "csv_HCS": row.get("HCS"),
        "rec_FR_K": extracted.get("FR_K"),
        "rec_CSC": extracted.get("CSC"),
        "rec_HCS": extracted.get("HCS"),
        "rec_FR_start_phases": json.dumps(extracted.get("FR_start_phases")),
        "rec_CSC_intervals": json.dumps(extracted.get("CSC_intervals")),
        "status": row.get("status"),
        "solidification_record": row.get("solidification_record")
    })
audit_solid = pd.DataFrame(audit_solid_rows)
audit_solid.to_csv(follow_outdir / "solid_record_audit.csv", index=False)

solid_fixed = solid.copy()
for idx, r in solid_fixed.iterrows():
    rec = load_record(r.get("solidification_record"))
    ex = extract_solid_from_record(rec)
    for src, dst in [
        ("ternary_terminal_molpct", "ternary_terminal_molpct"),
        ("L12_terminal_molpct", "L12_terminal_molpct"),
        ("Al3Ni_terminal_molpct", "Al3Ni_terminal_molpct"),
        ("Al3Zr_terminal_molpct", "Al3Zr_terminal_molpct"),
        ("FR_K", "FR_K"),
        ("CSC", "CSC"),
        ("HCS", "HCS"),
        ("terminal_solid_molpct", "terminal_solid_molpct"),
        ("FR_start_phases", "FR_start_phases"),
        ("CSC_intervals", "CSC_intervals"),
    ]:
        if (dst not in solid_fixed.columns or pd.isna(r.get(dst))) and src in ex:
            if dst not in solid_fixed.columns:
                solid_fixed[dst] = None
            solid_fixed.at[idx, dst] = ex[src]
solid_fixed.to_csv(follow_outdir / "solidification_screening_fixed.csv", index=False)

# Merge repaired previous evidence
coars_small = coars[["composition_id", "Ni", "Er", "Zr", "status", "J", "coarsening_record"]].copy()
phase_small = phase_fixed[["composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
                           "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
                           "liquid_at_upper_boundary", "phase_record"]].copy()
solid_small = solid_fixed[["composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
                           "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "FR_start_phases",
                           "CSC_intervals", "solidification_record"]].copy()
merged = coars_small.merge(phase_small, on="composition_id", how="left").merge(solid_small, on="composition_id", how="left")
merged.to_csv(follow_outdir / "merged_repaired_previous_evidence.csv", index=False)

# Diagnose current situation
feasible = merged[(merged["J"].notna()) & (merged["phase_feasible_660C"] == True)].copy()
infeasible = merged[(merged["J"].notna()) & (merged["phase_feasible_660C"] == False)].copy()
unknown_phase = merged[(merged["J"].notna()) & (merged["phase_feasible_660C"].isna())].copy()

note(f"After record audit: feasible valid-J candidates = {len(feasible)}; infeasible valid-J candidates = {len(infeasible)}; unknown-phase valid-J candidates = {len(unknown_phase)}.")

# ------------------------------------------------------------
# Step 3: choose small evidence-driven follow-up set
# ------------------------------------------------------------
# Decision logic:
# - If we already have feasible candidates, only fill consequential gaps:
#   * exact preferred branch Ni points if missing downstream evidence
#   * one boundary-bracketing Er/Zr point if feasibility is near a promising low-J branch.
# - If no feasible candidates yet, target phase-bracketing compositions near lowest-J measured points.
#
# We keep within remaining allowances conservatively: <=6 new search compositions,
# <=6 new phase profiles, <=6 new Scheil calculations.
# ------------------------------------------------------------
follow_candidates = []

# Sort promising low-J points
promising = merged[merged["J"].notna()].sort_values("J").copy()

if len(feasible) == 0:
    note("No feasible valid-J candidate yet after record audit; selecting follow-up points to bracket the 660 C L12 feasibility boundary on the lowest-J branches.")
    # choose up to 2 lowest-J measured points, then reduce Er and/or Ni slightly, and one lower-Zr point if branch suggests Al3Zr cost
    # pull exact top measured compositions
    top = promising.head(3)
    seen = set()
    for _, r in top.iterrows():
        ni, er, zr = float(r["Ni"]), float(r["Er"]), float(r["Zr"])
        trials = [
            {"Ni": ni, "Er": max(0.0, round(er - 0.2, 2)), "Zr": zr},
            {"Ni": max(0.0, round(ni - 0.5, 2)), "Er": er, "Zr": zr},
            {"Ni": ni, "Er": er, "Zr": max(0.0, round(zr - 0.2, 2))},
        ]
        for c in trials:
            k = comp_id(c)
            if k not in seen and k not in set(merged["composition_id"]):
                follow_candidates.append(c)
                seen.add(k)
            if len(follow_candidates) >= 6:
                break
        if len(follow_candidates) >= 6:
            break
else:
    note("At least one feasible valid-J candidate exists after record audit; selecting small consequential follow-up only.")
    bestJ = feasible["J"].min()
    tieband = feasible[(feasible["J"] - bestJ).abs() < 0.0005].copy()
    # If a retained Er/Zr branch exists, add one intermediate Ni only if missing and could affect selection
    branch = merged[(merged["Er"].round(2) == 0.70) & (merged["Zr"].round(2) == 0.50)].copy()
    if not ((branch["Ni"].round(2) == 2.00).any() and (branch["Ni"].round(2) == 2.50).any() and (branch["Ni"].round(2) == 1.50).any()):
        for ni in [1.5, 2.0, 2.5]:
            c = {"Ni": ni, "Er": 0.7, "Zr": 0.5}
            if comp_id(c) not in set(merged["composition_id"]):
                follow_candidates.append(c)
    # One Er/Zr midpoint if a feasible/infeasible pair brackets the limit on a promising branch
    # simple heuristic on Ni=2 branch
    branch2 = merged[(merged["Ni"].round(2) == 2.00) & merged["L12_660C_molpct"].notna()].copy()
    if len(branch2) >= 2:
        branch2 = branch2.sort_values(["Er", "Zr"])
        # propose (0.75,0.45) if missing
        c = {"Ni": 2.0, "Er": 0.75, "Zr": 0.45}
        if comp_id(c) not in set(merged["composition_id"]):
            follow_candidates.append(c)

# Deduplicate and clip
fc_unique = []
seen = set()
for c in follow_candidates:
    k = comp_id(c)
    if k not in seen:
        seen.add(k)
        fc_unique.append(c)
follow_candidates = fc_unique[:6]

pd.DataFrame(follow_candidates).to_csv(follow_outdir / "chosen_followup_candidates.csv", index=False)
note(f"Chosen follow-up candidates ({len(follow_candidates)}): {[comp_id(c) for c in follow_candidates]}")

# ------------------------------------------------------------
# Step 4: run follow-up calculations
# ------------------------------------------------------------
new_rows = {"coarsening": [], "phase": [], "solidification": []}

if follow_candidates:
    note("Running follow-up coarsening.")
    for raw in batch_run(tools.evaluate_coarsening, follow_candidates):
        new_rows["coarsening"].extend(parse_results(raw, "coarsening"))

    note("Running follow-up phase stability.")
    for raw in batch_run(tools.evaluate_phase_stability, follow_candidates):
        new_rows["phase"].extend(parse_results(raw, "phase"))

    # Run Scheil on at most 4 most consequential follow-up candidates after phase screen:
    phase_new_df = pd.DataFrame(new_rows["phase"])
    if len(phase_new_df):
        for col in ["L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C", "Al3Zr_boundary"]:
            if col in phase_new_df.columns:
                phase_new_df[col] = pd.to_numeric(phase_new_df[col], errors="coerce")
        phase_new_df["phase_feasible_660C"] = phase_new_df["L12_660C"].apply(lambda x: None if pd.isna(x) else (float(x) <= 1.05))
        # select most consequential: feasible with valid L12, lowest 660C among them, else lowest L12_660C overall
        scheil_targets = phase_new_df.copy()
        scheil_targets["cid"] = scheil_targets.apply(lambda r: comp_id({"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])}), axis=1)
        scheil_targets = scheil_targets.sort_values(["phase_feasible_660C", "L12_660C"], ascending=[False, True], na_position="last")
        scheil_targets = scheil_targets.head(4)
        scheil_comps = [{"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])} for _, r in scheil_targets.iterrows()]
    else:
        scheil_comps = []

    pd.DataFrame(scheil_comps).to_csv(follow_outdir / "chosen_followup_scheil_targets.csv", index=False)
    note(f"Running follow-up solidification on {[comp_id(c) for c in scheil_comps]}")
    for raw in batch_run(tools.evaluate_solidification, scheil_comps):
        new_rows["solidification"].extend(parse_results(raw, "solidification"))

# Save raw follow-up tables
for fam in ["coarsening", "phase", "solidification"]:
    pd.DataFrame(new_rows[fam]).to_csv(follow_outdir / f"followup_{fam}_raw.csv", index=False)

# ------------------------------------------------------------
# Step 5: integrate old and new evidence
# ------------------------------------------------------------
coars_new = pd.DataFrame(new_rows["coarsening"])
if len(coars_new):
    coars_new["composition_id"] = coars_new.apply(lambda r: comp_id({"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])}), axis=1)
    coars_new["J"] = pd.to_numeric(coars_new["J"], errors="coerce")
    coars_new = coars_new[["composition_id", "Ni", "Er", "Zr", "status", "J", "record"]].rename(columns={"record": "coarsening_record"})
else:
    coars_new = pd.DataFrame(columns=["composition_id", "Ni", "Er", "Zr", "status", "J", "coarsening_record"])

phase_new = pd.DataFrame(new_rows["phase"])
if len(phase_new):
    phase_new["composition_id"] = phase_new.apply(lambda r: comp_id({"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])}), axis=1)
    for c in ["L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C", "Al3Zr_boundary"]:
        if c in phase_new.columns:
            phase_new[c] = pd.to_numeric(phase_new[c], errors="coerce")
    phase_new["phase_feasible_660C"] = phase_new["L12_660C"].apply(lambda x: None if pd.isna(x) else (float(x) <= 1.05))
    phase_new = phase_new.rename(columns={
        "L12_250C": "L12_250C_molpct",
        "L12_660C": "L12_660C_molpct",
        "Al3Zr_250C": "Al3Zr_250C_molpct",
        "Al3Zr_660C": "Al3Zr_660C_molpct",
        "Al3Zr_boundary": "Al3Zr_boundary_C",
        "record": "phase_record"
    })
    keepcols = [c for c in ["composition_id","L12_250C_molpct","L12_660C_molpct","phase_feasible_660C",
                            "Al3Zr_250C_molpct","Al3Zr_660C_molpct","Al3Zr_boundary_C",
                            "liquid_at_upper_boundary","lower_disappearance_intervals","phase_record"] if c in phase_new.columns]
    phase_new = phase_new[keepcols]
else:
    phase_new = pd.DataFrame(columns=["composition_id"])

solid_new = pd.DataFrame(new_rows["solidification"])
if len(solid_new):
    solid_new["composition_id"] = solid_new.apply(lambda r: comp_id({"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])}), axis=1)
    for c in ["ternary_terminal_molpct","L12_terminal_molpct","Al3Ni_terminal_molpct","Al3Zr_terminal_molpct","FR_K","CSC","HCS","terminal_solid_molpct"]:
        if c in solid_new.columns:
            solid_new[c] = pd.to_numeric(solid_new[c], errors="coerce")
    solid_new = solid_new.rename(columns={"record": "solidification_record"})
    keepcols = [c for c in ["composition_id","ternary_terminal_molpct","L12_terminal_molpct","Al3Ni_terminal_molpct",
                            "Al3Zr_terminal_molpct","FR_K","CSC","HCS","terminal_solid_molpct",
                            "FR_start_phases","CSC_intervals","solidification_record"] if c in solid_new.columns]
    solid_new = solid_new[keepcols]
else:
    solid_new = pd.DataFrame(columns=["composition_id"])

# combine by replacing/adding rows from new tables
def combine(old, new, key="composition_id"):
    if len(new) == 0:
        return old.copy()
    old2 = old.copy()
    old2 = old2[~old2[key].isin(new[key])]
    return pd.concat([old2, new], ignore_index=True, sort=False)

coars_all = combine(coars_small, coars_new)
phase_all = combine(phase_small, phase_new)
solid_all = combine(solid_small, solid_new)

updated = coars_all.merge(phase_all, on="composition_id", how="left").merge(solid_all, on="composition_id", how="left")
updated["qualified_all_three"] = updated["J"].notna() & updated["phase_feasible_660C"].fillna(False).astype(bool) & updated["HCS"].notna()
updated.to_csv(follow_outdir / "updated_all_candidate_evidence.csv", index=False)

# Ranking
feasible_all = updated[(updated["J"].notna()) & (updated["phase_feasible_660C"] == True)].copy().sort_values("J")
if len(feasible_all):
    bestJ = float(feasible_all.iloc[0]["J"])
    updated["J_tie_to_best_feasible"] = updated["J"].apply(lambda x: None if pd.isna(x) else (abs(float(x) - bestJ) < 0.0005))
    updated["J_disadvantage_vs_best_feasible"] = updated["J"].apply(lambda x: None if pd.isna(x) else (float(x) - bestJ))
    tieband = updated[(updated["phase_feasible_660C"] == True) & (updated["J_tie_to_best_feasible"] == True)].copy()
    tieband = tieband.sort_values(["L12_250C_molpct", "Al3Ni_terminal_molpct", "HCS", "CSC", "FR_K"],
                                  ascending=[False, True, True, True, True], na_position="last")
    preferred = tieband.iloc[0].to_dict() if len(tieband) else feasible_all.iloc[0].to_dict()
else:
    bestJ = None
    updated["J_tie_to_best_feasible"] = None
    updated["J_disadvantage_vs_best_feasible"] = None
    preferred = None

balanced = None
if preferred is not None:
    qual = updated[updated["qualified_all_three"] == True].copy()
    qual = qual[qual["composition_id"] != preferred["composition_id"]]
    pref = pd.Series(preferred)
    cand_rows = []
    for _, r in qual.iterrows():
        improvements = 0
        for a, better in [("HCS", "low"), ("CSC", "low"), ("FR_K", "low"), ("Al3Ni_terminal_molpct", "low")]:
            if pd.notna(r.get(a)) and pd.notna(pref.get(a)):
                if better == "low" and float(r[a]) < float(pref[a]):
                    improvements += 1
        if pd.notna(r.get("L12_250C_molpct")) and pd.notna(pref.get("L12_250C_molpct")) and float(r["L12_250C_molpct"]) > float(pref["L12_250C_molpct"]):
            improvements += 1
        dj = None if bestJ is None or pd.isna(r.get("J")) else float(r["J"]) - bestJ
        cand_rows.append((improvements, 1e9 if dj is None else abs(dj), r.to_dict()))
    cand_rows.sort(key=lambda x: (-x[0], x[1]))
    if cand_rows and cand_rows[0][0] >= 2:
        balanced = cand_rows[0][2]

# Fresh verification if preferred exists and is new or not previously fresh-verified here
verification_result = {}
if preferred is not None:
    note("Running fresh verification for preferred candidate and reference.")
    fresh = AlloyDesignTools(OUTPUT / "fresh_verification_g1_followup")
    comp = [{"Ni": float(preferred["Ni"]), "Er": float(preferred["Er"]), "Zr": float(preferred["Zr"])}]
    verification_result["coarsening"] = fresh.evaluate_coarsening(comp)
    verification_result["phase"] = fresh.evaluate_phase_stability(comp)
    verification_result["solidification"] = fresh.evaluate_solidification(comp)
    (follow_outdir / "fresh_verification_followup.json").write_text(json.dumps(verification_result, indent=2, default=str))

# Compact matched-pair table from updated evidence
pair_defs = [
    ("Er at fixed Ni=2.0, Zr=0.5", "N2.00_E0.70_Z0.50", "N2.00_E0.90_Z0.50"),
    ("Ni at fixed Er=0.7, Zr=0.5", "N1.50_E0.70_Z0.50", "N2.50_E0.70_Z0.50"),
    ("coupled Er/Zr at fixed Ni=2.0", "N2.00_E0.80_Z0.40", "N2.00_E0.60_Z0.60"),
]
pair_rows = []
for desc, a, b in pair_defs:
    ra = updated[updated["composition_id"] == a]
    rb = updated[updated["composition_id"] == b]
    if len(ra) and len(rb):
        ra = ra.iloc[0]
        rb = rb.iloc[0]
        pair_rows.append({
            "comparison": desc,
            "case_A": a,
            "case_B": b,
            "service_L12_change_molpct": None if pd.isna(ra.get("L12_250C_molpct")) or pd.isna(rb.get("L12_250C_molpct")) else float(rb["L12_250C_molpct"]) - float(ra["L12_250C_molpct"]),
            "highT_L12_change_molpct": None if pd.isna(ra.get("L12_660C_molpct")) or pd.isna(rb.get("L12_660C_molpct")) else float(rb["L12_660C_molpct"]) - float(ra["L12_660C_molpct"]),
            "A_margin_to_limit": None if pd.isna(ra.get("L12_660C_molpct")) else 1.05 - float(ra["L12_660C_molpct"]),
            "B_margin_to_limit": None if pd.isna(rb.get("L12_660C_molpct")) else 1.05 - float(rb["L12_660C_molpct"]),
            "competing_phase_cost": f"Al3Zr@250C {ra.get('Al3Zr_250C_molpct')} -> {rb.get('Al3Zr_250C_molpct')}; boundary {ra.get('Al3Zr_boundary_C')} -> {rb.get('Al3Zr_boundary_C')}"
        })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(follow_outdir / "updated_matched_pair_table.csv", index=False)

# Final comparison table with evidence IDs
comparison_cols = [
    "composition_id", "Ni", "Er", "Zr", "status", "J", "J_tie_to_best_feasible", "J_disadvantage_vs_best_feasible",
    "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
    "Al3Zr_boundary_C", "liquid_at_upper_boundary", "ternary_terminal_molpct", "L12_terminal_molpct",
    "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS",
    "coarsening_record", "phase_record", "solidification_record"
]
for c in comparison_cols:
    if c not in updated.columns:
        updated[c] = None
updated[comparison_cols].sort_values(["J", "L12_250C_molpct"], na_position="last").to_csv(
    follow_outdir / "updated_comparison_table_with_evidence_ids.csv", index=False
)

# Narrative summary
lines = []
lines.append("Follow-up audit and refinement summary")
lines.append("")
lines.append("Audit outcome")
lines.append(f"- Previously saved outputs were re-read and scientific records were audited directly from returned evidence paths in {prev_outdir}.")
lines.append(f"- After repair from record content, feasible valid-J candidates found: {len(feasible_all)}.")
if len(feasible_all):
    lines.append(f"- Best measured feasible J = {bestJ:.6f}; practical ties use |ΔJ| < 0.0005.")
else:
    lines.append("- No measured candidate yet satisfies both valid J and the 660 C L12 <= 1.05 mol% feasibility condition.")
lines.append("")
lines.append("Follow-up candidate selection logic")
if len(follow_candidates):
    for c in follow_candidates:
        lines.append(f"- {comp_id(c)} selected because it tests a measured low-J branch against the blocking 660 C phase limit or fills a consequential retained-branch gap.")
else:
    lines.append("- No additional follow-up calculations were needed once the repaired evidence was sufficient.")
lines.append("")
if preferred is not None:
    lines.append("Recommended shortlist")
    lines.append(f"- Coarsening-first preferred candidate: {preferred['composition_id']}")
    lines.append(f"  J={preferred.get('J')}, L12@250C={preferred.get('L12_250C_molpct')} mol%, L12@660C={preferred.get('L12_660C_molpct')} mol%, Al3Zr@250C={preferred.get('Al3Zr_250C_molpct')} mol%")
    lines.append(f"  Scheil terminal: ternary={preferred.get('ternary_terminal_molpct')} mol%, L12={preferred.get('L12_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')} mol%, Al3Zr={preferred.get('Al3Zr_terminal_molpct')} mol%")
    lines.append(f"  FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        lines.append(f"- Balanced alternative: {balanced['composition_id']} with explicit J tradeoff ΔJ={balanced.get('J_disadvantage_vs_best_feasible')}")
        lines.append(f"  J={balanced.get('J')}, L12@250C={balanced.get('L12_250C_molpct')} mol%, L12@660C={balanced.get('L12_660C_molpct')} mol%")
        lines.append(f"  terminal ternary={balanced.get('ternary_terminal_molpct')} mol%, Al3Ni={balanced.get('Al3Ni_terminal_molpct')} mol%, FR_K={balanced.get('FR_K')}, CSC={balanced.get('CSC')}, HCS={balanced.get('HCS')}")
else:
    lines.append("No shortlist recommendation is yet defensible from measured evidence because the hard 660 C L12 feasibility condition remains unmet or unqualified on the low-J branches.")
lines.append("")
lines.append("Limits")
lines.append("- This remains an in-silico CALPHAD-based design screen, not an experimental strength or printability prediction.")
lines.append("- Equilibrium phase amounts and terminal Scheil phase amounts are distinct quantities and were not interchanged.")
lines.append("- If no feasible preferred candidate emerges, the next concrete work is a narrower phase-bracketing study around the lowest-J branch identified in the updated comparison table.")
(follow_outdir / "followup_summary.txt").write_text("\n".join(lines))

# File check
files = [
    follow_outdir / "phase_record_audit.csv",
    follow_outdir / "solid_record_audit.csv",
    follow_outdir / "phase_screening_fixed.csv",
    follow_outdir / "solidification_screening_fixed.csv",
    follow_outdir / "merged_repaired_previous_evidence.csv",
    follow_outdir / "chosen_followup_candidates.csv",
    follow_outdir / "updated_all_candidate_evidence.csv",
    follow_outdir / "updated_matched_pair_table.csv",
    follow_outdir / "updated_comparison_table_with_evidence_ids.csv",
    follow_outdir / "followup_summary.txt",
]
if (follow_outdir / "fresh_verification_followup.json").exists():
    files.append(follow_outdir / "fresh_verification_followup.json")
pd.DataFrame([{"file": str(f), "exists": f.exists(), "size": f.stat().st_size if f.exists() else None} for f in files]).to_csv(
    follow_outdir / "created_files_check.csv", index=False
)

# Console summary
note("")
note("Follow-up complete. Created files:")
for f in files:
    note(f"  - {f.name}: {'OK' if f.exists() else 'MISSING'}")
note("")
if preferred is not None:
    note(f"Preferred candidate: {preferred['composition_id']}")
    note(f"  J={preferred.get('J')}, L12@250C={preferred.get('L12_250C_molpct')} mol%, L12@660C={preferred.get('L12_660C_molpct')} mol%")
    note(f"  terminal ternary={preferred.get('ternary_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')} mol%, FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        note(f"Balanced alternative: {balanced['composition_id']} (ΔJ={balanced.get('J_disadvantage_vs_best_feasible')})")
else:
    note("No defensible preferred candidate yet from measured evidence.")
note(f"Outputs saved in: {follow_outdir}")