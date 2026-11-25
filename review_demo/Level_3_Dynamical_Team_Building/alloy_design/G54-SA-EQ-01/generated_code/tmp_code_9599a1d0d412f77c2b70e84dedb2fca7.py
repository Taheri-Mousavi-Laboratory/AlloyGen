import json
from pathlib import Path
import pandas as pd

from science_api import tools, REPO, OUTPUT, AlloyDesignTools

# ------------------------------------------------------------
# Debugged follow-up audit with defensive phase/solid parsing
# ------------------------------------------------------------
# Interpreted conditions, composition basis, outputs, units, methods:
# - Candidates: Ni, Er, Zr directly in at.% ; Al balance; candidate Y=Yb=0.
# - Hard shortlist feasibility: equilibrium L12 at 660 C <= 1.05 mol%.
# - Coarsening: normalized J at 250 C; lower is better.
# - Phase and terminal Scheil amounts: normalized mol% quantities.
# - FR_K, CSC, HCS compared separately; lower preferred.
# - Numerical methods fixed by evaluate_coarsening, evaluate_phase_stability, evaluate_solidification.
# - Preserve partial/unsuccessful calculations and save outputs to files.
# ------------------------------------------------------------

workdir = Path.cwd()
prev_outdir = workdir / "g1_design_outputs"
follow_outdir = workdir / "g1_followup_outputs"
follow_outdir.mkdir(exist_ok=True)

def note(msg):
    print(msg)

def comp_id(c):
    return f"N{c['Ni']:.2f}_E{c['Er']:.2f}_Z{c['Zr']:.2f}"

def safe_float(x):
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def load_csv(name):
    p = prev_outdir / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def load_record(relpath):
    if relpath is None or (isinstance(relpath, float) and pd.isna(relpath)):
        return None
    p = REPO / str(relpath)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def recursive_find_items(obj, keys):
    hits = []
    def rec(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys:
                    hits.append((path + "." + k, v))
                rec(v, path + "." + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, path + f"[{i}]")
    rec(obj)
    return hits

def extract_composition_from_record(rec):
    if rec is None:
        return {}
    out = {}
    for key in ["Ni", "Er", "Zr"]:
        hits = recursive_find_items(rec, [key])
        vals = [safe_float(v) for _, v in hits]
        vals = [v for v in vals if v is not None]
        if vals:
            out[key] = vals[0]
    return out

def extract_phase_record_values(rec):
    if rec is None:
        return {}
    keymap = {
        "L12_250C_molpct": ["L12_250C", "l12_250C", "L12_250", "eq_L12_250C_molpct"],
        "L12_660C_molpct": ["L12_660C", "l12_660C", "L12_660", "eq_L12_660C_molpct"],
        "Al3Zr_250C_molpct": ["Al3Zr_250C", "al3zr_250C", "eq_Al3Zr_250C_molpct"],
        "Al3Zr_660C_molpct": ["Al3Zr_660C", "al3zr_660C", "eq_Al3Zr_660C_molpct"],
        "Al3Zr_boundary_C": ["Al3Zr_boundary", "al3zr_boundary", "upper_boundary_C", "upper_disappearance_temperature_C"],
        "liquid_at_upper_boundary": ["liquid_at_upper_boundary"],
        "lower_disappearance_intervals": ["lower_disappearance_intervals"],
    }
    out = {}
    for std, keys in keymap.items():
        hits = recursive_find_items(rec, keys)
        if hits:
            out[std] = hits[0][1]
    return out

def extract_solid_record_values(rec):
    if rec is None:
        return {}
    keymap = {
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
    for std, keys in keymap.items():
        hits = recursive_find_items(rec, keys)
        if hits:
            out[std] = hits[0][1]
    return out

def parse_results(raw, batch):
    rows = []
    if not isinstance(raw, dict):
        return rows
    results = raw.get("results", [])
    for i, r in enumerate(results):
        row = {}
        comp = r.get("composition", {}) if isinstance(r.get("composition", {}), dict) else {}
        for k in ["Ni", "Er", "Zr"]:
            if k in r:
                row[k] = r[k]
            elif k in comp:
                row[k] = comp[k]
            elif i < len(batch):
                row[k] = batch[i][k]
        row["status"] = r.get("status")
        for k in [
            "J", "L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C", "Al3Zr_boundary",
            "liquid_at_upper_boundary", "lower_disappearance_intervals",
            "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
            "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
            "FR_start_phases", "CSC_intervals"
        ]:
            if k in r:
                row[k] = r[k]
        rec = None
        if "record" in r:
            rec = r["record"]
        elif isinstance(r.get("records"), dict):
            rec = r["records"].get("summary") or r["records"].get("record") or next(iter(r["records"].values()), None)
        row["record"] = rec
        rows.append(row)
    return rows

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def combine(old, new, key="composition_id"):
    if len(new) == 0:
        return old.copy()
    old2 = old[~old[key].isin(new[key])].copy()
    return pd.concat([old2, new], ignore_index=True, sort=False)

# --------------------------
# Load previous outputs
# --------------------------
coars = load_csv("coarsening_screening.csv")
phase = load_csv("phase_screening.csv")
solid = load_csv("solidification_screening.csv")
all_prev = load_csv("all_candidate_comparison.csv")

note("Loaded previous outputs.")
note(f"  coarsening rows: {len(coars)}")
note(f"  phase rows: {len(phase)}")
note(f"  solidification rows: {len(solid)}")
note(f"  merged rows: {len(all_prev)}")

# --------------------------
# Repair previous saved evidence from records
# --------------------------
phase_fixed = phase.copy()
phase_fixed = ensure_cols(phase_fixed, [
    "L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
    "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"
])
phase_audit = []
for idx, r in phase_fixed.iterrows():
    rec = load_record(r.get("phase_record"))
    ex = extract_phase_record_values(rec)
    comp_ex = extract_composition_from_record(rec)
    phase_audit.append({
        "composition_id": r["composition_id"],
        "phase_record": r.get("phase_record"),
        "record_composition": json.dumps(comp_ex),
        "record_phase_keys": ",".join(sorted(ex.keys()))
    })
    for c in ["L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
              "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"]:
        if pd.isna(r.get(c)) and c in ex:
            phase_fixed.at[idx, c] = ex[c]
phase_fixed["phase_feasible_660C"] = phase_fixed["L12_660C_molpct"].apply(lambda x: None if pd.isna(x) else (float(x) <= 1.05))
pd.DataFrame(phase_audit).to_csv(follow_outdir / "phase_record_audit.csv", index=False)
phase_fixed.to_csv(follow_outdir / "phase_screening_fixed.csv", index=False)

solid_fixed = solid.copy()
solid_fixed = ensure_cols(solid_fixed, [
    "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
    "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"
])
solid_audit = []
for idx, r in solid_fixed.iterrows():
    rec = load_record(r.get("solidification_record"))
    ex = extract_solid_record_values(rec)
    solid_audit.append({
        "composition_id": r["composition_id"],
        "solidification_record": r.get("solidification_record"),
        "record_solid_keys": ",".join(sorted(ex.keys()))
    })
    for c in ["ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
              "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"]:
        if pd.isna(r.get(c)) and c in ex:
            solid_fixed.at[idx, c] = ex[c]
pd.DataFrame(solid_audit).to_csv(follow_outdir / "solid_record_audit.csv", index=False)
solid_fixed.to_csv(follow_outdir / "solidification_screening_fixed.csv", index=False)

coars_small = coars[["composition_id", "Ni", "Er", "Zr", "status", "J", "coarsening_record"]].copy()
phase_small = phase_fixed[["composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
                           "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
                           "liquid_at_upper_boundary", "phase_record"]].copy()
solid_small = solid_fixed[["composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
                           "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "FR_start_phases",
                           "CSC_intervals", "solidification_record"]].copy()

merged_prev = coars_small.merge(phase_small, on="composition_id", how="left").merge(solid_small, on="composition_id", how="left")
merged_prev.to_csv(follow_outdir / "merged_repaired_previous_evidence.csv", index=False)

feasible_prev = merged_prev[(merged_prev["J"].notna()) & (merged_prev["phase_feasible_660C"] == True)].copy()
infeasible_prev = merged_prev[(merged_prev["J"].notna()) & (merged_prev["phase_feasible_660C"] == False)].copy()
unknown_prev = merged_prev[(merged_prev["J"].notna()) & (merged_prev["phase_feasible_660C"].isna())].copy()

note(f"After record audit: feasible valid-J candidates = {len(feasible_prev)}; infeasible valid-J candidates = {len(infeasible_prev)}; unknown-phase valid-J candidates = {len(unknown_prev)}.")

# --------------------------
# Choose evidence-driven follow-up set
# --------------------------
promising = merged_prev[merged_prev["J"].notna()].sort_values("J").copy()
existing_ids = set(merged_prev["composition_id"].tolist())
follow_candidates = []

if len(feasible_prev) == 0:
    note("No feasible valid-J candidate yet after record audit; selecting follow-up points to bracket the blocking 660 C phase limit on low-J branches.")
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
            cid = comp_id(c)
            if cid not in seen and cid not in existing_ids:
                follow_candidates.append(c)
                seen.add(cid)
            if len(follow_candidates) >= 6:
                break
        if len(follow_candidates) >= 6:
            break
else:
    note("Feasible valid-J candidates already exist; adding only consequential follow-up.")
    for c in [
        {"Ni": 1.5, "Er": 0.7, "Zr": 0.5},
        {"Ni": 2.0, "Er": 0.7, "Zr": 0.5},
        {"Ni": 2.5, "Er": 0.7, "Zr": 0.5},
        {"Ni": 2.0, "Er": 0.75, "Zr": 0.45},
    ]:
        cid = comp_id(c)
        if cid not in existing_ids and cid not in [comp_id(x) for x in follow_candidates]:
            follow_candidates.append(c)

pd.DataFrame(follow_candidates).to_csv(follow_outdir / "chosen_followup_candidates.csv", index=False)
note(f"Chosen follow-up candidates ({len(follow_candidates)}): {[comp_id(c) for c in follow_candidates]}")

# --------------------------
# Run follow-up calculations
# --------------------------
coars_rows, phase_rows, solid_rows = [], [], []
coars_batches, phase_batches, solid_batches = [], [], []

if follow_candidates:
    note("Running follow-up coarsening.")
    for i in range(0, len(follow_candidates), 3):
        batch = follow_candidates[i:i+3]
        raw = tools.evaluate_coarsening(batch)
        coars_batches.append({"batch": batch, "raw": raw})
        coars_rows.extend(parse_results(raw, batch))

    note("Running follow-up phase stability.")
    for i in range(0, len(follow_candidates), 3):
        batch = follow_candidates[i:i+3]
        raw = tools.evaluate_phase_stability(batch)
        phase_batches.append({"batch": batch, "raw": raw})
        phase_rows.extend(parse_results(raw, batch))

(follow_outdir / "followup_coarsening_batches.json").write_text(json.dumps(coars_batches, indent=2, default=str))
(follow_outdir / "followup_phase_batches.json").write_text(json.dumps(phase_batches, indent=2, default=str))

# Build follow-up phase dataframe defensively
phase_new_df = pd.DataFrame(phase_rows)
phase_new_df = ensure_cols(phase_new_df, [
    "Ni", "Er", "Zr", "status", "record",
    "L12_250C", "L12_660C", "Al3Zr_250C", "Al3Zr_660C", "Al3Zr_boundary",
    "liquid_at_upper_boundary", "lower_disappearance_intervals"
])

phase_new_df = phase_new_df.rename(columns={
    "L12_250C": "L12_250C_molpct",
    "L12_660C": "L12_660C_molpct",
    "Al3Zr_250C": "Al3Zr_250C_molpct",
    "Al3Zr_660C": "Al3Zr_660C_molpct",
    "Al3Zr_boundary": "Al3Zr_boundary_C",
})

phase_new_df = ensure_cols(phase_new_df, [
    "L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
    "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"
])

for idx, r in phase_new_df.iterrows():
    rec = load_record(r.get("record"))
    comp_ex = extract_composition_from_record(rec)
    for k in ["Ni", "Er", "Zr"]:
        if safe_float(r.get(k)) is None and k in comp_ex:
            phase_new_df.at[idx, k] = comp_ex[k]
    ex = extract_phase_record_values(rec)
    for c in ["L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
              "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"]:
        if pd.isna(r.get(c)) and c in ex:
            phase_new_df.at[idx, c] = ex[c]

for c in ["Ni", "Er", "Zr", "L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C"]:
    phase_new_df[c] = pd.to_numeric(phase_new_df[c], errors="coerce")
phase_new_df["composition_id"] = phase_new_df.apply(
    lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
    axis=1
)
phase_new_df["phase_feasible_660C"] = phase_new_df["L12_660C_molpct"].apply(
    lambda x: None if pd.isna(x) else (float(x) <= 1.05)
)
phase_new_df.to_csv(follow_outdir / "followup_phase_parsed.csv", index=False)

# Choose Scheil targets from known-composition phase rows
scheil_targets = []
sel = phase_new_df[phase_new_df["composition_id"].notna()].copy()
if len(sel):
    if sel["phase_feasible_660C"].notna().any():
        sel = sel.sort_values(["phase_feasible_660C", "L12_660C_molpct"], ascending=[False, True], na_position="last")
    else:
        sel = sel.sort_values(["L12_660C_molpct"], na_position="last")
    for _, r in sel.head(4).iterrows():
        scheil_targets.append({"Ni": float(r["Ni"]), "Er": float(r["Er"]), "Zr": float(r["Zr"])})
pd.DataFrame(scheil_targets).to_csv(follow_outdir / "chosen_followup_scheil_targets.csv", index=False)

if scheil_targets:
    note(f"Running follow-up solidification on {[comp_id(c) for c in scheil_targets]}")
    for i in range(0, len(scheil_targets), 3):
        batch = scheil_targets[i:i+3]
        raw = tools.evaluate_solidification(batch)
        solid_batches.append({"batch": batch, "raw": raw})
        solid_rows.extend(parse_results(raw, batch))

(follow_outdir / "followup_solidification_batches.json").write_text(json.dumps(solid_batches, indent=2, default=str))

# Save raw parsed rows
pd.DataFrame(coars_rows).to_csv(follow_outdir / "followup_coarsening_raw.csv", index=False)
pd.DataFrame(phase_rows).to_csv(follow_outdir / "followup_phase_raw.csv", index=False)
pd.DataFrame(solid_rows).to_csv(follow_outdir / "followup_solidification_raw.csv", index=False)

# --------------------------
# Assemble new coarsening evidence
# --------------------------
coars_new = pd.DataFrame(coars_rows)
coars_new = ensure_cols(coars_new, ["Ni", "Er", "Zr", "status", "J", "record"])
for c in ["Ni", "Er", "Zr", "J"]:
    coars_new[c] = pd.to_numeric(coars_new[c], errors="coerce")
coars_new["composition_id"] = coars_new.apply(
    lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
    axis=1
)
coars_new = coars_new[coars_new["composition_id"].notna()].copy()
coars_new = coars_new[["composition_id", "Ni", "Er", "Zr", "status", "J", "record"]].rename(columns={"record": "coarsening_record"})

# --------------------------
# Assemble new phase evidence
# --------------------------
phase_new = phase_new_df.copy()
phase_new = phase_new[phase_new["composition_id"].notna()].copy()
phase_new = ensure_cols(phase_new, [
    "composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
    "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
    "liquid_at_upper_boundary", "lower_disappearance_intervals", "record"
])
phase_new = phase_new[[
    "composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
    "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
    "liquid_at_upper_boundary", "lower_disappearance_intervals", "record"
]].rename(columns={"record": "phase_record"})

# --------------------------
# Assemble new solidification evidence defensively
# --------------------------
solid_new = pd.DataFrame(solid_rows)
solid_new = ensure_cols(solid_new, [
    "Ni", "Er", "Zr", "status", "record",
    "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
    "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
    "FR_start_phases", "CSC_intervals"
])

for idx, r in solid_new.iterrows():
    rec = load_record(r.get("record"))
    comp_ex = extract_composition_from_record(rec)
    for k in ["Ni", "Er", "Zr"]:
        if safe_float(r.get(k)) is None and k in comp_ex:
            solid_new.at[idx, k] = comp_ex[k]
    ex = extract_solid_record_values(rec)
    for c in ["ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
              "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"]:
        if pd.isna(r.get(c)) and c in ex:
            solid_new.at[idx, c] = ex[c]

for c in ["Ni", "Er", "Zr", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
          "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct"]:
    solid_new[c] = pd.to_numeric(solid_new[c], errors="coerce")

solid_new["composition_id"] = solid_new.apply(
    lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
    axis=1
)
solid_new = solid_new[solid_new["composition_id"].notna()].copy()
solid_new = ensure_cols(solid_new, [
    "composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
    "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
    "FR_start_phases", "CSC_intervals", "record"
])
solid_new = solid_new[[
    "composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
    "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
    "FR_start_phases", "CSC_intervals", "record"
]].rename(columns={"record": "solidification_record"})

# --------------------------
# Combine prior and new evidence
# --------------------------
coars_all = combine(coars_small, coars_new)
phase_all = combine(phase_small, phase_new)
solid_all = combine(solid_small, solid_new)

updated = coars_all.merge(phase_all, on="composition_id", how="left").merge(solid_all, on="composition_id", how="left")
updated["qualified_all_three"] = updated["J"].notna() & updated["phase_feasible_660C"].fillna(False).astype(bool) & updated["HCS"].notna()
updated.to_csv(follow_outdir / "updated_all_candidate_evidence.csv", index=False)

# --------------------------
# Ranking and shortlist logic
# --------------------------
feasible_all = updated[(updated["J"].notna()) & (updated["phase_feasible_660C"] == True)].copy().sort_values("J")
bestJ = None
preferred = None
balanced = None

if len(feasible_all):
    bestJ = float(feasible_all.iloc[0]["J"])
    updated["J_tie_to_best_feasible"] = updated["J"].apply(lambda x: None if pd.isna(x) else (abs(float(x) - bestJ) < 0.0005))
    updated["J_disadvantage_vs_best_feasible"] = updated["J"].apply(lambda x: None if pd.isna(x) else (float(x) - bestJ))
    tieband = updated[(updated["phase_feasible_660C"] == True) & (updated["J_tie_to_best_feasible"] == True)].copy()
    tieband = tieband.sort_values(
        ["L12_250C_molpct", "Al3Ni_terminal_molpct", "HCS", "CSC", "FR_K"],
        ascending=[False, True, True, True, True],
        na_position="last"
    )
    preferred = tieband.iloc[0].to_dict() if len(tieband) else feasible_all.iloc[0].to_dict()

    qual = updated[updated["qualified_all_three"] == True].copy()
    qual = qual[qual["composition_id"] != preferred["composition_id"]]
    pref = pd.Series(preferred)
    candidates = []
    for _, r in qual.iterrows():
        improvements = 0
        for col in ["HCS", "CSC", "FR_K", "Al3Ni_terminal_molpct"]:
            if pd.notna(r.get(col)) and pd.notna(pref.get(col)) and float(r[col]) < float(pref[col]):
                improvements += 1
        if pd.notna(r.get("L12_250C_molpct")) and pd.notna(pref.get("L12_250C_molpct")) and float(r["L12_250C_molpct"]) > float(pref["L12_250C_molpct"]):
            improvements += 1
        dj = None if pd.isna(r.get("J")) else float(r["J"]) - bestJ
        candidates.append((improvements, abs(dj) if dj is not None else 1e9, r.to_dict()))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    if candidates and candidates[0][0] >= 2:
        balanced = candidates[0][2]
else:
    updated["J_tie_to_best_feasible"] = None
    updated["J_disadvantage_vs_best_feasible"] = None

updated.to_csv(follow_outdir / "updated_comparison_table_with_evidence_ids.csv", index=False)

# --------------------------
# Matched-pair table
# --------------------------
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
pd.DataFrame(pair_rows).to_csv(follow_outdir / "updated_matched_pair_table.csv", index=False)

# --------------------------
# Fresh verification if preferred exists
# --------------------------
if preferred is not None:
    note("Running fresh verification for preferred candidate and reference.")
    fresh = AlloyDesignTools(OUTPUT / "fresh_verification_g1_followup")
    comp = [{"Ni": float(preferred["Ni"]), "Er": float(preferred["Er"]), "Zr": float(preferred["Zr"])}]
    verification = {
        "coarsening": fresh.evaluate_coarsening(comp),
        "phase": fresh.evaluate_phase_stability(comp),
        "solidification": fresh.evaluate_solidification(comp),
    }
    (follow_outdir / "fresh_verification_followup.json").write_text(json.dumps(verification, indent=2, default=str))

# --------------------------
# Summary report
# --------------------------
summary = []
summary.append("Follow-up audit and refinement summary")
summary.append("")
summary.append("Implemented calculation vs task statement")
summary.append("- Candidate compositions were handled directly in at.% Ni/Er/Zr, with Al balance.")
summary.append("- The fixed scientific methods evaluate_coarsening, evaluate_phase_stability and evaluate_solidification were used.")
summary.append("- Phase feasibility was checked as equilibrium L12 at 660 C <= 1.05 mol%.")
summary.append("- Saved outputs include unsuccessful/partial calculations and repaired record-based parsing.")
summary.append("")
summary.append("Audit outcome")
summary.append(f"- Previous records audited: coarsening={len(coars)}, phase={len(phase)}, solidification={len(solid)}.")
summary.append(f"- After audit and follow-up, feasible valid-J candidates found: {len(feasible_all)}.")
if bestJ is not None:
    summary.append(f"- Best measured feasible J = {bestJ:.6f}; practical ties use |ΔJ| < 0.0005.")
else:
    summary.append("- No measured candidate has yet been shown to satisfy both valid J and the 660 C L12 feasibility requirement.")
summary.append("")
summary.append("Follow-up candidate reasons")
for c in follow_candidates:
    summary.append(f"- {comp_id(c)}: chosen to bracket the blocking 660 C feasibility boundary on the lowest-J measured branches or fill a consequential evidence gap.")
summary.append("")
if preferred is not None:
    summary.append("Shortlist")
    summary.append(f"- Coarsening-first preferred candidate: {preferred['composition_id']}")
    summary.append(f"  J={preferred.get('J')}, L12@250C={preferred.get('L12_250C_molpct')} mol%, L12@660C={preferred.get('L12_660C_molpct')} mol%")
    summary.append(f"  Al3Zr@250C={preferred.get('Al3Zr_250C_molpct')} mol%, boundary={preferred.get('Al3Zr_boundary_C')} C, liquid_at_upper_boundary={preferred.get('liquid_at_upper_boundary')}")
    summary.append(f"  Terminal Scheil ternary={preferred.get('ternary_terminal_molpct')} mol%, L12={preferred.get('L12_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')} mol%, Al3Zr={preferred.get('Al3Zr_terminal_molpct')} mol%")
    summary.append(f"  FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        deltaJ = None if bestJ is None or pd.isna(balanced.get("J")) else float(balanced["J"]) - bestJ
        summary.append(f"- Balanced alternative: {balanced['composition_id']} with explicit J tradeoff ΔJ={deltaJ}")
else:
    summary.append("No defensible shortlist recommendation yet from measured evidence; the hard 660 C L12 feasibility screen remains unresolved or unmet on the measured low-J branches.")
summary.append("")
summary.append("Limits")
summary.append("- This is an in-silico design investigation, not an experimental strength or printability prediction.")
summary.append("- Terminal Scheil phase amounts were not interpreted as aged precipitate fractions.")
summary.append("- If shortlist evidence remains insufficient, the next concrete work is narrower phase-bracketing centered on the lowest-J branches in the updated comparison table.")
(follow_outdir / "followup_summary.txt").write_text("\n".join(summary))

# file check
files = [
    follow_outdir / "phase_record_audit.csv",
    follow_outdir / "solid_record_audit.csv",
    follow_outdir / "phase_screening_fixed.csv",
    follow_outdir / "solidification_screening_fixed.csv",
    follow_outdir / "merged_repaired_previous_evidence.csv",
    follow_outdir / "chosen_followup_candidates.csv",
    follow_outdir / "followup_coarsening_batches.json",
    follow_outdir / "followup_phase_batches.json",
    follow_outdir / "followup_solidification_batches.json",
    follow_outdir / "followup_coarsening_raw.csv",
    follow_outdir / "followup_phase_raw.csv",
    follow_outdir / "followup_phase_parsed.csv",
    follow_outdir / "followup_solidification_raw.csv",
    follow_outdir / "chosen_followup_scheil_targets.csv",
    follow_outdir / "updated_all_candidate_evidence.csv",
    follow_outdir / "updated_comparison_table_with_evidence_ids.csv",
    follow_outdir / "updated_matched_pair_table.csv",
    follow_outdir / "followup_summary.txt",
]
if (follow_outdir / "fresh_verification_followup.json").exists():
    files.append(follow_outdir / "fresh_verification_followup.json")
pd.DataFrame([{"file": str(f), "exists": f.exists(), "size_bytes": f.stat().st_size if f.exists() else None} for f in files]).to_csv(
    follow_outdir / "created_files_check.csv", index=False
)

# console summary
note("")
note("Debugged follow-up complete. Created files:")
for f in files:
    note(f"  - {f.name}: {'OK' if f.exists() else 'MISSING'}")
note("")
if preferred is not None:
    note(f"Preferred candidate: {preferred['composition_id']}")
    note(f"  J={preferred.get('J')}, L12@250C={preferred.get('L12_250C_molpct')} mol%, L12@660C={preferred.get('L12_660C_molpct')} mol%")
    note(f"  terminal ternary={preferred.get('ternary_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')}, FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        deltaJ = None if bestJ is None or pd.isna(balanced.get('J')) else float(balanced['J']) - bestJ
        note(f"Balanced alternative: {balanced['composition_id']} (ΔJ={deltaJ})")
else:
    note("No defensible preferred candidate yet from measured evidence.")
note(f"Outputs saved in: {follow_outdir}")