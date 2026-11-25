import json
from pathlib import Path
import pandas as pd

from science_api import tools, REPO, OUTPUT, AlloyDesignTools

# ------------------------------------------------------------
# Debugged follow-up audit with robust composition recovery
# ------------------------------------------------------------
# Interpreted conditions:
# - Candidate basis: Ni, Er, Zr in at.% directly; Al balance; Y=Yb=0.
# - Hard phase feasibility for main shortlist: equilibrium L12 at 660 C <= 1.05 mol%.
# - Coarsening metric: normalized J at 250 C, lower is better.
# - Phase and terminal Scheil amounts: normalized mol%.
# - Use the fixed supplied scientific methods only.
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
        numeric = []
        for _, v in hits:
            fv = safe_float(v)
            if fv is not None:
                numeric.append(fv)
        if numeric:
            # use first numeric occurrence
            out[key] = numeric[0]
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
            out[std + "_path"] = hits[0][0]
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
            out[std + "_path"] = hits[0][0]
    return out

def parse_results(raw, family, compositions_batch):
    rows = []
    if not isinstance(raw, dict):
        return rows
    results = raw.get("results", [])
    for i, r in enumerate(results):
        row = {"family": family}
        comp = r.get("composition", {}) if isinstance(r.get("composition", {}), dict) else {}
        # composition from explicit result
        for k in ["Ni", "Er", "Zr"]:
            if k in r:
                row[k] = r[k]
            elif k in comp:
                row[k] = comp[k]
        # fallback to requested batch order
        if i < len(compositions_batch):
            for k in ["Ni", "Er", "Zr"]:
                if row.get(k) is None:
                    row[k] = compositions_batch[i][k]
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

def run_batched(func, compositions):
    payloads = []
    all_rows = []
    for i in range(0, len(compositions), 3):
        batch = compositions[i:i+3]
        raw = func(batch)
        payloads.append({"batch": batch, "raw": raw})
        all_rows.extend(parse_results(raw, "unknown", batch))
    return payloads, all_rows

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
# Audit previous saved records
# --------------------------
phase_fixed = phase.copy()
phase_audit_rows = []
for idx, r in phase_fixed.iterrows():
    rec = load_record(r.get("phase_record"))
    ex = extract_phase_record_values(rec)
    comp_ex = extract_composition_from_record(rec)
    phase_audit_rows.append({
        "composition_id": r["composition_id"],
        "csv_L12_250C": r.get("L12_250C_molpct"),
        "csv_L12_660C": r.get("L12_660C_molpct"),
        "rec_keys_found": ",".join(sorted(ex.keys())),
        "rec_comp_found": json.dumps(comp_ex),
        "phase_record": r.get("phase_record"),
    })
    for col in ["L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
                "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"]:
        if ((col not in phase_fixed.columns) or pd.isna(r.get(col))) and col in ex:
            if col not in phase_fixed.columns:
                phase_fixed[col] = None
            phase_fixed.at[idx, col] = ex[col]
phase_fixed["phase_feasible_660C"] = phase_fixed["L12_660C_molpct"].apply(lambda x: None if pd.isna(x) else (float(x) <= 1.05))
pd.DataFrame(phase_audit_rows).to_csv(follow_outdir / "phase_record_audit.csv", index=False)
phase_fixed.to_csv(follow_outdir / "phase_screening_fixed.csv", index=False)

solid_fixed = solid.copy()
solid_audit_rows = []
for idx, r in solid_fixed.iterrows():
    rec = load_record(r.get("solidification_record"))
    ex = extract_solid_record_values(rec)
    solid_audit_rows.append({
        "composition_id": r["composition_id"],
        "rec_keys_found": ",".join(sorted(ex.keys())),
        "solidification_record": r.get("solidification_record"),
    })
    for col in ["ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
                "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"]:
        if ((col not in solid_fixed.columns) or pd.isna(r.get(col))) and col in ex:
            if col not in solid_fixed.columns:
                solid_fixed[col] = None
            solid_fixed.at[idx, col] = ex[col]
pd.DataFrame(solid_audit_rows).to_csv(follow_outdir / "solid_record_audit.csv", index=False)
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
# Evidence-driven follow-up set
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
    note("Feasible valid-J candidates already exist; only consequential follow-up will be added.")
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
# Run follow-up calculations in small batches
# --------------------------
coars_batches = []
phase_batches = []
solid_batches = []
coars_rows = []
phase_rows = []
solid_rows = []

# coarsening
if follow_candidates:
    note("Running follow-up coarsening.")
    for i in range(0, len(follow_candidates), 3):
        batch = follow_candidates[i:i+3]
        raw = tools.evaluate_coarsening(batch)
        coars_batches.append({"batch": batch, "raw": raw})
        coars_rows.extend(parse_results(raw, "coarsening", batch))

    note("Running follow-up phase stability.")
    for i in range(0, len(follow_candidates), 3):
        batch = follow_candidates[i:i+3]
        raw = tools.evaluate_phase_stability(batch)
        phase_batches.append({"batch": batch, "raw": raw})
        phase_rows.extend(parse_results(raw, "phase", batch))

(follow_outdir / "followup_coarsening_batches.json").write_text(json.dumps(coars_batches, indent=2, default=str))
(follow_outdir / "followup_phase_batches.json").write_text(json.dumps(phase_batches, indent=2, default=str))

# Robust phase dataframe
phase_new_df = pd.DataFrame(phase_rows)
if len(phase_new_df):
    for c in ["Ni", "Er", "Zr", "status", "record"]:
        if c not in phase_new_df.columns:
            phase_new_df[c] = None

    rename_map = {}
    if "L12_250C" in phase_new_df.columns:
        rename_map["L12_250C"] = "L12_250C_molpct"
    if "L12_660C" in phase_new_df.columns:
        rename_map["L12_660C"] = "L12_660C_molpct"
    if "Al3Zr_250C" in phase_new_df.columns:
        rename_map["Al3Zr_250C"] = "Al3Zr_250C_molpct"
    if "Al3Zr_660C" in phase_new_df.columns:
        rename_map["Al3Zr_660C"] = "Al3Zr_660C_molpct"
    if "Al3Zr_boundary" in phase_new_df.columns:
        rename_map["Al3Zr_boundary"] = "Al3Zr_boundary_C"
    phase_new_df = phase_new_df.rename(columns=rename_map)

    for c in ["L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct",
              "Al3Zr_boundary_C", "liquid_at_upper_boundary", "lower_disappearance_intervals"]:
        if c not in phase_new_df.columns:
            phase_new_df[c] = None

    # Recover missing composition or phase fields from records
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

    # Last-resort composition recovery from requested follow-up list order if still missing
    fallback_map = {}
    for c in follow_candidates:
        fallback_map[comp_id(c)] = c
    for idx, r in phase_new_df.iterrows():
        ni, er, zr = safe_float(r.get("Ni")), safe_float(r.get("Er")), safe_float(r.get("Zr"))
        if ni is None or er is None or zr is None:
            # use same row index if possible from flattened phase_rows source, already seeded from batch order in parse_results
            src = phase_rows[idx] if idx < len(phase_rows) else {}
            ni = safe_float(src.get("Ni"))
            er = safe_float(src.get("Er"))
            zr = safe_float(src.get("Zr"))
            if ni is not None:
                phase_new_df.at[idx, "Ni"] = ni
            if er is not None:
                phase_new_df.at[idx, "Er"] = er
            if zr is not None:
                phase_new_df.at[idx, "Zr"] = zr

    for c in ["Ni", "Er", "Zr", "L12_250C_molpct", "L12_660C_molpct", "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C"]:
        phase_new_df[c] = pd.to_numeric(phase_new_df[c], errors="coerce")

    # Keep partial rows even if composition or fields remain unknown
    phase_new_df["composition_id"] = phase_new_df.apply(
        lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
        axis=1
    )
    phase_new_df["phase_feasible_660C"] = phase_new_df["L12_660C_molpct"].apply(
        lambda x: None if pd.isna(x) else (float(x) <= 1.05)
    )

phase_new_df.to_csv(follow_outdir / "followup_phase_parsed.csv", index=False)

# choose Scheil targets only among rows with known composition_id
scheil_targets = []
if len(phase_new_df):
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
        solid_rows.extend(parse_results(raw, "solidification", batch))
(follow_outdir / "followup_solidification_batches.json").write_text(json.dumps(solid_batches, indent=2, default=str))

# Save raw parsed rows
pd.DataFrame(coars_rows).to_csv(follow_outdir / "followup_coarsening_raw.csv", index=False)
pd.DataFrame(phase_rows).to_csv(follow_outdir / "followup_phase_raw.csv", index=False)
pd.DataFrame(solid_rows).to_csv(follow_outdir / "followup_solidification_raw.csv", index=False)

# Build coarsening dataframe
coars_new = pd.DataFrame(coars_rows)
if len(coars_new):
    for c in ["Ni", "Er", "Zr", "J", "status", "record"]:
        if c not in coars_new.columns:
            coars_new[c] = None
    for c in ["Ni", "Er", "Zr", "J"]:
        coars_new[c] = pd.to_numeric(coars_new[c], errors="coerce")
    coars_new["composition_id"] = coars_new.apply(
        lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
        axis=1
    )
    coars_new = coars_new[coars_new["composition_id"].notna()].copy()
    coars_new = coars_new[["composition_id", "Ni", "Er", "Zr", "status", "J", "record"]].rename(columns={"record": "coarsening_record"})
else:
    coars_new = pd.DataFrame(columns=["composition_id", "Ni", "Er", "Zr", "status", "J", "coarsening_record"])

# Build phase dataframe
if len(phase_new_df):
    phase_new = phase_new_df[phase_new_df["composition_id"].notna()].copy()
    phase_new = phase_new[[
        "composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
        "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
        "liquid_at_upper_boundary", "lower_disappearance_intervals", "record"
    ]].rename(columns={"record": "phase_record"})
else:
    phase_new = pd.DataFrame(columns=[
        "composition_id", "L12_250C_molpct", "L12_660C_molpct", "phase_feasible_660C",
        "Al3Zr_250C_molpct", "Al3Zr_660C_molpct", "Al3Zr_boundary_C",
        "liquid_at_upper_boundary", "lower_disappearance_intervals", "phase_record"
    ])

# Build solidification dataframe
solid_new = pd.DataFrame(solid_rows)
if len(solid_new):
    for c in ["Ni", "Er", "Zr", "status", "record"]:
        if c not in solid_new.columns:
            solid_new[c] = None
    for idx, r in solid_new.iterrows():
        rec = load_record(r.get("record"))
        comp_ex = extract_composition_from_record(rec)
        for k in ["Ni", "Er", "Zr"]:
            if safe_float(r.get(k)) is None and k in comp_ex:
                solid_new.at[idx, k] = comp_ex[k]
        ex = extract_solid_record_values(rec)
        for c in ["ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct", "Al3Zr_terminal_molpct",
                  "FR_K", "CSC", "HCS", "terminal_solid_molpct", "FR_start_phases", "CSC_intervals"]:
            if ((c not in solid_new.columns) or pd.isna(r.get(c))) and c in ex:
                if c not in solid_new.columns:
                    solid_new[c] = None
                solid_new.at[idx, c] = ex[c]
    for c in ["Ni", "Er", "Zr", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
              "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct"]:
        if c in solid_new.columns:
            solid_new[c] = pd.to_numeric(solid_new[c], errors="coerce")
    solid_new["composition_id"] = solid_new.apply(
        lambda r: comp_id({"Ni": r["Ni"], "Er": r["Er"], "Zr": r["Zr"]}) if pd.notna(r["Ni"]) and pd.notna(r["Er"]) and pd.notna(r["Zr"]) else None,
        axis=1
    )
    solid_new = solid_new[solid_new["composition_id"].notna()].copy()
    solid_new = solid_new[[
        "composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
        "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
        "FR_start_phases", "CSC_intervals", "record"
    ]].rename(columns={"record": "solidification_record"})
else:
    solid_new = pd.DataFrame(columns=[
        "composition_id", "ternary_terminal_molpct", "L12_terminal_molpct", "Al3Ni_terminal_molpct",
        "Al3Zr_terminal_molpct", "FR_K", "CSC", "HCS", "terminal_solid_molpct",
        "FR_start_phases", "CSC_intervals", "solidification_record"
    ])

# combine
coars_all = combine(coars_small, coars_new)
phase_all = combine(phase_small, phase_new)
solid_all = combine(solid_small, solid_new)

updated = coars_all.merge(phase_all, on="composition_id", how="left").merge(solid_all, on="composition_id", how="left")
updated["qualified_all_three"] = updated["J"].notna() & updated["phase_feasible_660C"].fillna(False).astype(bool) & updated["HCS"].notna()
updated.to_csv(follow_outdir / "updated_all_candidate_evidence.csv", index=False)

# ranking
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
    cands = []
    for _, r in qual.iterrows():
        improvements = 0
        for col in ["HCS", "CSC", "FR_K", "Al3Ni_terminal_molpct"]:
            if pd.notna(r.get(col)) and pd.notna(pref.get(col)) and float(r[col]) < float(pref[col]):
                improvements += 1
        if pd.notna(r.get("L12_250C_molpct")) and pd.notna(pref.get("L12_250C_molpct")) and float(r["L12_250C_molpct"]) > float(pref["L12_250C_molpct"]):
            improvements += 1
        dj = None if pd.isna(r.get("J")) else float(r["J"]) - bestJ
        cands.append((improvements, abs(dj) if dj is not None else 1e9, r.to_dict()))
    cands.sort(key=lambda x: (-x[0], x[1]))
    if cands and cands[0][0] >= 2:
        balanced = cands[0][2]
else:
    updated["J_tie_to_best_feasible"] = None
    updated["J_disadvantage_vs_best_feasible"] = None

updated.to_csv(follow_outdir / "updated_comparison_table_with_evidence_ids.csv", index=False)

# matched-pair table
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

# fresh verification
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

# summary
summary = []
summary.append("Follow-up audit and refinement summary")
summary.append("")
summary.append("Implemented conditions and numerical methods")
summary.append("- Candidate compositions were handled directly in at.% Ni/Er/Zr, with Al balance.")
summary.append("- Coarsening, phase stability and solidification were evaluated using the supplied fixed scientific functions only.")
summary.append("- Phase feasibility was interpreted strictly as equilibrium L12 at 660 C <= 1.05 mol%.")
summary.append("- Returned phase quantities are normalized mol%; they are not extensive sample totals.")
summary.append("")
summary.append("Evidence status")
summary.append(f"- Previous records audited: coarsening rows={len(coars)}, phase rows={len(phase)}, solidification rows={len(solid)}.")
summary.append(f"- After audit and follow-up, feasible valid-J candidates found: {len(feasible_all)}.")
if bestJ is not None:
    summary.append(f"- Best measured feasible J = {bestJ:.6f}; practical ties use |ΔJ| < 0.0005.")
else:
    summary.append("- No measured candidate has yet been shown to satisfy both a valid J and the 660 C L12 feasibility requirement.")
summary.append("")
summary.append("Follow-up candidate reasons")
for c in follow_candidates:
    summary.append(f"- {comp_id(c)}: selected to bracket the blocking 660 C feasibility boundary on the lowest-J measured branches or to fill a consequential gap.")
summary.append("")
if preferred is not None:
    summary.append("Shortlist")
    summary.append(f"- Coarsening-first preferred candidate: {preferred['composition_id']}")
    summary.append(f"  J={preferred.get('J')}, L12@250C={preferred.get('L12_250C_molpct')} mol%, L12@660C={preferred.get('L12_660C_molpct')} mol%")
    summary.append(f"  Al3Zr@250C={preferred.get('Al3Zr_250C_molpct')} mol%, Al3Zr@660C={preferred.get('Al3Zr_660C_molpct')} mol%, boundary={preferred.get('Al3Zr_boundary_C')} C")
    summary.append(f"  Terminal Scheil ternary={preferred.get('ternary_terminal_molpct')} mol%, L12={preferred.get('L12_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')} mol%, Al3Zr={preferred.get('Al3Zr_terminal_molpct')} mol%")
    summary.append(f"  FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        deltaJ = None if bestJ is None or pd.isna(balanced.get('J')) else float(balanced['J']) - bestJ
        summary.append(f"- Balanced alternative: {balanced['composition_id']} with explicit J tradeoff ΔJ={deltaJ}")
else:
    summary.append("No defensible shortlist recommendation yet from measured evidence. The blocking issue remains the hard 660 C L12 feasibility qualification on low-J branches.")
summary.append("")
summary.append("Limits")
summary.append("- This is an in-silico design investigation, not an experimental prediction of strength or printability.")
summary.append("- Terminal Scheil phase amounts were not interpreted as aged precipitate fractions.")
summary.append("- If shortlist evidence remains insufficient, the next concrete work is narrower phase-bracketing around the lowest-J branches listed in the updated comparison table.")
(follow_outdir / "followup_summary.txt").write_text("\n".join(summary))

# file existence check
files = [
    follow_outdir / "phase_record_audit.csv",
    follow_outdir / "solid_record_audit.csv",
    follow_outdir / "phase_screening_fixed.csv",
    follow_outdir / "solidification_screening_fixed.csv",
    follow_outdir / "merged_repaired_previous_evidence.csv",
    follow_outdir / "chosen_followup_candidates.csv",
    follow_outdir / "followup_coarsening_batches.json",
    follow_outdir / "followup_phase_batches.json",
    follow_outdir / "chosen_followup_scheil_targets.csv",
    follow_outdir / "followup_coarsening_raw.csv",
    follow_outdir / "followup_phase_raw.csv",
    follow_outdir / "followup_phase_parsed.csv",
    follow_outdir / "followup_solidification_raw.csv",
    follow_outdir / "followup_solidification_batches.json",
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
    note(f"  terminal ternary={preferred.get('ternary_terminal_molpct')} mol%, Al3Ni={preferred.get('Al3Ni_terminal_molpct')} mol%, FR_K={preferred.get('FR_K')}, CSC={preferred.get('CSC')}, HCS={preferred.get('HCS')}")
    if balanced is not None:
        deltaJ = None if bestJ is None or pd.isna(balanced.get('J')) else float(balanced['J']) - bestJ
        note(f"Balanced alternative: {balanced['composition_id']} (ΔJ={deltaJ})")
else:
    note("No defensible preferred candidate yet from measured evidence.")
note(f"Outputs saved in: {follow_outdir}")