#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — automated Arrhenius parameter fitting for mcPolymer kMC simulations
=============================================================================

Single entry point for the s-BuLi / styrene living anionic polymerization fit.
Run it on the workstation where the mcPolymer engine lives:

    python main.py --setup          # build one folder per experiment (+ experiment.json)
    python main.py --single TW60    # run ONE folder once, print Mn(t)  (validation)
    python main.py --eval-once      # one parallel fan-out for a fixed theta -> global loss
    python main.py --screen         # ki/kp sensitivity + stochastic noise floor
    python main.py --stage1         # decoupled per-temperature k warm start + Arrhenius seed
    python main.py --stage2         # coupled global (A_ki,Ea_ki,A_kp,Ea_kp) refinement
    python main.py --verify         # re-run best fit K times at verification resolution
    python main.py --all            # setup -> stage1 -> stage2 -> verify -> report
    python main.py --report         # (re)write report.md + plots from the latest results

The stages are deliberately independent so each can be validated on its own
(see the task's staged-validation plan).

--------------------------------------------------------------------------------
WHAT THIS PROGRAM DOES / DOES NOT TOUCH
--------------------------------------------------------------------------------
* It NEVER imports or edits mcPolymer, kineticModel.py, modelInterpreter.py, or
  TWXX.py.  The engine is only ever exercised by launching the *provided* driver
  TWXX.py as a subprocess (cwd = experiment folder).  main.py itself only needs
  numpy / pandas / scipy / matplotlib and the validated analyzer mwd_analyzerv2.
* It drives the driver purely through the two JSON files TWXX.py reads:
  experiment.json (recipe + temperature + sim settings, written once at --setup)
  and coeffs.json (the {"ki":.., "kp":..} for the current evaluation, rewritten
  every optimizer iteration).
* It fits Mn ONLY at 10/20/40/60 min.  The 720-min row is NOT a residual; it is
  used (as n_sbuli_eff_mol, precomputed in the CSV) to set the effective
  initiator so each sim reproduces its final Mn by construction.
* Conversion is a DIAGNOSTIC only (never in the objective).

--------------------------------------------------------------------------------
API ASSUMPTIONS  (stated explicitly; verify locally against the real engine)
--------------------------------------------------------------------------------
A1. Coefficient injection — TEMPLATE rewrite is the reliable path (TWXX default).
    modelInterpreter.interpreteModelFile() PREPENDS any addCoefficient lines to
    the model-file text and interprets the concatenation.  Because ip.mcPolymer
    ALREADY defines `ki = ...` and `kp = ...`, addCoefficient would leave TWO
    definitions of each coefficient in the merged file, and which one the engine
    honours is undefined.  The template method (TWXX.write_active_model) rewrites
    the existing `ki =`/`kp =` lines in place, so exactly one definition carries
    the injected value.  kineticModel(modelFile=...) then re-reads the interpreted
    model, so the value MUST live in the file (a Python-side addCoefficient alone
    is not enough).  => keep injection_method="template"; do not edit TWXX.py.

A2. exportMMD output vs analyzer format.  kineticModel.exportMMD writes the two-
    column MMD to  <folder>/mcPolymerSimulationresults-ID_<id>/MMD-S-<sec>.dat
    (NOTE: a per-simulation subfolder, not the cwd directly).  We therefore
    search each run dir RECURSIVELY for MMD-S-<sec>.dat and CLEAN stale outputs
    before every run.  mwd_analyzerv2 auto-orients columns to x=log10(M),
    y=dw/dlog10(M); if your engine build emits the opposite orientation, set
    FORCE_X_COL below to "c0"/"c1"/"swap" instead of editing the analyzer.

A3. True simulated conversion (DIAGNOSTIC only): X_sim(t) = 1 - mol_styrene(t)/
    mol_styrene(0), read via kMC.getMol("styrene") (monomer balance, NOT
    concentration — the volume contracts).  TWXX.py already writes this to
    sim_conversion.csv; we read it only for the diagnostic table.

A4. The two fit coefficients are exactly `ki` (initiation) and `kp` (propagation);
    the model has only these two reactions.

A5. Common-random-numbers (CRN): kineticModel calls mcPolymer.py_startRandomGenerator()
    with no seed argument exposed through the Python wrapper, so CRN across
    evaluations cannot be enforced from here without an engine hook.  We instead
    reduce stochastic noise by averaging K replicates per evaluation and report
    the measured noise floor (see --screen).  If your engine build accepts a seed,
    that is the one place to wire CRN in — but do it in the engine, not here.
--------------------------------------------------------------------------------
"""

import os
import re
import sys
import csv
import json
import math
import time
import shutil
import argparse
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# The validated MMD analyzer (imports only numpy/pandas -> safe to import here).
import mwd_analyzerv2 as mwd

# scipy is used for the optimizers; import lazily-tolerant so --setup/--single
# work even on a machine without scipy.
try:
    from scipy.optimize import differential_evolution
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# matplotlib only for diagnostic plots; degrade gracefully if absent.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


# =============================================================================
# CONFIG  — everything tunable lives here (no hard-coded temperatures/codes/paths
# elsewhere; codes, temperatures and data times are read from the CSV).
# =============================================================================
CONFIG = {
    # ---- inputs / layout -----------------------------------------------------
    "repo_dir":            str(Path(__file__).resolve().parent),
    "data_csv":            "experimental_data.csv",
    "driver_template":     "TWXX.py",         # the fixed generic driver (copied into each folder)
    "model_template":      "ip.mcPolymer",    # the two-reaction model (copied into each folder)
    "work_root":           "work",            # experiment folders (TW60/, ...) are created here
    "results_root":        "results",         # timestamped run outputs land here

    # ---- physical constants --------------------------------------------------
    "R":                   8.314,             # J/(mol K)
    "T_ref_C":             0.0,               # T[K] = T[C] + 273.15
    "T_offset_K":          273.15,
    "monomer_mw":          104.15,            # styrene, g/mol (for Mn/DP diagnostics)
    "charge_mass_g":       4.0000,            # constant styrene charge; n_I,eff = charge/Mn720

    # ---- simulation settings written into experiment.json --------------------
    "dt_s":                1,                 # volume-balance update step (integer seconds)
    "mmd_raster_points":   800,
    "injection_method":    "template",        # keep "template" (see assumption A1)
    "numMolecules_fit":    100_000_000,       # 1e8 — fast/noisy, for fitting
    "numMolecules_verify": 1_000_000_000,     # 1e9 — final verification resolution

    # ---- replicates (noise averaging) ---------------------------------------
    "K_replicates_fit":     1,                # replicates per evaluation during fitting
    "K_replicates_verify":  5,                # replicates for the final verification scatter

    # ---- Mn extraction -------------------------------------------------------
    "force_x_col":          "auto",           # "auto"|"c0"|"c1"|"swap" (assumption A2)
    "mmd_glob":             "MMD-S-{t}.dat",

    # ---- objective -----------------------------------------------------------
    # residual on Mn: "log"  -> ln(Mn_sim) - ln(Mn_exp)   (default; scale-free)
    #                 "rel"  -> (Mn_sim - Mn_exp)/Mn_exp
    "residual_kind":        "log",
    # weighting so 30 C (4 exps) does not swamp 20 C (2 exps):
    #   "per_experiment"  -> every experiment weighted equally (default)
    #   "per_temperature" -> average within a T level, then average the two levels
    "loss_normalization":   "per_experiment",
    # DISABLED hook for a second observable (dispersity / Mw). Set weight > 0 and
    # populate exp_D in the CSV to switch it on; strongly constrains ki.
    "use_dispersity_term":  False,
    "dispersity_weight":    0.0,
    "penalty_loss":         1.0e3,            # returned when a sim eval fails (keeps DE alive)

    # ---- parallelism ---------------------------------------------------------
    "max_workers":          6,                # concurrent driver processes (folders x reps)
    "subprocess_timeout_s": 7200,             # per driver run

    # ---- Stage 1: decoupled per-temperature k fit ---------------------------
    "stage1_k_bounds_log10": [-6.0, 3.0],     # bounds on log10(k) [L/(mol s)] per coefficient
    "stage1_maxiter":        30,
    "stage1_popsize":        12,
    "stage1_tol":            1e-3,

    # ---- Stage 2: coupled global Arrhenius fit ------------------------------
    # transformed params: x = [log10 A_ki, Ea_ki(kJ/mol), log10 A_kp, Ea_kp(kJ/mol)]
    "log10A_bounds":         [-5.0, 40.0],
    "Ea_bounds_kJ":          [0.0, 150.0],
    "stage2_optimizer":      "differential_evolution",  # or "cma" if the `cma` pkg is installed
    "stage2_maxiter":        40,
    "stage2_popsize":        12,
    "stage2_tol":            1e-3,
    "stage2_mutation":       [0.5, 1.0],
    "stage2_recombination":  0.7,
    "cma_sigma0":            0.5,             # CMA-ES only (in normalized [0,1] box units)

    # ---- reproducibility -----------------------------------------------------
    "seed":                  20260704,

    # ---- sensitivity screen --------------------------------------------------
    "screen_perturb_factor": 1.5,            # multiply/divide ki or kp by this
    "screen_noise_reps":     5,              # replicates to estimate the Mn noise floor
}

# Physical bound for a "fast enough to be unresolved" initiation, used when we
# report the practical lower bound on ki (diagnostic text only).
DATA_TIME_COL = "time_s"

LOG = logging.getLogger("arrhenius_fit")


# =============================================================================
# small helpers
# =============================================================================
def setup_logging(logfile: Path | None = None, verbose: bool = True):
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


def T_K(T_C: float) -> float:
    return float(T_C) + CONFIG["T_offset_K"]


def arrhenius_k(A: float, Ea_J: float, T_C: float) -> float:
    """k = A * exp(-Ea/(R T)),  Ea in J/mol, T in Kelvin."""
    return float(A) * math.exp(-float(Ea_J) / (CONFIG["R"] * T_K(T_C)))


def repo_path(name: str) -> Path:
    return Path(CONFIG["repo_dir"]) / name


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", CONFIG["repo_dir"], "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# experimental data
# =============================================================================
class Experiment:
    """One experiment (one `code`): its recipe, temperature, and Mn(t) targets."""
    def __init__(self, code: str, temperature_C: float, n_styrene_mol: float,
                 n_sbuli_charged_mol: float, n_cyclohexane_mol: float,
                 n_sbuli_eff_mol: float, fit_times_s, Mn_by_time: dict,
                 Mn720: float, D_by_time: dict | None = None):
        self.code = code
        self.temperature_C = float(temperature_C)
        self.n_styrene_mol = float(n_styrene_mol)
        self.n_sbuli_charged_mol = float(n_sbuli_charged_mol)
        self.n_cyclohexane_mol = float(n_cyclohexane_mol)
        self.n_sbuli_eff_mol = float(n_sbuli_eff_mol)
        self.fit_times_s = [int(t) for t in fit_times_s]
        self.Mn_by_time = {int(t): float(v) for t, v in Mn_by_time.items()}
        self.Mn720 = float(Mn720)
        self.D_by_time = D_by_time or {}

    @property
    def titer_ratio(self) -> float:
        return self.n_sbuli_eff_mol / self.n_sbuli_charged_mol

    def n_eff_check(self) -> float:
        """Cross-check: n_I,eff should equal charge_mass / Mn720."""
        return CONFIG["charge_mass_g"] / self.Mn720


def load_experiments(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"code", "temperature_C", "n_styrene_mol", "n_sbuli_charged_mol",
                "n_cyclohexane_mol", "n_sbuli_eff_mol", "time_s", "Mn", "is_full_conversion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    exps = []
    for code, g in df.groupby("code", sort=False):
        g = g.sort_values("time_s")
        anchor = g[g["is_full_conversion"] == 1]
        fit_rows = g[g["is_full_conversion"] == 0]
        if anchor.empty:
            raise ValueError(f"{code}: no is_full_conversion=1 (720 min) anchor row")
        if fit_rows.empty:
            raise ValueError(f"{code}: no fit rows (is_full_conversion=0)")
        Mn720 = float(anchor["Mn"].iloc[0])
        fit_times = [int(t) for t in fit_rows["time_s"].tolist()]
        Mn_by_time = {int(t): float(m) for t, m in zip(fit_rows["time_s"], fit_rows["Mn"])}
        D_by_time = {}
        if "exp_D" in df.columns:
            D_by_time = {int(t): float(d) for t, d in zip(fit_rows["time_s"], fit_rows["exp_D"])
                         if pd.notna(d)}
        r0 = g.iloc[0]
        exp = Experiment(
            code=str(code),
            temperature_C=r0["temperature_C"],
            n_styrene_mol=r0["n_styrene_mol"],
            n_sbuli_charged_mol=r0["n_sbuli_charged_mol"],
            n_cyclohexane_mol=r0["n_cyclohexane_mol"],
            n_sbuli_eff_mol=r0["n_sbuli_eff_mol"],
            fit_times_s=fit_times,
            Mn_by_time=Mn_by_time,
            Mn720=Mn720,
            D_by_time=D_by_time,
        )
        # sanity: consistent effective-initiator definition
        chk = exp.n_eff_check()
        rel = abs(chk - exp.n_sbuli_eff_mol) / exp.n_sbuli_eff_mol
        if rel > 0.02:
            LOG.warning("%s: n_sbuli_eff_mol=%.6g disagrees with charge/Mn720=%.6g (%.1f%%)",
                        code, exp.n_sbuli_eff_mol, chk, 100 * rel)
        exps.append(exp)
    return exps


def temperatures(exps):
    """Sorted unique temperatures (°C) present in the data."""
    return sorted({e.temperature_C for e in exps})


def all_fit_times(exps):
    ts = set()
    for e in exps:
        ts.update(e.fit_times_s)
    return sorted(ts)


# =============================================================================
# folder setup  (--setup)
# =============================================================================
def experiment_folder(exp: Experiment) -> Path:
    return Path(CONFIG["repo_dir"]) / CONFIG["work_root"] / exp.code


def make_experiment_json(exp: Experiment, numMolecules: int) -> dict:
    return {
        "code": exp.code,
        "temperature_C": exp.temperature_C,
        "n_styrene_mol": exp.n_styrene_mol,
        # EFFECTIVE initiator (n_I,eff = charge/Mn720) — NOT the charged titer:
        "n_sbuli_mol": exp.n_sbuli_eff_mol,
        "n_cyclohexane_mol": exp.n_cyclohexane_mol,
        "export_times_s": list(exp.fit_times_s),
        "dt_s": CONFIG["dt_s"],
        "numMolecules": int(numMolecules),
        "mmd_raster_points": CONFIG["mmd_raster_points"],
        "injection_method": CONFIG["injection_method"],
        "model_template": CONFIG["model_template"],
    }


def setup_folders(exps, numMolecules: int):
    driver = repo_path(CONFIG["driver_template"])
    model = repo_path(CONFIG["model_template"])
    for f in (driver, model):
        if not f.exists():
            raise FileNotFoundError(f"required source file not found: {f}")

    for exp in exps:
        folder = experiment_folder(exp)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(driver, folder / CONFIG["driver_template"])
        shutil.copy2(model, folder / CONFIG["model_template"])
        with open(folder / "experiment.json", "w") as fh:
            json.dump(make_experiment_json(exp, numMolecules), fh, indent=2)
        LOG.info("[setup] %-6s  T=%4.1f C  n_I,eff=%.4e  titer(eff/charged)=%5.1f%%  -> %s",
                 exp.code, exp.temperature_C, exp.n_sbuli_eff_mol,
                 100 * exp.titer_ratio, folder)
    LOG.info("[setup] built %d experiment folders under %s (numMolecules=%.2e)",
             len(exps), Path(CONFIG["repo_dir"]) / CONFIG["work_root"], numMolecules)


# =============================================================================
# subprocess execution of the provided driver TWXX.py
# =============================================================================
def _driver_env() -> dict:
    """Ensure the driver (run from cwd=folder) can import the core modules and
    the mcPolymer engine that live at the repo root / user's PYTHONPATH."""
    env = dict(os.environ)
    extra = CONFIG["repo_dir"]
    env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _clean_outputs(run_dir: Path):
    """Remove stale sim outputs so a crashed run cannot masquerade as success.
    (Essential: folders are reused across every optimizer iteration.)"""
    for p in run_dir.glob("MMD-S-*.dat"):
        p.unlink(missing_ok=True)
    for p in run_dir.glob("mcPolymerSimulationresults-ID_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for name in ("sim_conversion.csv", "run_meta.json", "ip_active.mcPolymer",
                 "ip_active.mcPolymer.json"):
        (run_dir / name).unlink(missing_ok=True)


def _ensure_rundir(exp: Experiment, run_dir: Path, numMolecules: int, coeffs: dict):
    """Populate a run directory with everything the driver needs, then clean stale
    outputs.  run_dir may be the experiment folder itself (K=1) or a rep subdir."""
    folder = experiment_folder(exp)
    run_dir.mkdir(parents=True, exist_ok=True)
    for fn in (CONFIG["driver_template"], CONFIG["model_template"]):
        src = folder / fn
        if not src.exists():
            src = repo_path(fn)
        dst = run_dir / fn
        if (not dst.exists()) or (dst.stat().st_mtime < src.stat().st_mtime):
            shutil.copy2(src, dst)
    with open(run_dir / "experiment.json", "w") as fh:
        json.dump(make_experiment_json(exp, numMolecules), fh, indent=2)
    with open(run_dir / "coeffs.json", "w") as fh:
        json.dump({"ki": float(coeffs["ki"]), "kp": float(coeffs["kp"])}, fh, indent=2)
    _clean_outputs(run_dir)


def run_driver(run_dir: Path) -> tuple:
    """Launch `python TWXX.py` in run_dir. Returns (ok, message)."""
    try:
        proc = subprocess.run(
            [sys.executable, CONFIG["driver_template"]],
            cwd=str(run_dir),
            env=_driver_env(),
            capture_output=True, text=True,
            timeout=CONFIG["subprocess_timeout_s"],
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {CONFIG['subprocess_timeout_s']}s"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        return False, "driver exited %d:\n    %s" % (proc.returncode, "\n    ".join(tail))
    return True, "ok"


# =============================================================================
# Mn extraction from MMD-S-<sec>.dat via mwd_analyzerv2
# =============================================================================
def find_mmd_file(run_dir: Path, t_s: int) -> Path | None:
    """MMD lands either directly in run_dir or in the per-sim subfolder
    mcPolymerSimulationresults-ID_<id>/ (assumption A2).  We search exactly those
    two levels — NOT rglob — so a K=1 run (run_dir = experiment folder) never
    picks up stale MMDs from sibling rep*/ dirs left by a prior K>1 verify run.
    If several match (should not, after cleaning), take the newest."""
    name = CONFIG["mmd_glob"].format(t=t_s)
    cands = list(run_dir.glob(name)) + list(run_dir.glob(f"mcPolymerSimulationresults-ID_*/{name}"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def mn_from_mmd(path: Path) -> dict:
    df2 = mwd.load_first_two_numeric_cols(path)
    df_xy, swapped = mwd.choose_xy_as_logm(df2, force=CONFIG["force_x_col"])
    m = mwd.compute_mwd_metrics(df_xy)
    return {"Mn": m["Mn_gmol"], "Mw": m["Mw_gmol"], "D": m["D"],
            "peak_M": m["peak_M_gmol"], "swapped": swapped}


def extract_metrics(run_dir: Path, times_s) -> dict:
    """{t_s: {Mn,Mw,D,...}} for every requested export time; raises if any missing."""
    out = {}
    for t in times_s:
        p = find_mmd_file(run_dir, t)
        if p is None:
            raise FileNotFoundError(f"no MMD-S-{t}.dat under {run_dir}")
        out[int(t)] = mn_from_mmd(p)
    return out


# =============================================================================
# evaluate one theta  (parallel fan-out across folders x replicates)
# =============================================================================
def coeffs_for(exp: Experiment, params: dict) -> dict:
    """params = {A_ki, Ea_ki, A_kp, Ea_kp} (Ea in J/mol) -> {ki, kp} at exp's T."""
    return {
        "ki": arrhenius_k(params["A_ki"], params["Ea_ki"], exp.temperature_C),
        "kp": arrhenius_k(params["A_kp"], params["Ea_kp"], exp.temperature_C),
    }


def run_all(exps, coeffs_by_code: dict, numMolecules: int, K: int) -> dict:
    """Fan out: for each experiment, run K replicates, average Mn/Mw/D per time.

    coeffs_by_code: {code: {"ki":.., "kp":..}}
    Returns {code: {"per_time": {t: {Mn,Mw,D}}, "reps": [...], "errors": [...]}}.
    """
    # build the flat task list: (exp, rep_index, run_dir)
    tasks = []
    for exp in exps:
        folder = experiment_folder(exp)
        for r in range(K):
            run_dir = folder if K == 1 else (folder / f"rep{r:02d}")
            _ensure_rundir(exp, run_dir, numMolecules, coeffs_by_code[exp.code])
            tasks.append((exp, r, run_dir))

    results = {exp.code: {"per_time_reps": [], "errors": []} for exp in exps}

    max_workers = max(1, min(CONFIG["max_workers"], len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2task = {pool.submit(run_driver, rd): (exp, r, rd) for (exp, r, rd) in tasks}
        for fut in as_completed(fut2task):
            exp, r, rd = fut2task[fut]
            ok, msg = fut.result()
            if not ok:
                results[exp.code]["errors"].append(f"rep{r}: {msg}")
                LOG.warning("[run] %s rep%d FAILED: %s", exp.code, r, msg)
                continue
            try:
                metrics = extract_metrics(rd, exp.fit_times_s)
                results[exp.code]["per_time_reps"].append(metrics)
            except Exception as e:
                results[exp.code]["errors"].append(f"rep{r}: extract: {e}")
                LOG.warning("[run] %s rep%d extract FAILED: %s", exp.code, r, e)

    # average replicates
    for exp in exps:
        reps = results[exp.code]["per_time_reps"]
        per_time = {}
        if reps:
            for t in exp.fit_times_s:
                Mns = [rep[t]["Mn"] for rep in reps if t in rep]
                Mws = [rep[t]["Mw"] for rep in reps if t in rep]
                Ds = [rep[t]["D"] for rep in reps if t in rep]
                per_time[t] = {
                    "Mn": float(np.mean(Mns)), "Mn_std": float(np.std(Mns, ddof=0)),
                    "Mw": float(np.mean(Mws)),
                    "D": float(np.mean(Ds)),
                    "n_rep": len(Mns),
                }
        results[exp.code]["per_time"] = per_time
    return results


# =============================================================================
# objective / loss
# =============================================================================
def residual(mn_sim: float, mn_exp: float) -> float:
    if CONFIG["residual_kind"] == "log":
        return math.log(mn_sim) - math.log(mn_exp)
    return (mn_sim - mn_exp) / mn_exp


def per_experiment_sse(exp: Experiment, per_time: dict) -> float:
    """Mean squared residual over this experiment's fit times."""
    rs = []
    for t in exp.fit_times_s:
        if t not in per_time:
            return float("nan")
        rs.append(residual(per_time[t]["Mn"], exp.Mn_by_time[t]))
        if CONFIG["use_dispersity_term"] and t in exp.D_by_time and CONFIG["dispersity_weight"] > 0:
            d_sim = per_time[t]["D"]
            d_exp = exp.D_by_time[t]
            rs.append(CONFIG["dispersity_weight"] * (math.log(d_sim) - math.log(d_exp)))
    return float(np.mean(np.square(rs)))


def aggregate_loss(exps, results) -> tuple:
    """Return (total_loss, breakdown{code: sse})."""
    breakdown = {}
    failed = False
    for exp in exps:
        pt = results[exp.code]["per_time"]
        sse = per_experiment_sse(exp, pt) if pt else float("nan")
        if not math.isfinite(sse):
            failed = True
        breakdown[exp.code] = sse

    if failed:
        # a robust penalty keeps the global optimizer moving instead of crashing
        finite = [v for v in breakdown.values() if math.isfinite(v)]
        base = (max(finite) if finite else 0.0)
        total = base + CONFIG["penalty_loss"]
        return total, breakdown

    if CONFIG["loss_normalization"] == "per_temperature":
        by_T = {}
        for exp in exps:
            by_T.setdefault(exp.temperature_C, []).append(breakdown[exp.code])
        total = float(np.mean([np.mean(v) for v in by_T.values()]))
    else:  # per_experiment (default)
        total = float(np.mean([breakdown[e.code] for e in exps]))
    return total, breakdown


# --- evaluation log ---------------------------------------------------------
class EvalLogger:
    def __init__(self, path: Path, codes):
        self.path = path
        self.codes = list(codes)
        self.n = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eval", "stage", "wall_s", "total_loss"]
                       + [f"A_ki", "Ea_ki_kJ", "A_kp", "Ea_kp_kJ"]
                       + [f"sse_{c}" for c in self.codes])

    def log(self, stage: str, wall_s: float, total_loss: float, params: dict, breakdown: dict):
        self.n += 1
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.n, stage, f"{wall_s:.1f}", f"{total_loss:.6g}",
                        f"{params.get('A_ki', float('nan')):.6g}",
                        f"{params.get('Ea_ki', float('nan'))/1000:.4f}",
                        f"{params.get('A_kp', float('nan')):.6g}",
                        f"{params.get('Ea_kp', float('nan'))/1000:.4f}"]
                       + [f"{breakdown.get(c, float('nan')):.6g}" for c in self.codes])


def evaluate_params(exps, params: dict, numMolecules: int, K: int,
                    stage: str, evallog: EvalLogger | None) -> tuple:
    """Full pipeline for one Arrhenius parameter set: fan-out -> Mn -> loss."""
    t0 = time.time()
    coeffs_by_code = {e.code: coeffs_for(e, params) for e in exps}
    results = run_all(exps, coeffs_by_code, numMolecules, K)
    total, breakdown = aggregate_loss(exps, results)
    wall = time.time() - t0
    if evallog is not None:
        evallog.log(stage, wall, total, params, breakdown)
    LOG.info("[eval:%s] loss=%.5g  (%.1fs)  A_ki=%.3e Ea_ki=%.1f kJ  A_kp=%.3e Ea_kp=%.1f kJ",
             stage, total, wall, params["A_ki"], params["Ea_ki"]/1000,
             params["A_kp"], params["Ea_kp"]/1000)
    return total, breakdown, results


# =============================================================================
# Stage 1 — decoupled per-temperature k warm start  (--stage1)
# =============================================================================
def stage1_level_loss(exps_at_T, ki: float, kp: float, numMolecules: int, K: int) -> tuple:
    """Loss for a single temperature level given effective (ki, kp) directly."""
    coeffs_by_code = {e.code: {"ki": ki, "kp": kp} for e in exps_at_T}
    results = run_all(exps_at_T, coeffs_by_code, numMolecules, K)
    total, breakdown = aggregate_loss(exps_at_T, results)
    return total, breakdown


def stage1_fit_level(exps_at_T, T_C: float, numMolecules: int, K: int) -> dict:
    """Fit effective ki, kp for one temperature level (2D global search in log10 k)."""
    lo, hi = CONFIG["stage1_k_bounds_log10"]

    def obj(x):
        ki = 10.0 ** x[0]
        kp = 10.0 ** x[1]
        loss, _ = stage1_level_loss(exps_at_T, ki, kp, numMolecules, K)
        LOG.debug("[stage1 T=%.0f] ki=%.4g kp=%.4g loss=%.5g", T_C, ki, kp, loss)
        return loss

    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for stage1/stage2 optimization")

    LOG.info("[stage1] fitting effective (ki,kp) at T=%.1f C over %d experiments",
             T_C, len(exps_at_T))
    res = differential_evolution(
        obj, bounds=[(lo, hi), (lo, hi)],
        maxiter=CONFIG["stage1_maxiter"], popsize=CONFIG["stage1_popsize"],
        tol=CONFIG["stage1_tol"], seed=CONFIG["seed"], polish=False,
        mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
    )
    ki = 10.0 ** res.x[0]
    kp = 10.0 ** res.x[1]
    LOG.info("[stage1] T=%.1f C -> ki=%.5g  kp=%.5g  (loss=%.5g)", T_C, ki, kp, res.fun)
    return {"T_C": T_C, "ki": ki, "kp": kp, "loss": float(res.fun)}


def arrhenius_seed_from_two_levels(level_lo: dict, level_hi: dict) -> dict:
    """Analytic Arrhenius parameters from k at two temperatures (assumes T_hi>T_lo).
       Ea = R ln(k_hi/k_lo) / (1/T_lo - 1/T_hi);  A = k_hi exp(Ea/(R T_hi))."""
    R = CONFIG["R"]
    Tlo = T_K(level_lo["T_C"]); Thi = T_K(level_hi["T_C"])
    inv = (1.0 / Tlo - 1.0 / Thi)
    seed = {}
    for j in ("ki", "kp"):
        k_lo = level_lo[j]; k_hi = level_hi[j]
        Ea = R * math.log(k_hi / k_lo) / inv           # J/mol
        A = k_hi * math.exp(Ea / (R * Thi))
        seed[f"A_{j}"] = A
        seed[f"Ea_{j}"] = Ea
    return seed


def run_stage1(exps, numMolecules: int, K: int) -> dict:
    temps = temperatures(exps)
    if len(temps) < 2:
        raise RuntimeError("stage1 needs >= 2 temperature levels")
    levels = {}
    for T in temps:
        exps_at_T = [e for e in exps if e.temperature_C == T]
        levels[T] = stage1_fit_level(exps_at_T, T, numMolecules, K)
    T_lo, T_hi = temps[0], temps[-1]
    seed = arrhenius_seed_from_two_levels(levels[T_lo], levels[T_hi])
    out = {"levels": {str(T): levels[T] for T in temps},
           "arrhenius_seed": seed}
    LOG.info("[stage1] Arrhenius seed: A_ki=%.4e Ea_ki=%.1f kJ  A_kp=%.4e Ea_kp=%.1f kJ",
             seed["A_ki"], seed["Ea_ki"]/1000, seed["A_kp"], seed["Ea_kp"]/1000)
    return out


# =============================================================================
# Stage 2 — coupled global Arrhenius refinement  (--stage2)
# =============================================================================
def x_to_params(x) -> dict:
    """x = [log10 A_ki, Ea_ki(kJ), log10 A_kp, Ea_kp(kJ)] -> params (Ea in J/mol)."""
    return {
        "A_ki": 10.0 ** x[0], "Ea_ki": x[1] * 1000.0,
        "A_kp": 10.0 ** x[2], "Ea_kp": x[3] * 1000.0,
    }


def params_to_x(params: dict):
    return [math.log10(params["A_ki"]), params["Ea_ki"] / 1000.0,
            math.log10(params["A_kp"]), params["Ea_kp"] / 1000.0]


def stage2_bounds():
    la, ha = CONFIG["log10A_bounds"]
    le, he = CONFIG["Ea_bounds_kJ"]
    return [(la, ha), (le, he), (la, ha), (le, he)]


def run_stage2(exps, numMolecules: int, K: int, x0=None, evallog=None) -> dict:
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for stage2 optimization")
    bounds = stage2_bounds()

    def obj(x):
        params = x_to_params(x)
        total, _, _ = evaluate_params(exps, params, numMolecules, K, "stage2", evallog)
        return total

    optimizer = CONFIG["stage2_optimizer"]
    LOG.info("[stage2] optimizer=%s  bounds=%s", optimizer, bounds)

    if optimizer == "cma":
        best = _run_cma(obj, bounds, x0)
    else:
        # seed the initial DE population around the warm start (if provided)
        init = _seeded_population(bounds, x0, CONFIG["stage2_popsize"] * 4)
        res = differential_evolution(
            obj, bounds=bounds,
            maxiter=CONFIG["stage2_maxiter"], popsize=CONFIG["stage2_popsize"],
            tol=CONFIG["stage2_tol"], seed=CONFIG["seed"], polish=False,
            mutation=tuple(CONFIG["stage2_mutation"]),
            recombination=CONFIG["stage2_recombination"],
            updating="deferred",
            init=init if init is not None else "sobol",
        )
        best = {"x": list(res.x), "fun": float(res.fun), "nit": int(getattr(res, "nit", -1))}

    params = x_to_params(best["x"])
    LOG.info("[stage2] BEST loss=%.5g  A_ki=%.4e Ea_ki=%.1f kJ  A_kp=%.4e Ea_kp=%.1f kJ",
             best["fun"], params["A_ki"], params["Ea_ki"]/1000,
             params["A_kp"], params["Ea_kp"]/1000)
    return {"params": params, "loss": best["fun"], "x": best["x"]}


def _seeded_population(bounds, x0, n):
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(CONFIG["seed"])
    if x0 is None:
        return None
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
    span = (ub - lb)
    pop = x0[None, :] + 0.15 * span[None, :] * rng.standard_normal((n, len(bounds)))
    pop[0] = x0                       # keep the exact warm start
    return np.clip(pop, lb, ub)


def _run_cma(obj, bounds, x0):
    import cma  # optional dependency
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    span = ub - lb
    if x0 is None:
        x0 = (lb + ub) / 2.0
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
    # optimize in a normalized [0,1] box for good CMA conditioning
    def norm(x):  return (np.asarray(x) - lb) / span
    def denorm(z): return lb + np.clip(np.asarray(z), 0, 1) * span
    es = cma.CMAEvolutionStrategy(
        list(norm(x0)), CONFIG["cma_sigma0"],
        {"bounds": [0, 1], "seed": CONFIG["seed"],
         "maxiter": CONFIG["stage2_maxiter"], "verbose": -9})
    while not es.stop():
        zs = es.ask()
        es.tell(zs, [obj(denorm(z)) for z in zs])
    z = es.result.xbest
    return {"x": list(denorm(z)), "fun": float(es.result.fbest), "nit": int(es.result.iterations)}


# =============================================================================
# sensitivity screen + noise floor  (--screen)
# =============================================================================
def run_screen(exps, params: dict, numMolecules: int) -> dict:
    """Perturb ki and kp about the nominal params; quantify d ln Mn / d ln k, and
    measure the stochastic Mn noise floor from replicates."""
    f = CONFIG["screen_perturb_factor"]
    base = {e.code: coeffs_for(e, params) for e in exps}

    def eval_with(mod_key=None, factor=1.0):
        cb = {}
        for e in exps:
            c = dict(base[e.code])
            if mod_key is not None:
                c[mod_key] = c[mod_key] * factor
            cb[e.code] = c
        return run_all(exps, cb, numMolecules, K=1)

    LOG.info("[screen] baseline evaluation")
    res_base = eval_with()

    sens = {}
    for key in ("ki", "kp"):
        LOG.info("[screen] perturbing %s by x%.2f and /%.2f", key, f, f)
        res_up = eval_with(key, f)
        res_dn = eval_with(key, 1.0 / f)
        # central finite difference of ln Mn wrt ln k, averaged over exps & times
        slopes = []
        for e in exps:
            for t in e.fit_times_s:
                try:
                    mup = res_up[e.code]["per_time"][t]["Mn"]
                    mdn = res_dn[e.code]["per_time"][t]["Mn"]
                    slopes.append((math.log(mup) - math.log(mdn)) / (2 * math.log(f)))
                except Exception:
                    pass
        # also isolate the 10-min (earliest) sensitivity — where ki lives
        t0 = min(all_fit_times(exps))
        slopes_t0 = []
        for e in exps:
            try:
                mup = res_up[e.code]["per_time"][t0]["Mn"]
                mdn = res_dn[e.code]["per_time"][t0]["Mn"]
                slopes_t0.append((math.log(mup) - math.log(mdn)) / (2 * math.log(f)))
            except Exception:
                pass
        sens[key] = {
            "d_lnMn_d_lnk_mean": float(np.mean(slopes)) if slopes else float("nan"),
            "d_lnMn_d_lnk_earliest": float(np.mean(slopes_t0)) if slopes_t0 else float("nan"),
        }

    # noise floor: replicate scatter at the baseline
    LOG.info("[screen] noise floor from %d replicates", CONFIG["screen_noise_reps"])
    res_reps = run_all(exps, base, numMolecules, K=CONFIG["screen_noise_reps"])
    rel_stds = []
    for e in exps:
        for t in e.fit_times_s:
            pt = res_reps[e.code]["per_time"].get(t)
            if pt and pt["Mn"] > 0:
                rel_stds.append(pt["Mn_std"] / pt["Mn"])
    noise_floor = float(np.mean(rel_stds)) if rel_stds else float("nan")

    LOG.info("[screen] d lnMn/d lnkp (mean)=%.3f ; d lnMn/d lnki (mean)=%.3f (earliest=%.3f)",
             sens["kp"]["d_lnMn_d_lnk_mean"], sens["ki"]["d_lnMn_d_lnk_mean"],
             sens["ki"]["d_lnMn_d_lnk_earliest"])
    LOG.info("[screen] Mn stochastic noise floor (relative std) ~ %.3g at numMol=%.1e",
             noise_floor, numMolecules)
    return {"sensitivity": sens, "noise_floor_rel": noise_floor,
            "numMolecules": numMolecules, "perturb_factor": f}


# =============================================================================
# verification  (--verify)
# =============================================================================
def run_verify(exps, params: dict) -> dict:
    K = CONFIG["K_replicates_verify"]
    nMol = CONFIG["numMolecules_verify"]
    LOG.info("[verify] %d replicates at numMolecules=%.1e", K, nMol)
    coeffs_by_code = {e.code: coeffs_for(e, params) for e in exps}
    results = run_all(exps, coeffs_by_code, nMol, K)
    total, breakdown = aggregate_loss(exps, results)
    LOG.info("[verify] verification loss=%.5g", total)
    return {"loss": total, "breakdown": breakdown, "results": results,
            "numMolecules": nMol, "K": K}


# =============================================================================
# plots + report
# =============================================================================
def _mn_table(exps, results):
    rows = []
    for e in exps:
        pt = results[e.code]["per_time"]
        for t in e.fit_times_s:
            sim = pt.get(t, {})
            rows.append({
                "code": e.code, "temperature_C": e.temperature_C, "time_s": t,
                "Mn_exp": e.Mn_by_time[t],
                "Mn_sim": sim.get("Mn", float("nan")),
                "Mn_sim_std": sim.get("Mn_std", float("nan")),
                "D_sim": sim.get("D", float("nan")),
                "n_rep": sim.get("n_rep", 0),
            })
    return pd.DataFrame(rows)


def make_plots(exps, results, params, stage1, out_dir: Path):
    if not _HAVE_MPL:
        LOG.warning("[plots] matplotlib not available; skipping plots")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # (1) Mn sim-vs-exp overlays
    n = len(exps)
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for i, e in enumerate(exps):
        ax = axes[i // ncol][i % ncol]
        ts = e.fit_times_s
        mn_exp = [e.Mn_by_time[t] for t in ts]
        pt = results[e.code]["per_time"]
        mn_sim = [pt.get(t, {}).get("Mn", float("nan")) for t in ts]
        mn_std = [pt.get(t, {}).get("Mn_std", 0.0) for t in ts]
        tmin = [t / 60 for t in ts]
        ax.plot(tmin, mn_exp, "o-", label="exp", color="k")
        ax.errorbar(tmin, mn_sim, yerr=mn_std, fmt="s--", label="sim", color="C0", capsize=3)
        ax.set_title(f"{e.code}  ({e.temperature_C:.0f} C)")
        ax.set_xlabel("t (min)"); ax.set_ylabel("Mn (g/mol)")
        ax.legend(fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    p = out_dir / "mn_overlays.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # (2) residuals
    tbl = _mn_table(exps, results)
    fig, ax = plt.subplots(figsize=(7, 4))
    for e in exps:
        sub = tbl[tbl["code"] == e.code]
        res = np.log(sub["Mn_sim"]) - np.log(sub["Mn_exp"])
        ax.plot(sub["time_s"] / 60, res, "o-", label=e.code)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("t (min)"); ax.set_ylabel("ln(Mn_sim) - ln(Mn_exp)")
    ax.set_title("Mn residuals"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = out_dir / "residuals.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # (3) Arrhenius plot
    if params is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        temps = temperatures(exps)
        invT = np.linspace(1 / T_K(max(temps)) * 0.98, 1 / T_K(min(temps)) * 1.02, 50)
        for j, col in (("ki", "C0"), ("kp", "C1")):
            A = params[f"A_{j}"]; Ea = params[f"Ea_{j}"]
            lnk = np.log(A) - Ea / CONFIG["R"] * invT
            ax.plot(invT * 1000, lnk, "-", color=col, label=f"{j} fit")
            if stage1 is not None:
                for T in temps:
                    lv = stage1["levels"][str(T)]
                    ax.plot(1000 / T_K(T), math.log(lv[j]), "o", color=col)
        ax.set_xlabel("1000/T (1/K)"); ax.set_ylabel("ln k")
        ax.set_title("Arrhenius (lines=fit, markers=stage-1 k)")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "arrhenius.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # (4) convergence history
    log_csv = out_dir.parent / "eval_log.csv"
    if log_csv.exists():
        try:
            ev = pd.read_csv(log_csv)
            ev2 = ev[ev["stage"] == "stage2"]
            if not ev2.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                best = ev2["total_loss"].cummin()
                ax.plot(ev2["eval"], ev2["total_loss"], ".", alpha=0.4, label="eval")
                ax.plot(ev2["eval"], best, "-", color="C3", label="best-so-far")
                ax.set_yscale("log")
                ax.set_xlabel("evaluation"); ax.set_ylabel("total loss")
                ax.set_title("Stage-2 convergence"); ax.legend()
                fig.tight_layout()
                p = out_dir / "convergence.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
        except Exception as e:
            LOG.warning("[plots] convergence plot failed: %s", e)

    LOG.info("[plots] wrote %d figures to %s", len(paths), out_dir)
    return paths


def write_report(exps, params, stage1, screen, verify, out_dir: Path, run_dir: Path):
    temps = temperatures(exps)
    # implied k at each temperature
    k_at_T = {}
    if params is not None:
        for T in temps:
            k_at_T[T] = {j: arrhenius_k(params[f"A_{j}"], params[f"Ea_{j}"], T)
                         for j in ("ki", "kp")}

    lines = []
    lines.append("# mcPolymer Arrhenius fit — report\n")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')} · "
                 f"git {git_commit()[:10]}_\n")

    lines.append("## Best-fit Arrhenius parameters\n")
    if params is not None:
        lines.append("| coeff | A | Ea (kJ/mol) |")
        lines.append("|---|---|---|")
        lines.append(f"| ki | {params['A_ki']:.4e} | {params['Ea_ki']/1000:.2f} |")
        lines.append(f"| kp | {params['A_kp']:.4e} | {params['Ea_kp']/1000:.2f} |\n")
        lines.append("### Implied rate coefficients\n")
        lines.append("| T (°C) | ki | kp |")
        lines.append("|---|---|---|")
        for T in temps:
            lines.append(f"| {T:.0f} | {k_at_T[T]['ki']:.5g} | {k_at_T[T]['kp']:.5g} |")
        lines.append("")
    else:
        lines.append("_No fitted parameters yet (run --stage1/--stage2)._\n")

    lines.append("## Effective-initiator / titer diagnostic\n")
    lines.append("n_I,eff is set from the 720-min Mn (n_I,eff = 4.0000 g / Mn₇₂₀), so each "
                 "simulation reproduces its final Mn by construction; the 10–60 min Mn shape "
                 "is the kinetic signal.\n")
    lines.append("| code | T (°C) | Mn₇₂₀ | n_I,eff (mol) | n_charged (mol) | eff/charged |")
    lines.append("|---|---|---|---|---|---|")
    for e in exps:
        lines.append(f"| {e.code} | {e.temperature_C:.0f} | {e.Mn720:.0f} | "
                     f"{e.n_sbuli_eff_mol:.4e} | {e.n_sbuli_charged_mol:.4e} | "
                     f"{100*e.titer_ratio:.1f}% |")
    lines.append("\nEfficiencies straddling 100% (some >100%, impossible for a living ki/kp "
                 "model) confirm charged-titer / SEC-calibration scatter, not kinetics — which "
                 "is exactly why the charged titer is discarded and n_I,eff is used instead.\n")

    if stage1 is not None:
        lines.append("## Stage 1 — decoupled per-temperature effective k (warm start)\n")
        lines.append("| T (°C) | ki_eff | kp_eff | level loss |")
        lines.append("|---|---|---|---|")
        for T in temps:
            lv = stage1["levels"][str(T)]
            lines.append(f"| {T:.0f} | {lv['ki']:.5g} | {lv['kp']:.5g} | {lv['loss']:.4g} |")
        s = stage1["arrhenius_seed"]
        lines.append(f"\nAnalytic Arrhenius seed → A_ki={s['A_ki']:.3e}, "
                     f"Ea_ki={s['Ea_ki']/1000:.1f} kJ/mol; A_kp={s['A_kp']:.3e}, "
                     f"Ea_kp={s['Ea_kp']/1000:.1f} kJ/mol.\n")

    lines.append("## Identifiability / sensitivity\n")
    if screen is not None:
        sk = screen["sensitivity"]
        lines.append(f"- Noise floor (relative Mn std) ≈ **{screen['noise_floor_rel']:.2%}** at "
                     f"numMolecules={screen['numMolecules']:.1e} → set optimizer tolerance above this.")
        lines.append(f"- d ln Mn / d ln kp (all times) = **{sk['kp']['d_lnMn_d_lnk_mean']:.3f}** "
                     f"→ kp is strongly identified by the Mn growth rate.")
        lines.append(f"- d ln Mn / d ln ki (all times) = **{sk['ki']['d_lnMn_d_lnk_mean']:.3f}**, "
                     f"earliest (10 min) = **{sk['ki']['d_lnMn_d_lnk_earliest']:.3f}** "
                     f"→ ki only shifts the earliest Mn.")
        ki_slope = abs(sk["ki"]["d_lnMn_d_lnk_earliest"])
        nf = screen["noise_floor_rel"]
        if math.isfinite(ki_slope) and ki_slope > 0 and math.isfinite(nf):
            resolvable = nf / ki_slope
            lines.append(f"- Practical ki resolution: a change in ki is only resolvable if it moves "
                         f"the 10-min Mn by more than the ~{nf:.2%} noise floor, i.e. |Δln ki| ≳ "
                         f"**{resolvable:.2f}** (≈ ×{math.exp(resolvable):.2f}). Below the transient "
                         f"threshold ki is effectively unidentified.\n")
    else:
        lines.append("_Run --screen to quantify ki/kp sensitivity and the noise floor._\n")
    lines.append("**Recommendation:** with Mn-only data ki is weakly identified (early-time "
                 "transient only). If a dispersity (Đ = Mw/Mn) or Mw time series is available, "
                 "add it as a second observable (hook already present: set "
                 "`use_dispersity_term=True`, `dispersity_weight>0`, and add an `exp_D` column) — "
                 "Đ carries the initiation-broadening signal and strongly constrains ki.\n")

    if verify is not None:
        lines.append("## Verification (high-resolution replicates)\n")
        lines.append(f"Re-ran the best fit K={verify['K']} times at numMolecules="
                     f"{verify['numMolecules']:.1e}; verification loss = {verify['loss']:.5g}.\n")
        lines.append("| code | time (min) | Mn_exp | Mn_sim | ±std | Đ_sim |")
        lines.append("|---|---|---|---|---|---|")
        for e in exps:
            pt = verify["results"][e.code]["per_time"]
            for t in e.fit_times_s:
                s = pt.get(t, {})
                lines.append(f"| {e.code} | {t//60} | {e.Mn_by_time[t]:.0f} | "
                             f"{s.get('Mn', float('nan')):.0f} | {s.get('Mn_std', 0):.0f} | "
                             f"{s.get('D', float('nan')):.3f} |")
        lines.append("")

    lines.append("## Method notes & caveats\n")
    lines.append("- **Objective:** squared residuals on **Mn** at 10/20/40/60 min, "
                 f"`{CONFIG['residual_kind']}` scale, normalized `{CONFIG['loss_normalization']}` "
                 "so the four 30 °C experiments do not swamp the two 20 °C experiments. The "
                 "720-min point is excluded from the residuals (used only to set n_I,eff).")
    lines.append("- **Conversion is a diagnostic only** (X(t)=Mn(t)/Mn₇₂₀ in the data carries no "
                 "information independent of Mn); the simulated monomer-balance conversion in "
                 "`sim_conversion.csv` is never scored.")
    lines.append("- **Two temperatures cannot test the Arrhenius form.** With exactly two T "
                 "levels, (A, Ea) is an *exact reparametrization* of (k(20 °C), k(30 °C)); the fit "
                 "*assumes* Arrhenius. Acquiring a **third temperature validates the form** — and "
                 "adding it here is just more rows in the CSV + more folders, no code changes.")
    lines.append("- **Coefficient injection** uses the template rewrite (assumption A1); "
                 "**MMD files** are read recursively from the per-sim subfolder (A2); "
                 "**conversion** via monomer balance (A3).")
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    LOG.info("[report] wrote %s", report_path)
    return report_path


def save_best_params(params, breakdown, run_dir: Path):
    if params is None:
        return
    j = {
        "A_ki": params["A_ki"], "Ea_ki_J": params["Ea_ki"], "Ea_ki_kJ": params["Ea_ki"] / 1000,
        "A_kp": params["A_kp"], "Ea_kp_J": params["Ea_kp"], "Ea_kp_kJ": params["Ea_kp"] / 1000,
        "loss_breakdown": breakdown or {},
    }
    (run_dir / "best_params.json").write_text(json.dumps(j, indent=2))
    with open(run_dir / "best_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A_ki", "Ea_ki_kJ", "A_kp", "Ea_kp_kJ"])
        w.writerow([params["A_ki"], params["Ea_ki"] / 1000,
                    params["A_kp"], params["Ea_kp"] / 1000])
    LOG.info("[out] wrote best_params.json/csv to %s", run_dir)


# =============================================================================
# run-directory / manifest bookkeeping
# =============================================================================
def new_run_dir() -> Path:
    # microsecond suffix so back-to-back invocations never share (and clobber) a dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d = Path(CONFIG["repo_dir"]) / CONFIG["results_root"] / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_dir: Path, args):
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": (getattr(__import__("scipy"), "__version__", "n/a") if _HAVE_SCIPY else "absent"),
        "argv": vars(args),
        "config": CONFIG,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_latest_stage1():
    """Best-effort: read the most recent stage1 result for warm-starting stage2."""
    root = Path(CONFIG["repo_dir"]) / CONFIG["results_root"]
    if not root.exists():
        return None
    for d in sorted(root.iterdir(), reverse=True):
        p = d / "stage1.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def load_latest_best_params():
    root = Path(CONFIG["repo_dir"]) / CONFIG["results_root"]
    if not root.exists():
        return None
    for d in sorted(root.iterdir(), reverse=True):
        p = d / "best_params.json"
        if p.exists():
            try:
                j = json.loads(p.read_text())
                return {"A_ki": j["A_ki"], "Ea_ki": j["Ea_ki_J"],
                        "A_kp": j["A_kp"], "Ea_kp": j["Ea_kp_J"]}
            except Exception:
                continue
    return None


def default_params_from_model() -> dict:
    """Fallback theta from the model file's nominal ki=0.25, kp=0.20 at the mean T.
    Uses Ea=0 so k(T)=A=nominal — a neutral starting point when no stage1 exists."""
    return {"A_ki": 0.25, "Ea_ki": 0.0, "A_kp": 0.20, "Ea_kp": 0.0}


# =============================================================================
# CLI
# =============================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Automated Arrhenius fitting for mcPolymer kMC simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--setup", action="store_true",
                    help="build one folder per experiment (+ experiment.json)")
    ap.add_argument("--single", metavar="CODE",
                    help="run ONE experiment folder once and print Mn(t) (validation)")
    ap.add_argument("--eval-once", action="store_true",
                    help="one parallel fan-out for a fixed theta -> single global loss")
    ap.add_argument("--screen", action="store_true",
                    help="ki/kp sensitivity + stochastic noise floor")
    ap.add_argument("--stage1", action="store_true",
                    help="decoupled per-temperature k warm start + Arrhenius seed")
    ap.add_argument("--stage2", action="store_true",
                    help="coupled global Arrhenius refinement")
    ap.add_argument("--verify", action="store_true",
                    help="re-run best fit K times at verification resolution")
    ap.add_argument("--report", action="store_true",
                    help="(re)write report.md + plots from the latest results")
    ap.add_argument("--all", action="store_true",
                    help="setup -> stage1 -> stage2 -> verify -> report")
    ap.add_argument("--numMolecules", type=float, default=None,
                    help="override fitting numMolecules for this run")
    ap.add_argument("--reps", type=int, default=None,
                    help="override K replicates for fitting")
    ap.add_argument("--workers", type=int, default=None,
                    help="override max concurrent driver processes")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.numMolecules is not None:
        CONFIG["numMolecules_fit"] = int(args.numMolecules)
    if args.reps is not None:
        CONFIG["K_replicates_fit"] = int(args.reps)
    if args.workers is not None:
        CONFIG["max_workers"] = int(args.workers)

    run_dir = new_run_dir()
    setup_logging(run_dir / "run.log", verbose=True)
    write_manifest(run_dir, args)
    LOG.info("results dir: %s", run_dir)

    data_csv = repo_path(CONFIG["data_csv"])
    if not data_csv.exists():
        LOG.error("data CSV not found: %s", data_csv)
        return 2
    exps = load_experiments(data_csv)
    LOG.info("loaded %d experiments: %s", len(exps), ", ".join(e.code for e in exps))
    LOG.info("temperatures: %s C ; fit times: %s s",
             temperatures(exps), all_fit_times(exps))

    evallog = EvalLogger(run_dir / "eval_log.csv", [e.code for e in exps])
    nMol_fit = CONFIG["numMolecules_fit"]
    K_fit = CONFIG["K_replicates_fit"]

    did_something = False
    stage1 = None
    best = None
    screen = None
    verify = None
    last_results = None
    last_breakdown = None

    # nothing selected -> show help
    if not any([args.setup, args.single, args.eval_once, args.screen,
                args.stage1, args.stage2, args.verify, args.report, args.all]):
        build_argparser().print_help()
        return 0

    # ---- setup --------------------------------------------------------------
    if args.setup or args.all:
        setup_folders(exps, nMol_fit)
        did_something = True

    # ---- single folder validation ------------------------------------------
    if args.single:
        code = args.single
        exp = next((e for e in exps if e.code == code), None)
        if exp is None:
            LOG.error("unknown code %s", code); return 2
        setup_folders([exp], nMol_fit)
        # use the model's nominal coefficients for a smoke test
        cb = {exp.code: {"ki": 0.25, "kp": 0.20}}
        LOG.info("[single] running %s once with nominal ki=0.25 kp=0.20 ...", code)
        res = run_all([exp], cb, nMol_fit, K=1)
        pt = res[code]["per_time"]
        if res[code]["errors"]:
            LOG.error("[single] errors: %s", res[code]["errors"]); return 1
        for t in exp.fit_times_s:
            LOG.info("[single] %s t=%4ds  Mn_sim=%.0f  Mn_exp=%.0f  D=%.3f",
                     code, t, pt[t]["Mn"], exp.Mn_by_time[t], pt[t]["D"])
        did_something = True

    # ---- stage 1 ------------------------------------------------------------
    if args.stage1 or args.all:
        stage1 = run_stage1(exps, nMol_fit, K_fit)
        (run_dir / "stage1.json").write_text(json.dumps(stage1, indent=2))
        best = dict(stage1["arrhenius_seed"])
        did_something = True
    else:
        stage1 = load_latest_stage1()

    # ---- one fixed-theta fan-out -------------------------------------------
    if args.eval_once:
        params = best or load_latest_best_params() or (
            stage1["arrhenius_seed"] if stage1 else default_params_from_model())
        total, breakdown, last_results = evaluate_params(
            exps, params, nMol_fit, K_fit, "eval_once", evallog)
        last_breakdown = breakdown
        LOG.info("[eval-once] GLOBAL LOSS = %.6g", total)
        did_something = True

    # ---- screen -------------------------------------------------------------
    if args.screen or args.all:
        params = best or (stage1["arrhenius_seed"] if stage1 else
                          load_latest_best_params() or default_params_from_model())
        screen = run_screen(exps, params, nMol_fit)
        (run_dir / "screen.json").write_text(json.dumps(screen, indent=2))
        did_something = True

    # ---- stage 2 ------------------------------------------------------------
    if args.stage2 or args.all:
        seed_params = best or (stage1["arrhenius_seed"] if stage1 else None) \
            or load_latest_best_params()
        x0 = params_to_x(seed_params) if seed_params else None
        st2 = run_stage2(exps, nMol_fit, K_fit, x0=x0, evallog=evallog)
        best = st2["params"]
        # one clean evaluation at the optimum to capture per-exp breakdown + Mn
        total, last_breakdown, last_results = evaluate_params(
            exps, best, nMol_fit, K_fit, "stage2_best", evallog)
        save_best_params(best, last_breakdown, run_dir)
        did_something = True
    elif best is None:
        best = load_latest_best_params()

    # ---- verify -------------------------------------------------------------
    if args.verify or args.all:
        params = best or load_latest_best_params()
        if params is None:
            LOG.error("[verify] no best params available (run --stage2 first)")
        else:
            verify = run_verify(exps, params)
            (run_dir / "verify.json").write_text(
                json.dumps({"loss": verify["loss"], "breakdown": verify["breakdown"],
                            "numMolecules": verify["numMolecules"], "K": verify["K"]}, indent=2))
            last_results = verify["results"]
            did_something = True

    # ---- report + plots -----------------------------------------------------
    if args.report or args.all or args.stage2 or args.verify or args.eval_once:
        if best is None:
            best = load_latest_best_params()
        if last_results is None and best is not None:
            _, last_breakdown, last_results = evaluate_params(
                exps, best, nMol_fit, K_fit, "report_eval", evallog)
        if last_results is not None:
            make_plots(exps, last_results, best, stage1, run_dir / "plots")
            write_report(exps, best, stage1, screen, verify, run_dir, run_dir)
            save_best_params(best, last_breakdown, run_dir)
        else:
            LOG.warning("[report] no simulation results available to plot/report")
        did_something = True

    if not did_something:
        build_argparser().print_help()
    LOG.info("done. outputs in %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
