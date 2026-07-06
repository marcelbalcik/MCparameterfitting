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
    python main.py --stage-ki       # (opt-in) determine ki from the 10-min MMD shape (stage1 < here < stage2)
    python main.py --stage-ki-mw    # (opt-in) determine ki from experimental Mw in the CSV (exp_Mw column)
    python main.py --stage2         # coupled Arrhenius refinement (holds ki fixed if a ki stage ran)
    python main.py --verify         # re-run best fit K times at verification resolution
    python main.py --all            # setup -> stage1 -> screen -> stage2 -> verify -> report  (NO ki stage)
    python main.py --all-ki         # same as --all but WITH the MMD ki stage
    python main.py --all-ki-mw      # same as --all but WITH the Mw ki stage (CSV exp_Mw)
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
    "charge_mass_g":       4.0000,            # (informational) nominal styrene charge; the n_I,eff
                                              # cross-check uses M0*n_styrene per experiment, not this

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
    # SEC band-broadening applied to the SIMULATED MMD (Gaussian in log10 M) before
    # computing Mn/Mw/Đ, so the sim is comparable to the broadened SEC data and the
    # breadth-based ki is unbiased. 0 = off. For the TW067 column set we estimated
    # ~0.063 from the internal-standard peak. Affects every Mn/Mw/Đ the code reads.
    "sec_broadening_sigma_log10M": 0.0,

    # ---- objective -----------------------------------------------------------
    # residual on Mn: "log"  -> ln(Mn_sim) - ln(Mn_exp)   (default; scale-free)
    #                 "rel"  -> (Mn_sim - Mn_exp)/Mn_exp
    "residual_kind":        "log",
    # weighting so 30 C (4 exps) does not swamp 20 C (2 exps):
    #   "per_experiment"  -> every experiment weighted equally (default)
    #   "per_temperature" -> average within a T level, then average the two levels
    "loss_normalization":   "per_experiment",
    # ---- JOINT objective: fit ki AND kp together (one method, symmetric) ------
    # When enabled, the SAME optimizer fits all four Arrhenius params against a
    # combined loss = Mn(t) residual + a breadth residual (Đ or Mw). ki is then
    # determined exactly like kp (one coupled regression), with the breadth term
    # supplying the ki constraint that Mn alone lacks. Needs exp_Mw (or exp_D) in
    # the CSV. Recommended path instead of the separate hard-fix ki stage:
    #   set joint_breadth_enable=True, run plain --stage2 / --all (fits all four).
    "joint_breadth_enable":     False,
    "joint_breadth_observable": "dispersity",  # "dispersity" (Đ=Mw/Mn) | "mw"
    "joint_breadth_weight":     2.0,     # weight of EACH breadth residual vs a single Mn residual
    "joint_breadth_times_s":    [600],   # times whose breadth to score (10 min carries the ki signal)
    "penalty_loss":         1.0e3,            # returned when a sim eval fails (keeps DE alive)

    # ---- parallelism ---------------------------------------------------------
    "max_workers":          6,                # concurrent driver processes (folders x reps)
    "subprocess_timeout_s": 7200,             # per driver run

    # ---- Stage 1: decoupled per-temperature k fit ---------------------------
    # Worst-case sim evaluations per temperature level = (maxiter+1)*popsize*2
    # (2-D: ki,kp). (14+1)*6*2 = 180 (< 200); it usually stops earlier via tol.
    "stage1_k_bounds_log10": [-6.0, 3.0],     # bounds on log10(k) [L/(mol s)] per coefficient
    "stage1_maxiter":        14,
    "stage1_popsize":        6,
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

    # ---- Stage 3 (MMD): fit ki from the early-time molar-mass DISTRIBUTION ---
    # You provide the experimental curve(s) as two-column log10(M) vs dw/dlog10(M)
    # files (same format as the sim output), one per experiment/time, named by
    # exp_mmd_pattern inside exp_mmd_dir, e.g. exp_mmd/TW60_MMD-600.dat.
    # Any experiment without a file is silently skipped (per-experiment optional).
    "exp_mmd_dir":           "exp_mmd",
    "exp_mmd_pattern":       "{code}_MMD-{t}.dat",
    "mmd_fit_times_s":       [600],            # 10 min; add more times to use them too
    "mmd_metric":            "l2",             # "l2" | "wasserstein" | "dispersity"
    "mmd_grid_points":       400,              # shared log10(M) grid resolution
    "mmd_force_x_col":       "auto",           # orientation of the EXP file (auto|c0|c1|swap)
    "mmd_mn_weight":         0.0,              # optional Mn term alongside the shape term (0=shape only)
    # This stage runs BETWEEN stage 1 and stage 2: it determines an effective
    # ki(T) per temperature from the MMD shape (holding kp(T) at the stage-1
    # value), THEN stage 2 sets the Arrhenius constants with ki held fixed.
    # With MMDs at >=2 temperatures Ea_ki is identifiable from the ki(T) points;
    # with one temperature Ea_ki is held at mmd_fixed_Ea_kJ (None -> the stage-1
    # seed's Ea_ki) and only ki(T) (hence A_ki) is determined.
    "mmd_fixed_Ea_kJ":       None,
    "mmd_ki_maxiter":        30,
    "mmd_ki_popsize":        12,
    "mmd_ki_tol":            1e-2,

    # ---- Stage-ki (Mw): fit ki from experimental Mw in the CSV ----------------
    # ALTERNATIVE to the MMD stage when you have Mw but not full curves. Add an
    # `exp_Mw` column (g/mol) to experimental_data.csv on the fit rows; this stage
    # determines ki(T) per temperature so simulated Mw matches, holding kp(T) at
    # the stage-1 value, then ki(T) -> Arrhenius (fed to stage 2 as a fixed ki).
    "mw_ki_times_s":         [600],            # which times' Mw to use (10 min carries the ki signal)
    "mw_metric":             "mw",             # "mw" (fit Mw) | "dispersity" (fit Đ=Mw/Mn, scale-free)
    "mw_fixed_Ea_kJ":        None,             # 1-temperature fallback (None -> stage-1 seed Ea_ki)
    "mw_ki_maxiter":         20,
    "mw_ki_popsize":         8,
    "mw_ki_tol":             1e-2,

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
                 Mn720: float, D_by_time: dict | None = None,
                 Mw_by_time: dict | None = None):
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
        self.Mw_by_time = Mw_by_time or {}   # experimental Mw(t) from the CSV `exp_Mw` column

    @property
    def titer_ratio(self) -> float:
        return self.n_sbuli_eff_mol / self.n_sbuli_charged_mol

    def n_eff_check(self) -> float:
        """Cross-check: n_I,eff should equal the monomer MASS / Mn720, where the
        monomer mass is M0 * n_styrene for THIS experiment (not a fixed 4 g — the
        charge can differ per experiment)."""
        return (CONFIG["monomer_mw"] * self.n_styrene_mol) / self.Mn720


def _read_table(csv_path: Path):
    """Read the experimental CSV, auto-detecting the field separator (',' or ';';
    many European/GPC exports use ';'). Column names are stripped of whitespace."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(csv_path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sep


def load_experiments(csv_path: Path):
    df, sep = _read_table(csv_path)
    if sep != ",":
        LOG.info("[data] detected '%s'-separated CSV", sep)
    required = {"code", "temperature_C", "n_styrene_mol", "n_sbuli_charged_mol",
                "n_cyclohexane_mol", "n_sbuli_eff_mol", "time_s", "Mn", "is_full_conversion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}  "
                         f"(found: {sorted(df.columns)}; separator detected: '{sep}')")

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
        Mw_by_time = {}
        if "exp_Mw" in df.columns:
            Mw_by_time = {int(t): float(w) for t, w in zip(fit_rows["time_s"], fit_rows["exp_Mw"])
                          if pd.notna(w)}
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
            Mw_by_time=Mw_by_time,
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


def apply_sec_broadening(df_xy):
    """Convolve a distribution w(log10 M) with a Gaussian of width
    sec_broadening_sigma_log10M (in log10 M), emulating SEC axial dispersion so the
    simulated Mn/Mw/Đ become comparable to broadened SEC data. No-op when sigma=0."""
    sigma = float(CONFIG.get("sec_broadening_sigma_log10M", 0.0) or 0.0)
    if sigma <= 0:
        return df_xy
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    if x.size < 3:
        return df_xy
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        return df_xy
    half = max(1, int(math.ceil(4.0 * sigma / dx)))
    half = min(half, (x.size - 1) // 2)   # keep kernel <= curve length so mode="same" preserves length
    if half < 1:
        return df_xy
    k = np.arange(-half, half + 1) * dx
    kern = np.exp(-0.5 * (k / sigma) ** 2)
    kern /= kern.sum()
    yb = np.convolve(y, kern, mode="same")
    return pd.DataFrame({"x": x, "y": yb})


def mn_from_mmd(path: Path) -> dict:
    df2 = mwd.load_first_two_numeric_cols(path)
    df_xy, swapped = mwd.choose_xy_as_logm(df2, force=CONFIG["force_x_col"])
    df_xy = apply_sec_broadening(df_xy)     # no-op unless sec_broadening_sigma_log10M > 0
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


def _launch_fanout(exps, coeffs_by_code: dict, numMolecules: int, K: int) -> dict:
    """Run the driver for every (experiment x replicate) in parallel.
    Returns {code: [(rep_idx, run_dir, ok, msg), ...]}.  Shared by the Mn path
    (run_all) and the MMD-shape path (collect_curves) so the fan-out logic and
    error handling live in exactly one place."""
    tasks = []
    for exp in exps:
        folder = experiment_folder(exp)
        for r in range(K):
            run_dir = folder if K == 1 else (folder / f"rep{r:02d}")
            _ensure_rundir(exp, run_dir, numMolecules, coeffs_by_code[exp.code])
            tasks.append((exp, r, run_dir))

    out = {exp.code: [] for exp in exps}
    max_workers = max(1, min(CONFIG["max_workers"], len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2task = {pool.submit(run_driver, rd): (exp, r, rd) for (exp, r, rd) in tasks}
        for fut in as_completed(fut2task):
            exp, r, rd = fut2task[fut]
            ok, msg = fut.result()
            if not ok:
                LOG.warning("[run] %s rep%d FAILED: %s", exp.code, r, msg)
            out[exp.code].append((r, rd, ok, msg))
    return out


def run_all(exps, coeffs_by_code: dict, numMolecules: int, K: int) -> dict:
    """Fan out: for each experiment, run K replicates, average Mn/Mw/D per time.

    coeffs_by_code: {code: {"ki":.., "kp":..}}
    Returns {code: {"per_time": {t: {Mn,Mw,D}}, "reps": [...], "errors": [...]}}.
    """
    fanout = _launch_fanout(exps, coeffs_by_code, numMolecules, K)
    results = {exp.code: {"per_time_reps": [], "errors": []} for exp in exps}

    for exp in exps:
        for (r, rd, ok, msg) in fanout[exp.code]:
            if not ok:
                results[exp.code]["errors"].append(f"rep{r}: {msg}")
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
# MMD-shape observable  (fit ki from the early-time molar-mass DISTRIBUTION)
# =============================================================================
# The 10-min distribution SHAPE (peak position + breadth), not just its Mn,
# carries the initiation-broadening signal that pins ki.  The sim already
# exports MMD-S-600.dat every run, so the simulated side is free; we only add
# the experimental curve + a distribution distance.
#
# Consistency note on the M-axis: n_I,eff is derived from the SEC Mn(720), so
# the simulation's absolute molar-mass axis is tied to the SAME SEC calibration
# as the experimental curve — L2 over log10(M) is therefore meaningful. If your
# SEC M-axis is only relative/PS-equivalent, prefer mmd_metric="dispersity" or
# "wasserstein" (less sensitive to an absolute peak-position offset).

def load_norm_curve(path: Path, force: str):
    """Load a two-column MMD -> (x=log10 M, w=dw/dlog10 M) normalized to unit area."""
    df2 = mwd.load_first_two_numeric_cols(path)
    df_xy, _ = mwd.choose_xy_as_logm(df2, force=force)
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    area = float(np.trapezoid(y, x))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("invalid/zero area in MMD curve")
    return x, y / area


def exp_mmd_path(code: str, t_s: int) -> Path:
    pat = CONFIG["exp_mmd_pattern"].format(code=code, t=int(t_s))
    return Path(CONFIG["repo_dir"]) / CONFIG["exp_mmd_dir"] / pat


def load_exp_curves(exps, times):
    """{code: {t: (x, w)}} for every experiment/time that HAS a provided exp MMD.
    Experiments without a file are simply omitted (per-experiment optional)."""
    curves = {}
    for e in exps:
        per_t = {}
        for t in times:
            p = exp_mmd_path(e.code, t)
            if p.exists():
                try:
                    per_t[int(t)] = load_norm_curve(p, CONFIG["mmd_force_x_col"])
                except Exception as ex:
                    LOG.warning("[mmd] %s t=%ss: could not read %s (%s)", e.code, t, p, ex)
        if per_t:
            curves[e.code] = per_t
    return curves


def _common_grid_interp(x_a, w_a, x_b, w_b, npts):
    """Put two normalized densities on a shared log10(M) grid (0-filled tails),
    renormalized to unit area on that grid."""
    lo = min(float(x_a.min()), float(x_b.min()))
    hi = max(float(x_a.max()), float(x_b.max()))
    grid = np.linspace(lo, hi, int(npts))
    wa = np.interp(grid, x_a, w_a, left=0.0, right=0.0)
    wb = np.interp(grid, x_b, w_b, left=0.0, right=0.0)
    aa = float(np.trapezoid(wa, grid)); ab = float(np.trapezoid(wb, grid))
    if aa > 0:
        wa = wa / aa
    if ab > 0:
        wb = wb / ab
    return grid, wa, wb


def shape_distance(sim_curve, exp_curve, metric: str, npts: int) -> float:
    """Distance between two normalized MMD curves.  sim/exp = (x=log10 M, w)."""
    x_s, w_s = sim_curve
    x_e, w_e = exp_curve
    grid, ws, we = _common_grid_interp(x_s, w_s, x_e, w_e, npts)
    if metric == "l2":
        return float(np.trapezoid((ws - we) ** 2, grid))
    if metric == "wasserstein":
        # 1-Wasserstein between 1-D densities on a shared grid = integral |CDF diff|
        cdf_s = np.concatenate([[0.0], np.cumsum(0.5 * (ws[1:] + ws[:-1]) * np.diff(grid))])
        cdf_e = np.concatenate([[0.0], np.cumsum(0.5 * (we[1:] + we[:-1]) * np.diff(grid))])
        return float(np.trapezoid(np.abs(cdf_s - cdf_e), grid))
    if metric == "dispersity":
        # compare Mw/Mn implied by each curve (throws away peak position)
        def disp(x, w):
            M = np.power(10.0, x)
            Mw = float(np.trapezoid(M * w, x))
            invMn = float(np.trapezoid(w / M, x))
            return (Mw * invMn) if invMn > 0 else float("nan")
        d_s = disp(x_s, w_s); d_e = disp(x_e, w_e)
        return float((math.log(d_s) - math.log(d_e)) ** 2)
    raise ValueError(f"unknown mmd_metric: {metric}")


def collect_curves(exps, coeffs_by_code, numMolecules, K, times) -> dict:
    """Fan out and return the (replicate-averaged) SIMULATED normalized MMD curve
    per experiment/time: {code: {t: (grid, w_mean)}}."""
    fanout = _launch_fanout(exps, coeffs_by_code, numMolecules, K)
    out = {}
    for exp in exps:
        per_t = {}
        for t in times:
            rep_curves = []
            for (r, rd, ok, msg) in fanout[exp.code]:
                if not ok:
                    continue
                p = find_mmd_file(rd, t)
                if p is None:
                    continue
                try:
                    rep_curves.append(load_norm_curve(p, CONFIG["force_x_col"]))
                except Exception as ex:
                    LOG.warning("[mmd] %s rep%d t=%ss curve read failed: %s",
                                exp.code, r, t, ex)
            if rep_curves:
                # average replicate curves on a shared grid spanning all of them
                lo = min(float(x.min()) for x, _ in rep_curves)
                hi = max(float(x.max()) for x, _ in rep_curves)
                grid = np.linspace(lo, hi, int(CONFIG["mmd_grid_points"]))
                stack = []
                for x, w in rep_curves:
                    wi = np.interp(grid, x, w, left=0.0, right=0.0)
                    a = float(np.trapezoid(wi, grid))
                    stack.append(wi / a if a > 0 else wi)
                per_t[int(t)] = (grid, np.mean(stack, axis=0))
        out[exp.code] = per_t
    return out


# =============================================================================
# objective / loss
# =============================================================================
def residual(mn_sim: float, mn_exp: float) -> float:
    if CONFIG["residual_kind"] == "log":
        return math.log(mn_sim) - math.log(mn_exp)
    return (mn_sim - mn_exp) / mn_exp


def exp_breadth_target(exp: Experiment, t: int):
    """Experimental breadth observable at time t for the joint objective:
    Đ = exp_Mw/exp_Mn (preferred), or exp_D if that's what the CSV carries, or Mw."""
    obs = CONFIG["joint_breadth_observable"]
    if obs == "mw":
        return exp.Mw_by_time.get(int(t))
    # dispersity
    Mw = exp.Mw_by_time.get(int(t)); Mn = exp.Mn_by_time.get(int(t))
    if Mw is not None and Mn:
        return Mw / Mn
    return exp.D_by_time.get(int(t))    # fall back to an exp_D column if provided


def per_experiment_sse(exp: Experiment, per_time: dict) -> float:
    """Mean squared residual over this experiment's fit times.
    With joint_breadth_enable, also adds a breadth (Đ or Mw) residual at the
    configured times — this is what lets one coupled fit determine ki AND kp."""
    rs = []
    for t in exp.fit_times_s:
        if t not in per_time:
            return float("nan")
        rs.append(residual(per_time[t]["Mn"], exp.Mn_by_time[t]))
    if CONFIG["joint_breadth_enable"] and CONFIG["joint_breadth_weight"] > 0:
        w = CONFIG["joint_breadth_weight"]
        obs = CONFIG["joint_breadth_observable"]
        for t in CONFIG["joint_breadth_times_s"]:
            tgt = exp_breadth_target(exp, int(t))
            if tgt is None or int(t) not in per_time:
                continue
            sim = per_time[int(t)]["D"] if obs != "mw" else per_time[int(t)]["Mw"]
            if sim > 0 and tgt > 0:
                rs.append(w * (math.log(sim) - math.log(tgt)))
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


def run_stage2(exps, numMolecules: int, K: int, x0=None, evallog=None, fixed_ki=None) -> dict:
    """Coupled Arrhenius refinement against Mn(t).

    If `fixed_ki` = {"A_ki":.., "Ea_ki":..} is given (the MMD-determined ki from
    the stage-ki step), ki is HELD FIXED and only (A_kp, Ea_kp) are optimized —
    i.e. ki is set BEFORE the Arrhenius constants are fit.  Otherwise all four
    parameters are optimized (original behaviour, when no MMD data was provided)."""
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for stage2 optimization")
    la, ha = CONFIG["log10A_bounds"]
    le, he = CONFIG["Ea_bounds_kJ"]

    if fixed_ki is not None:
        bounds = [(la, ha), (le, he)]                    # (log10 A_kp, Ea_kp kJ) only
        def to_params(x):
            return {"A_ki": fixed_ki["A_ki"], "Ea_ki": fixed_ki["Ea_ki"],
                    "A_kp": 10.0 ** x[0], "Ea_kp": x[1] * 1000.0}
        x0r = [x0[2], x0[3]] if x0 is not None else None
        LOG.info("[stage2] ki HELD FIXED from the ki stage (A_ki=%.4e, Ea_ki=%.1f kJ); "
                 "fitting only (A_kp, Ea_kp)", fixed_ki["A_ki"], fixed_ki["Ea_ki"]/1000)
    else:
        bounds = [(la, ha), (le, he), (la, ha), (le, he)]
        to_params = x_to_params
        x0r = x0

    def obj(x):
        params = to_params(x)
        total, _, _ = evaluate_params(exps, params, numMolecules, K, "stage2", evallog)
        return total

    optimizer = CONFIG["stage2_optimizer"]
    LOG.info("[stage2] optimizer=%s  bounds=%s", optimizer, bounds)

    if optimizer == "cma":
        best = _run_cma(obj, bounds, x0r)
    else:
        init = _seeded_population(bounds, x0r, CONFIG["stage2_popsize"] * 4)
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

    params = to_params(best["x"])
    LOG.info("[stage2] BEST loss=%.5g  A_ki=%.4e Ea_ki=%.1f kJ  A_kp=%.4e Ea_kp=%.1f kJ%s",
             best["fun"], params["A_ki"], params["Ea_ki"]/1000,
             params["A_kp"], params["Ea_kp"]/1000,
             "  (ki fixed from the ki stage)" if fixed_ki is not None else "")
    return {"params": params, "loss": best["fun"], "x": params_to_x(params)}


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
# Stage 3 — fit ki from the early-time molar-mass DISTRIBUTION  (--stage-ki)
# =============================================================================
def mmd_shape_loss(exps, params: dict, numMolecules: int, K: int,
                   exp_curves: dict, times) -> tuple:
    """Fan out at `params`, compare each simulated MMD curve to the provided
    experimental curve, and reduce to a single (per-experiment-normalized) loss.
    Optionally adds an Mn term (mmd_mn_weight) so the shape fit can't drift Mn."""
    avail = [e for e in exps if e.code in exp_curves]
    coeffs_by_code = {e.code: coeffs_for(e, params) for e in avail}
    sim_curves = collect_curves(avail, coeffs_by_code, numMolecules, K, times)

    # optional Mn term reuses the already-run MMD files (no extra sims)
    metric = CONFIG["mmd_metric"]
    npts = CONFIG["mmd_grid_points"]
    per_exp = {}
    failed = False
    for e in avail:
        terms = []
        for t in times:
            if t not in exp_curves[e.code]:
                continue
            sc = sim_curves.get(e.code, {}).get(int(t))
            if sc is None:
                failed = True
                continue
            d = shape_distance(sc, exp_curves[e.code][t], metric, npts)
            terms.append(d)
            if CONFIG["mmd_mn_weight"] > 0 and t in e.Mn_by_time:
                # Mn implied by the simulated curve vs experimental Mn(t)
                gx, gw = sc
                M = np.power(10.0, gx)
                invMn = float(np.trapezoid(gw / M, gx))
                if invMn > 0:
                    mn_sim = 1.0 / invMn
                    terms.append(CONFIG["mmd_mn_weight"] *
                                 (math.log(mn_sim) - math.log(e.Mn_by_time[t])) ** 2)
        per_exp[e.code] = float(np.mean(terms)) if terms else float("nan")
        if not per_exp[e.code] or not math.isfinite(per_exp[e.code]):
            failed = True

    if failed or not per_exp:
        finite = [v for v in per_exp.values() if math.isfinite(v)]
        return (max(finite) if finite else 0.0) + CONFIG["penalty_loss"], per_exp

    if CONFIG["loss_normalization"] == "per_temperature":
        by_T = {}
        for e in avail:
            by_T.setdefault(e.temperature_C, []).append(per_exp[e.code])
        total = float(np.mean([np.mean(v) for v in by_T.values()]))
    else:
        total = float(np.mean([per_exp[e.code] for e in avail]))
    return total, per_exp


def run_ki_stage(exps, stage1: dict, numMolecules: int, K: int, evallog=None) -> dict:
    """Determine ki FIRST, from the experimental early-time MMD shape — BEFORE the
    Arrhenius constants are set (this runs between stage 1 and stage 2).

    For every temperature level that has MMD data we fit ONE effective ki(T) that
    best reproduces the 10-min distribution shape, holding kp(T) at the stage-1
    value (kp is well determined by Mn, ki is not).  Then:
      * >=2 temperatures with MMD  -> A_ki, Ea_ki analytically from the ki(T) points
      * exactly 1 temperature      -> hold Ea_ki (mmd_fixed_Ea_kJ, default = stage-1
                                      seed) and back out A_ki from the single ki(T).
    Returns the MMD-determined ki Arrhenius (A_ki, Ea_ki) + the per-level ki(T),
    which the orchestration feeds into stage 2 as a FIXED ki."""
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for the MMD ki stage")
    times = [int(t) for t in CONFIG["mmd_fit_times_s"]]
    exp_curves = load_exp_curves(exps, times)
    if not exp_curves:
        LOG.error("[stage-ki] no experimental MMD files under %s/ (pattern %s). Skipping — "
                  "stage 2 will fit ki from Mn as before.",
                  CONFIG["exp_mmd_dir"], CONFIG["exp_mmd_pattern"])
        return {"ki_by_T": {}, "A_ki": None, "Ea_ki": None, "n_used": 0,
                "codes_used": [], "temps_with_data": []}

    used = sorted(exp_curves.keys())
    # stage-1 gives us the well-determined effective kp per temperature level
    kp_by_T = {float(T): lv["kp"] for T, lv in stage1["levels"].items()} if stage1 else {}
    temps_with = sorted({e.temperature_C for e in exps if e.code in exp_curves})
    LOG.info("[stage-ki] determining ki FIRST from MMD (%s) at times %s ; temperatures: %s C",
             used, times, temps_with)

    lo, hi = CONFIG["stage1_k_bounds_log10"]
    ki_by_T = {}
    loss_by_T = {}
    for T in temps_with:
        exps_here = [e for e in exps if e.temperature_C == T and e.code in exp_curves]
        kpT = kp_by_T.get(T)
        if kpT is None:
            raise RuntimeError(f"[stage-ki] no stage-1 kp for T={T} C (run --stage1 first)")

        def obj(u, exps_here=exps_here, kpT=kpT):
            ki = 10.0 ** u[0]
            # hold kp(T) fixed; coeffs_for expects Arrhenius params, so encode k as A, Ea=0
            params = {"A_ki": ki, "Ea_ki": 0.0, "A_kp": kpT, "Ea_kp": 0.0}
            loss, _ = mmd_shape_loss(exps_here, params, numMolecules, K, exp_curves, times)
            LOG.debug("[stage-ki] T=%.0fC ki=%.4g (kp=%.4g) shape_loss=%.5g", T, ki, kpT, loss)
            return loss

        res = differential_evolution(
            obj, bounds=[(lo, hi)],
            maxiter=CONFIG["mmd_ki_maxiter"], popsize=CONFIG["mmd_ki_popsize"],
            tol=CONFIG["mmd_ki_tol"], seed=CONFIG["seed"], polish=False,
            updating="deferred")
        ki_by_T[T] = 10.0 ** res.x[0]
        loss_by_T[T] = float(res.fun)
        LOG.info("[stage-ki] T=%.0f C -> ki(MMD)=%.5g  (shape_loss=%.4g, kp held=%.5g)",
                 T, ki_by_T[T], loss_by_T[T], kpT)

    A_ki, Ea_ki = _ki_points_to_arrhenius(ki_by_T, temps_with, stage1,
                                          CONFIG["mmd_fixed_Ea_kJ"], tag="stage-ki")
    return {"method": "MMD-shape", "metric": CONFIG["mmd_metric"], "times_s": times,
            "ki_by_T": {str(T): ki_by_T[T] for T in temps_with},
            "loss_by_T": {str(T): loss_by_T[T] for T in temps_with},
            "A_ki": A_ki, "Ea_ki": Ea_ki, "n_used": len(used),
            "codes_used": used, "temps_with_data": temps_with}


def _ki_points_to_arrhenius(ki_by_T, temps_with, stage1, fixed_Ea_kJ, tag):
    """Turn per-temperature ki(T) points into ki's Arrhenius (A_ki, Ea_ki).
    >=2 temperatures -> exact analytic; 1 temperature -> hold Ea_ki and back-solve
    A_ki (Ea_ki not identifiable from one temperature).  Shared by every ki stage."""
    R = CONFIG["R"]
    if len(temps_with) >= 2:
        Tlo, Thi = temps_with[0], temps_with[-1]
        inv = 1.0 / T_K(Tlo) - 1.0 / T_K(Thi)
        Ea_ki = R * math.log(ki_by_T[Thi] / ki_by_T[Tlo]) / inv
        A_ki = ki_by_T[Thi] * math.exp(Ea_ki / (R * T_K(Thi)))
        LOG.info("[%s] ki Arrhenius from ki(T) points: A_ki=%.4e  Ea_ki=%.1f kJ/mol",
                 tag, A_ki, Ea_ki / 1000)
        Ea_hi = CONFIG["Ea_bounds_kJ"][1] * 1000.0
        if Ea_ki < 0 or Ea_ki > Ea_hi:
            LOG.warning("[%s] Ea_ki=%.1f kJ/mol is outside [0, %.0f] — the two ki(%.0fC)=%.4g "
                        "and ki(%.0fC)=%.4g don't follow a physical Arrhenius trend (likely noise "
                        "in ki(T)). Treat ki's temperature dependence with caution; more "
                        "replicates or a 3rd temperature helps.",
                        tag, Ea_ki / 1000, Ea_hi / 1000, Tlo, ki_by_T[Tlo], Thi, ki_by_T[Thi])
    else:
        T = temps_with[0]
        Ea_kJ = fixed_Ea_kJ
        if Ea_kJ is None and stage1 is not None:
            Ea_kJ = stage1["arrhenius_seed"]["Ea_ki"] / 1000.0
        if Ea_kJ is None:
            Ea_kJ = 0.0
        Ea_ki = float(Ea_kJ) * 1000.0
        A_ki = ki_by_T[T] * math.exp(Ea_ki / (R * T_K(T)))
        LOG.warning("[%s] only ONE temperature has data -> Ea_ki NOT identifiable; held at "
                    "%.1f kJ/mol, ki(%.0fC)=%.4g -> A_ki back-solved (%.4e).",
                    tag, Ea_kJ, T, ki_by_T[T], A_ki)
    return A_ki, Ea_ki


# =============================================================================
# Stage-ki (Mw) — determine ki from experimental Mw fed in the CSV  (--stage-ki-mw)
# =============================================================================
# Alternative to the MMD-shape stage when you have Mw (not full curves).  Since
# n_I,eff fixes the chain count and kp(T) is held at the stage-1 value, the ONLY
# free lever left in Mw(early time) is the distribution breadth — i.e. ki.  Same
# structure: fit ki(T) per temperature, then ki(T) -> Arrhenius, fed to stage 2
# as a fixed ki.  Provide Mw in the CSV as an `exp_Mw` column (g/mol) on the
# 10-min (and optionally 20/40/60-min) rows.  Optionally uses Đ = Mw/Mn instead
# of Mw (config mw_metric="dispersity"), which cancels the absolute-scale part.
#
# SEC caveat: instrumental band-broadening inflates measured Mw/Đ but the raw kMC
# distribution has none, so raw-sim vs exp comparison biases ki LOW.  Flagged in
# the log/report; use mw_metric="dispersity" or more temperatures to mitigate.

def mw_target(exp, t, metric):
    """Experimental target for the Mw stage: Mw (g/mol) or Đ = Mw/Mn."""
    Mw = exp.Mw_by_time.get(int(t))
    if Mw is None:
        return None
    if metric == "dispersity":
        Mn = exp.Mn_by_time.get(int(t))
        return (Mw / Mn) if (Mn and Mn > 0) else None
    return Mw


def mw_loss(exps_here, params, numMolecules, K, times, metric):
    """Fan out at params; compare simulated Mw (or Đ) to the CSV targets."""
    coeffs_by_code = {e.code: coeffs_for(e, params) for e in exps_here}
    results = run_all(exps_here, coeffs_by_code, numMolecules, K)
    per_exp = {}
    failed = False
    for e in exps_here:
        terms = []
        pt = results[e.code]["per_time"]
        for t in times:
            tgt = mw_target(e, t, metric)
            if tgt is None or t not in pt:
                continue
            sim = pt[t]["D"] if metric == "dispersity" else pt[t]["Mw"]
            if sim <= 0:
                failed = True
                continue
            terms.append((math.log(sim) - math.log(tgt)) ** 2)
        per_exp[e.code] = float(np.mean(terms)) if terms else float("nan")
        if not per_exp[e.code] or not math.isfinite(per_exp[e.code]):
            failed = True
    if failed or not per_exp:
        finite = [v for v in per_exp.values() if math.isfinite(v)]
        return (max(finite) if finite else 0.0) + CONFIG["penalty_loss"], per_exp
    total = float(np.mean([per_exp[e.code] for e in exps_here]))
    return total, per_exp


def run_ki_stage_mw(exps, stage1: dict, numMolecules: int, K: int, evallog=None) -> dict:
    """Determine ki from experimental Mw (CSV `exp_Mw`), between stage 1 and 2.
    Mirrors the MMD stage but scores simulated Mw (or Đ) against the CSV values."""
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for the Mw ki stage")
    metric = CONFIG["mw_metric"]
    times = [int(t) for t in CONFIG["mw_ki_times_s"]]
    avail = [e for e in exps if any(mw_target(e, t, metric) is not None for t in times)]
    if not avail:
        LOG.error("[stage-ki-mw] no experimental Mw found in the CSV (need an `exp_Mw` column "
                  "with values at times %s). Skipping — stage 2 will fit ki from Mn as before.",
                  times)
        return {"ki_by_T": {}, "A_ki": None, "Ea_ki": None, "n_used": 0,
                "codes_used": [], "temps_with_data": []}

    used = sorted(e.code for e in avail)
    kp_by_T = {float(T): lv["kp"] for T, lv in stage1["levels"].items()} if stage1 else {}
    temps_with = sorted({e.temperature_C for e in avail})
    LOG.info("[stage-ki-mw] determining ki from experimental %s (%s) at times %s ; temps: %s C",
             "Đ=Mw/Mn" if metric == "dispersity" else "Mw", used, times, temps_with)
    LOG.info("[stage-ki-mw] NOTE: SEC band-broadening inflates measured %s but the raw kMC "
             "distribution has none, so this can bias ki low; treat as a practical estimate.",
             "Đ" if metric == "dispersity" else "Mw")

    lo, hi = CONFIG["stage1_k_bounds_log10"]
    ki_by_T, loss_by_T = {}, {}
    for T in temps_with:
        exps_here = [e for e in avail if e.temperature_C == T]
        kpT = kp_by_T.get(T)
        if kpT is None:
            raise RuntimeError(f"[stage-ki-mw] no stage-1 kp for T={T} C (run --stage1 first)")

        def obj(u, exps_here=exps_here, kpT=kpT):
            ki = 10.0 ** u[0]
            params = {"A_ki": ki, "Ea_ki": 0.0, "A_kp": kpT, "Ea_kp": 0.0}
            loss, _ = mw_loss(exps_here, params, numMolecules, K, times, metric)
            LOG.debug("[stage-ki-mw] T=%.0fC ki=%.4g (kp=%.4g) mw_loss=%.5g", T, ki, kpT, loss)
            return loss

        res = differential_evolution(
            obj, bounds=[(lo, hi)],
            maxiter=CONFIG["mw_ki_maxiter"], popsize=CONFIG["mw_ki_popsize"],
            tol=CONFIG["mw_ki_tol"], seed=CONFIG["seed"], polish=False, updating="deferred")
        ki_by_T[T] = 10.0 ** res.x[0]
        loss_by_T[T] = float(res.fun)
        LOG.info("[stage-ki-mw] T=%.0f C -> ki(Mw)=%.5g  (loss=%.4g, kp held=%.5g)",
                 T, ki_by_T[T], loss_by_T[T], kpT)

    A_ki, Ea_ki = _ki_points_to_arrhenius(ki_by_T, temps_with, stage1,
                                          CONFIG["mw_fixed_Ea_kJ"], tag="stage-ki-mw")
    return {"method": "Mw" if metric != "dispersity" else "dispersity(Mw/Mn)",
            "metric": metric, "times_s": times,
            "ki_by_T": {str(T): ki_by_T[T] for T in temps_with},
            "loss_by_T": {str(T): loss_by_T[T] for T in temps_with},
            "A_ki": A_ki, "Ea_ki": Ea_ki, "n_used": len(used),
            "codes_used": used, "temps_with_data": temps_with}


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


def make_mmd_plot(exps, params, out_dir: Path):
    """Overlay simulated vs experimental normalized MMD curves at the MMD fit
    times, for every experiment that has a provided experimental curve."""
    if not _HAVE_MPL or params is None:
        return None
    times = [int(t) for t in CONFIG["mmd_fit_times_s"]]
    exp_curves = load_exp_curves(exps, times)
    if not exp_curves:
        return None
    used = [e for e in exps if e.code in exp_curves]
    coeffs_by_code = {e.code: coeffs_for(e, params) for e in used}
    sim_curves = collect_curves(used, coeffs_by_code, CONFIG["numMolecules_fit"],
                                CONFIG["K_replicates_fit"], times)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(used)
    ncol = min(3, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for i, e in enumerate(used):
        ax = axes[i // ncol][i % ncol]
        for t in times:
            if t in exp_curves[e.code]:
                xe, we = exp_curves[e.code][t]
                ax.plot(xe, we / max(np.trapezoid(we, xe), 1e-30), "k-", label=f"exp {t}s")
            sc = sim_curves.get(e.code, {}).get(t)
            if sc is not None:
                xs, ws = sc
                ax.plot(xs, ws, "C0--", label=f"sim {t}s")
        ax.set_title(f"{e.code} ({e.temperature_C:.0f} °C)")
        ax.set_xlabel("log10 M"); ax.set_ylabel("dw/dlog10 M (norm.)")
        ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    p = out_dir / "mmd_overlays.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    LOG.info("[plots] wrote MMD overlay %s", p)
    return p


def write_report(exps, params, stage1, screen, verify, out_dir: Path, run_dir: Path,
                 ki_stage=None):
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
        if math.isfinite(ki_slope) and ki_slope > 1e-6 and math.isfinite(nf):
            resolvable = nf / ki_slope
            if resolvable > 20:   # exp() would overflow / is physically "unbounded"
                lines.append(f"- Practical ki resolution: the 10-min Mn barely responds to ki "
                             f"(slope {ki_slope:.2g} ≲ noise floor {nf:.2%}), so **ki is effectively "
                             f"unidentified from Mn alone** — the MMD stage below is the way to pin it.\n")
            else:
                lines.append(f"- Practical ki resolution: a change in ki is only resolvable if it moves "
                             f"the 10-min Mn by more than the ~{nf:.2%} noise floor, i.e. |Δln ki| ≳ "
                             f"**{resolvable:.2f}** (≈ ×{math.exp(resolvable):.2f}). Below the transient "
                             f"threshold ki is effectively unidentified.\n")
        else:
            lines.append(f"- The 10-min Mn shows no resolvable ki sensitivity → **ki is unidentified "
                         f"from Mn alone**; use the MMD stage to determine it.\n")
    else:
        lines.append("_Run --screen to quantify ki/kp sensitivity and the noise floor._\n")
    lines.append("**Recommendation:** with Mn-only data ki is weakly identified (early-time "
                 "transient only). Two ways to resolve it: (a) the **joint objective** "
                 "(`joint_breadth_enable=True`) fits ki AND kp in one coupled stage-2 regression "
                 "against Mn + a breadth term (Đ or Mw from the CSV) — ki treated exactly like kp; "
                 "or (b) the separate **ki stage** (`--stage-ki[-mw]`) that pins ki first and holds "
                 "it. Both need early-time breadth data (exp_Mw / MMD).\n")

    if ki_stage is not None and ki_stage.get("n_used", 0) > 0:
        method = ki_stage.get("method", "MMD-shape")
        obs = "MMD" if "MMD" in method else ("Đ=Mw/Mn" if "dispersity" in method else "Mw")
        lines.append(f"## Stage 1.5 — ki determined from experimental {obs} (before Arrhenius)\n")
        lines.append(f"**ki is pinned first**, from the experimental {obs} at t={ki_stage['times_s']} s "
                     f"(method: `{method}`), holding kp(T) at the stage-1 value — *then* the "
                     f"Arrhenius constants are set (stage 2 fits only kp, with ki held fixed). "
                     f"Experiments used: {', '.join(ki_stage['codes_used'])}.\n")
        lines.append(f"| T (°C) | ki({obs}) | kp(stage-1, held) | fit loss |")
        lines.append("|---|---|---|---|")
        for T in ki_stage["temps_with_data"]:
            kiT = ki_stage["ki_by_T"][str(T)]
            kpT = (stage1["levels"][str(T)]["kp"] if stage1 else float("nan"))
            lossT = ki_stage["loss_by_T"][str(T)]
            lines.append(f"| {T:.0f} | {kiT:.5g} | {kpT:.5g} | {lossT:.4g} |")
        lines.append(f"\n→ ki Arrhenius from these points: **A_ki = {ki_stage['A_ki']:.4e}**, "
                     f"**Ea_ki = {ki_stage['Ea_ki']/1000:.2f} kJ/mol** (carried into stage 2 as a "
                     f"FIXED ki).")
        if len(ki_stage["temps_with_data"]) < 2:
            lines.append(f"\n> ⚠️ Only one temperature had {obs} data, so **Ea_ki is not identifiable** "
                         f"— it was held (fixed_Ea config, default = the stage-1 seed) and only "
                         f"ki(T) (hence A_ki) was determined. Provide {obs} at the other temperature "
                         f"to identify Ea_ki too.")
        if obs != "MMD":
            lines.append(f"\n> Note: SEC band-broadening inflates measured {obs} while the raw kMC "
                         f"distribution has none, so this ki is a **practical estimate** (biased low). "
                         f"Use `mw_metric=\"dispersity\"` or add temperatures to mitigate.")
        else:
            lines.append("\nSee `plots/mmd_overlays.png` for the sim-vs-exp distribution overlays.")
        lines.append("")

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
    if CONFIG["joint_breadth_enable"]:
        obs = "Đ = Mw/Mn" if CONFIG["joint_breadth_observable"] != "mw" else "Mw"
        lines.append(f"- **Joint objective (ON):** ki and kp were fit **together** in one coupled "
                     f"stage-2 regression against Mn **plus** a breadth term ({obs}, weight "
                     f"{CONFIG['joint_breadth_weight']}, at t={CONFIG['joint_breadth_times_s']} s). "
                     f"ki is treated exactly like kp — the breadth term supplies the ki constraint "
                     f"that Mn alone lacks." +
                     (f" Simulated Mn/Mw/Đ were SEC-broadened by σ="
                      f"{CONFIG['sec_broadening_sigma_log10M']:.3f} log10M to match the SEC data."
                      if CONFIG.get('sec_broadening_sigma_log10M', 0.0) else ""))
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


def load_latest_fixed_ki():
    """Read the MMD-determined ki (A_ki, Ea_ki) from the most recent stage_ki.json
    so a separate `--stage2` invocation still holds ki fixed at what the MMD stage
    found."""
    root = Path(CONFIG["repo_dir"]) / CONFIG["results_root"]
    if not root.exists():
        return None
    for d in sorted(root.iterdir(), reverse=True):
        p = d / "stage_ki.json"
        if p.exists():
            try:
                j = json.loads(p.read_text())
                if j.get("A_ki") is not None:
                    return {"A_ki": float(j["A_ki"]), "Ea_ki": float(j["Ea_ki"])}
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
    ap.add_argument("--stage-ki", dest="stage_ki", action="store_true",
                    help="determine ki from the 10-min MMD shape (runs between stage1 & stage2)")
    ap.add_argument("--stage-ki-mw", dest="stage_ki_mw", action="store_true",
                    help="determine ki from experimental Mw in the CSV (exp_Mw column; between stage1 & stage2)")
    ap.add_argument("--verify", action="store_true",
                    help="re-run best fit K times at verification resolution")
    ap.add_argument("--report", action="store_true",
                    help="(re)write report.md + plots from the latest results")
    ap.add_argument("--all", action="store_true",
                    help="setup -> stage1 -> screen -> stage2 -> verify -> report (NO ki/MMD stage)")
    ap.add_argument("--all-ki", dest="all_ki", action="store_true",
                    help="like --all but WITH the MMD ki stage: setup -> stage1 -> stage-ki "
                         "-> screen -> stage2 -> verify -> report")
    ap.add_argument("--all-ki-mw", dest="all_ki_mw", action="store_true",
                    help="like --all but WITH the Mw ki stage (from the CSV exp_Mw column)")
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

    # --all runs the full pipeline WITHOUT any ki stage; --all-ki / --all-ki-mw add one.
    run_full = args.all or args.all_ki or args.all_ki_mw   # every stage except (maybe) a ki stage
    run_stage_ki = args.stage_ki or args.all_ki            # MMD-shape ki stage
    run_stage_ki_mw = args.stage_ki_mw or args.all_ki_mw   # Mw-based ki stage

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

    if CONFIG["joint_breadth_enable"]:
        obs = "Đ=Mw/Mn" if CONFIG["joint_breadth_observable"] != "mw" else "Mw"
        nbt = [e.code for e in exps if any(exp_breadth_target(e, t) is not None
                                           for t in CONFIG["joint_breadth_times_s"])]
        LOG.info("[joint] JOINT objective ON: fitting ki AND kp together against Mn + %s "
                 "(weight %.2f) at t=%s. Breadth data present for: %s",
                 obs, CONFIG["joint_breadth_weight"], CONFIG["joint_breadth_times_s"],
                 nbt or "NONE (add exp_Mw to the CSV!)")
        if CONFIG.get("sec_broadening_sigma_log10M", 0.0):
            LOG.info("[joint] SEC broadening σ=%.3f log10M applied to simulated Mn/Mw/Đ.",
                     CONFIG["sec_broadening_sigma_log10M"])

    did_something = False
    stage1 = None
    best = None
    screen = None
    verify = None
    last_results = None
    last_breakdown = None

    # nothing selected -> show help
    if not any([args.setup, args.single, args.eval_once, args.screen,
                args.stage1, args.stage2, args.stage_ki, args.stage_ki_mw, args.verify,
                args.report, args.all, args.all_ki, args.all_ki_mw]):
        build_argparser().print_help()
        return 0

    # ---- setup --------------------------------------------------------------
    if args.setup or run_full:
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
    if args.stage1 or run_full:
        stage1 = run_stage1(exps, nMol_fit, K_fit)
        (run_dir / "stage1.json").write_text(json.dumps(stage1, indent=2))
        best = dict(stage1["arrhenius_seed"])
        did_something = True
    else:
        stage1 = load_latest_stage1()

    # ---- stage-ki: determine ki FIRST (between stage 1 & 2) ------------------
    # OPTIONAL — two alternative methods, each opt-in (NOT run by plain --all):
    #   --stage-ki    / --all-ki     : from the early-time MMD *shape*
    #   --stage-ki-mw / --all-ki-mw  : from experimental Mw in the CSV (exp_Mw)
    # Either pins ki BEFORE the Arrhenius constants are set; stage 2 then holds ki
    # fixed and fits only kp. If both are requested, the MMD (richer) result wins.
    ki_stage = None
    fixed_ki = None

    def _adopt_ki(ks):
        nonlocal best, fixed_ki, ki_stage
        ki_stage = ks
        if ks.get("A_ki") is not None:
            fixed_ki = {"A_ki": ks["A_ki"], "Ea_ki": ks["Ea_ki"]}
            base = dict(best) if best is not None else dict(stage1["arrhenius_seed"])
            base["A_ki"] = fixed_ki["A_ki"]; base["Ea_ki"] = fixed_ki["Ea_ki"]
            best = base
            (run_dir / "stage_ki.json").write_text(json.dumps(ks, indent=2, default=str))

    if (run_stage_ki or run_stage_ki_mw) and CONFIG["joint_breadth_enable"]:
        LOG.warning("[config] joint_breadth_enable=True AND a hard-fix ki stage were both "
                    "requested. These are two different designs — the ki stage will PIN ki "
                    "(stage 2 can't move it), making the joint breadth term redundant. Pick one: "
                    "joint objective (drop --stage-ki*) OR the ki stage (set joint_breadth_enable=False).")
    if run_stage_ki or run_stage_ki_mw:
        if stage1 is None:
            LOG.error("[stage-ki] needs stage 1 for kp(T). Run --stage1 first (or --all-ki[/-mw]).")
        else:
            if run_stage_ki:
                _adopt_ki(run_ki_stage(exps, stage1, nMol_fit, K_fit, evallog))
            if run_stage_ki_mw:
                if fixed_ki is not None:
                    LOG.warning("[stage-ki-mw] MMD ki stage already determined ki; skipping the "
                                "Mw stage (MMD shape is the richer observable). Run only "
                                "--stage-ki-mw / --all-ki-mw to use Mw instead.")
                else:
                    _adopt_ki(run_ki_stage_mw(exps, stage1, nMol_fit, K_fit, evallog))
            did_something = True
    # Persist the determined ki across separate invocations (e.g. --stage-ki[-mw]
    # then a later --stage2 holds ki fixed).  NEVER for plain --all, which
    # deliberately excludes any ki stage and must fit all four params from Mn.
    if fixed_ki is None and not args.all:
        fixed_ki = load_latest_fixed_ki()
        if fixed_ki is not None:
            LOG.info("[stage2] using ki fixed from a previous ki stage "
                     "(A_ki=%.4e, Ea_ki=%.1f kJ). Use plain --all to ignore it.",
                     fixed_ki["A_ki"], fixed_ki["Ea_ki"] / 1000)

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
    if args.screen or run_full:
        params = best or (stage1["arrhenius_seed"] if stage1 else
                          load_latest_best_params() or default_params_from_model())
        screen = run_screen(exps, params, nMol_fit)
        (run_dir / "screen.json").write_text(json.dumps(screen, indent=2))
        did_something = True

    # ---- stage 2 ------------------------------------------------------------
    if args.stage2 or run_full:
        seed_params = best or (stage1["arrhenius_seed"] if stage1 else None) \
            or load_latest_best_params()
        x0 = params_to_x(seed_params) if seed_params else None
        st2 = run_stage2(exps, nMol_fit, K_fit, x0=x0, evallog=evallog, fixed_ki=fixed_ki)
        best = st2["params"]
        # one clean evaluation at the optimum to capture per-exp breakdown + Mn
        total, last_breakdown, last_results = evaluate_params(
            exps, best, nMol_fit, K_fit, "stage2_best", evallog)
        save_best_params(best, last_breakdown, run_dir)
        did_something = True
    elif best is None:
        best = load_latest_best_params()

    # ---- verify -------------------------------------------------------------
    if args.verify or run_full:
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
    if args.report or run_full or args.stage2 or args.stage_ki or args.verify or args.eval_once:
        if best is None:
            best = load_latest_best_params()
        if last_results is None and best is not None:
            _, last_breakdown, last_results = evaluate_params(
                exps, best, nMol_fit, K_fit, "report_eval", evallog)
        if last_results is not None:
            make_plots(exps, last_results, best, stage1, run_dir / "plots")
            make_mmd_plot(exps, best, run_dir / "plots")
            write_report(exps, best, stage1, screen, verify, run_dir, run_dir, ki_stage=ki_stage)
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
