#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERIC per-experiment driver for mcPolymer Arrhenius fitting.

One identical copy of this file lives in each experiment folder (TW60/, TW62/, ...).
Nothing experiment-specific is hard-coded: everything comes from two small JSON
files the fitting harness writes into the folder before each run.

Harness <-> driver contract
---------------------------
INPUTS  (harness places these in the folder, then launches this script with cwd = folder):
    ip.mcPolymer        the kMC model template (defines coefficients ki, kp + reactions)
    experiment.json     recipe + temperature + sim settings for THIS experiment (constant across the fit)
    coeffs.json         the rate coefficients for THIS evaluation, e.g. {"ki": 0.25, "kp": 0.20}

OUTPUTS (this script writes into the same folder):
    MMD-S-<seconds>.dat one molecular-mass distribution per export time (e.g. MMD-S-3600.dat = 60 min)
    sim_conversion.csv  time_s, conversion_true, mol_styrene   (true monomer-balance conversion; DIAGNOSTIC only)
    run_meta.json       echo of inputs + status, for debugging

The harness extracts Mn(t) from MMD-S-*.dat with mwd_analyzerv2 and does the scoring.
This driver never reads experimental data and never computes Mn — it only runs the
simulation and exports the distributions.

Standalone debugging: if experiment.json / coeffs.json are absent, the DEFAULTS below
are used, so `python TWXX.py` runs a single experiment on its own.

IMPORTANT for the master code: this file is the fixed interface. Do NOT edit it except
to switch `injection_method` if Step-0 inspection shows the template rewrite is unnecessary
(then "addCoefficient" can be used instead). Build the orchestration AROUND this contract.
"""

import os
import re
import csv
import json
from pathlib import Path

# mcPolymer Python wrappers (local modules, same as TW013.py)
from kineticModel import kineticModel
from modelInterpreter import modelInterpreter

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)  # so mcPolymer's relative-path outputs land in THIS folder


# ---------------------------------------------------------------------------
# Shared physical constants — properties of the chemicals, identical for every
# experiment. These do NOT belong in the per-experiment table. (From TW013.py.)
# ---------------------------------------------------------------------------
STYRENE_CONST = {
    "M": 104.15, "rho_a0": 924.3, "rho_a1": 0.900,
    "rhoPolymer_a0": 1057.8, "rhoPolymer_a1": 0.251,
}
CYCLOHEXANE_CONST = {"M": 84.16, "rho_a0": 797.7, "rho_a1": 0.955}


# ---------------------------------------------------------------------------
# DEFAULTS — used ONLY when experiment.json / coeffs.json are missing.
# Values shown are TW60, so the file runs standalone for debugging.
# The harness always overrides these via the JSON files.
# ---------------------------------------------------------------------------
DEFAULT_EXPERIMENT = {
    "code": "TW60",
    "temperature_C": 30.0,
    "n_styrene_mol": 0.0384061450,
    "n_sbuli_mol": 7.963844e-05,        # EFFECTIVE initiator (= 4.0000 / Mn_720min)
    "n_cyclohexane_mol": 0.553707224,
    "export_times_s": [600, 1200, 2400, 3600],   # 10, 20, 40, 60 min
    "dt_s": 1,                          # volume-balance update step (integer seconds)
    "numMolecules": 100_000_000,        # fitting resolution; raise for final verification
    "mmd_raster_points": 800,
    "injection_method": "template",     # "template" (rewrite model file) | "addCoefficient"
    "model_template": "ip.mcPolymer",
}
DEFAULT_COEFFS = {"ki": 0.25, "kp": 0.20}


def load_json(name, fallback):
    p = SCRIPT_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return dict(fallback)


# ---------------------------------------------------------------------------
# Volume balance — reactor volume as monomer converts to polymer (density
# change). Logic lifted verbatim from TW013.py, trimmed. Keep this coupling.
# ---------------------------------------------------------------------------
class VolumeBalance:
    def __init__(self, recipe):
        self.componentData = []
        for item in recipe:
            if "rho_a0" not in item:      # e.g. sbuli: only mol, no density -> skip
                continue
            data = {
                "name": item["name"], "mol": float(item["mol"]), "M": float(item["M"]),
                "rho_a0": float(item["rho_a0"]), "rho_a1": float(item["rho_a1"]),
                "isMonomer": "rhoPolymer_a0" in item,
            }
            if data["isMonomer"]:
                data["rhoPolymer_a0"] = float(item["rhoPolymer_a0"])
                data["rhoPolymer_a1"] = float(item["rhoPolymer_a1"])
                data["molPolymer"] = 0.0
            self.componentData.append(data)

    def getVolume(self, temperature):
        volume = 0.0
        for it in self.componentData:
            rho = it["rho_a0"] - it["rho_a1"] * temperature
            v = it["mol"] * it["M"] / rho
            if it["isMonomer"]:
                rhoP = it["rhoPolymer_a0"] - it["rhoPolymer_a1"] * temperature
                v += it["molPolymer"] * it["M"] / rhoP
            volume += v
        return volume

    def updateMol(self, species, mol, molPolymer=None):
        for it in self.componentData:
            if it["name"] == species:
                it["mol"] = mol
                if it["isMonomer"] and molPolymer is not None:
                    it["molPolymer"] = molPolymer


def build_recipe(exp):
    """Assemble the mcPolymer recipe from shared constants + this row's mole numbers.
    NOTE: sbuli uses the EFFECTIVE initiator (n_sbuli_mol here = 4.0000/Mn_720min)."""
    return [
        {"name": "styrene", "mol": float(exp["n_styrene_mol"]), **STYRENE_CONST},
        {"name": "cyclohexane", "mol": float(exp["n_cyclohexane_mol"]), **CYCLOHEXANE_CONST},
        {"name": "sbuli", "mol": float(exp["n_sbuli_mol"])},
    ]


def write_active_model(template, active, coeffs):
    """Rewrite the 'name = value' coefficient-definition lines with injected values.
    Robust to wrapper internals; only touches definition lines, never reaction lines
    (those start with a species, so the anchored regex won't match them)."""
    text = Path(template).read_text()
    for name, val in coeffs.items():
        pat = re.compile(rf"^\s*{re.escape(name)}\s*=\s*[^\n]+$", re.MULTILINE)
        if not pat.search(text):
            raise ValueError(f"coefficient '{name}' not found in model template {template}")
        text = pat.sub(f"{name} = {float(val):.10g}", text)
    Path(active).write_text(text)


def main():
    exp = load_json("experiment.json", DEFAULT_EXPERIMENT)
    coeffs = load_json("coeffs.json", DEFAULT_COEFFS)

    T = float(exp["temperature_C"])
    dt = int(exp.get("dt_s", 1))
    export_times = sorted(int(t) for t in exp["export_times_s"])
    reaction_time = max(export_times)
    raster = int(exp.get("mmd_raster_points", 800))
    numMol = int(exp.get("numMolecules", DEFAULT_EXPERIMENT["numMolecules"]))
    method = exp.get("injection_method", "template")
    template = exp.get("model_template", "ip.mcPolymer")

    recipe = build_recipe(exp)
    styrene0 = float(exp["n_styrene_mol"])

    vB = VolumeBalance(recipe)
    currentVolume = vB.getVolume(T)

    # ---- coefficient injection --------------------------------------------
    if method == "template":
        model_file = "ip_active.mcPolymer"
        write_active_model(template, model_file, coeffs)
        interpreter = modelInterpreter(modelFile=model_file)
        interpreter.addRecipe(recipe=recipe, volume=currentVolume)
        interpreter.interpreteModelFile()
    else:  # "addCoefficient" — mirrors the commented hook in TW013.py
        model_file = template
        interpreter = modelInterpreter(modelFile=model_file)
        interpreter.addRecipe(recipe=recipe, volume=currentVolume)
        for name, val in coeffs.items():
            interpreter.addCoefficient(name=name, value=float(val))
        interpreter.interpreteModelFile()

    kMC = kineticModel(temperature=T, volume=currentVolume, recipe=recipe,
                       modelFile=model_file, numMolecules=numMol)

    # ---- run loop: advance on a grid, keep volume coupled, export at data times
    grid = sorted(set(range(dt, reaction_time + 1, dt)) | set(export_times))
    export_set = set(export_times)
    rows = []
    for t in grid:
        kMC.runTo(time=t)
        molStyrene = kMC.getMol(species="styrene")
        segs = kMC.getPolymerSegments(species="styrene")
        vB.updateMol(species="styrene", mol=molStyrene, molPolymer=segs)
        currentVolume = vB.getVolume(T)
        kMC.updateVolume(volume=currentVolume)
        if t in export_set:
            kMC.exportMMD(filename=f"MMD-S-{t}.dat", numRasterPoints=raster)
            rows.append((t, 1.0 - molStyrene / styrene0, molStyrene))

    with open("sim_conversion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion_true", "mol_styrene"])
        w.writerows(rows)

    with open("run_meta.json", "w") as f:
        json.dump({"experiment": exp, "coeffs": coeffs,
                   "final_volume": currentVolume, "status": "ok"}, f, indent=2)


if __name__ == "__main__":
    main()
