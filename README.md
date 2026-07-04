# Automated Arrhenius fitting for mcPolymer kMC (s‑BuLi / styrene)

Fits the Arrhenius parameters `A_ki, Ea_ki, A_kp, Ea_kp` of the two‑reaction
living‑anionic model so the mcPolymer kMC simulations reproduce experimental
`Mn(t)` across six experiments at two temperatures (20 °C and 30 °C),
**simultaneously**.

```
k_j(T) = A_j · exp(−Ea_j / (R·T))        R = 8.314 J/(mol·K),  T in K = °C + 273.15
```

Everything is driven through the fixed generic driver **`TWXX.py`** (never
modified) via two JSON files it reads (`experiment.json`, `coeffs.json`).
`main.py` never imports mcPolymer — it only launches `TWXX.py` as a subprocess.

## Layout

| file | role |
|---|---|
| `main.py` | **the one entry point** — folder setup, parallel fan‑out, Mn extraction, two‑stage fit, plots, `report.md` |
| `TWXX.py` | provided generic per‑experiment driver (fixed contract; copied into each folder) |
| `ip.mcPolymer` | two‑reaction model template (`ki`, `kp`) |
| `mwd_analyzerv2.py` | validated MMD → `Mn/Mw/Đ` analyzer (imported by `main.py`) |
| `experimental_data.csv` | tidy long‑format data (codes, T, moles, `Mn(t)`, 720‑min anchor) |
| `kineticModel.py`, `modelInterpreter.py`, `mcPolymer` | core engine + wrappers (read‑only; must be importable) |

Generated at runtime (git‑ignored): `work/<CODE>/` experiment folders and
`results/<timestamp>/` outputs (`best_params.*`, `report.md`, `plots/`,
`eval_log.csv`, `manifest.json`).

## Install

```bash
pip install -r requirements.txt      # numpy pandas scipy matplotlib
```

The mcPolymer engine (`mcPolymer`, `kineticModel.py`, `modelInterpreter.py`)
must be importable in the same Python environment.

## Run — staged so each step is independently verifiable

```bash
python main.py --setup            # 1. build work/TW60 … work/TW56 (+ experiment.json)
python main.py --single TW60      # 2. run ONE folder once → prints Mn(t) (smoke test)
python main.py --eval-once        # 3. one parallel fan-out for a fixed θ → single global loss
python main.py --screen           # 4. ki/kp sensitivity + stochastic noise floor
python main.py --stage1           # 5. per-temperature effective-k warm start + Arrhenius seed
python main.py --stage-ki         # 6. determine ki from the 10-min MMD shape (BETWEEN stage1 & stage2)
python main.py --stage2           # 7. coupled Arrhenius refinement (holds ki fixed if stage-ki ran)
python main.py --verify           # 8. re-run best fit K× at high numMolecules
python main.py --all              # setup → stage1 → stage-ki → screen → stage2 → verify → report
```

### Stage-ki — determine `ki` FIRST, from the early-time molar-mass distribution

`kp` is well constrained by `Mn(t)`, but `ki` is not — it only shifts the earliest
point. So **`ki` is pinned before the Arrhenius constants are set**: this stage
runs *between* stage 1 and stage 2 and fits an effective **`ki(T)` per temperature**
against the 10-min **distribution shape** (peak position + breadth, which carries
the initiation-broadening signal), **holding `kp(T)` at the stage-1 value**. Stage 2
then sets the Arrhenius constants with **`ki` held fixed**, fitting only
`(A_kp, Ea_kp)` against `Mn(t)`. This cleanly decouples the two observables:
`kp` from Mn(t), `ki` from the early-time shape.

**To use it, drop your experimental MMD file(s) here:**

```
exp_mmd/TW60_MMD-600.dat        # two columns: log10(M)   dw/dlog10(M)   (same format as the sim output)
exp_mmd/TW56_MMD-600.dat        # 600 = 10 min; one file per experiment you have data for
...
```

Filename pattern and folder are config (`exp_mmd_pattern`, `exp_mmd_dir`). Any
experiment without a file is silently skipped, so you can start with just one.
**Identifiability of `Ea_ki`:** with MMDs at **both** temperatures the two `ki(T)`
points give `A_ki, Ea_ki` analytically; with only **one** temperature it determines
`ki(T)` there and holds `Ea_ki` (config `mmd_fixed_Ea_kJ`, default = the stage-1
seed), warning that `Ea_ki` isn't identifiable from one temperature. Distance
metric is config `mmd_metric` (`l2` default, or `wasserstein` / `dispersity`).
If **no** MMD file is present, stage-ki is skipped and stage 2 fits all four
parameters from `Mn(t)` as before (fully backward compatible). Output:
`ki` pinned in `best_params.*`, `stage_ki.json` (with the per-temperature
`ki(T)`), and `plots/mmd_overlays.png` (sim vs exp curves).

> M-axis consistency: because `n_I,eff` is derived from the SEC `Mn(720)`, the
> simulation's absolute molar-mass axis is tied to the **same SEC calibration** as
> your experimental curve, so comparing them on `log10(M)` is meaningful. If your
> SEC axis is only relative / PS-equivalent, prefer `mmd_metric="dispersity"` or
> `"wasserstein"` (less sensitive to an absolute peak-position offset).

Before wiring the optimizer, validate the driver alone:
`cd work/TW60 && python TWXX.py` (uses the folder's JSONs) should produce
`MMD-S-{600,1200,2400,3600}.dat`.

Useful flags: `--numMolecules 1e8` (fit resolution), `--reps K` (replicate
averaging), `--workers N` (concurrent driver processes).

## Configuration

All knobs live in the `CONFIG` block at the top of `main.py` — paths,
`numMolecules` (fit vs verify), replicates `K`, residual kind (`log`/`rel`),
loss normalization (`per_experiment`/`per_temperature`), optimizer choice and
bounds, seed. **Nothing** experiment‑specific is hard‑coded: codes,
temperatures and data times are read from `experimental_data.csv`. Adding a
**third temperature is just more rows + folders**, no code change.

## Key modeling decisions (see `main.py` docstring for the full assumptions)

- **Fit Mn against Mn** at 10/20/40/60 min. The **720‑min** point is *not* a
  residual — it sets the **effective initiator** `n_I,eff = 4.0000 g / Mn₇₂₀`
  (column `n_sbuli_eff_mol`), so each sim matches its final Mn by construction
  and the intermediate shape is pure `ki`/`kp` kinetics.
- **Conversion is a diagnostic only** (the data’s `X(t)=Mn(t)/Mn₇₂₀` carries no
  information independent of Mn).
- **`kp`** is well constrained (Mn growth rate); **`ki`** is weak (early‑time
  transient only). If dispersity (Đ/Mw) becomes available, enable the second
  observable (`use_dispersity_term=True`, add an `exp_D` column) to pin `ki`.
- With exactly two temperatures the fit **assumes** Arrhenius (it is an exact
  reparametrization of `k(20 °C), k(30 °C)`); ≥3 temperatures validate the form.
- **Coefficient injection uses the template rewrite** (`injection_method="template"`),
  the reliable path — see assumption A1 in `main.py`.
