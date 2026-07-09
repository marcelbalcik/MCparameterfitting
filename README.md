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
python main.py --stage-ki         # 6a. (optional) determine ki from the 10-min MMD shape
python main.py --stage-ki-mw      # 6b. (optional) determine ki from experimental Mw in the CSV
python main.py --stage2           # 7. coupled Arrhenius refinement (holds ki fixed if a ki stage ran)
python main.py --verify           # 8. re-run best fit K× at high numMolecules
python main.py --predict          # predict Mn(t) for NEW recipes (T + moles only) — see below
python main.py --all              # setup → stage1 → screen → stage2 → verify → report   (NO ki stage)
python main.py --all-ki           # same as --all but WITH the MMD ki stage
python main.py --all-ki-mw        # same as --all but WITH the Mw ki stage (CSV exp_Mw)
```

### Predicting new experiments (`--predict`)

Once you have a fit (`best_params.json` under `results/`), predict `Mn(t)` for
**new recipes you have not run** — you supply only temperature and initial moles,
no Mn data. Put the recipes in **`predict.csv`**:

```
code,temperature_C,n_styrene_mol,n_sbuli_mol,n_cyclohexane_mol
NEW_20C,20,0.0384,7.5e-5,0.60
NEW_30C,30,0.0384,7.5e-5,0.60
NEW_dilute,30,0.0384,4.0e-5,1.0
```

`n_sbuli_mol` is the initiator used as-is (charged). Then:

```bash
python main.py --predict                    # uses predict.csv + the latest best_params.json
python main.py --predict --predict-csv other.csv
```

For each recipe it computes `ki(T), kp(T)` from the fitted Arrhenius params, runs
the kMC forward, and writes **`predictions.csv`** (`code, T, ki, kp, time_s,
Mn_pred, Mn_std, Mw_pred, D_pred`) plus `plots/predictions.png`. Export times are
`predict_times_s` in CONFIG (default 10/20/40/60 min — set any list, including
beyond 60 min). Resolution/replicates: `predict_numMolecules`, `predict_reps`.
This mode needs no `experimental_data.csv`.

> Predictions are only as good as the fit: at a `[P*]`/dilution far from the
> calibration set the single-`kp` (no-aggregation) bias applies, and extrapolating
> outside the fitted temperatures assumes Arrhenius holds.

### Two ways to determine `ki` (both need early-time breadth data)

`ki` is not identifiable from `Mn(t)` alone. There are two designs — pick one:

**(A) Joint objective — *recommended*, symmetric.** One coupled stage-2 regression
fits **all four** Arrhenius params against `Mn(t)` **plus a breadth term** (`Đ = Mw/Mn`
or `Mw`). `ki` is treated exactly like `kp` — same optimizer, same method — with the
breadth term supplying the `ki` constraint `Mn` lacks. Turn it on in CONFIG:

```python
"joint_breadth_enable":     True,
"joint_breadth_observable": "dispersity",   # "dispersity" (Đ=Mw/Mn) | "mw"
"joint_breadth_weight":     2.0,            # weight of each breadth residual vs one Mn residual
"joint_breadth_times_s":    [600],          # 10 min carries the ki signal
```

Then just run `python main.py --all` (or `--stage2`) — no ki stage, `ki` and `kp`
come out of the same fit. Needs `exp_Mw` (or `exp_D`) in the CSV, at **both**
temperatures for `Ea_ki` to be identifiable.

**(B) Separate ki stage — pins `ki` first, then holds it.** Plain `--all` runs
*neither* ki stage (and ignores any leftover `stage_ki.json`) — it fits all four
from `Mn(t)`. Opt in with:
- `--all-ki` / `--stage-ki` → `ki` from the 10-min **MMD shape** (needs `exp_mmd/` curve files).
- `--all-ki-mw` / `--stage-ki-mw` → `ki` from experimental **Mw** in the CSV (`exp_Mw`).

These run between stage 1 and stage 2, pin `ki` per temperature (holding stage-1's
`kp`), then stage 2 sets the Arrhenius constants with `ki` **held fixed**. A later
bare `--stage2` keeps holding that `ki`. If both MMD and Mw are requested, MMD wins.

> Use **A or B, not both** — B hard-fixes `ki` so B's breadth term can't move it.
> The code warns if you enable both.

### SEC band-broadening correction (for either breadth method)

Measured `Mw`/`Đ` are inflated by SEC axial dispersion, but the raw kMC distribution
has none — so a raw comparison biases `ki` low. Set `sec_broadening_sigma_log10M`
(CONFIG, default `0` = off) to Gaussian-broaden the *simulated* MMD before computing
`Mn/Mw/Đ`, making sim and exp comparable. We estimated **≈ 0.063** for the TW067
column set from its internal-standard peak; use your own if you have narrow standards.

### `--stage-ki-mw` — determine `ki` from Mw in the CSV

When you have `Mw` but not full curves, add an **`exp_Mw`** column (g/mol) to
`experimental_data.csv` on the fit rows (at least the 10-min row):

```
code,temperature_C,...,time_s,Mn,...,exp_Mw
TW60,30,...,600,7484,...,8100
TW60,30,...,1200,12359,...,13050
...
```

The stage fits an effective `ki(T)` per temperature so simulated `Mw` matches the
CSV, holding `kp(T)` from stage 1, then turns the `ki(T)` points into `A_ki, Ea_ki`
(needs both temperatures for `Ea_ki`; one temperature holds `Ea_ki`). Config:
`mw_ki_times_s` (default `[600]` = 10 min), `mw_metric` (`"mw"` default, or
`"dispersity"` to fit `Đ = Mw/Mn`, which cancels the absolute-scale part),
`mw_fixed_Ea_kJ`, and `mw_ki_*` optimizer settings.

> **SEC caveat (both breadth-based methods):** instrumental band-broadening inflates
> measured `Mw`/`Đ` while the raw kMC distribution has none, so a raw comparison
> biases `ki` **low**. Treat the result as a practical estimate; `mw_metric="dispersity"`
> and/or more temperatures mitigate it. Flagged in the log and report.

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
