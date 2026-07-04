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
python main.py --stage2           # 6. coupled global (A,Ea) refinement (warm-started from stage1)
python main.py --verify           # 7. re-run best fit K× at high numMolecules
python main.py --all              # setup → stage1 → screen → stage2 → verify → report
```

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
