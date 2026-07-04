#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# Filename time parsing (robust, optional)
# ----------------------------

EXP_TIME_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*[-_ ]*\s*(?P<unit>min|mins|minute|minutes|m|h|hr|hrs|hour|hours)(?=$|[^a-zA-Z])",
    re.IGNORECASE,
)

SIM_S_TOKEN_RE = re.compile(r"(?:^|[-_ ])s\s*[-_ ]*\s*(?P<sec>\d+(?:\.\d+)?)\b", re.IGNORECASE)
SIM_S_INLINE_RE = re.compile(r"(?:^|[-_ ])s(?P<sec>\d+(?:\.\d+)?)\b", re.IGNORECASE)
SIM_UNIT_RE = re.compile(
    r"(?P<sec>\d+(?:\.\d+)?)\s*[-_ ]*\s*(?:s|sec|secs|second|seconds)(?=$|[^a-zA-Z])",
    re.IGNORECASE,
)


def parse_time_seconds_from_name(stem: str):
    """
    Returns (kind, seconds, parsed_flag)
      kind: "exp" | "sim" | "unknown"
      seconds: float or np.nan
      parsed_flag: True/False

    Time parsing is optional; unknowns still get processed.
    """
    s = stem.lower()

    m = SIM_S_TOKEN_RE.search(s)
    if m:
        return "sim", float(m.group("sec")), True
    m = SIM_S_INLINE_RE.search(s)
    if m:
        return "sim", float(m.group("sec")), True
    m = SIM_UNIT_RE.search(s)
    if m:
        return "sim", float(m.group("sec")), True

    m = EXP_TIME_RE.search(s)
    if m:
        val = float(m.group("val"))
        unit = m.group("unit").lower()
        if unit in ("min", "mins", "minute", "minutes", "m"):
            return "exp", val * 60.0, True
        if unit in ("h", "hr", "hrs", "hour", "hours"):
            return "exp", val * 3600.0, True

    return "unknown", float("nan"), False


# ----------------------------
# Reading + robust x/y detection
# ----------------------------

def load_first_two_numeric_cols(path: Path) -> pd.DataFrame:
    """
    Reads first two columns and keeps only numeric rows.
    Allows whitespace/tab/comma/semicolon separated.
    """
    raw = pd.read_csv(path, sep=r"\s+|\t+|,|;", engine="python", header=None, comment="#")
    if raw.shape[1] < 2:
        raise ValueError("expected >=2 columns")
    df = raw.iloc[:, :2].copy()
    df.columns = ["c0", "c1"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("no numeric rows after parsing")
    return df


def score_as_log10M(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    spread = xmax - xmin

    score = 0.0
    if -1.0 <= xmin <= 12.0 and 0.0 <= xmax <= 15.0:
        score += 2.0
    if 1.0 <= xmin <= 9.0 and 2.0 <= xmax <= 10.0:
        score += 1.5
    if spread >= 0.5:
        score += 1.0
    if spread >= 2.0:
        score += 0.5
    return score


def choose_xy_as_logm(df2: pd.DataFrame, force: str = "auto"):
    """
    Returns (df_xy, swapped_flag)
      df_xy columns: x(log10M), y(dw/dlog10M)
      sorted by x
    """
    c0 = df2["c0"].to_numpy()
    c1 = df2["c1"].to_numpy()

    if force == "c0":
        x, y = c0, c1
        swapped = False
    elif force in ("c1", "swap"):
        x, y = c1, c0
        swapped = True
    else:
        s0 = score_as_log10M(c0)
        s1 = score_as_log10M(c1)
        if s0 >= s1:
            x, y = c0, c1
            swapped = False
        else:
            x, y = c1, c0
            swapped = True

    out = pd.DataFrame({"x": x, "y": y}).sort_values("x").reset_index(drop=True)

    # Merge duplicates in x (if any)
    if np.any(np.diff(out["x"].to_numpy()) == 0):
        out = out.groupby("x", as_index=False)["y"].mean()

    return out, swapped


# ----------------------------
# MWD metrics (NumPy 2.x compatible)
# ----------------------------

def compute_mwd_metrics(df_xy: pd.DataFrame):
    """
    df_xy: x=log10(M), y=dw/dlog10(M) (not necessarily normalized)

    Returns:
      area_raw, peak_log10M, peak_M, Mn, Mw, D
    """
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)

    if x.size < 3:
        raise ValueError("not enough points")

    area_raw = float(np.trapezoid(y, x))
    if not np.isfinite(area_raw) or area_raw == 0.0:
        raise ValueError("invalid/zero area under curve")

    w = y / area_raw  # normalized weight fraction density vs log10M

    idx = int(np.argmax(w))
    peak_log10M = float(x[idx])
    peak_M = float(10.0 ** peak_log10M)

    M = np.power(10.0, x)
    Mw = float(np.trapezoid(M * w, x))

    invMn = float(np.trapezoid(w / M, x))
    if invMn <= 0 or not np.isfinite(invMn):
        raise ValueError("invalid Mn integral")
    Mn = float(1.0 / invMn)

    D = float(Mw / Mn) if Mn > 0 else float("nan")

    return {
        "area_raw": area_raw,
        "peak_log10M": peak_log10M,
        "peak_M_gmol": peak_M,
        "Mn_gmol": Mn,
        "Mw_gmol": Mw,
        "D": D,
    }


def compute_living_quantities(Mn_gmol: float, monomer_mw: float, initiator_mol: float, monomer_mol: float):
    """
    Ideal living:
      DPn = Mn / M0
      monomer_consumed = DPn * n_I
      conversion X = consumed / n_M0
    """
    DPn = Mn_gmol / monomer_mw if monomer_mw > 0 else float("nan")
    monomer_consumed = DPn * initiator_mol
    X = monomer_consumed / monomer_mol if monomer_mol > 0 else float("nan")
    return float(DPn), float(monomer_consumed), float(X)


# ----------------------------
# Main
# ----------------------------

def iter_files(in_dir: Path, pattern: str, recursive: bool):
    if recursive:
        yield from in_dir.rglob(pattern)
    else:
        yield from in_dir.glob(pattern)


def main():
    ap = argparse.ArgumentParser(description="Summarize MWD curves (exp+sim) to CSV: peak, Mn, Mw, conversion.")
    ap.add_argument("--input", "-i", default=".", help="Folder containing files")
    ap.add_argument("--pattern", default="*.dat", help='Glob pattern, e.g. "*.dat" or "*mwd*" or "*.txt"')
    ap.add_argument("--recursive", action="store_true", help="Search subfolders")
    ap.add_argument("--out", "-o", default="mwd_summary.csv", help="Output CSV path")

    ap.add_argument("--monomer-mw", type=float, required=True, help="Monomer MW (g/mol), e.g. styrene 104.15")
    ap.add_argument("--initiator-mol", type=float, required=True, help="Initiator moles (mol)")
    ap.add_argument("--monomer-mol", type=float, required=True, help="Initial monomer moles (mol)")

    ap.add_argument("--force-x-col", choices=["auto", "c0", "c1", "swap"], default="auto",
                    help="Force which column is x=log10(M).")
    ap.add_argument("--debug", action="store_true", help="Verbose per-file report.")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()

    files = [p for p in iter_files(in_dir, args.pattern, args.recursive) if p.is_file()]
    print(f"Input folder: {in_dir}")
    print(f"Pattern: {args.pattern}  Recursive: {args.recursive}")
    print(f"Found {len(files)} file(s)")

    rows = []
    skipped = []

    for p in sorted(files):
        kind, tsec, t_ok = parse_time_seconds_from_name(p.stem)

        try:
            df2 = load_first_two_numeric_cols(p)
            df_xy, swapped = choose_xy_as_logm(df2, force=args.force_x_col)
            metrics = compute_mwd_metrics(df_xy)
            DPn, mon_cons, X = compute_living_quantities(
                Mn_gmol=metrics["Mn_gmol"],
                monomer_mw=args.monomer_mw,
                initiator_mol=args.initiator_mol,
                monomer_mol=args.monomer_mol,
            )

            xr = (float(df_xy["x"].min()), float(df_xy["x"].max()))
            if args.debug:
                print(f"[OK] {p.name:35s} kind={kind:7s} time_parsed={t_ok} time_s={tsec if t_ok else 'NA'} "
                      f"swapped={swapped} x_range=[{xr[0]:.3g},{xr[1]:.3g}] Mn={metrics['Mn_gmol']:.3g}")

            rows.append({
                "file": p.name,
                "source": kind,                 # exp / sim / unknown
                "time_parsed": bool(t_ok),
                "time_s": float(tsec) if t_ok else float("nan"),
                "swapped_cols": bool(swapped),
                "x_min_log10M": xr[0],
                "x_max_log10M": xr[1],

                **metrics,

                "monomer_MW_gmol": float(args.monomer_mw),
                "initiator_mol": float(args.initiator_mol),
                "monomer_mol": float(args.monomer_mol),

                "DPn": DPn,
                "monomer_consumed_mol": mon_cons,
                "conversion_X": X,
            })

        except Exception as e:
            skipped.append((p.name, str(e), kind, t_ok))
            if args.debug:
                print(f"[SKIP] {p.name:35s} reason={e}")

    if not rows:
        print("\nNo usable curves found. Here is why files were skipped:\n")
        if not files:
            print("  - No files matched your --input/--pattern.")
            print("    Try: --pattern '*mwd*' or use --recursive if files are in subfolders.")
        else:
            for fn, why, kind, t_ok in skipped[:50]:
                print(f"  - {fn}  |  kind={kind} time_parsed={t_ok}  |  {why}")
            if len(skipped) > 50:
                print(f"  ... and {len(skipped)-50} more.")
        raise SystemExit(1)

    df_out = pd.DataFrame(rows)
    df_out["time_sort"] = df_out["time_s"].fillna(1e99)
    df_out = df_out.sort_values(["time_sort", "source", "file"]).drop(columns=["time_sort"]).reset_index(drop=True)

    df_out.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df_out)} row(s) to: {out_csv}")

    if skipped and args.debug:
        print("\nSkipped files:")
        for fn, why, kind, t_ok in skipped:
            print(f"  - {fn} | kind={kind} time_parsed={t_ok} | {why}")


if __name__ == "__main__":
    main()
