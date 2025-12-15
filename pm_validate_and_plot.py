import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# IO helpers
# ----------------------------

def safe_read_parquet_or_csv(p: Path) -> pd.DataFrame:
    pq = p / "snapshots.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = p / "snapshots.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No snapshots.parquet/csv in {p}")

def find_market_dirs(data_dir: Path):
    # data_dir/markets/<league>/<market_id__slug>/day=YYYY-MM-DD
    root = data_dir / "markets"
    if not root.exists():
        raise FileNotFoundError(f"Missing {root} (is --data-dir correct?)")

    out = []
    for league_dir in root.iterdir():
        if not league_dir.is_dir():
            continue
        for mdir in league_dir.iterdir():
            if not mdir.is_dir():
                continue
            for daydir in mdir.glob("day=*"):
                if daydir.is_dir():
                    out.append((league_dir.name, mdir.name, daydir))
    return out

def corr(a: pd.Series, b: pd.Series):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 10:
        return None
    return float(df.corr().iloc[0, 1])

# ----------------------------
# Validation logic
# ----------------------------

CORE_COLS = ["ts_ms", "yes_mid", "no_mid", "yes_bid", "yes_ask", "no_bid", "no_ask"]

def validate_df(
    df: pd.DataFrame,
    depth: int,
    *,
    sum_tol_median: float,
    sum_tol_p95: float,
    sum_tol_pct: float,
    sum_tol_pct_threshold: float,
    max_gap_ms: int,
    stale_ms_info: int,
):
    """
    Returns:
      - ok (bool): core pass/fail
      - issues (list[str]): issues and warnings
      - metrics (dict): load-bearing stats for report
    """
    issues = []
    metrics = {}

    # Required columns
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        issues += [f"missing_col:{c}" for c in missing]
        return False, issues, metrics

    # Price bounds
    for col in ["yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask", "no_mid"]:
        s = df[col].dropna()
        bad = s[(s < 0) | (s > 1)]
        if len(bad) > 0:
            issues.append(f"price_out_of_bounds:{col}:{len(bad)}")

    # Crossed books (if both sides exist)
    d = df.dropna(subset=["yes_bid", "yes_ask"])
    crossed_yes = int((d["yes_bid"] > d["yes_ask"]).sum())
    if crossed_yes:
        issues.append(f"crossed_book:yes:{crossed_yes}")

    d = df.dropna(subset=["no_bid", "no_ask"])
    crossed_no = int((d["no_bid"] > d["no_ask"]).sum())
    if crossed_no:
        issues.append(f"crossed_book:no:{crossed_no}")

    # Snapshot liveness: look at time gaps between consecutive snapshots
    df2 = df.dropna(subset=["ts_ms"]).sort_values("ts_ms").copy()
    if len(df2) >= 2:
        gap = df2["ts_ms"].diff()
        max_gap = float(gap.max())
        metrics["max_snapshot_gap_ms"] = int(max_gap) if pd.notna(max_gap) else None
        big_gaps = int((gap > max_gap_ms).sum())
        if big_gaps:
            # This is a WARNING (can happen if the process stalled briefly),
            # but you asked for "legit" so we keep it visible.
            issues.append(f"snapshot_gaps_over_{max_gap_ms}ms:{big_gaps}")
    else:
        issues.append("too_few_rows_for_gap_check")

    # Sum-to-1 legitimacy check (your definition)
    # Use mid prices because they are the cleanest.
    sum_mid = (df2["yes_mid"] + df2["no_mid"]).astype("float64")
    sum_err = (sum_mid - 1.0).abs()

    metrics["sum_err_median"] = float(sum_err.median()) if len(sum_err) else None
    metrics["sum_err_p95"] = float(sum_err.quantile(0.95)) if len(sum_err) else None
    metrics["sum_err_pct_under_thresh"] = (
        float((sum_err <= sum_tol_pct_threshold).mean()) if len(sum_err) else None
    )

    # Pass/fail based on tolerances (tunable)
    sum_ok = True
    if metrics["sum_err_median"] is not None and metrics["sum_err_median"] > sum_tol_median:
        sum_ok = False
        issues.append(f"sum_not_close_to_1:median>{sum_tol_median:.4f}")

    if metrics["sum_err_p95"] is not None and metrics["sum_err_p95"] > sum_tol_p95:
        sum_ok = False
        issues.append(f"sum_not_close_to_1:p95>{sum_tol_p95:.4f}")

    if metrics["sum_err_pct_under_thresh"] is not None and metrics["sum_err_pct_under_thresh"] < sum_tol_pct:
        sum_ok = False
        issues.append(f"sum_not_close_to_1:pct_under_{sum_tol_pct_threshold:.3f}<{sum_tol_pct:.2f}")

    # Keep your old stale check as INFO only (do not fail the market on this)
    for side in ("yes", "no"):
        c = f"{side}_book_ts_ms"
        if c in df2.columns:
            dd = df2.dropna(subset=["ts_ms", c]).copy()
            if len(dd):
                age = (dd["ts_ms"] - dd[c]).astype("float64")
                stale_count = int((age > stale_ms_info).sum())
                if stale_count:
                    issues.append(f"info:{side}_book_ts_age_over_{stale_ms_info}ms:{stale_count}")

    # Correlation helper columns (optional)
    imb_col = f"yes_imb{depth}"
    pres_col = f"yes_pressure{depth}"
    if imb_col in df2.columns:
        df2["yes_ret"] = df2["yes_mid"].pct_change()
        metrics["corr_ret_vs_imb"] = corr(df2["yes_ret"], df2[imb_col])
    else:
        metrics["corr_ret_vs_imb"] = None

    if pres_col in df2.columns:
        if "yes_ret" not in df2.columns:
            df2["yes_ret"] = df2["yes_mid"].pct_change()
        metrics["corr_ret_vs_pressure"] = corr(df2["yes_ret"], df2[pres_col])
    else:
        metrics["corr_ret_vs_pressure"] = None

    # Define "core ok":
    # Must have: no missing cols, no out-of-bounds, no crossed books, AND sum-to-1 test passes.
    hard_fail = any(x.startswith(("missing_col:", "price_out_of_bounds:", "crossed_book:")) for x in issues)
    ok = (not hard_fail) and sum_ok

    return ok, issues, metrics

# ----------------------------
# Plotting
# ----------------------------

def plot_market(df: pd.DataFrame, out_png: Path, depth: int, title: str):
    df = df.dropna(subset=["ts_ms"]).sort_values("ts_ms").copy()
    df["t"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)

    yes_mid = df["yes_mid"].astype("float64")
    one_minus_no = (1.0 - df["no_mid"].astype("float64")) if "no_mid" in df.columns else None

    imb = df.get(f"yes_imb{depth}", pd.Series(index=df.index, dtype="float64"))
    pres = df.get(f"yes_pressure{depth}", pd.Series(index=df.index, dtype="float64"))

    sum_err = ((df["yes_mid"].astype("float64") + df["no_mid"].astype("float64")) - 1.0).abs()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df["t"], yes_mid, label="YES mid")
    if one_minus_no is not None:
        ax1.plot(df["t"], one_minus_no, label="1 - NO mid", linestyle="--")

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Price / Probability")
    ax1.set_title(title)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(df["t"], imb, label=f"YES imbalance@{depth}", alpha=0.8)
    ax2.plot(df["t"], pres, label=f"YES pressure@{depth}", alpha=0.8, linestyle=":")
    ax2.plot(df["t"], sum_err, label="abs(YES+NO-1)", alpha=0.8, linestyle="-.")

    ax2.set_ylabel("Imbalance / Pressure / Sum error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to pm_data_v2")
    ap.add_argument("--out-dir", default="pm_check")
    ap.add_argument("--depth", type=int, default=10)

    # Your “legit” thresholds (tunable)
    ap.add_argument("--sum-tol-median", type=float, default=0.010, help="median abs(YES+NO-1) must be <= this")
    ap.add_argument("--sum-tol-p95", type=float, default=0.030, help="p95 abs(YES+NO-1) must be <= this")
    ap.add_argument("--sum-tol-pct", type=float, default=0.90, help="fraction of samples under --sum-tol-pct-threshold")
    ap.add_argument("--sum-tol-pct-threshold", type=float, default=0.020, help="threshold used for pct check")

    # Liveness / gaps
    ap.add_argument("--max-gap-ms", type=int, default=5000, help="warn if snapshot gap larger than this")
    ap.add_argument("--stale-ms-info", type=int, default=15000, help="only informational; does not fail market")

    ap.add_argument("--only-league", default=None)
    ap.add_argument("--only-date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--max-markets", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_dirs = find_market_dirs(data_dir)

    rows = []
    n = 0
    for league, market_folder, daydir in sorted(market_dirs):
        if args.only_league and league != args.only_league:
            continue
        if args.only_date and f"day={args.only_date}" != daydir.name:
            continue

        day = daydir.name.replace("day=", "")

        try:
            df = safe_read_parquet_or_csv(daydir)
        except Exception as e:
            rows.append({
                "league": league,
                "market_folder": market_folder,
                "day": day,
                "ok": False,
                "plot_ok": False,
                "nrows": 0,
                "issues": f"read_fail:{type(e).__name__}:{e}",
            })
            continue

        ok, issues, metrics = validate_df(
            df,
            depth=args.depth,
            sum_tol_median=args.sum_tol_median,
            sum_tol_p95=args.sum_tol_p95,
            sum_tol_pct=args.sum_tol_pct,
            sum_tol_pct_threshold=args.sum_tol_pct_threshold,
            max_gap_ms=args.max_gap_ms,
            stale_ms_info=args.stale_ms_info,
        )

        title = f"{league} | {market_folder} | {day}"
        png = out_dir / "plots" / league / f"{market_folder}__{day}.png"
        try:
            plot_market(df, png, depth=args.depth, title=title)
            plot_ok = True
        except Exception as e:
            plot_ok = False
            issues.append(f"plot_fail:{type(e).__name__}:{e}")

        row = {
            "league": league,
            "market_folder": market_folder,
            "day": day,
            "ok": ok,
            "plot_ok": plot_ok,
            "nrows": int(len(df)),
            "issues": "|".join(issues) if issues else "",
            "plot_file": str(png),
        }

        # metrics into report
        for k, v in metrics.items():
            row[k] = v if v is None else (round(v, 6) if isinstance(v, float) else v)

        rows.append(row)

        n += 1
        if args.max_markets and n >= args.max_markets:
            break

    report = pd.DataFrame(rows)
    report = report.sort_values(["ok", "plot_ok", "league", "day"], ascending=[True, True, True, True])

    report_path = out_dir / "recording_validation_report_v2.csv"
    report.to_csv(report_path, index=False)

    hard_fail = report[report["ok"] == False]
    if len(hard_fail) == 0:
        print("VERDICT: YES (sum-to-1 + core sanity checks pass).")
    else:
        print("VERDICT: NO (some markets fail your legit criteria).")

    print("Report:", report_path)
    print("Plots :", out_dir / "plots")

if __name__ == "__main__":
    main()
