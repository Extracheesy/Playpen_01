import argparse
from pathlib import Path
import shutil
import pandas as pd


def read_snapshots(daydir: Path) -> tuple[pd.DataFrame, Path, str]:
    pq = daydir / "snapshots.parquet"
    if pq.exists():
        return pd.read_parquet(pq), pq, "parquet"
    csv = daydir / "snapshots.csv"
    if csv.exists():
        return pd.read_csv(csv), csv, "csv"
    raise FileNotFoundError(f"No snapshots.parquet or snapshots.csv in {daydir}")


def write_snapshots(df: pd.DataFrame, target: Path, fmt: str):
    if fmt == "parquet":
        df.to_parquet(target, index=False)
    else:
        df.to_csv(target, index=False)


def find_day_dirs(data_dir: Path):
    root = data_dir / "markets"
    if not root.exists():
        raise FileNotFoundError(f"Missing {root} (is --data-dir correct?)")

    for league_dir in root.iterdir():
        if not league_dir.is_dir():
            continue
        for market_dir in league_dir.iterdir():
            if not market_dir.is_dir():
                continue
            for daydir in market_dir.glob("day=*"):
                if daydir.is_dir():
                    yield league_dir.name, market_dir.name, daydir


def detect_completion_index(
    df_sorted: pd.DataFrame,
    *,
    prob_hi: float,
    prob_lo: float,
    stable_n: int,
    max_sum_err: float | None,
) -> int | None:
    if not {"ts_ms", "yes_mid", "no_mid"}.issubset(df_sorted.columns):
        return None

    d = df_sorted.dropna(subset=["ts_ms", "yes_mid", "no_mid"]).copy()
    if len(d) < stable_n:
        return None

    yes = d["yes_mid"].astype("float64")
    no = d["no_mid"].astype("float64")

    done = ((yes >= prob_hi) & (no <= prob_lo)) | ((yes <= prob_lo) & (no >= prob_hi))

    if max_sum_err is not None:
        sum_err = (yes + no - 1.0).abs()
        done = done & (sum_err <= max_sum_err)

    run = done.rolling(stable_n).sum()
    hit = run[run >= stable_n]
    if hit.empty:
        return None

    end_pos = int(hit.index[0])
    start_pos = end_pos - stable_n + 1
    return max(0, start_pos)


def detect_activity_start_pos_v3(
    df_sorted: pd.DataFrame,
    *,
    activity_range: float,
    activity_window: int,
    min_rows: int,
    depth: int,
    activity_imb: float | None,
    activity_pressure: float | None,
) -> int | None:
    """
    Activity means: in some window, price range (max-min) >= activity_range.
    This is robust to starting mid-game and to not relying on the first tick.

    Returns the first position where activity becomes true.
    """
    if "yes_mid" not in df_sorted.columns or len(df_sorted) < max(min_rows, activity_window):
        return None

    y = df_sorted["yes_mid"].astype("float64")

    # Rolling range
    roll_max = y.rolling(activity_window).max()
    roll_min = y.rolling(activity_window).min()
    rng = roll_max - roll_min

    hit = rng[rng >= activity_range]
    if not hit.empty:
        return int(hit.index[0])  # first moment we see enough movement

    # Optional imbalance/pressure activity triggers
    candidates = []
    imb_col = f"yes_imb{depth}"
    pres_col = f"yes_pressure{depth}"

    if activity_imb is not None and imb_col in df_sorted.columns:
        imb = df_sorted[imb_col].astype("float64").abs()
        t = imb >= activity_imb
        if t.any():
            candidates.append(int(t.idxmax()))

    if activity_pressure is not None and pres_col in df_sorted.columns:
        pres = df_sorted[pres_col].astype("float64").abs()
        t = pres >= activity_pressure
        if t.any():
            candidates.append(int(t.idxmax()))

    if candidates:
        return min(candidates)

    return None


def detect_flatline_cut_pos(
    df_sorted: pd.DataFrame,
    *,
    eps: float,
    stable_n: int,
    min_rows_after_activity: int,
    activity_start_pos: int,
) -> int | None:
    if "yes_mid" not in df_sorted.columns or len(df_sorted) < stable_n + 2:
        return None

    start_search = activity_start_pos + min_rows_after_activity
    if start_search >= len(df_sorted) - stable_n:
        return None

    y = df_sorted["yes_mid"].astype("float64")
    dy = y.diff().abs()

    flat = dy <= eps
    flat.iloc[:start_search] = False

    run = flat.rolling(stable_n).sum()
    hit = run[run >= stable_n]
    if hit.empty:
        return None

    end_pos = int(hit.index[0])
    start_pos = end_pos - stable_n + 1
    return max(0, start_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--only-league", default=None)
    ap.add_argument("--only-date", default=None)
    ap.add_argument("--max-markets", type=int, default=None)

    # completion
    ap.add_argument("--prob-hi", type=float, default=0.995)
    ap.add_argument("--prob-lo", type=float, default=0.005)
    ap.add_argument("--stable-n", type=int, default=15)
    ap.add_argument("--max-sum-err", type=float, default=0.02, help="set <0 to disable")

    # flatline
    ap.add_argument("--flat-eps", type=float, default=1e-4)
    ap.add_argument("--flat-stable-n", type=int, default=30)
    ap.add_argument("--flat-min-after-activity", type=int, default=60)

    # activity (NEW robust version)
    ap.add_argument("--activity-range", type=float, default=0.01, help="rolling(max-min) >= this triggers activity")
    ap.add_argument("--activity-window", type=int, default=120, help="window length (rows) for rolling range")
    ap.add_argument("--activity-min-rows", type=int, default=300, help="need at least this many rows to attempt activity detection")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--activity-imb", type=float, default=None)
    ap.add_argument("--activity-pressure", type=float, default=None)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    max_sum_err = None if args.max_sum_err < 0 else args.max_sum_err

    total = 0
    trimmed = 0
    reason_counts = {"completion": 0, "flatline": 0, "skip": 0}

    for league, market_folder, daydir in find_day_dirs(data_dir):
        if args.only_league and league != args.only_league:
            continue
        if args.only_date and daydir.name != f"day={args.only_date}":
            continue

        df, src_path, fmt = read_snapshots(daydir)
        if "ts_ms" not in df.columns:
            print(f"[SKIP] {league} | {market_folder} | {daydir.name}: missing ts_ms")
            reason_counts["skip"] += 1
            continue

        df_sorted = df.sort_values("ts_ms").reset_index(drop=True)

        cut_pos = detect_completion_index(
            df_sorted,
            prob_hi=args.prob_hi,
            prob_lo=args.prob_lo,
            stable_n=args.stable_n,
            max_sum_err=max_sum_err,
        )

        cut_reason = None

        if cut_pos is None:
            act_pos = detect_activity_start_pos_v3(
                df_sorted,
                activity_range=args.activity_range,
                activity_window=args.activity_window,
                min_rows=args.activity_min_rows,
                depth=args.depth,
                activity_imb=args.activity_imb,
                activity_pressure=args.activity_pressure,
            )

            if act_pos is None:
                print(f"[SKIP] {league} | {market_folder} | {daydir.name}: no activity detected")
                reason_counts["skip"] += 1
                total += 1
                if args.max_markets and total >= args.max_markets:
                    break
                continue

            flat_cut = detect_flatline_cut_pos(
                df_sorted,
                eps=args.flat_eps,
                stable_n=args.flat_stable_n,
                min_rows_after_activity=args.flat_min_after_activity,
                activity_start_pos=act_pos,
            )
            if flat_cut is not None:
                cut_pos = flat_cut
                cut_reason = "flatline"
        else:
            cut_reason = "completion"

        total += 1

        if cut_pos is None:
            print(f"[SKIP] {league} | {market_folder} | {daydir.name}: no completion/flatline found")
            reason_counts["skip"] += 1
        else:
            df_clean = df_sorted.iloc[: cut_pos + 1].copy()
            dropped = len(df_sorted) - len(df_clean)

            if args.inplace:
                out_path = src_path
                if args.backup:
                    bak = src_path.with_suffix(src_path.suffix + ".bak")
                    if not bak.exists():
                        shutil.copy2(src_path, bak)
                write_snapshots(df_clean, out_path, fmt)
            else:
                out_path = daydir / ("snapshots.cleaned.parquet" if fmt == "parquet" else "snapshots.cleaned.csv")
                write_snapshots(df_clean, out_path, fmt)

            trimmed += 1
            reason_counts[cut_reason] += 1
            print(f"[TRIM:{cut_reason}] {league} | {market_folder} | {daydir.name}: kept={len(df_clean)} dropped={dropped} -> {out_path.name}")

        if args.max_markets and total >= args.max_markets:
            break

    print("\nDone.")
    print(f"Markets scanned={total}, trimmed={trimmed}")
    print("Trim reasons:", reason_counts)


if __name__ == "__main__":
    main()
