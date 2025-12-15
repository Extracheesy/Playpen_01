import argparse
from pathlib import Path
import json
import shutil
import time
from datetime import timedelta

import pandas as pd
import requests


GAMMA_BASE = "https://gamma-api.polymarket.com"


def parse_dt(s: str | None):
    if not s:
        return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")


def extract_markets_from_grouped_json(markets_json_path: Path) -> list[dict]:
    data = json.loads(markets_json_path.read_text(encoding="utf-8"))
    markets = []

    # expected structure: {"meta":..., "byDate": {date: {league: [market,...]}}}
    by_date = data.get("byDate", {})
    for date, leagues in by_date.items():
        if not isinstance(leagues, dict):
            continue
        for league, lst in leagues.items():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if isinstance(m, dict) and m.get("id") and m.get("slug"):
                    markets.append(
                        {
                            "id": str(m["id"]),
                            "slug": str(m["slug"]),
                            "league": str(m.get("league", league)),
                            "date": str(m.get("date", date)),
                        }
                    )
    return markets


def fetch_market_from_gamma(market_id: str | None, slug: str | None, session: requests.Session) -> dict | None:
    """
    Gamma /markets is a list endpoint. We'll try several ways:
    - /markets?id=<id>
    - /markets?slug=<slug>
    - fallback: /markets?limit=100&query=<slug> and filter exact slug
    """
    # 1) try id
    if market_id:
        r = session.get(f"{GAMMA_BASE}/markets", params={"id": market_id}, timeout=20)
        if r.ok:
            j = r.json()
            # could be a dict or list depending on API behavior
            if isinstance(j, dict) and j.get("id"):
                return j
            if isinstance(j, list) and len(j) > 0:
                return j[0]

    # 2) try slug
    if slug:
        r = session.get(f"{GAMMA_BASE}/markets", params={"slug": slug}, timeout=20)
        if r.ok:
            j = r.json()
            if isinstance(j, dict) and j.get("slug"):
                return j
            if isinstance(j, list):
                # if list, pick exact match if possible
                for it in j:
                    if isinstance(it, dict) and it.get("slug") == slug:
                        return it
                if len(j) > 0:
                    return j[0]

    # 3) query fallback
    if slug:
        r = session.get(f"{GAMMA_BASE}/markets", params={"limit": 100, "query": slug}, timeout=20)
        if r.ok:
            j = r.json()
            if isinstance(j, list):
                for it in j:
                    if isinstance(it, dict) and it.get("slug") == slug:
                        return it

    return None


def build_schedule_map(markets: list[dict], rate_sleep: float = 0.05) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns {slug: (start_dt, end_dt)}
    """
    out = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": "pm_cleanup_by_schedule/1.0"})

        for m in markets:
            slug = m["slug"]
            mid = m["id"]

            info = fetch_market_from_gamma(mid, slug, session=session)
            if not info:
                continue

            # These keys depend on Gamma’s schema; we try common variants.
            start = parse_dt(info.get("startDate") or info.get("startTime") or info.get("eventStartDate"))
            end = parse_dt(info.get("endDate") or info.get("resolveDate") or info.get("eventEndDate"))

            # If still missing, keep NaT (caller will skip)
            out[slug] = (start, end)

            time.sleep(rate_sleep)
    return out


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


def read_snapshots(daydir: Path):
    pq = daydir / "snapshots.parquet"
    if pq.exists():
        return pd.read_parquet(pq), pq, "parquet"
    csv = daydir / "snapshots.csv"
    if csv.exists():
        return pd.read_csv(csv), csv, "csv"
    raise FileNotFoundError(f"No snapshots.parquet/csv in {daydir}")


def write_snapshots(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def extract_slug(market_folder: str) -> str | None:
    # folder looks like "783811__epl-cry-mac-2025-12-14-mac"
    if "__" not in market_folder:
        return None
    return market_folder.split("__", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--markets-json", required=True)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--backup", action="store_true")

    ap.add_argument("--pre-min", type=int, default=30)
    ap.add_argument("--post-min", type=int, default=180)

    ap.add_argument("--only-league", default=None)
    ap.add_argument("--only-date", default=None)

    ap.add_argument("--gamma-sleep", type=float, default=0.05, help="sleep between Gamma calls to be polite")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # 1) load all (id, slug) from your grouped json
    markets = extract_markets_from_grouped_json(Path(args.markets_json))

    # optional filtering here (based on the json metadata)
    if args.only_league:
        markets = [m for m in markets if m.get("league") == args.only_league]
    if args.only_date:
        markets = [m for m in markets if m.get("date") == args.only_date]

    if not markets:
        print("[ERR] No markets extracted from markets json (check file path/format).")
        return

    # 2) build schedule map from Gamma
    print(f"[INFO] Fetching schedules from Gamma for {len(markets)} markets ...")
    sched = build_schedule_map(markets, rate_sleep=args.gamma_sleep)

    total = trimmed = skipped = 0
    pre = timedelta(minutes=args.pre_min)
    post = timedelta(minutes=args.post_min)

    for league, market_folder, daydir in find_day_dirs(data_dir):
        if args.only_league and league != args.only_league:
            continue
        if args.only_date and daydir.name != f"day={args.only_date}":
            continue

        total += 1
        slug = extract_slug(market_folder)
        if not slug:
            print(f"[SKIP] {league} | {market_folder} | {daydir.name}: cannot parse slug from folder name")
            skipped += 1
            continue

        if slug not in sched:
            print(f"[SKIP] {league} | {market_folder} | {daydir.name}: slug not in Gamma schedule map")
            skipped += 1
            continue

        start_dt, end_dt = sched[slug]
        if pd.isna(start_dt) or pd.isna(end_dt):
            print(f"[SKIP] {league} | {slug} | {daydir.name}: missing start/end from Gamma (start={start_dt}, end={end_dt})")
            skipped += 1
            continue

        df, src_path, fmt = read_snapshots(daydir)
        if "ts_ms" not in df.columns:
            print(f"[SKIP] {league} | {slug} | {daydir.name}: missing ts_ms")
            skipped += 1
            continue

        df = df.sort_values("ts_ms").copy()
        df["t"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)

        t0 = start_dt - pre
        t1 = end_dt + post

        df_clean = df[(df["t"] >= t0) & (df["t"] <= t1)].drop(columns=["t"])
        if df_clean.empty:
            print(f"[SKIP] {league} | {slug} | {daydir.name}: window produced 0 rows")
            skipped += 1
            continue

        dropped = len(df) - len(df_clean)

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
        print(f"[TRIM:schedule] {league} | {slug} | {daydir.name}: kept={len(df_clean)} dropped={dropped}")

    print(f"\nDone. scanned={total} trimmed={trimmed} skipped={skipped}")


if __name__ == "__main__":
    main()
