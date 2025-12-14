#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

GAMMA = "https://gamma-api.polymarket.com"
LIMIT = 200

DEFAULT_SLUG_EXCLUDE = ["btts", "spread", "total"]  # must NOT appear in slug

LEAGUE_TO_SLUG_PREFIX = {
    # England
    "premier": "epl-",
    "epl": "epl-",

    # Spain
    "liga": "lal-",
    "laliga": "lal-",
    "la-liga": "lal-",

    # Germany
    "bun": "bun-",

    # Italy
    "sea": "sea-",

    # France
    "fl1": "fl1-",

    # Netherlands
    "ere": "ere-",

    # Mexico
    "mex": "mex-",
}


def fetch_markets(offset: int) -> list[dict]:
    params = {
        "limit": LIMIT,
        "offset": offset,
        "closed": False,
        "archived": False,
        "active": True,
    }
    r = requests.get(f"{GAMMA}/markets", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_active_markets() -> list[dict]:
    offset = 0
    out: list[dict] = []
    while True:
        batch = fetch_markets(offset)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < LIMIT:
            break
        offset += LIMIT
    return out


def compute_dates(args_date: str | None, when: str | None, tz: ZoneInfo) -> list[str]:
    if args_date:
        return [args_date]

    when = (when or "today").lower()
    today = datetime.now(tz).date()

    if when == "today":
        return [today.strftime("%Y-%m-%d")]
    if when == "tomorrow":
        return [(today + timedelta(days=1)).strftime("%Y-%m-%d")]
    if when == "both":
        return [
            today.strftime("%Y-%m-%d"),
            (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]

    raise ValueError(f"Invalid --when value: {when}")


def normalize_leagues(raw_leagues: list[str]) -> list[str]:
    leagues = [x.strip().lower() for x in raw_leagues if x.strip()]
    if not leagues:
        return []

    if "all" in leagues:
        return sorted(set(LEAGUE_TO_SLUG_PREFIX.keys()))

    seen = set()
    out = []
    for l in leagues:
        if l not in LEAGUE_TO_SLUG_PREFIX:
            raise ValueError(
                f"Unknown league '{l}'. Allowed: {sorted(LEAGUE_TO_SLUG_PREFIX.keys())} or 'all'."
            )
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Find Polymarket markets by league slug prefix and date embedded in slug. Supports multi-league + today/tomorrow."
    )

    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--date",
        default=None,
        help="Date to match in slug, format YYYY-MM-DD. If set, overrides --when.",
    )
    g.add_argument(
        "--when",
        default=None,
        choices=["today", "tomorrow", "both"],
        help="Use Europe/Paris to compute date(s). Default: today.",
    )

    ap.add_argument(
        "--league",
        required=True,
        nargs="+",
        help="One or more leagues (space-separated), or 'all'. Example: --league epl bun fl1 | --league all",
    )

    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON filename (default auto).",
    )

    ap.add_argument(
        "--exclude",
        default=",".join(DEFAULT_SLUG_EXCLUDE),
        help="Comma-separated slug fragments to exclude (default: btts,spread,total)",
    )

    ap.add_argument(
        "--print",
        action="store_true",
        help="Print each match to stdout (in addition to JSON save).",
    )

    args = ap.parse_args()

    tz = ZoneInfo("Europe/Paris")
    dates = compute_dates(args.date, args.when, tz)
    leagues = normalize_leagues(args.league)

    date_patterns = [f"-{d}-".lower() for d in dates]
    slug_exclude = [x.strip().lower() for x in args.exclude.split(",") if x.strip()]
    league_prefixes = {lk: LEAGUE_TO_SLUG_PREFIX[lk].lower() for lk in leagues}

    all_markets = fetch_all_active_markets()

    matches: list[dict] = []
    date_regex = re.compile(r"-(\d{4}-\d{2}-\d{2})-")

    for m in all_markets:
        slug = m.get("slug") or ""
        if not isinstance(slug, str) or not slug:
            continue
        slug_l = slug.lower()

        matched_league = None
        matched_prefix = None
        for lk, pref in league_prefixes.items():
            if slug_l.startswith(pref):
                matched_league = lk
                matched_prefix = pref
                break
        if not matched_league:
            continue

        matched_date = None
        for pat in date_patterns:
            if pat in slug_l:
                matched_date = pat.strip("-")
                break
        if not matched_date:
            continue

        if any(x in slug_l for x in slug_exclude):
            continue

        mdate = matched_date
        mm = date_regex.search(slug_l)
        if mm:
            mdate = mm.group(1)

        matches.append(
            {
                "id": m.get("id"),
                "league": matched_league,
                "slugPrefix": matched_prefix,
                "date": mdate,
                "slug": slug,
                "question": m.get("question"),
                "active": m.get("active"),
                "closed": m.get("closed"),
                "archived": m.get("archived"),
                "outcomes": m.get("outcomes"),
                "clobTokenIds": m.get("clobTokenIds"),
                "url": f"https://polymarket.com/market/{slug}",
            }
        )

    matches.sort(key=lambda x: (x.get("date") or "", x.get("league") or "", x.get("slug") or ""))

    # ✅ GROUPED OUTPUT: byDate[date][league] = [markets...]
    by_date: dict[str, dict[str, list[dict]]] = {}
    for mm in matches:
        d = mm.get("date") or "unknown"
        l = mm.get("league") or "unknown"
        by_date.setdefault(d, {}).setdefault(l, []).append(mm)

    payload = {
        "meta": {
            "generatedAtParis": datetime.now(tz).isoformat(),
            "leagues": leagues,
            "dates": dates,
            "datePatternsInSlug": date_patterns,
            "slugExclude": slug_exclude,
            "source": {
                "api": "gamma",
                "endpoint": f"{GAMMA}/markets",
                "filters": {"active": True, "closed": False, "archived": False},
            },
            "count": len(matches),
        },
        "byDate": by_date,
    }

    if args.out:
        out_file = args.out
    else:
        leagues_part = "all" if len(leagues) == len(set(LEAGUE_TO_SLUG_PREFIX.keys())) else "-".join(leagues)
        dates_part = dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}"
        out_file = f"markets_grouped_{leagues_part}_{dates_part}.json"

    print(f"[OK] Found {len(matches)} market(s) for leagues={leagues} dates={dates}")
    print(f"     excludes={slug_exclude}")
    print(f"     grouped as byDate[date][league]")

    if args.print:
        for i, mm in enumerate(matches, 1):
            print(f"\n--- #{i} ---")
            print("league:", mm["league"])
            print("date:", mm["date"])
            print("slug:", mm["slug"])
            print("question:", mm["question"])
            print("tokens:", mm["clobTokenIds"])
            print("url:", mm["url"])

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_file}")


if __name__ == "__main__":
    main()
