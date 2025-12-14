#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import requests

GAMMA = "https://gamma-api.polymarket.com"
LIMIT = 200

SLUG_PREFIX = "epl-"
SLUG_EXCLUDE = ["btts", "spread", "total"]  # must NOT appear in slug


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Date to match in slug, format YYYY-MM-DD (example: 2025-12-14)")
    ap.add_argument("--out", default=None, help="Output JSON filename (default auto)")
    args = ap.parse_args()

    # pattern must be inside slug like "-2025-12-14-"
    slug_must_contain = f"-{args.date}-".lower()

    offset = 0
    matches: list[dict] = []

    while True:
        batch = fetch_markets(offset)
        if not batch:
            break

        for m in batch:
            slug = m.get("slug") or ""
            if not isinstance(slug, str) or not slug:
                continue

            slug_l = slug.lower()

            if not slug_l.startswith(SLUG_PREFIX):
                continue
            if slug_must_contain not in slug_l:
                continue
            if any(x in slug_l for x in SLUG_EXCLUDE):
                continue

            matches.append({
                "id": m.get("id"),
                "slug": slug,
                "question": m.get("question"),
                "active": m.get("active"),
                "closed": m.get("closed"),
                "archived": m.get("archived"),
                "outcomes": m.get("outcomes"),
                "clobTokenIds": m.get("clobTokenIds"),
                "url": f"https://polymarket.com/market/{slug}",
            })

        if len(batch) < LIMIT:
            break
        offset += LIMIT

    matches.sort(key=lambda x: x["slug"])

    out_file = args.out or f"epl_main_markets_{args.date}.json"

    print(f"[OK] Found {len(matches)} market(s) for date={args.date}")
    print(f"     slug startswith '{SLUG_PREFIX}', contains '{slug_must_contain}', excludes {SLUG_EXCLUDE}")

    for i, mm in enumerate(matches, 1):
        print(f"\n--- #{i} ---")
        print("slug:", mm["slug"])
        print("question:", mm["question"])
        print("tokens:", mm["clobTokenIds"])
        print("url:", mm["url"])

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_file}")


if __name__ == "__main__":
    main()
