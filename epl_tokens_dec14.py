#!/usr/bin/env python3
"""
Fetch Polymarket token IDs for a list of EPL fixtures on a target date.

- Uses Gamma Markets API: https://gamma-api.polymarket.com/markets
- Optionally auto-resolves EPL tag_id via /tags
- Filters markets by tag_id + active + not closed + not archived
- Filters to markets whose startDate/gameStartTime/eventStartTime fall within the requested day window
- Fuzzy-matches fixture strings against market question/groupItemTitle
- Outputs tokens mapped to outcomes (team names) for each matched market
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


GAMMA_BASE = "https://gamma-api.polymarket.com"


FIXTURES_DEC14 = [
    "Crystal Palace vs Manchester City",
    "Nottingham Forest vs Tottenham Hotspur",
    "Sunderland vs Newcastle United",
    "West Ham United vs Aston Villa",
    "Brentford vs Leeds United",
]


# ---- helpers ----

def iso_z(dt: datetime) -> str:
    """ISO8601 with Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def try_parse_json_maybe_string(v: Any) -> Any:
    """
    Gamma fields like outcomes/clobTokenIds sometimes come back as JSON-encoded strings.
    If v is a string containing JSON, parse it; else return v unchanged.
    """
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def norm(s: str) -> str:
    s = s.lower()
    s = s.replace("&", "and")
    s = re.sub(r"\bvs\.?\b", "vs", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # common football name variants
    s = s.replace("manchester city", "man city")
    s = s.replace("manchester united", "man utd")
    s = s.replace("tottenham hotspur", "tottenham spurs")
    s = s.replace("newcastle united", "newcastle")
    s = s.replace("nottingham forest", "forest")
    s = s.replace("aston villa", "villa")
    s = s.replace("west ham united", "west ham")
    return s


def fixture_tokens(fixture: str) -> List[str]:
    """
    tokens for matching: includes both raw club names and normalized variations.
    """
    n = norm(fixture)
    parts = [p.strip() for p in n.split("vs")]
    toks = []
    for p in parts:
        if not p:
            continue
        toks.append(p)
        # add some extra short tokens for robustness
        toks.extend([w for w in p.split() if len(w) >= 4])
    # de-dup while preserving order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_match(fixture: str, market_text: str) -> float:
    """
    Simple fuzzy score based on presence of team tokens.
    """
    ftoks = fixture_tokens(fixture)
    mt = norm(market_text)

    hits = 0
    for t in ftoks:
        if t in mt:
            hits += 1

    # weight: must include both sides' main tokens (best effort)
    sides = [p.strip() for p in norm(fixture).split("vs") if p.strip()]
    side_hits = sum(1 for s in sides if s and s in mt)

    # base score: token hit ratio + side match bonus
    if not ftoks:
        return 0.0
    return (hits / len(ftoks)) + (0.75 * side_hits)


def pick_best_market(fixture: str, markets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Choose the best matching market for a fixture.
    """
    best = None
    best_score = 0.0

    for m in markets:
        text = " ".join([
            str(m.get("question") or ""),
            str(m.get("groupItemTitle") or ""),
            str(m.get("slug") or ""),
        ])
        sc = score_match(fixture, text)
        if sc > best_score:
            best_score = sc
            best = m

    # threshold to avoid nonsense matches
    return best if best is not None and best_score >= 1.0 else None


def parse_market_start(m: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to extract a start time from common Gamma fields.
    """
    for k in ("eventStartTime", "gameStartTime", "startDate", "startDateIso"):
        v = m.get(k)
        if not v:
            continue
        if isinstance(v, str):
            try:
                # handle trailing Z
                vv = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(vv)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None


def in_window(dt: Optional[datetime], start_utc: datetime, end_utc: datetime) -> bool:
    if dt is None:
        return False
    return start_utc <= dt < end_utc


# ---- gamma calls ----

def gamma_get(path: str, params: Dict[str, Any], timeout: int = 30) -> Any:
    url = f"{GAMMA_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def resolve_tag_id(label_hint: str = "epl") -> Optional[int]:
    """
    Try to find an EPL tag by scanning /tags.
    """
    tags = gamma_get("/tags", params={"limit": 1000, "offset": 0})
    # tags can be a list
    best = None
    best_score = 0
    for t in tags if isinstance(tags, list) else []:
        slug = str(t.get("slug") or "").lower()
        label = str(t.get("label") or "").lower()
        tid = t.get("id")
        if tid is None:
            continue
        sc = 0
        if label_hint in slug:
            sc += 2
        if "premier" in label and "league" in label:
            sc += 3
        if label_hint == "epl" and "epl" in (slug + " " + label):
            sc += 4
        if sc > best_score:
            best_score = sc
            best = int(tid)
    return best


def fetch_epl_markets_for_day(
    date_yyyy_mm_dd: str,
    tag_id: int,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Fetch markets for a specific UTC day window (00:00 to 24:00 UTC).
    We also filter client-side using eventStartTime/gameStartTime/startDate since
    Gamma provides multiple relevant time fields.
    """
    day = datetime.fromisoformat(date_yyyy_mm_dd).replace(tzinfo=timezone.utc)
    start_utc = day
    end_utc = day + timedelta(days=1)

    all_markets: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "tag_id": tag_id,
            "active": True,
            "closed": False,
            "archived": False,
            # coarse server-side filter; still do strict client-side below
            "start_date_min": iso_z(start_utc),
            "start_date_max": iso_z(end_utc),
            "include_tag": True,
        }
        batch = gamma_get("/markets", params=params)
        if not isinstance(batch, list) or not batch:
            break

        all_markets.extend(batch)

        if len(batch) < limit:
            break
        offset += limit

    # strict filter by whichever start field exists
    filtered: List[Dict[str, Any]] = []
    for m in all_markets:
        dt = parse_market_start(m)
        if in_window(dt, start_utc, end_utc):
            filtered.append(m)

    return filtered


# ---- output ----

@dataclass
class Matched:
    fixture: str
    market_id: str
    slug: str
    question: str
    start_time_utc: str
    outcomes: List[str]
    token_ids: List[str]


def extract_tokens(m: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    outcomes = try_parse_json_maybe_string(m.get("outcomes")) or []
    token_ids = try_parse_json_maybe_string(m.get("clobTokenIds")) or []

    # normalize to lists of strings
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    if isinstance(token_ids, str):
        token_ids = [token_ids]

    outcomes = [str(x) for x in outcomes] if isinstance(outcomes, list) else []
    token_ids = [str(x) for x in token_ids] if isinstance(token_ids, list) else []

    return outcomes, token_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-12-14", help="UTC day (YYYY-MM-DD). Default: 2025-12-14")
    ap.add_argument("--tag-id", type=int, default=306, help="Gamma tag_id for EPL. Default: 306")
    ap.add_argument("--auto-tag", action="store_true", help="Auto-resolve EPL tag_id from /tags (overrides --tag-id)")
    ap.add_argument("--min-matches", type=int, default=5, help="Expected number of fixtures to match")
    ap.add_argument("--json-out", default="epl_tokens_2025-12-14.json", help="JSON output file")
    ap.add_argument("--csv-out", default="epl_tokens_2025-12-14.csv", help="CSV output file")
    args = ap.parse_args()

    fixtures = FIXTURES_DEC14

    tag_id = args.tag_id
    if args.auto_tag:
        resolved = resolve_tag_id("epl")
        if resolved is None:
            print("[ERR] Could not auto-resolve EPL tag_id from /tags", file=sys.stderr)
            return 2
        tag_id = resolved
        print(f"[INFO] Auto tag_id={tag_id}")

    print(f"[INFO] Fetching EPL markets for {args.date} (tag_id={tag_id}) ...")
    markets = fetch_epl_markets_for_day(args.date, tag_id=tag_id)
    print(f"[INFO] Candidate markets found: {len(markets)}")

    matched: List[Matched] = []
    unmatched: List[str] = []

    for fx in fixtures:
        best = pick_best_market(fx, markets)
        if not best:
            unmatched.append(fx)
            continue

        dt = parse_market_start(best)
        outcomes, token_ids = extract_tokens(best)

        matched.append(Matched(
            fixture=fx,
            market_id=str(best.get("id")),
            slug=str(best.get("slug") or ""),
            question=str(best.get("question") or ""),
            start_time_utc=iso_z(dt) if dt else "",
            outcomes=outcomes,
            token_ids=token_ids,
        ))

    # write JSON
    out_json = {
        "date_utc": args.date,
        "tag_id": tag_id,
        "matched": [
            {
                "fixture": m.fixture,
                "market_id": m.market_id,
                "slug": m.slug,
                "question": m.question,
                "start_time_utc": m.start_time_utc,
                "outcomes": m.outcomes,
                "token_ids": m.token_ids,
                "outcome_to_token": dict(zip(m.outcomes, m.token_ids)) if m.outcomes and m.token_ids else {},
            }
            for m in matched
        ],
        "unmatched": unmatched,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # write CSV (one row per outcome/token)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date_utc", "fixture", "market_id", "slug", "question", "start_time_utc", "outcome", "token_id"])
        for m in matched:
            if m.outcomes and m.token_ids and len(m.outcomes) == len(m.token_ids):
                for o, tid in zip(m.outcomes, m.token_ids):
                    w.writerow([args.date, m.fixture, m.market_id, m.slug, m.question, m.start_time_utc, o, tid])
            else:
                # still emit one row with raw lists for debugging
                w.writerow([args.date, m.fixture, m.market_id, m.slug, m.question, m.start_time_utc,
                            json.dumps(m.outcomes, ensure_ascii=False), json.dumps(m.token_ids)])

    print(f"[OK] Matched {len(matched)} fixtures. JSON={args.json_out} CSV={args.csv_out}")
    if unmatched:
        print("[WARN] Unmatched fixtures:")
        for u in unmatched:
            print("  -", u)

    # enforce expected count if you want it strict
    if len(matched) < args.min_matches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
