#!/usr/bin/env python3
r"""
pm_market_status_check.py

Read a Polymarket markets JSON (like nba_today_tomorrow.json) and query Polymarket's
Gamma Markets API to determine:
- which markets are completed vs still live
- which markets are tradeable right now (acceptingOrders)

Works with inputs that are:
- a flat list of markets
- a dict grouped by date/league -> list[markets]
- a dict with a top-level "markets" list

Gamma API docs:
- GET market by id:    https://gamma-api.polymarket.com/markets/{id}
- GET market by slug:  https://gamma-api.polymarket.com/markets/slug/{slug}
- List markets (filters): https://gamma-api.polymarket.com/markets?slug=<slug>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd


GAMMA_BASE = "https://gamma-api.polymarket.com"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _flatten_input(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize common shapes into a list[dict] of market-ish entries.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("markets"), list):
            return [x for x in data["markets"] if isinstance(x, dict)]

        # our grouped export shape: {"meta": {...}, "byDate": {date: {league: [markets...]}}}
        if isinstance(data.get("byDate"), dict):
            return _flatten_input(data["byDate"])

        # grouped by date -> league -> list
        # or date -> list
        out: List[Dict[str, Any]] = []
        for k, v in data.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        it2 = dict(it)
                        it2.setdefault("date", k)
                        out.append(it2)
            elif isinstance(v, dict):
                # nested groups
                for k2, v2 in v.items():
                    if isinstance(v2, list):
                        for it in v2:
                            if isinstance(it, dict):
                                it2 = dict(it)
                                it2.setdefault("date", k)
                                it2.setdefault("league", k2)
                                out.append(it2)
        if out:
            return out

    raise ValueError("Unsupported input JSON shape. Expected list or dict with markets/grouped lists.")


def _extract_market_key(m: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (id, slug) as strings if present.
    """
    mid = m.get("id") or m.get("market_id") or m.get("marketId")
    if mid is not None:
        mid = str(mid)

    slug = m.get("slug") or m.get("market_slug") or m.get("marketSlug")
    if slug is not None:
        slug = str(slug)

    return mid, slug


def _gamma_get_market_by_id(session: requests.Session, market_id: str, timeout: float) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/markets/{market_id}"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _gamma_get_market_by_slug(session: requests.Session, slug: str, timeout: float) -> Dict[str, Any]:
    # Prefer the dedicated endpoint (1 market)
    url = f"{GAMMA_BASE}/markets/slug/{slug}"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _classify_market(g: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns fields for completion + tradeability.
    """
    active = g.get("active")
    closed = g.get("closed")
    archived = g.get("archived")
    accepting = g.get("acceptingOrders")  # key used by Gamma (boolean) in market object

    # Completion bucket
    if closed is True or archived is True or active is False:
        bucket = "COMPLETED"
    elif active is True and closed is False and archived is False:
        bucket = "LIVE"
    else:
        bucket = "UNKNOWN"

    # Trade bucket (more granular)
    if bucket == "COMPLETED":
        trade_bucket = "COMPLETED"
        tradeable_now = False
    else:
        if accepting is True and bucket == "LIVE":
            trade_bucket = "TRADEABLE_NOW"
            tradeable_now = True
        elif accepting is False and bucket in ("LIVE", "UNKNOWN"):
            trade_bucket = "NOT_TRADEABLE_NOW"
            tradeable_now = False
        else:
            trade_bucket = "UNKNOWN"
            tradeable_now = None

    return {
        "active": active,
        "closed": closed,
        "archived": archived,
        "acceptingOrders": accepting,
        "bucket": bucket,
        "trade_bucket": trade_bucket,
        "tradeable_now": tradeable_now,
    }


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True, help="Path to markets json (e.g. .\\nba_today_tomorrow.json)")
    ap.add_argument("--out-dir", required=True, help="Directory to write status reports")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds (default 10)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between API calls (rate-limit friendly)")
    ap.add_argument("--print", action="store_true", help="Print a console summary")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    raw = _load_json(in_path)
    markets = _flatten_input(raw)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "pm_market_status_check/1.1",
        "Accept": "application/json",
    })

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for m in markets:
        mid, slug = _extract_market_key(m)
        league = _pick(m, "league", "sport", "competition")
        date = _pick(m, "date", "day", "game_date")
        question = _pick(m, "question", "title")
        input_slug = slug

        g: Optional[Dict[str, Any]] = None
        try:
            if mid:
                g = _gamma_get_market_by_id(session, mid, timeout=args.timeout)
            elif slug:
                g = _gamma_get_market_by_slug(session, slug, timeout=args.timeout)
            else:
                raise ValueError("Market entry has neither id nor slug")

            c = _classify_market(g)

            row = {
                "id": str(g.get("id")) if g.get("id") is not None else (mid or ""),
                "slug": str(g.get("slug")) if g.get("slug") is not None else (input_slug or ""),
                "question": g.get("question") or question,
                "date": date,
                "league": league,
                "endDate": g.get("endDate"),
                "startDate": g.get("startDate"),
                "category": g.get("category"),
                **c,
                "fetched_at": _now_iso(),
            }
            rows.append(row)

        except Exception as e:
            errors.append({
                "id": mid,
                "slug": slug,
                "league": league,
                "date": date,
                "error": repr(e),
            })

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    # DataFrame + sorting (robust)
    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["bucket", "trade_bucket", "date", "league", "id", "slug"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols))

    # Write outputs
    csv_path = out_dir / "status_report.csv"
    json_path = out_dir / "status_report.json"
    summary_path = out_dir / "status_summary.json"
    errors_path = out_dir / "status_errors.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    _dump_json(json_path, rows)

    summary = {
        "generated_at": _now_iso(),
        "input": str(in_path),
        "total_in_input": len(markets),
        "fetched_ok": len(rows),
        "errors": len(errors),
        "by_bucket": df["bucket"].value_counts(dropna=False).to_dict() if "bucket" in df.columns else {},
        "by_trade_bucket": df["trade_bucket"].value_counts(dropna=False).to_dict() if "trade_bucket" in df.columns else {},
        "tradeable_now_count": int(df["tradeable_now"].fillna(False).sum()) if "tradeable_now" in df.columns else 0,
    }
    _dump_json(summary_path, summary)
    _dump_json(errors_path, errors)

    if args.print:
        print(f"[OK] Input markets: {len(markets)}")
        print(f"[OK] Fetched:      {len(rows)}")
        print(f"[OK] Errors:       {len(errors)}")
        if summary["by_bucket"]:
            print("\n== Completion ==")
            for k, v in summary["by_bucket"].items():
                print(f"  {k:>12}: {v}")
        if summary["by_trade_bucket"]:
            print("\n== Tradeability ==")
            for k, v in summary["by_trade_bucket"].items():
                print(f"  {k:>15}: {v}")
        print(f"\n[OUT] {csv_path}")
        print(f"[OUT] {json_path}")
        print(f"[OUT] {summary_path}")
        if errors:
            print(f"[OUT] {errors_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        sys.exit(130)
