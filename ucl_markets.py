import requests
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


GAMMA_BASE = "https://gamma-api.polymarket.com"


# ------------------------------------------------
# 0) HUMAN NAME -> TAG SLUG RESOLVER
# ------------------------------------------------
def resolve_market_tag_slug(market_name: str) -> str:
    """
    Convert a human-friendly market name (e.g. 'champions league games',
    'NBA games', 'Premier League games') into a Polymarket tag slug
    (e.g. 'ucl', 'nba', 'epl').
    """
    if not market_name:
        raise ValueError("market_name cannot be empty")

    name = market_name.strip().lower()

    alias_map = {
        # --- Football / Soccer ---
        "epl": "epl",
        "pl": "epl",
        "premier league": "epl",
        "english premier league": "epl",
        "premier league games": "epl",
        "english premier league games": "epl",

        # --- Existing mappings ---
        "ucl": "ucl",
        "champions league": "ucl",
        "champions league games": "ucl",
        "uefa champions league": "ucl",

        "nba": "nba",
        "nba games": "nba",
        "basketball": "nba",
        "basketball games": "nba",

        "nfl": "nfl",
        "nfl games": "nfl",
        "american football": "nfl",

        "tennis": "tennis",
        "tennis matches": "tennis",
        "tennis games": "tennis",

        "rugby": "rugby",
        "rugby games": "rugby",
    }

    if name in alias_map:
        return alias_map[name]

    # Heuristic fallbacks
    if "premier league" in name or name == "pl" or "epl" in name:
        return "epl"
    if "champions league" in name:
        return "ucl"
    if "nba" in name or "basketball" in name:
        return "nba"
    if "nfl" in name or "american football" in name:
        return "nfl"
    if "tennis" in name:
        return "tennis"
    if "rugby" in name:
        return "rugby"

    # Default: assume user passed a valid slug already
    return name


class PolymarketSportsFetcher:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def get_tag_id(self, slug: str) -> str:
        slug = slug.strip().lower()

        # First: direct lookup by slug
        url = f"{GAMMA_BASE}/tags/slug/{slug}"
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                tag_id = data.get("id")
                if tag_id:
                    return tag_id
        except Exception:
            pass

        # Fallback: list tags & search
        offset = 0
        page_size = 500

        while True:
            params = {"limit": page_size, "offset": offset}
            r = self.session.get(f"{GAMMA_BASE}/tags", params=params, timeout=10)
            r.raise_for_status()
            tags = r.json()

            if not tags:
                break

            for t in tags:
                t_slug = (t.get("slug") or "").lower()
                t_label = (t.get("label") or "").lower()
                if t_slug == slug or slug in t_slug or slug in t_label:
                    tag_id = t.get("id")
                    if tag_id:
                        return tag_id

            if len(tags) < page_size:
                break

            offset += page_size

        raise RuntimeError(f"Could not find tag_id for slug/name '{slug}' from Gamma /tags endpoint")

    def get_markets_for_tag(
        self,
        tag_slug: str,
        sports_market_types: Optional[List[str]] = None,
        limit_per_page: int = 200,
        start_utc_min: Optional[datetime] = None,
        start_utc_max: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        if sports_market_types is None:
            sports_market_types = ["moneyline"]

        tag_id = self.get_tag_id(tag_slug)

        all_markets: List[Dict[str, Any]] = []
        offset = 0

        while True:
            params: Dict[str, Any] = {
                "limit": limit_per_page,
                "offset": offset,
                "tag_id": tag_id,
                "closed": "false",
                "include_tag": "false",
                "ascending": "false",
                "order": "-liquidity_num",
            }

            params["sports_market_types"] = sports_market_types

            r = self.session.get(f"{GAMMA_BASE}/markets", params=params, timeout=10)
            r.raise_for_status()
            page = r.json()

            if not page:
                break

            for m in page:
                if m.get("closed"):
                    continue
                if m.get("active") is False:
                    continue

                if start_utc_min is not None or start_utc_max is not None:
                    raw_start = m.get("gameStartTime") or m.get("eventStartTime")
                    if not raw_start:
                        continue

                    try:
                        ts = str(raw_start).replace("Z", "+00:00")
                        dt = datetime.fromisoformat(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                    except Exception:
                        continue

                    if start_utc_min is not None and dt < start_utc_min:
                        continue
                    if start_utc_max is not None and dt >= start_utc_max:
                        continue

                all_markets.append(m)

            if len(page) < limit_per_page:
                break

            offset += limit_per_page

        return all_markets


def pretty_print_sports_markets(
    markets: List[Dict[str, Any]],
    tag_name: str,
    resolved_slug: str,
    date_info: str,
) -> None:
    print(f"Requested market name : '{tag_name}'")
    print(f"Resolved tag slug     : '{resolved_slug}'")
    print(f"Date filter (UTC)     : {date_info if date_info else '<none>'}")
    print(f"Found {len(markets)} markets.\n")

    for m in markets:
        question = m.get("question")
        slug = m.get("slug")
        game_start = m.get("gameStartTime") or m.get("eventStartTime")
        category = m.get("category")
        sports_type = m.get("sportsMarketType")
        clob_token_ids = m.get("clobTokenIds")
        outcomes = m.get("outcomes")
        outcome_prices = m.get("outcomePrices")

        print("────────────────────────────────────────────")
        print(f"Question      : {question}")
        print(f"Slug          : {slug}")
        print(f"Category      : {category}")
        print(f"Sports type   : {sports_type}")
        print(f"Game start    : {game_start}")
        print(f"CLOB tokens   : {clob_token_ids}")
        print(f"Outcomes      : {outcomes}")
        print(f"OutcomePrices : {outcome_prices}")
        print("")


def build_utc_range_for_local_hour(date_yyyymmdd: str, local_tz: str, start_hour: int) -> tuple[datetime, datetime, str]:
    """
    Convert local date + local hour into a UTC [min,max) window of 1 hour.
    Example: 2025-12-12 @ 21:00 Europe/Paris -> UTC range.
    """
    if ZoneInfo is None:
        raise SystemExit("zoneinfo not available. Use --today/--date filters only, or install Python 3.9+.")

    tz = ZoneInfo(local_tz)
    d = datetime.strptime(date_yyyymmdd, "%Y-%m-%d").date()

    local_start = datetime(d.year, d.month, d.day, start_hour, 0, 0, tzinfo=tz)
    local_end = local_start + timedelta(hours=1)

    utc_start = local_start.astimezone(timezone.utc)
    utc_end = local_end.astimezone(timezone.utc)

    info = f"{date_yyyymmdd} {start_hour:02d}:00 ({local_tz}) => UTC [{utc_start.isoformat()} .. {utc_end.isoformat()})"
    return utc_start, utc_end, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket sports markets by tag slug/human name; optionally filter by date/time."
    )
    parser.add_argument("--market-name", "-m", type=str, default="ucl")
    parser.add_argument("--sports-types", "-t", nargs="*", default=["moneyline"])
    parser.add_argument("--json-out", "-o", type=str, default=None)

    parser.add_argument("--today", action="store_true", help="Filter markets to games happening today (UTC).")
    parser.add_argument("--date", type=str, default=None, help="Filter markets to a specific date (UTC), format YYYY-MM-DD.")

    # Optional “21h Paris” style filter
    parser.add_argument("--local-date", type=str, default=None, help="Local date YYYY-MM-DD for --start-hour filter.")
    parser.add_argument("--local-tz", type=str, default="Europe/Paris", help="IANA timezone for --start-hour (default Europe/Paris).")
    parser.add_argument("--start-hour", type=int, default=None, help="Local hour (0-23) window filter (1 hour window). Example: 21.")

    args = parser.parse_args()

    resolved_slug = resolve_market_tag_slug(args.market_name)

    start_utc_min: Optional[datetime] = None
    start_utc_max: Optional[datetime] = None
    date_info = ""

    if args.start_hour is not None:
        if not args.local_date:
            raise SystemExit("--local-date is required when using --start-hour.")
        start_utc_min, start_utc_max, date_info = build_utc_range_for_local_hour(
            args.local_date, args.local_tz, int(args.start_hour)
        )

    elif args.date:
        try:
            d = datetime.strptime(args.date, "%Y-%m-%d").date()
            start_utc_min = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            start_utc_max = start_utc_min + timedelta(days=1)
            date_info = f"{d.isoformat()} (00:00:00 to 24:00:00 UTC)"
        except ValueError:
            raise SystemExit("Invalid --date format, expected YYYY-MM-DD (e.g. 2025-12-10).")

    elif args.today:
        today_utc = datetime.now(timezone.utc).date()
        start_utc_min = datetime(today_utc.year, today_utc.month, today_utc.day, tzinfo=timezone.utc)
        start_utc_max = start_utc_min + timedelta(days=1)
        date_info = f"{today_utc.isoformat()} (00:00:00 to 24:00:00 UTC)"

    fetcher = PolymarketSportsFetcher()
    markets = fetcher.get_markets_for_tag(
        tag_slug=resolved_slug,
        sports_market_types=args.sports_types,
        limit_per_page=200,
        start_utc_min=start_utc_min,
        start_utc_max=start_utc_max,
    )

    pretty_print_sports_markets(markets, tag_name=args.market_name, resolved_slug=resolved_slug, date_info=date_info)

    out_file = args.json_out or f"{resolved_slug}_markets.json"

    payload = {
        "__meta__": {
            "requested_market_name": args.market_name,
            "resolved_slug": resolved_slug,
            "sports_types": args.sports_types,
            "date_filter": date_info or None,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "markets": markets,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nJSON saved to {out_file}")


if __name__ == "__main__":
    main()
