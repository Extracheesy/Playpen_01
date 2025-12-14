import requests
import csv
import time
from datetime import datetime, timezone

BASE = "https://gamma-api.polymarket.com"
LIMIT = 100
SLEEP = 0.15

def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(value):
    """
    Parse ISO-ish datetime strings from Gamma.
    Returns aware datetime in UTC, or None if missing/unparseable.
    """
    if not value:
        return None
    if isinstance(value, (int, float)):
        # if API ever returns unix seconds (rare)
        return datetime.fromtimestamp(value, tz=timezone.utc)
    s = str(value).strip()

    # Normalize Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # fromisoformat handles: 2025-12-13T23:00:00+00:00
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def paged_get(endpoint, closed=False):
    out = []
    offset = 0
    while True:
        params = {
            "limit": LIMIT,
            "offset": offset,
            "closed": str(closed).lower(),  # closed=false
        }
        r = requests.get(f"{BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        offset += LIMIT
        time.sleep(SLEEP)
    return out

def is_truthy(x):
    return x is True or (isinstance(x, str) and x.lower() == "true")

def has_nonempty_clob_token_ids(market):
    ids = market.get("clobTokenIds")
    # sometimes could be list, sometimes string-ish; handle both
    if ids is None:
        return False
    if isinstance(ids, list):
        return len(ids) > 0 and any(str(v).strip() for v in ids)
    s = str(ids).strip()
    return s not in ("", "[]", "null", "None")

def enddate_not_passed(obj):
    # Gamma often has endDate on events; markets sometimes too.
    end = obj.get("endDate") or obj.get("end_date") or obj.get("eventEndDate")
    dt = parse_dt(end)
    if dt is None:
        # If no endDate, we DON'T filter it out (you can change to False if you want strict)
        return True
    return dt > now_utc()

def filter_events(events):
    kept = []
    for e in events:
        if e.get("closed") is True:
            continue
        if "active" in e and not is_truthy(e.get("active")):
            continue
        if not enddate_not_passed(e):
            continue
        kept.append(e)
    return kept

def filter_markets(markets):
    kept = []
    for m in markets:
        if m.get("closed") is True:
            continue

        # Inactive filters
        if "active" in m and not is_truthy(m.get("active")):
            continue
        if "enableOrderBook" in m and not is_truthy(m.get("enableOrderBook")):
            continue
        if "acceptingOrders" in m and not is_truthy(m.get("acceptingOrders")):
            continue

        # Your new requirements
        if not has_nonempty_clob_token_ids(m):
            continue
        if not enddate_not_passed(m):
            continue

        kept.append(m)
    return kept

def dump_csv(filename, rows):
    if not rows:
        print(f"[WARN] {filename}: no rows")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[OK] wrote {len(rows)} rows -> {filename}")

if __name__ == "__main__":
    print("Fetching OPEN events (closed=false)...")
    events = paged_get("/events", closed=False)
    events2 = filter_events(events)
    print(f"Events: {len(events)} fetched -> {len(events2)} kept (open+active+endDate future)")
    dump_csv("events_open_active_future.csv", events2)

    print("Fetching OPEN markets (closed=false)...")
    markets = paged_get("/markets", closed=False)
    markets2 = filter_markets(markets)
    print(f"Markets: {len(markets)} fetched -> {len(markets2)} kept (tradable+tokens+endDate future)")
    dump_csv("markets_open_active_future.csv", markets2)
