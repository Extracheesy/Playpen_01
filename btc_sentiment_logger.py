
#!/usr/bin/env python
"""
BTC social sentiment (Utradea) + Fear & Greed logger.

- Every 5 minutes (aligned to :00 / :05 / :10 / :15 / ...):
    * Calls Utradea /v1/get-social for BTC
    * Calls Alternative.me Fear & Greed index
    * Appends a row to a CSV file

Usage:
    1) Put your Utradea API key in UTRADEA_API_KEY.
    2) Run:  python btc_sentiment_logger.py
"""

import csv
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from requests.exceptions import RequestException, ConnectTimeout

# ===================== CONFIG =====================

# ⚠️ Put your Utradea API key here
UTRADEA_API_KEY = "gV2nTMvCqpo7th4v3e5DLT2t7svqtLU5psb"

# Utradea BTC ticker symbol
# 👉 We now use "BTC" instead of "BTC-USD"
BTC_TICKER = "BTC"

# Output CSV file
LOG_FILE = "btc_social_sentiment_log.csv"

# Interval in minutes (you said you want 5)
INTERVAL_MINUTES = 5

# Utradea endpoint
UTRADEA_URL = "https://api.utradea.com/v1/get-social"

# Fear & Greed endpoint
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# CSV columns
FIELDNAMES = [
    "run_ts_utc",
    "run_ts_local",
    "symbol",

    "twitterPosts",
    "twitterComments",
    "twitterLikes",
    "twitterImpressions",

    "redditPosts",
    "redditComments",
    "redditLikes",
    "redditImpressions",

    "stocktwitsPosts",
    "stocktwitsComments",
    "stocktwitsLikes",
    "stocktwitsImpressions",

    "utradea_timestamp",

    "fng_value",
    "fng_classification",
    "fng_timestamp_unix",
    "fng_timestamp_utc",
    "fng_time_until_update",
]


# ===================== HELPERS =====================

def ensure_log_file(path: str) -> None:
    """Create CSV file with header if it does not exist or is empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def fetch_utradea_btc(api_key: str, ticker: str) -> dict:
    """
    Fetch latest social sentiment snapshot from Utradea for one ticker.

    Uses:

        GET https://api.utradea.com/v1/get-social
            ?tickers=BTC
            &social=twitter,reddit,stocktwits
            &charts=posts,comments,likes,impressions

    Returns the latest record (by timestamp).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }

    # NOTE: param name MUST be 'tickers' (plural).
    params = {
        "tickers": ticker,
        "social": "twitter,reddit,stocktwits",
        "charts": "posts,comments,likes,impressions",
    }

    try:
        resp = requests.get(UTRADEA_URL, headers=headers, params=params, timeout=30)
    except ConnectTimeout as e:
        raise RuntimeError(f"Utradea request timed out: {e}") from e
    except RequestException as e:
        raise RuntimeError(f"Utradea request error: {e}") from e

    if resp.status_code != 200:
        # Show body to debug 4xx/5xx if needed
        try:
            body = resp.text
        except Exception:
            body = "<no body>"
        raise RuntimeError(
            f"Utradea returned HTTP {resp.status_code} for {resp.url} "
            f"Body: {body}"
        )

    data = resp.json()

    # Handle {"statusCode":200, "output":[...]} or just [...]
    if isinstance(data, dict) and "output" in data:
        rows = data.get("output", [])
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected Utradea response format: {data!r}")

    if not rows:
        raise ValueError("Utradea returned empty output list")

    latest = max(rows, key=lambda x: x.get("timestamp") or "")
    return latest


def fetch_fear_greed() -> dict:
    """
    Fetch latest Fear & Greed index value.

    Returns:
    {
        "value": int,
        "classification": str,
        "timestamp_unix": int,
        "timestamp_utc": str,
        "time_until_update": int or None,
    }
    """
    params = {
        "limit": 1,
        "format": "json",
    }
    resp = requests.get(FEAR_GREED_URL, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data")
    if not data:
        raise ValueError(f"Unexpected Fear & Greed payload: {payload!r}")

    item = data[0]
    value = int(item["value"])
    classification = item.get("value_classification")
    ts_unix = int(item["timestamp"])
    ts_utc = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
    time_until_update = item.get("time_until_update")
    if time_until_update is not None:
        try:
            time_until_update = int(time_until_update)
        except (TypeError, ValueError):
            time_until_update = None

    return {
        "value": value,
        "classification": classification,
        "timestamp_unix": ts_unix,
        "timestamp_utc": ts_utc,
        "time_until_update": time_until_update,
    }


def build_row(run_ts_utc: datetime, utradea: dict, fng: dict) -> dict:
    """Merge Utradea and Fear & Greed data into one flat row for CSV."""
    def g(key: str) -> int:
        return int(utradea.get(key, 0) or 0)

    row = {
        "run_ts_utc": run_ts_utc.isoformat(),
        "run_ts_local": run_ts_utc.astimezone().isoformat(),
        "symbol": utradea.get("symbol", BTC_TICKER),

        "twitterPosts": g("twitterPosts"),
        "twitterComments": g("twitterComments"),
        "twitterLikes": g("twitterLikes"),
        "twitterImpressions": g("twitterImpressions"),

        "redditPosts": g("redditPosts"),
        "redditComments": g("redditComments"),
        "redditLikes": g("redditLikes"),
        "redditImpressions": g("redditImpressions"),

        "stocktwitsPosts": g("stocktwitsPosts"),
        "stocktwitsComments": g("stocktwitsComments"),
        "stocktwitsLikes": g("stocktwitsLikes"),
        "stocktwitsImpressions": g("stocktwitsImpressions"),

        "utradea_timestamp": utradea.get("timestamp"),

        "fng_value": fng["value"],
        "fng_classification": fng["classification"],
        "fng_timestamp_unix": fng["timestamp_unix"],
        "fng_timestamp_utc": fng["timestamp_utc"],
        "fng_time_until_update": fng["time_until_update"],
    }
    return row


def append_row(path: str, row: dict) -> None:
    """Append a single row to the CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def seconds_until_next_interval(interval_minutes: int) -> float:
    """
    Seconds until the next interval boundary (e.g. every 5 minutes).

    Example for 5 minutes:
      If now is 10:02 -> next run at 10:05
      If now is 10:05:05 -> next run at 10:10
    """
    now = datetime.now(timezone.utc)
    minutes_since_hour = now.minute
    intervals_passed = minutes_since_hour // interval_minutes
    next_interval_minute = (intervals_passed + 1) * interval_minutes

    base = now.replace(second=0, microsecond=0, minute=0)
    next_run = base + timedelta(minutes=next_interval_minute)

    delta = (next_run - now).total_seconds()
    if delta < 1:
        delta = 1.0
    return delta


# ===================== MAIN LOOP =====================

def main() -> None:
    if not UTRADEA_API_KEY or "YOUR_UTRADEA_API_KEY_HERE" in UTRADEA_API_KEY:
        raise SystemExit("Please set UTRADEA_API_KEY at the top of the script.")

    ensure_log_file(LOG_FILE)

    print(f"Logging to {LOG_FILE}")
    print(f"Interval: every {INTERVAL_MINUTES} minutes (aligned to multiples of {INTERVAL_MINUTES})")
    print("Press Ctrl+C to stop.\n")

    while True:
        run_ts_utc = datetime.now(timezone.utc)

        try:
            utradea_data = fetch_utradea_btc(UTRADEA_API_KEY, BTC_TICKER)
            fng_data = fetch_fear_greed()
            row = build_row(run_ts_utc, utradea_data, fng_data)
            append_row(LOG_FILE, row)

            # Small debug to see if stuff is moving
            tp = row["twitterPosts"]
            rp = row["redditPosts"]
            sp = row["stocktwitsPosts"]

            print(
                f"[{row['run_ts_local']}] Logged row "
                f"(posts T/R/S = {tp}/{rp}/{sp}, "
                f"F&G = {row['fng_value']} {row['fng_classification']})"
            )
        except Exception as exc:
            print(f"[{run_ts_utc.isoformat()}] ERROR: {exc}")

        sleep_s = seconds_until_next_interval(INTERVAL_MINUTES)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
