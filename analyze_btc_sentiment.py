#!/usr/bin/env python
"""
Analyze BTC social sentiment log and build features for ML / signals.

- Reads btc_social_sentiment_log.csv (created by btc_sentiment_logger.py)
- Assumes a fixed logging interval (5 minutes by default)
- Creates:
    * Total social posts per sample
    * 1-step (Δ last 5 min) changes
    * 1-hour (Δ last 60 min) changes
    * Rolling means over 1h
    * Normalized Fear & Greed index

- Writes result to btc_social_sentiment_features.csv
"""

import os
from datetime import datetime

import pandas as pd

# ===================== CONFIG =====================

# Input log file from the logger script
INPUT_CSV = "btc_social_sentiment_log.csv"

# Output features file
OUTPUT_CSV = "btc_social_sentiment_features.csv"

# Base logging interval in minutes
# ⚠️ You said you will set INTERVAL_MINUTES = 5 in the logger,
# so set this to 5 here as well.
BASE_INTERVAL_MINUTES = 5

# Derived: number of rows corresponding to ~1 hour
ROWS_PER_HOUR = int(round(60 / BASE_INTERVAL_MINUTES))


# ===================== FEATURE ENGINEERING =====================

def load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)

    # Parse timestamps
    if "run_ts_utc" not in df.columns:
        raise ValueError("Column 'run_ts_utc' missing from log file.")

    df["run_ts_utc"] = pd.to_datetime(df["run_ts_utc"], utc=True, errors="coerce")
    df = df.sort_values("run_ts_utc").reset_index(drop=True)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Base helpers ----
    social_cols_posts = [
        "twitterPosts",
        "redditPosts",
        "stocktwitsPosts",
    ]
    social_cols_engagement = [
        "twitterComments",
        "twitterLikes",
        "twitterImpressions",
        "redditComments",
        "redditLikes",
        "redditImpressions",
        "stocktwitsComments",
        "stocktwitsLikes",
        "stocktwitsImpressions",
    ]

    # Make sure columns exist, fill missing with 0
    for col in social_cols_posts + social_cols_engagement:
        if col not in df.columns:
            df[col] = 0

    # Total social activity metrics
    df["total_posts"] = df[social_cols_posts].sum(axis=1)
    df["total_engagement"] = df[social_cols_engagement].sum(axis=1)

    # Normalized Fear & Greed (0-1)
    if "fng_value" in df.columns:
        df["fng_norm"] = df["fng_value"] / 100.0
    else:
        df["fng_value"] = None
        df["fng_norm"] = None

    # ---- 1-step (Δ over last interval ~ 5 min) ----
    # These are good for "momentum" in sentiment / social activity

    diff_cols = [
        "total_posts",
        "total_engagement",
        "fng_value",
        "fng_norm",
        "twitterPosts",
        "redditPosts",
        "stocktwitsPosts",
    ]

    for col in diff_cols:
        if col in df.columns:
            df[f"{col}_diff_1step"] = df[col].diff()

    # ---- 1-hour changes ----
    # ROWS_PER_HOUR rows correspond to ~ 60 minutes of data

    if ROWS_PER_HOUR >= 1:
        for col in diff_cols:
            if col in df.columns:
                df[f"{col}_diff_1h"] = df[col] - df[col].shift(ROWS_PER_HOUR)
                df[f"{col}_pct_1h"] = df[f"{col}_diff_1h"] / df[col].shift(ROWS_PER_HOUR)

        # Rolling mean over 1h
        for col in ["total_posts", "total_engagement"]:
            if col in df.columns:
                df[f"{col}_rollmean_1h"] = df[col].rolling(ROWS_PER_HOUR).mean()

    # Drop rows with no timestamp (should not happen)
    df = df.dropna(subset=["run_ts_utc"]).reset_index(drop=True)

    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n=== BTC Social Sentiment Features Summary ===\n")
    print(f"Rows: {len(df)}")
    print(f"Time range: {df['run_ts_utc'].min()}  -->  {df['run_ts_utc'].max()}")
    print(f"Logging interval (assumed): {BASE_INTERVAL_MINUTES} minutes")
    print(f"Rows per hour: {ROWS_PER_HOUR}")

    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    # Simple sanity stats
    if "total_posts" in df.columns:
        print("\nTotal posts stats (last 10 rows):")
        print(df[["run_ts_utc", "total_posts"]].tail(10))

    if "fng_value" in df.columns:
        print("\nFear & Greed (last 10 rows):")
        print(df[["run_ts_utc", "fng_value", "fng_norm"]].tail(10))

    print("\nFeatures saved to:", OUTPUT_CSV)
    print()


# ===================== MAIN =====================

def main():
    print(f"[{datetime.now()}] Loading log from: {INPUT_CSV}")
    df_log = load_log(INPUT_CSV)

    print(f"[{datetime.now()}] Building features (interval={BASE_INTERVAL_MINUTES} min, rows_per_hour={ROWS_PER_HOUR})")
    df_feat = build_features(df_log)

    print(f"[{datetime.now()}] Saving features to: {OUTPUT_CSV}")
    df_feat.to_csv(OUTPUT_CSV, index=False)

    summarize(df_feat)


if __name__ == "__main__":
    main()
