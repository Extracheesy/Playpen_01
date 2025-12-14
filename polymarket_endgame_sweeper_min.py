# ============================================================
# Polymarket Endgame Sweeper (TEST MODE)
# Adaptive Trailing Stop + Gamma Market Resolution
# ============================================================

import json
import time
import threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import requests
from websocket import WebSocketApp

from dotenv import load_dotenv
import os

# ===================== TELEGRAM =====================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = True

TG_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else None
)

def log(msg: str, level: str = "INFO"):
    line = f"[{level}] {msg}"
    print(line)
    if TELEGRAM_ENABLED and TG_URL:
        try:
            requests.post(
                TG_URL,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": line},
                timeout=4
            )
        except Exception:
            pass


# ===================== USER CONFIG =====================

MARKETS_FILES = [Path("nba_markets.json")]

BANKROLL = 10000.0
RISK_FRACTION = 1.0
GAME_RISK_FRACTION = 0.01

# Adaptive trailing stop
TRAIL_STOP_WIDE = 0.10   # 10% before strong dominance
TRAIL_STOP_TIGHT = 0.05  # 5% once price >= 0.95
TRAIL_TIGHTEN_LEVEL = 0.95

PRICE_TIERS = [
    {"threshold": 0.80, "weight": 0.30},
    {"threshold": 0.88, "weight": 0.40},
    {"threshold": 0.94, "weight": 0.30},
]

MAX_ENTRY_PRICE = 0.97

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_PING_SEC = 10
WS_RECONNECT_SEC = 3

GAMMA_BASE = "https://gamma-api.polymarket.com"
GAMMA_POLL_SEC = 20


# ===================== LOAD JSON =====================

def _safe_json_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def load_tokens(files):
    tokens = []
    seen = set()

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        markets = data["markets"] if isinstance(data, dict) else data
        log(f"Loaded {len(markets)} markets → {len(markets)*2} tokens", "INIT")

        for m in markets:
            if (m.get("sportsMarketType") or "").lower() != "moneyline":
                continue

            outcomes = _safe_json_list(m.get("outcomes"))
            clob = _safe_json_list(m.get("clobTokenIds"))
            if not outcomes or not clob:
                continue

            for name, tid in zip(outcomes, clob):
                if str(name).lower() == "draw":
                    continue
                key = (m["id"], tid)
                if key in seen:
                    continue
                seen.add(key)

                tokens.append({
                    "market_id": m["id"],
                    "token_id": str(tid),
                    "game": m["question"],
                    "label": f"{m['question']}: {name} win"
                })
    return tokens


TOKENS = load_tokens(MARKETS_FILES)

# ===================== PORTFOLIO =====================

usable_capital = BANKROLL * RISK_FRACTION
cash = usable_capital

def per_token_budget():
    spread = usable_capital / len(TOKENS)
    cap = BANKROLL * GAME_RISK_FRACTION
    return min(spread, cap)

PER_TOKEN_BUDGET = per_token_budget()

state = {}
game_map = defaultdict(list)
resolved_markets = set()
reported_games = set()

for t in TOKENS:
    tid = t["token_id"]
    state[tid] = {
        "market_id": t["market_id"],
        "game": t["game"],
        "label": t["label"],
        "size": 0.0,
        "cost": 0.0,
        "avg_entry": 0.0,
        "last_bid": 0.0,
        "max_bid_seen": 0.0,
        "entered": False,
        "closed": False,
        "tiers": [
            {"threshold": x["threshold"], "usdc": PER_TOKEN_BUDGET * x["weight"], "filled": False}
            for x in PRICE_TIERS
        ]
    }
    game_map[t["game"]].append(tid)


# ===================== ORDERS =====================

def buy_token(tid, price, usdc, tier):
    global cash
    s = state[tid]
    if cash < usdc or price >= 0.999 or price > MAX_ENTRY_PRICE:
        return

    size = usdc / price
    cash -= usdc

    s["size"] += size
    s["cost"] += usdc
    s["avg_entry"] = s["cost"] / s["size"]
    s["entered"] = True
    s["max_bid_seen"] = price

    log(f"BUY {s['label']} | tier={tier:.2f} | price={price:.3f} | usdc={usdc:.2f} | cash={cash:.2f}", "BUY")

def sell_all(tid, price, reason):
    global cash
    s = state[tid]
    if s["size"] <= 0:
        return 0.0

    payout = s["size"] * price
    pnl = payout - s["cost"]
    cash += payout

    log(f"SELL {s['label']} | reason={reason} | price={price:.3f} | pnl={pnl:.2f} | cash={cash:.2f}", "SELL")

    s["size"] = 0.0
    s["cost"] = 0.0
    s["closed"] = True
    return pnl


# ===================== STRATEGY =====================

def on_book(tid, bid):
    s = state[tid]
    s["last_bid"] = bid

    if s["market_id"] in resolved_markets or s["closed"]:
        return

    # Adaptive trailing stop
    if s["entered"] and s["size"] > 0:
        if bid > s["max_bid_seen"]:
            s["max_bid_seen"] = bid

        if s["max_bid_seen"] > 0:
            trail_pct = (
                TRAIL_STOP_TIGHT
                if s["max_bid_seen"] >= TRAIL_TIGHTEN_LEVEL
                else TRAIL_STOP_WIDE
            )
            dd = (s["max_bid_seen"] - bid) / s["max_bid_seen"]
            if dd >= trail_pct:
                sell_all(tid, bid, f"TRAIL_STOP {trail_pct*100:.0f}%")
                return

    # Tiered entries
    mid = bid
    for tier in s["tiers"]:
        if not tier["filled"] and mid >= tier["threshold"]:
            buy_token(tid, mid, tier["usdc"], tier["threshold"])
            tier["filled"] = True


# ===================== GAMMA POLLING =====================

def gamma_poll():
    log("Gamma polling started", "BOT")
    market_ids = {t["market_id"] for t in TOKENS}

    while True:
        for mid in market_ids:
            if mid in resolved_markets:
                continue
            try:
                r = requests.get(f"{GAMMA_BASE}/markets/{mid}", timeout=6)
                if r.status_code != 200:
                    continue
                m = r.json()
            except Exception:
                continue

            if m.get("closed") or not m.get("active", True) or m.get("resolved"):
                resolved_markets.add(mid)
                game = None
                pnl = 0.0
                for tid, s in state.items():
                    if s["market_id"] == mid:
                        game = s["game"]
                        pnl += sell_all(tid, s["last_bid"], "MARKET_RESOLVED")

                if game and game not in reported_games:
                    reported_games.add(game)
                    log(f"GAME_END {game} | pnl={pnl:.2f}", "PNL")
                    log(f"EQUITY equity={cash:.2f}", "EQUITY")

        time.sleep(GAMMA_POLL_SEC)


# ===================== WEBSOCKET =====================

class WS:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.ws = WebSocketApp(
            WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

    def on_open(self, ws):
        log("WS connected", "WS")
        ws.send(json.dumps({"type": "market", "assets_ids": self.token_ids}))
        log(f"WS subscribed to {len(self.token_ids)} assets", "WS")

        def ping():
            while True:
                try:
                    ws.send("PING")
                    time.sleep(WS_PING_SEC)
                except Exception:
                    break
        threading.Thread(target=ping, daemon=True).start()

    def on_message(self, ws, msg):
        if not msg or msg[0] not in "{[":
            return
        try:
            data = json.loads(msg)
        except Exception:
            return

        if isinstance(data, list):
            for d in data:
                self.handle(d)
        else:
            self.handle(data)

    def handle(self, d):
        if d.get("event_type") != "book":
            return
        tid = d.get("asset_id")
        bids = d.get("bids") or []
        if bids:
            best = max(float(b["price"]) for b in bids)
            on_book(tid, best)

    def on_error(self, ws, err):
        log(f"WS error: {err}", "WS")

    def on_close(self, ws, code, reason):
        log(f"WS closed ({code}, {reason}), reconnecting…", "WS")
        time.sleep(WS_RECONNECT_SEC)
        self.run()

    def run(self):
        self.ws.run_forever()


# ===================== MAIN =====================

def main():
    log("Starting Polymarket Endgame Sweeper (Adaptive Trailing Stop)", "BOT")
    threading.Thread(target=gamma_poll, daemon=True).start()
    WS([t["token_id"] for t in TOKENS]).run()

if __name__ == "__main__":
    main()
