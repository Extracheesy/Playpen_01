import argparse
import asyncio
import gzip
import json
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import websockets

CLOB_REST = "https://clob.polymarket.com"
CLOB_WSS_BASE = "wss://ws-subscriptions-clob.polymarket.com"


# ----------------- Helpers -----------------
def utc_ms() -> int:
    return int(time.time() * 1000)


def utc_iso(ms: Optional[int] = None) -> str:
    if ms is None:
        ms = utc_ms()
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_load_any(s: Any) -> Any:
    if isinstance(s, str) and s.startswith("[") and s.endswith("]"):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def sanitize_slug(s: str, maxlen: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:maxlen] if len(s) > maxlen else s


def topk_levels(levels: List[Dict[str, str]], k: int) -> List[Tuple[float, float]]:
    out = []
    for lvl in levels[:k]:
        try:
            out.append((float(lvl["price"]), float(lvl["size"])))
        except Exception:
            continue
    return out


def best_bid_ask(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    bid_sz = bids[0][1] if bids else 0.0
    ask_sz = asks[0][1] if asks else 0.0
    return bid, ask, bid_sz, ask_sz


def microprice(best_bid, best_ask, bid_sz, ask_sz):
    if best_bid is None or best_ask is None:
        return None
    denom = bid_sz + ask_sz
    if denom <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_ask * bid_sz + best_bid * ask_sz) / denom


def depth_sum(levels: List[Tuple[float, float]], k: int) -> float:
    return float(sum(sz for _, sz in levels[:k]))


def pressure_weighted(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], k: int) -> float:
    bid_p = sum((sz / (i + 1)) for i, (_, sz) in enumerate(bids[:k]))
    ask_p = sum((sz / (i + 1)) for i, (_, sz) in enumerate(asks[:k]))
    denom = bid_p + ask_p
    if denom <= 0:
        return 0.0
    return (bid_p - ask_p) / denom


# ----------------- Data model -----------------
@dataclass
class MarketSpec:
    date: str
    league: str
    market_id: str
    slug: str
    question: str
    yes_token: str
    no_token: str
    url: str


@dataclass
class BookState:
    ts_ms: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    hash: Optional[str] = None


# ----------------- Delayed Labeler -----------------
class DelayedLabeler:
    """
    No cheating: it labels row at time t only when we've reached t + max_horizon.
    Stores a small rolling buffer per market.
    """
    def __init__(self, horizons_s: List[int], spike_thresholds: Dict[int, float], buffer_seconds: int = 180):
        self.horizons_s = sorted(horizons_s)
        self.spike_thresholds = spike_thresholds
        self.max_h = max(self.horizons_s)
        self.buffer_seconds = buffer_seconds

        # market_id -> deque of rows (ordered by ts_ms)
        self.buffers: Dict[str, Deque[Dict[str, Any]]] = {}

    def push(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Push a new snapshot row. Returns newly-labeled rows (possibly empty).
        """
        mid = row.get("yes_mid")
        if mid is None:
            # still store, but labeling will be mostly None
            pass

        mid_id = row["market_id"]
        buf = self.buffers.setdefault(mid_id, deque())
        buf.append(row)

        # drop old beyond buffer_seconds
        cutoff_ms = row["ts_ms"] - (self.buffer_seconds * 1000)
        while buf and buf[0]["ts_ms"] < cutoff_ms:
            buf.popleft()

        # Try to label from the left as long as we have enough future
        labeled_rows = []
        now_ms = row["ts_ms"]
        while buf:
            t0 = buf[0]["ts_ms"]
            if now_ms < t0 + self.max_h * 1000:
                break
            labeled_rows.append(self._label_one(buf, t0))
            buf.popleft()

        return labeled_rows

    def _label_one(self, buf: Deque[Dict[str, Any]], t0_ms: int) -> Dict[str, Any]:
        """
        Compute labels for the row at t0_ms using future rows in buffer.
        """
        base = None
        for r in buf:
            if r["ts_ms"] == t0_ms:
                base = r
                break
        if base is None:
            base = buf[0]

        y0 = base.get("yes_mid")
        out = {
            "ts_ms": base["ts_ms"],
            "ts_iso": base["ts_iso"],
            "date": base["date"],
            "league": base["league"],
            "market_id": base["market_id"],
            "slug": base["slug"],
        }

        # For each horizon, find the first row with ts >= t0 + h
        for h in self.horizons_s:
            target_ms = t0_ms + h * 1000
            y_t = None
            # Also compute MFE within (t0, t0+h]
            max_y = None
            min_y = None

            for r in buf:
                if r["ts_ms"] > t0_ms and r.get("yes_mid") is not None:
                    yy = r["yes_mid"]
                    max_y = yy if max_y is None else max(max_y, yy)
                    min_y = yy if min_y is None else min(min_y, yy)

                if r["ts_ms"] >= target_ms:
                    y_t = r.get("yes_mid")
                    break

            # returns
            if y0 is None or y_t is None or y0 <= 0:
                ret = None
            else:
                ret = (y_t - y0) / y0

            out[f"ret_yes_{h}s"] = ret

            # spike labels (threshold-based)
            thr = self.spike_thresholds.get(h, 0.03)
            if ret is None:
                out[f"spike_up_yes_{h}s"] = None
                out[f"spike_dn_yes_{h}s"] = None
            else:
                out[f"spike_up_yes_{h}s"] = 1 if ret >= thr else 0
                out[f"spike_dn_yes_{h}s"] = 1 if ret <= -thr else 0

            # MFE (max favorable excursion within horizon window)
            if y0 is None or y0 <= 0 or max_y is None or min_y is None:
                out[f"mfe_up_yes_{h}s"] = None
                out[f"mfe_dn_yes_{h}s"] = None
            else:
                out[f"mfe_up_yes_{h}s"] = (max_y - y0) / y0
                out[f"mfe_dn_yes_{h}s"] = (min_y - y0) / y0

        return out


# ----------------- Recorder -----------------
class Recorder:
    def __init__(
        self,
        specs: List[MarketSpec],
        out_dir: Path,
        depth: int = 10,
        snap_interval_s: float = 1.0,
        flush_every_s: float = 10.0,
        bootstrap_rest: bool = True,
        raw_gzip: bool = True,
        write_index_all: bool = True,
        horizons_s: List[int] = [10, 30, 60],
        thresholds: Optional[Dict[int, float]] = None,
    ):
        self.specs = specs
        self.out_dir = out_dir
        self.depth = depth
        self.snap_interval_s = snap_interval_s
        self.flush_every_s = flush_every_s
        self.bootstrap_rest = bootstrap_rest
        self.raw_gzip = raw_gzip
        self.write_index_all = write_index_all

        self._stop = asyncio.Event()

        self.books: Dict[str, BookState] = {}
        self.asset_ids: List[str] = sorted({s.yes_token for s in specs} | {s.no_token for s in specs})

        # per-market snapshot buffer
        self.snap_buf: Dict[str, List[Dict[str, Any]]] = {s.market_id: [] for s in specs}
        self.label_buf: Dict[str, List[Dict[str, Any]]] = {s.market_id: [] for s in specs}

        self.last_flush_ms = utc_ms()

        # Labeler setup
        if thresholds is None:
            thresholds = {10: 0.02, 30: 0.035, 60: 0.05}
        self.labeler = DelayedLabeler(horizons_s=horizons_s, spike_thresholds=thresholds, buffer_seconds=240)

        # maps
        self.market_by_id: Dict[str, MarketSpec] = {s.market_id: s for s in specs}

        self.paths = self._init_dirs()

    def _init_dirs(self) -> Dict[str, Path]:
        safe_mkdir(self.out_dir)
        safe_mkdir(self.out_dir / "meta")
        safe_mkdir(self.out_dir / "markets")
        safe_mkdir(self.out_dir / "index")
        return {
            "meta": self.out_dir / "meta",
            "markets": self.out_dir / "markets",
            "index": self.out_dir / "index",
        }

    def request_stop(self):
        self._stop.set()

    def market_dir(self, spec: MarketSpec, day: str) -> Path:
        folder = f"{spec.market_id}__{sanitize_slug(spec.slug)}"
        p = self.paths["markets"] / spec.league / folder / f"day={day}"
        safe_mkdir(p)
        return p

    def index_dir(self, day: str) -> Path:
        p = self.paths["index"] / f"day={day}"
        safe_mkdir(p)
        return p

    def write_meta(self, input_json_path: Path):
        # copy input json
        dst = self.paths["meta"] / "input_markets.json"
        if not dst.exists():
            dst.write_text(input_json_path.read_text(encoding="utf-8"), encoding="utf-8")

        flat = []
        for s in self.specs:
            flat.append({
                "date": s.date, "league": s.league, "market_id": s.market_id, "slug": s.slug,
                "question": s.question, "yes_token": s.yes_token, "no_token": s.no_token, "url": s.url
            })
        pd.DataFrame(flat).to_csv(self.paths["meta"] / "markets_flat.csv", index=False)

        cfg = {
            "created_utc": utc_iso(),
            "depth": self.depth,
            "snap_interval_s": self.snap_interval_s,
            "flush_every_s": self.flush_every_s,
            "bootstrap_rest": self.bootstrap_rest,
            "raw_gzip": self.raw_gzip,
            "write_index_all": self.write_index_all,
            "assets": len(self.asset_ids),
            "markets": len(self.specs),
            "label_horizons_s": self.labeler.horizons_s,
            "label_thresholds": self.labeler.spike_thresholds,
        }
        (self.paths["meta"] / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    async def bootstrap_books(self):
        if not self.bootstrap_rest:
            return
        print(f"[BOOT] REST bootstrap /book for {len(self.asset_ids)} assets")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
            sem = asyncio.Semaphore(10)

            async def fetch_one(token_id: str):
                url = f"{CLOB_REST}/book"
                params = {"token_id": token_id}
                async with sem:
                    try:
                        async with sess.get(url, params=params) as r:
                            if r.status != 200:
                                return
                            js = await r.json()
                            bids = topk_levels(js.get("bids", []), self.depth)
                            asks = topk_levels(js.get("asks", []), self.depth)
                            self.books[token_id] = BookState(ts_ms=utc_ms(), bids=bids, asks=asks, hash=js.get("hash"))
                    except Exception:
                        return

            await asyncio.gather(*[fetch_one(t) for t in self.asset_ids])

    def _write_raw_ws(self, spec: MarketSpec, day: str, msg: Dict[str, Any]):
        md = self.market_dir(spec, day)
        fp = md / "raw_ws_book.jsonl"
        if self.raw_gzip:
            fp = fp.with_suffix(".jsonl.gz")
            with gzip.open(fp, "at", encoding="utf-8") as f:
                f.write(json.dumps(msg, separators=(",", ":")) + "\n")
        else:
            with open(fp, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, separators=(",", ":")) + "\n")

    def _compute_snapshot_row(self, spec: MarketSpec, now_ms: int) -> Dict[str, Any]:
        yes = self.books.get(spec.yes_token)
        no = self.books.get(spec.no_token)

        def pack(side: Optional[BookState], prefix: str) -> Dict[str, Any]:
            if not side:
                return {
                    f"{prefix}_bid": None, f"{prefix}_ask": None,
                    f"{prefix}_bid_sz": 0.0, f"{prefix}_ask_sz": 0.0,
                    f"{prefix}_mid": None, f"{prefix}_spread": None,
                    f"{prefix}_micro": None,
                    f"{prefix}_bid_depth{self.depth}": 0.0,
                    f"{prefix}_ask_depth{self.depth}": 0.0,
                    f"{prefix}_imb{self.depth}": 0.0,
                    f"{prefix}_pressure{self.depth}": 0.0,
                    f"{prefix}_book_ts_ms": None,
                }
            bb, ba, bb_sz, ba_sz = best_bid_ask(side.bids, side.asks)
            mid = None if (bb is None or ba is None) else (bb + ba) / 2.0
            spr = None if (bb is None or ba is None) else (ba - bb)
            mic = microprice(bb, ba, bb_sz, ba_sz)
            bid_d = depth_sum(side.bids, self.depth)
            ask_d = depth_sum(side.asks, self.depth)
            imb = (bid_d / (bid_d + ask_d)) if (bid_d + ask_d) > 0 else 0.0
            pres = pressure_weighted(side.bids, side.asks, self.depth)
            return {
                f"{prefix}_bid": bb, f"{prefix}_ask": ba,
                f"{prefix}_bid_sz": bb_sz, f"{prefix}_ask_sz": ba_sz,
                f"{prefix}_mid": mid, f"{prefix}_spread": spr,
                f"{prefix}_micro": mic,
                f"{prefix}_bid_depth{self.depth}": bid_d,
                f"{prefix}_ask_depth{self.depth}": ask_d,
                f"{prefix}_imb{self.depth}": imb,
                f"{prefix}_pressure{self.depth}": pres,
                f"{prefix}_book_ts_ms": side.ts_ms,
            }

        row = {
            "ts_ms": now_ms,
            "ts_iso": utc_iso(now_ms),
            "date": spec.date,
            "league": spec.league,
            "market_id": spec.market_id,
            "slug": spec.slug,
            "question": spec.question,
            "url": spec.url,
            "yes_token": spec.yes_token,
            "no_token": spec.no_token,
        }
        row.update(pack(yes, "yes"))
        row.update(pack(no, "no"))

        if row["yes_mid"] is not None and row["no_mid"] is not None:
            row["sum_mid_yes_no"] = row["yes_mid"] + row["no_mid"]
            row["divergence_yes_vs_1_minus_no"] = row["yes_mid"] - (1.0 - row["no_mid"])
        else:
            row["sum_mid_yes_no"] = None
            row["divergence_yes_vs_1_minus_no"] = None

        return row

    async def snapshot_loop(self):
        print(f"[SNAP] every {self.snap_interval_s}s depth={self.depth} | labels delayed {self.labeler.horizons_s}s")
        while not self._stop.is_set():
            now_ms = utc_ms()
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            for spec in self.specs:
                snap = self._compute_snapshot_row(spec, now_ms)
                self.snap_buf[spec.market_id].append(snap)

                # delayed labeling
                newly_labeled = self.labeler.push(snap)
                if newly_labeled:
                    self.label_buf[spec.market_id].extend(newly_labeled)

            if (now_ms - self.last_flush_ms) >= int(self.flush_every_s * 1000):
                await self.flush(day)
                self.last_flush_ms = now_ms

            await asyncio.sleep(self.snap_interval_s)

        # final flush
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await self.flush(day)

    async def flush(self, day: str):
        # write per market
        all_snap_rows = []
        all_label_rows = []

        for mid, rows in list(self.snap_buf.items()):
            if not rows and not self.label_buf[mid]:
                continue

            spec = self.market_by_id[mid]
            md = self.market_dir(spec, day)

            if rows:
                df = pd.DataFrame(rows)
                self.snap_buf[mid] = []
                all_snap_rows.append(df)

                # snapshots.parquet (overwrite per flush -> easiest and safe)
                # if you want appends, we can switch to partitioned dataset writing
                out = md / "snapshots.parquet"
                try:
                    # append by reading existing (simple, robust)
                    if out.exists():
                        prev = pd.read_parquet(out)
                        df = pd.concat([prev, df], ignore_index=True)
                    df.to_parquet(out, index=False)
                except Exception:
                    # fallback CSV append
                    out_csv = md / "snapshots.csv"
                    header = not out_csv.exists()
                    df.to_csv(out_csv, mode="a", header=header, index=False)

                # features-only view
                feat_cols = [c for c in df.columns if c.startswith(("yes_", "no_", "sum_mid", "divergence"))] + \
                            ["ts_ms", "ts_iso", "date", "league", "market_id", "slug"]
                feat_df = df[feat_cols].copy()
                feat_out = md / "features.parquet"
                try:
                    if feat_out.exists():
                        prev = pd.read_parquet(feat_out)
                        feat_df = pd.concat([prev, feat_df], ignore_index=True)
                    feat_df.to_parquet(feat_out, index=False)
                except Exception:
                    feat_csv = md / "features.csv"
                    header = not feat_csv.exists()
                    feat_df.to_csv(feat_csv, mode="a", header=header, index=False)

            # labels
            lab_rows = self.label_buf[mid]
            if lab_rows:
                ldf = pd.DataFrame(lab_rows)
                self.label_buf[mid] = []
                all_label_rows.append(ldf)

                lab_out = md / "labels.parquet"
                try:
                    if lab_out.exists():
                        prev = pd.read_parquet(lab_out)
                        ldf = pd.concat([prev, ldf], ignore_index=True)
                    ldf.to_parquet(lab_out, index=False)
                except Exception:
                    lab_csv = md / "labels.csv"
                    header = not lab_csv.exists()
                    ldf.to_csv(lab_csv, mode="a", header=header, index=False)

        # optional global index files for quick “all markets” ML training
        if self.write_index_all:
            idx = self.index_dir(day)
            if all_snap_rows:
                dfa = pd.concat(all_snap_rows, ignore_index=True)
                try:
                    out = idx / "snapshots_all.parquet"
                    if out.exists():
                        prev = pd.read_parquet(out)
                        dfa = pd.concat([prev, dfa], ignore_index=True)
                    dfa.to_parquet(out, index=False)
                except Exception:
                    out_csv = idx / "snapshots_all.csv"
                    header = not out_csv.exists()
                    dfa.to_csv(out_csv, mode="a", header=header, index=False)

            if all_label_rows:
                dla = pd.concat(all_label_rows, ignore_index=True)
                try:
                    out = idx / "labels_all.parquet"
                    if out.exists():
                        prev = pd.read_parquet(out)
                        dla = pd.concat([prev, dla], ignore_index=True)
                    dla.to_parquet(out, index=False)
                except Exception:
                    out_csv = idx / "labels_all.csv"
                    header = not out_csv.exists()
                    dla.to_csv(out_csv, mode="a", header=header, index=False)

        print(f"[FLUSH] day={day} @ {utc_iso()}")

    async def ws_loop(self):
        ws_url = f"{CLOB_WSS_BASE}/ws/market"
        sub = {"assets_ids": self.asset_ids, "type": "market"}

        print(f"[WS] connect {ws_url}")
        print(f"[WS] subscribe assets={len(self.asset_ids)}")

        while not self._stop.is_set():
            try:
                async with websockets.connect(ws_url, ping_interval=None, close_timeout=5) as ws:
                    await ws.send(json.dumps(sub))
                    last_ping = time.time()

                    while not self._stop.is_set():
                        if time.time() - last_ping > 10:
                            try:
                                await ws.send("PING")
                            except Exception:
                                break
                            last_ping = time.time()

                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue

                        if isinstance(msg, bytes):
                            msg = msg.decode("utf-8", errors="ignore")

                        if msg in ("PONG", "PING"):
                            continue

                        try:
                            js = json.loads(msg)
                        except Exception:
                            continue

                        # book updates
                        if isinstance(js, dict) and js.get("event_type") == "book":
                            token_id = js.get("asset_id")
                            ts = js.get("timestamp")
                            ts_ms = int(ts) if isinstance(ts, int) else utc_ms()

                            bids = topk_levels(js.get("bids", js.get("buys", [])), self.depth)
                            asks = topk_levels(js.get("asks", js.get("sells", [])), self.depth)
                            self.books[token_id] = BookState(ts_ms=ts_ms, bids=bids, asks=asks, hash=js.get("hash"))

                            # write raw message into all markets that include this token
                            # (yes it's duplicated, but that's what you want for per-market replay)
                            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            for spec in self.specs:
                                if token_id in (spec.yes_token, spec.no_token):
                                    self._write_raw_ws(spec, day, js)

            except Exception as e:
                print(f"[WS] reconnect after error: {type(e).__name__}: {e}")
                await asyncio.sleep(2)


def load_specs_from_grouped_json(path: Path, leagues: Optional[List[str]] = None, dates: Optional[List[str]] = None,
                                max_markets: Optional[int] = None) -> List[MarketSpec]:
    js = json.loads(path.read_text(encoding="utf-8"))
    byDate = js.get("byDate", {})
    out: List[MarketSpec] = []

    for d, leagues_map in byDate.items():
        if dates and d not in dates:
            continue
        for lg, markets in leagues_map.items():
            if leagues and lg not in leagues:
                continue
            for m in markets:
                market_id = str(m.get("id", ""))
                slug = m.get("slug", "")
                question = m.get("question", "")
                url = m.get("url", "")
                token_ids = json_load_any(m.get("clobTokenIds", "[]"))
                if not isinstance(token_ids, list) or len(token_ids) < 2:
                    continue
                out.append(MarketSpec(
                    date=d, league=lg, market_id=market_id, slug=slug, question=question,
                    yes_token=str(token_ids[0]), no_token=str(token_ids[1]), url=url
                ))

    if max_markets is not None:
        out = out[:max_markets]
    return out


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--out-dir", default="pm_data_v2")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--snap-interval", type=float, default=1.0)
    ap.add_argument("--flush-every", type=float, default=10.0)
    ap.add_argument("--league", action="append", default=None)
    ap.add_argument("--date", action="append", default=None)
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--raw-no-gzip", action="store_true")
    ap.add_argument("--no-index-all", action="store_true")

    ap.add_argument("--label-horizons", default="10,30,60", help="comma list, e.g. 10,30,60")
    ap.add_argument("--thr-10", type=float, default=0.02)
    ap.add_argument("--thr-30", type=float, default=0.035)
    ap.add_argument("--thr-60", type=float, default=0.05)

    args = ap.parse_args()
    input_path = Path(args.input_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    horizons = [int(x.strip()) for x in args.label_horizons.split(",") if x.strip()]
    thresholds = {10: args.thr_10, 30: args.thr_30, 60: args.thr_60}

    specs = load_specs_from_grouped_json(
        input_path,
        leagues=args.league if args.league else None,
        dates=args.date if args.date else None,
        max_markets=args.max_markets
    )
    if not specs:
        print("[ERR] No markets found after filters.")
        sys.exit(2)

    rec = Recorder(
        specs=specs,
        out_dir=out_dir,
        depth=args.depth,
        snap_interval_s=args.snap_interval,
        flush_every_s=args.flush_every,
        bootstrap_rest=not args.no_bootstrap,
        raw_gzip=not args.raw_no_gzip,
        write_index_all=not args.no_index_all,
        horizons_s=horizons,
        thresholds=thresholds,
    )
    rec.write_meta(input_path)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, rec.request_stop)
        except NotImplementedError:
            pass

    await rec.bootstrap_books()

    ws_task = asyncio.create_task(rec.ws_loop())
    snap_task = asyncio.create_task(rec.snapshot_loop())

    await asyncio.wait([ws_task, snap_task], return_when=asyncio.FIRST_COMPLETED)
    rec.request_stop()
    await asyncio.gather(ws_task, snap_task, return_exceptions=True)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
