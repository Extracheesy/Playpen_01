#!/usr/bin/env python3
from __future__ import annotations

"""
GUI wrapper for list_league_by_date.py (multi-league)

Features:
- Select ONE or MULTIPLE leagues (NBA and/or football leagues, plus "all")
- Date window: today / tomorrow / both / range (today -> end date)
- Default excludes identical; user can untick
- Fetch markets, then select which markets to export
- Export selected to JSON, with default name based on selected leagues + date window
"""

import json
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

import requests

# --- Core behavior (aligned with list_league_by_date.py) ---
GAMMA = "https://gamma-api.polymarket.com"
LIMIT = 200

DEFAULT_SLUG_EXCLUDE = ["btts", "spread", "total"]  # must NOT appear in slug (all leagues)
DEFAULT_NBA_SLUG_EXCLUDE = ["assists", "points", "rebounds", "spread", "total", "1h"]  # NBA-only (props)

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

    # Basketball
    "nba": "__NBA__",
}

DISPLAY_LEAGUES = [
    ("NBA", "nba"),
    ("Premier League (EPL)", "epl"),
    ("La Liga", "liga"),
    ("Bundesliga", "bun"),
    ("Serie A", "sea"),
    ("Ligue 1", "fl1"),
    ("Eredivisie", "ere"),
    ("Liga MX", "mex"),
    ("All (everything above)", "all"),
]


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


def normalize_leagues(raw_leagues: list[str]) -> list[str]:
    leagues = [x.strip().lower() for x in raw_leagues if x and x.strip()]
    if not leagues:
        return []

    if "all" in leagues:
        # keep it stable and predictable, but exclude "all" itself
        return sorted(set(k for k in LEAGUE_TO_SLUG_PREFIX.keys() if k != "all"))

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


def _as_lower_str(x) -> str:
    return (x or "").lower() if isinstance(x, str) else ""


def _as_lower_list(x) -> list[str]:
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str):
                out.append(v.lower())
        return out
    return []


def is_nba_market(m: dict) -> bool:
    """
    NBA markets are best detected via metadata rather than slug prefixes.
    We try several common fields that appear in Gamma responses.
    """
    slug = _as_lower_str(m.get("slug"))
    question = _as_lower_str(m.get("question"))
    category = _as_lower_str(m.get("category"))
    sport = _as_lower_str(m.get("sport"))
    league = _as_lower_str(m.get("league"))
    tags = _as_lower_list(m.get("tags"))

    if slug.startswith("nba-"):
        return True

    hay = " ".join([category, sport, league, question, " ".join(tags)])

    if " nba " in f" {hay} " or hay.startswith("nba") or "nba:" in hay:
        return True

    if "basketball" in hay and ("nba" in hay or "national basketball association" in hay):
        return True

    if any(t == "nba" or "nba" in t for t in tags):
        return True

    return False


def parse_market_date_from_start_time(m: dict, tz: ZoneInfo) -> str | None:
    st = (
        m.get("startTime")
        or m.get("gameStartTime")
        or m.get("eventStartTime")
        or m.get("startDate")
        or m.get("endDate")
    )
    if not isinstance(st, str) or not st:
        return None

    st_norm = st.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(st_norm)
    except ValueError:
        if len(st) >= 10 and st[4] == "-" and st[7] == "-":
            return st[:10]
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(tz).date().strftime("%Y-%m-%d")


@dataclass
class Match:
    id: str | int | None
    league: str
    slugPrefix: str
    date: str
    slug: str
    question: str | None
    active: bool | None
    closed: bool | None
    archived: bool | None
    outcomes: object
    clobTokenIds: object
    startTime: str | None
    url: str

    def display(self) -> str:
        q = (self.question or "").strip()
        if len(q) > 120:
            q = q[:117] + "..."
        return f"{self.date} | {self.league.upper():4s} | {self.slug} | {q}"


def compute_dates_gui(mode: str, tz: ZoneInfo, end_date_str: str | None) -> list[str]:
    today_d = datetime.now(tz).date()

    if mode == "today":
        return [today_d.strftime("%Y-%m-%d")]
    if mode == "tomorrow":
        return [(today_d + timedelta(days=1)).strftime("%Y-%m-%d")]
    if mode == "both":
        return [
            today_d.strftime("%Y-%m-%d"),
            (today_d + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]
    if mode == "range":
        if not end_date_str:
            raise ValueError("End date is required for range mode.")
        try:
            y, m, d = (int(x) for x in end_date_str.split("-"))
            end_d = date(y, m, d)
        except Exception as e:
            raise ValueError("End date must be in YYYY-MM-DD format.") from e

        if end_d < today_d:
            raise ValueError("End date must be today or later.")

        out = []
        cur = today_d
        while cur <= end_d:
            out.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return out

    raise ValueError(f"Invalid mode: {mode}")


def find_matches(
    leagues: list[str],
    dates: list[str],
    slug_exclude: list[str],
    nba_slug_exclude: list[str],
) -> tuple[list[Match], dict]:
    tz = ZoneInfo("Europe/Paris")
    date_patterns = [f"-{d}-".lower() for d in dates] + [f"-{d}".lower() for d in dates]
    league_prefixes = {lk: LEAGUE_TO_SLUG_PREFIX[lk].lower() for lk in leagues}
    date_regex = re.compile(r"-(\d{4}-\d{2}-\d{2})-")

    all_markets = fetch_all_active_markets()
    matches: list[Match] = []

    for m in all_markets:
        slug = m.get("slug") or ""
        if not isinstance(slug, str) or not slug:
            continue
        slug_l = slug.lower()

        if any(x in slug_l for x in slug_exclude):
            continue

        matched_league = None
        matched_prefix = None

        for lk, pref in league_prefixes.items():
            if lk == "nba":
                if is_nba_market(m):
                    matched_league = "nba"
                    matched_prefix = "nba"
                    break
            else:
                if slug_l.startswith(pref):
                    matched_league = lk
                    matched_prefix = pref
                    break

        if not matched_league:
            continue

        if matched_league == "nba":
            if any(x in slug_l for x in nba_slug_exclude):
                continue

        matched_date = None
        mm_date = date_regex.search(slug_l)
        if mm_date:
            matched_date = mm_date.group(1)
        else:
            for pat in date_patterns:
                if pat in slug_l:
                    matched_date = pat.strip("-")
                    break

        if not matched_date:
            if matched_league == "nba":
                d_from_time = parse_market_date_from_start_time(m, tz)
                if not d_from_time or d_from_time not in dates:
                    continue
                matched_date = d_from_time
            else:
                continue

        mdate = matched_date
        mm = date_regex.search(slug_l)
        if mm:
            mdate = mm.group(1)

        matches.append(
            Match(
                id=m.get("id"),
                league=matched_league,
                slugPrefix=str(matched_prefix or ""),
                date=str(mdate),
                slug=slug,
                question=m.get("question"),
                active=m.get("active"),
                closed=m.get("closed"),
                archived=m.get("archived"),
                outcomes=m.get("outcomes"),
                clobTokenIds=m.get("clobTokenIds"),
                startTime=m.get("startTime"),
                url=f"https://polymarket.com/market/{slug}",
            )
        )

    matches.sort(key=lambda x: (x.date or "", x.league or "", x.slug or ""))

    by_date: dict[str, dict[str, list[dict]]] = {}
    for mm in matches:
        d = mm.date or "unknown"
        l = mm.league or "unknown"
        by_date.setdefault(d, {}).setdefault(l, []).append(mm.__dict__)

    payload = {
        "meta": {
            "generatedAtParis": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
            "leagues": leagues,
            "dates": dates,
            "slugExclude": slug_exclude,
            "nbaSlugExclude": nba_slug_exclude,
            "source": {
                "api": "gamma",
                "endpoint": f"{GAMMA}/markets",
                "filters": {"active": True, "closed": False, "archived": False},
            },
            "count": len(matches),
        },
        "byDate": by_date,
    }

    return matches, payload


def make_default_filename(league_keys: list[str], mode: str, dates: list[str]) -> str:
    # Compact league part
    if not league_keys:
        league_part = "league"
    else:
        # If user picked a lot, keep it short
        uniq = []
        for k in league_keys:
            if k not in uniq:
                uniq.append(k)
        if len(uniq) <= 3:
            league_part = "+".join(uniq)
        else:
            league_part = f"multi{len(uniq)}"

    if mode in ("today", "tomorrow") and len(dates) == 1:
        dates_part = dates[0]
    elif mode in ("both", "range") and len(dates) >= 2:
        dates_part = f"{dates[0]}_to_{dates[-1]}"
    else:
        dates_part = dates[0] if dates else "dates"

    return f"{league_part}_{dates_part}.json"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Polymarket League-by-Date (GUI) — Multi-league")
        self.geometry("1300x760")
        self.minsize(1050, 660)

        self.tz = ZoneInfo("Europe/Paris")
        self.matches: list[Match] = []
        self._fetch_thread: threading.Thread | None = None

        # --- Top controls ---
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        # League multi-select
        league_frame = ttk.LabelFrame(top, text="Leagues (Ctrl/Shift to multi-select)", padding=8)
        league_frame.grid(row=0, column=0, rowspan=2, sticky="nw")

        self.league_listbox = tk.Listbox(league_frame, selectmode="extended", height=7, exportselection=False)
        self.league_listbox.pack(side="left", fill="y")
        sb = ttk.Scrollbar(league_frame, orient="vertical", command=self.league_listbox.yview)
        sb.pack(side="right", fill="y")
        self.league_listbox.configure(yscrollcommand=sb.set)

        self._league_keys_in_order: list[str] = []
        for label, key in DISPLAY_LEAGUES:
            self.league_listbox.insert(tk.END, label)
            self._league_keys_in_order.append(key)

        # Default select NBA
        try:
            idx = self._league_keys_in_order.index("nba")
            self.league_listbox.selection_set(idx)
        except ValueError:
            pass

        self.league_listbox.bind("<<ListboxSelect>>", lambda e: self._sync_nba_excludes_enabled())

        # Date controls
        date_frame = ttk.Frame(top)
        date_frame.grid(row=0, column=1, sticky="nw", padx=(14, 0))

        ttk.Label(date_frame, text="Date window:").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="both")
        self.mode_combo = ttk.Combobox(
            date_frame,
            textvariable=self.mode_var,
            values=["today", "tomorrow", "both", "range"],
            state="readonly",
            width=12,
        )
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 16))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self._sync_end_date_enabled())

        ttk.Label(date_frame, text="End date (YYYY-MM-DD):").grid(row=0, column=2, sticky="w")
        self.end_date_var = tk.StringVar(value="")
        self.end_date_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=14)
        self.end_date_entry.grid(row=0, column=3, sticky="w", padx=(6, 16))

        self.fetch_btn = ttk.Button(date_frame, text="Fetch markets", command=self.on_fetch)
        self.fetch_btn.grid(row=0, column=4, sticky="w")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(top, textvariable=self.status_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # --- Middle area: excludes + list ---
        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        excl = ttk.LabelFrame(mid, text="Exclude slug fragments (untick to allow)", padding=10)
        excl.pack(side="left", fill="y")

        ttk.Label(excl, text="Global excludes:").pack(anchor="w")
        self.excl_vars: dict[str, tk.BooleanVar] = {}
        for frag in DEFAULT_SLUG_EXCLUDE:
            v = tk.BooleanVar(value=True)
            self.excl_vars[frag] = v
            ttk.Checkbutton(excl, text=frag, variable=v).pack(anchor="w")

        ttk.Separator(excl, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(excl, text="NBA-only excludes (props):").pack(anchor="w")
        self.nba_excl_vars: dict[str, tk.BooleanVar] = {}
        for frag in DEFAULT_NBA_SLUG_EXCLUDE:
            v = tk.BooleanVar(value=True)
            self.nba_excl_vars[frag] = v
            ttk.Checkbutton(excl, text=frag, variable=v).pack(anchor="w")

        ttk.Separator(excl, orient="horizontal").pack(fill="x", pady=10)

        self.select_all_btn = ttk.Button(excl, text="Select all shown", command=self.select_all)
        self.select_all_btn.pack(fill="x", pady=(0, 6))
        self.clear_sel_btn = ttk.Button(excl, text="Clear selection", command=self.clear_selection)
        self.clear_sel_btn.pack(fill="x")

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        top_list = ttk.Frame(right)
        top_list.pack(fill="x")

        self.filter_var = tk.StringVar(value="")
        ttk.Label(top_list, text="Filter (contains):").pack(side="left")
        self.filter_entry = ttk.Entry(top_list, textvariable=self.filter_var, width=40)
        self.filter_entry.pack(side="left", padx=(6, 8))
        self.filter_entry.bind("<KeyRelease>", lambda e: self.refresh_listbox())

        self.count_var = tk.StringVar(value="0 markets")
        ttk.Label(top_list, textvariable=self.count_var).pack(side="right")

        self.listbox = tk.Listbox(right, selectmode="extended", activestyle="none")
        self.listbox.pack(fill="both", expand=True, pady=(8, 0))

        # --- Bottom: export ---
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")

        ttk.Label(bottom, text="Export filename:").pack(side="left")
        self.out_name_var = tk.StringVar(value="")
        self.out_entry = ttk.Entry(bottom, textvariable=self.out_name_var, width=60)
        self.out_entry.pack(side="left", padx=(6, 10))

        self.browse_btn = ttk.Button(bottom, text="Browse…", command=self.on_browse)
        self.browse_btn.pack(side="left")

        self.export_btn = ttk.Button(bottom, text="Export selected to JSON", command=self.on_export)
        self.export_btn.pack(side="right")

        self._sync_end_date_enabled()
        self._sync_nba_excludes_enabled()
        self._refresh_default_outname()

    # --- selection helpers ---
    def get_selected_league_keys(self) -> list[str]:
        idxs = list(self.league_listbox.curselection())
        keys = [self._league_keys_in_order[i] for i in idxs if 0 <= i < len(self._league_keys_in_order)]
        # If nothing selected, treat as empty
        return keys

    def _sync_end_date_enabled(self) -> None:
        mode = self.mode_var.get()
        self.end_date_entry.configure(state=("normal" if mode == "range" else "disabled"))

    def _walk_widgets(self, root):
        stack = [root]
        while stack:
            w = stack.pop()
            yield w
            stack.extend(getattr(w, "winfo_children", lambda: [])())

    def _sync_nba_excludes_enabled(self) -> None:
        # Enable NBA-only exclude checkboxes if NBA is among selected leagues (or all)
        selected = set(self.get_selected_league_keys())
        is_nba = ("nba" in selected) or ("all" in selected)
        state = "normal" if is_nba else "disabled"

        for w in self._walk_widgets(self):
            if isinstance(w, ttk.Checkbutton):
                txt = str(w.cget("text"))
                if txt in self.nba_excl_vars:
                    w.configure(state=state)

    def _refresh_default_outname(self) -> None:
        try:
            league_keys = self.get_selected_league_keys()
            mode = self.mode_var.get()
            dates = compute_dates_gui(mode, self.tz, self.end_date_var.get().strip() or None)
            default_name = make_default_filename(league_keys, mode, dates)
            cur = self.out_name_var.get().strip()
            if not cur:
                self.out_name_var.set(default_name)
        except Exception:
            pass

    # --- Actions ---
    def on_fetch(self) -> None:
        if self._fetch_thread and self._fetch_thread.is_alive():
            messagebox.showinfo("Busy", "Already fetching markets. Please wait.")
            return

        league_keys = self.get_selected_league_keys()
        if not league_keys:
            messagebox.showerror("No leagues", "Select at least one league (or 'all').")
            return

        mode = self.mode_var.get().strip().lower()
        end_date = self.end_date_var.get().strip() or None

        try:
            leagues = normalize_leagues(league_keys)
        except Exception as e:
            messagebox.showerror("Invalid leagues", str(e))
            return

        try:
            dates = compute_dates_gui(mode, self.tz, end_date)
        except Exception as e:
            messagebox.showerror("Invalid date window", str(e))
            return

        slug_exclude = [k for k, v in self.excl_vars.items() if v.get()]
        nba_slug_exclude = [k for k, v in self.nba_excl_vars.items() if v.get()]

        self.status_var.set("Fetching markets from Gamma…")
        self.fetch_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.listbox.delete(0, tk.END)
        self.matches = []
        self.count_var.set("0 markets")
        self._refresh_default_outname()

        def worker():
            try:
                ms, _payload = find_matches(
                    leagues=leagues,
                    dates=dates,
                    slug_exclude=slug_exclude,
                    nba_slug_exclude=nba_slug_exclude,
                )
                self.matches = ms
                self.after(0, lambda: self._after_fetch_ok(leagues, dates, len(ms)))
            except Exception as e:
                self.after(0, lambda: self._after_fetch_err(e))

        self._fetch_thread = threading.Thread(target=worker, daemon=True)
        self._fetch_thread.start()

    def _after_fetch_ok(self, leagues: list[str], dates: list[str], n: int) -> None:
        self.status_var.set(f"[OK] Found {n} market(s) for leagues={leagues} dates={dates}")
        self.fetch_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        self.refresh_listbox()
        # refresh default name after successful fetch (nice UX)
        league_keys = self.get_selected_league_keys()
        mode = self.mode_var.get().strip().lower()
        try:
            default_name = make_default_filename(league_keys, mode, dates)
            if not self.out_name_var.get().strip():
                self.out_name_var.set(default_name)
        except Exception:
            pass

    def _after_fetch_err(self, e: Exception) -> None:
        self.status_var.set("Error.")
        self.fetch_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        messagebox.showerror("Fetch error", f"{type(e).__name__}: {e}")

    def refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        filt = self.filter_var.get().strip().lower()
        shown = 0
        for m in self.matches:
            line = m.display()
            if filt and filt not in line.lower():
                continue
            self.listbox.insert(tk.END, line)
            shown += 1
        self.count_var.set(f"{shown} markets shown (of {len(self.matches)})")

    def select_all(self) -> None:
        self.listbox.selection_set(0, tk.END)

    def clear_selection(self) -> None:
        self.listbox.selection_clear(0, tk.END)

    def on_browse(self) -> None:
        initial = self.out_name_var.get().strip() or "markets.json"
        if not initial.lower().endswith(".json"):
            initial += ".json"

        path = filedialog.asksaveasfilename(
            title="Save JSON as…",
            defaultextension=".json",
            initialfile=initial,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.out_name_var.set(path)

    def on_export(self) -> None:
        if not self.matches:
            messagebox.showinfo("Nothing to export", "Fetch markets first.")
            return

        filt = self.filter_var.get().strip().lower()
        visible: list[Match] = []
        for m in self.matches:
            if filt and filt not in m.display().lower():
                continue
            visible.append(m)

        sel = list(self.listbox.curselection())
        if not sel:
            messagebox.showinfo("No selection", "Select at least one market in the list.")
            return

        selected_matches = [visible[i] for i in sel if 0 <= i < len(visible)]

        league_keys = self.get_selected_league_keys()
        mode = self.mode_var.get().strip().lower()
        end_date = self.end_date_var.get().strip() or None

        leagues = normalize_leagues(league_keys)
        dates = compute_dates_gui(mode, self.tz, end_date)

        slug_exclude = [k for k, v in self.excl_vars.items() if v.get()]
        nba_slug_exclude = [k for k, v in self.nba_excl_vars.items() if v.get()]

        by_date: dict[str, dict[str, list[dict]]] = {}
        for mm in selected_matches:
            d = mm.date or "unknown"
            l = mm.league or "unknown"
            by_date.setdefault(d, {}).setdefault(l, []).append(mm.__dict__)

        payload = {
            "meta": {
                "generatedAtParis": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                "leagues": leagues,
                "dates": dates,
                "slugExclude": slug_exclude,
                "nbaSlugExclude": nba_slug_exclude,
                "source": {
                    "api": "gamma",
                    "endpoint": f"{GAMMA}/markets",
                    "filters": {"active": True, "closed": False, "archived": False},
                },
                "count": len(selected_matches),
                "note": "This file contains only the markets you selected in the GUI.",
            },
            "byDate": by_date,
        }

        out = self.out_name_var.get().strip()
        if not out:
            out = make_default_filename(league_keys, mode, dates)

        if not out.lower().endswith(".json"):
            out += ".json"

        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Save error", f"{type(e).__name__}: {e}")
            return

        self.status_var.set(f"[SAVED] {out}  ({len(selected_matches)} selected)")
        messagebox.showinfo("Export complete", f"Saved:\n{out}\n\nMarkets exported: {len(selected_matches)}")


def main() -> None:
    # Better DPI on Windows
    try:
        import ctypes  # noqa
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
