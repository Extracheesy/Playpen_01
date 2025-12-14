import requests
import csv

BASE = "https://gamma-api.polymarket.com"

DATE_TOKEN = "-2025-12-14-"
SLUG_TOKEN = "btts"          # both teams to score
CLOSED = False               # keep only open markets
LIMIT = 200                  # gamma usually supports up to 200
MAX_PAGES = 200              # raise if needed (brute force crawl)


def get_json(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def is_openish(m: dict) -> bool:
    if m.get("closed") is True or m.get("isClosed") is True:
        return False
    if "active" in m and m.get("active") is False:
        return False
    status = str(m.get("status", "")).lower().strip()
    if status in {"closed", "inactive", "resolved", "canceled", "cancelled", "ended", "settled"}:
        return False
    return True


def scan_markets_slug_contains(date_token: str, slug_token: str, closed=False, limit=200, max_pages=200):
    out = []
    offset = 0

    for page in range(max_pages):
        params = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        batch = get_json("/markets", params=params)
        if not batch:
            break

        for m in batch:
            slug = str(m.get("slug", "")).lower()
            if date_token in slug and slug_token in slug:
                if is_openish(m):
                    out.append(m)

        offset += limit

        # small progress log every 10 pages
        if page % 10 == 0:
            print(f"[DEBUG] scanned pages={page+1}, offset={offset}, matches={len(out)}")

    return out


if __name__ == "__main__":
    print(f"[INFO] Searching markets where slug contains '{DATE_TOKEN}' AND '{SLUG_TOKEN}' ...")
    markets = scan_markets_slug_contains(DATE_TOKEN, SLUG_TOKEN, closed=CLOSED, limit=LIMIT, max_pages=MAX_PAGES)

    print(f"\n=== MATCHING MARKETS === {len(markets)}")
    for m in markets[:300]:
        q = m.get("question") or m.get("title") or m.get("slug")
        mid = m.get("id") or m.get("marketId")
        print(f"- {q} | slug={m.get('slug')} | market_id={mid}")

    # save for later use
    out_csv = "btts_2025_12_14_markets.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["market_id", "slug", "question"])
        w.writeheader()
        for m in markets:
            w.writerow({
                "market_id": m.get("id") or m.get("marketId"),
                "slug": m.get("slug"),
                "question": m.get("question") or m.get("title") or "",
            })

    print(f"\n[INFO] Wrote: {out_csv}")
