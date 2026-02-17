"""
windrun_collector.py

Purpose
-------
This module pulls two datasets from windrun.io for Dota 2 Ability Draft:

1) Ability "weights" (win rate style metrics) from:
      https://windrun.io/ability-high-skill

2) Ability pairs (pair win rate and synergy) from:
      https://windrun.io/ability-pairs

Why Playwright is needed
------------------------
Windrun is a React site. Some tables are not present in the raw HTML you get via
requests.get(...). Those tables are rendered in the browser with JavaScript.

Why "weights" needs scrolling too
---------------------------------
The ability-high-skill table is virtualized. Only a slice of rows exist in the DOM
at once. If we scrape without scrolling, we only get about 20 rows.
"""

import re
import requests
from bs4 import BeautifulSoup

ABILITY_WEIGHTS_URL = "https://windrun.io/ability-high-skill"
ABILITY_PAIRS_URL = "https://windrun.io/ability-pairs"


# --------------------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    """
    Convert a user-facing name (eg "Ball Lightning") into a stable key.
    """
    s = str(s).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)         # drop parenthetical notes
    s = s.replace("&", "and")             # deterministic for "&"
    s = re.sub(r"\s+", "_", s)            # spaces -> underscores
    s = re.sub(r"[^a-z0-9_]+", "", s)     # remove punctuation
    s = re.sub(r"_+", "_", s).strip("_")  # collapse underscores
    return s


def _to_float(x):
    """
    Convert numeric-ish strings into float.

    Examples:
      "52.4%"  -> 52.4
      "+19.4%" -> 19.4
      "1,234"  -> 1234.0
    """
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace(",", "").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def _href_to_name(href: str) -> str:
    """
    Convert a URL-ish href into a human-ish name.
    """
    if not href:
        return ""
    h = href.strip()
    h = h.split("?")[0].split("#")[0].rstrip("/")
    last = h.split("/")[-1]
    last = last.replace("-", " ").replace("_", " ").strip()
    return last


# --------------------------------------------------------------------------------------
# Table parsing: Playwright (shared cell extraction)
# --------------------------------------------------------------------------------------
def _parse_table_cells_playwright(page) -> tuple[list[str], list[list[str]]]:
    """
    Read headers and visible tbody rows from the first <table> on the page.
    Uses robust cell extraction:
      1) innerText
      2) aria-label
      3) img alt/title
      4) a[href] as HREF:...
    """
    headers = page.eval_on_selector_all(
        "table thead th",
        "ths => ths.map(th => (th.innerText || '').trim())"
    )

    rows = page.evaluate(
        r"""
        () => {
          const trs = Array.from(document.querySelectorAll('table tbody tr'));
          return trs.map(tr => {
            const tds = Array.from(tr.querySelectorAll('td'));
            return tds.map(td => {
              let txt = (td.innerText || '').trim();
              if (txt) return txt;

              const ariaSelf = (td.getAttribute('aria-label') || '').trim();
              if (ariaSelf) return ariaSelf;

              const img = td.querySelector('img');
              if (img) {
                const alt = (img.getAttribute('alt') || '').trim();
                if (alt) return alt;
                const title = (img.getAttribute('title') || '').trim();
                if (title) return title;
              }

              const a = td.querySelector('a[href]');
              if (a) {
                const href = (a.getAttribute('href') || '').trim();
                if (href) return 'HREF:' + href;
              }

              return '';
            });
          });
        }
        """
    )

    cleaned = []
    for r in rows:
        if not r:
            continue
        if all(str(x).strip() == "" for x in r):
            continue

        r2 = []
        for cell in r:
            if isinstance(cell, str) and cell.startswith("HREF:"):
                r2.append(_href_to_name(cell[5:]))
            else:
                r2.append(cell)
        cleaned.append(r2)

    return headers, cleaned


# --------------------------------------------------------------------------------------
# Table parsing: Playwright simple (no scroll)
# --------------------------------------------------------------------------------------
def _parse_simple_table_playwright(url: str):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to scrape Windrun pages.\n"
            "Install with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("table")

        headers, rows = _parse_table_cells_playwright(page)
        browser.close()

    return headers, rows


# --------------------------------------------------------------------------------------
# Table parsing: Playwright scroll + accumulate (for virtualized tables)
# --------------------------------------------------------------------------------------
def _parse_table_playwright_scroll(url: str, max_scroll_steps: int = 260, stale_limit: int = 12):
    """
    Generic virtualized-table scraper:
      - load page
      - repeatedly read visible rows
      - scroll
      - accumulate unique rows until stale
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to scrape Windrun pages.\n"
            "Install with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("table")

        # Wait until we see at least one non-empty cell
        for _ in range(80):
            ok = page.evaluate(
                r"""
                () => {
                  const tds = Array.from(document.querySelectorAll('table tbody td'));
                  for (const td of tds) {
                    const txt = (td.innerText || '').trim();
                    if (txt) return true;

                    const img = td.querySelector('img');
                    if (img) {
                      const alt = (img.getAttribute('alt') || '').trim();
                      const title = (img.getAttribute('title') || '').trim();
                      if (alt || title) return true;
                    }

                    const a = td.querySelector('a[href]');
                    if (a && (a.getAttribute('href') || '').trim()) return true;
                  }
                  return false;
                }
                """
            )
            if ok:
                break
            page.wait_for_timeout(250)

        headers, _ = _parse_table_cells_playwright(page)

        seen = set()
        out = []
        stale = 0

        page.hover("table")

        for _ in range(max_scroll_steps):
            _, rows = _parse_table_cells_playwright(page)

            new = 0
            for r in rows:
                if headers and len(r) > len(headers):
                    r = r[: len(headers)]
                key = tuple(r)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
                    new += 1

            if new == 0:
                stale += 1
            else:
                stale = 0

            if stale >= stale_limit:
                break

            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(200)

        browser.close()

    return headers, out


# --------------------------------------------------------------------------------------
# Table parsing: requests + bs4 fast path (kept, but not used for weights anymore)
# --------------------------------------------------------------------------------------
def _parse_first_table_requests(url: str):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return _parse_simple_table_playwright(url)

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    tbody = table.find("tbody")
    if tbody is None:
        return _parse_simple_table_playwright(url)

    rows = []
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if tds and any(str(x).strip() for x in tds):
            rows.append(tds)

    if not rows:
        return _parse_simple_table_playwright(url)

    return headers, rows


# --------------------------------------------------------------------------------------
# Public API: "weights" loader
# --------------------------------------------------------------------------------------
def load_windrun_ability_weights() -> dict:
    """
    Load "ability weights" from windrun ability-high-skill.

    Output:
      { ability_norm: wr_high_float }

    Important:
      ability-high-skill is virtualized. We must scroll to collect the full dataset.
    """
    headers, rows = _parse_table_playwright_scroll(ABILITY_WEIGHTS_URL)

    def hnorm(x: str) -> str:
        x = str(x).strip().lower()
        x = x.replace("Δ", "delta")
        x = re.sub(r"\s+", " ", x)
        return x

    header_map = {hnorm(h): i for i, h in enumerate(headers)}

    def col(*names: str):
        for n in names:
            i = header_map.get(hnorm(n))
            if i is not None:
                return i
        return None

    ability_i = col("ability", "name")
    if ability_i is None:
        ability_i = 0

    weight_i = col("wr all", "wr high", "win %", "win rate", "wr")
    if weight_i is None:
        for i, h in enumerate(headers):
            hl = hnorm(h)
            if "wr" in hl or "win" in hl:
                weight_i = i
                break

    if weight_i is None:
        raise RuntimeError(f"Could not find a win column in headers: {headers}")

    weights = {}
    for r in rows:
        if len(r) <= max(ability_i, weight_i):
            continue

        ability_raw = r[ability_i]
        weight_raw = r[weight_i]

        a = _norm(ability_raw)
        w = _to_float(weight_raw)

        if a:
            weights[a] = w

    return weights


# --------------------------------------------------------------------------------------
# Public API: "pairs" loader
# --------------------------------------------------------------------------------------
def load_windrun_ability_pairs():
    """
    Load ability pairs from windrun ability-pairs.
    """
    headers, rows = _parse_table_playwright_scroll(ABILITY_PAIRS_URL)

    headers_l = [h.strip().lower() for h in headers]

    def idx(name: str, fallback: int):
        name = name.strip().lower()
        return headers_l.index(name) if name in headers_l else fallback

    a_i = idx("ability 1", 0)
    b_i = idx("ability 2", 2)
    score_i = idx("pair wr", 4)

    out = []
    for r in rows:
        if len(r) <= max(a_i, b_i, score_i):
            continue

        a_raw = r[a_i]
        b_raw = r[b_i]
        score_raw = r[score_i]

        a = _norm(a_raw)
        b = _norm(b_raw)
        score = _to_float(score_raw)

        if not a or not b:
            continue

        out.append(
            {
                "a_raw": a_raw,
                "b_raw": b_raw,
                "a_norm": a,
                "b_norm": b,
                "score": score,
            }
        )

    return out


def load_windrun_ability_table():
    return load_windrun_ability_weights()
