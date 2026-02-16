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

So we use a two-stage approach:
- Try requests + BeautifulSoup first (fast).
- If the table is missing or empty (JS-rendered), fall back to Playwright
  (slower but reliable).

Why the "pairs" scraper is special
----------------------------------
The ability-pairs table is virtualized (only a portion is in the DOM at once).
To collect the full dataset we must:
- Scroll repeatedly
- Re-read the currently visible rows each time
- Accumulate unique rows until scrolling stops yielding new rows
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
    Convert a user-facing name (eg "Ball Lightning") into a stable key:

      "Ball Lightning" -> "ball_lightning"
      "Berserker's Blood" -> "berserkers_blood"
      "X&Y" -> "xandy"

    We use this as a join key between:
    - what your YOLO / inference returns
    - what Windrun returns
    - your icon filenames on disk
    """
    s = str(s).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)         # drop parenthetical notes: "Foo (Shard)" -> "Foo"
    s = s.replace("&", "and")             # keep deterministic behaviour for "&"
    s = re.sub(r"\s+", "_", s)            # spaces -> underscores
    s = re.sub(r"[^a-z0-9_]+", "", s)     # remove punctuation and weird characters
    s = re.sub(r"_+", "_", s).strip("_")  # collapse multiple underscores and trim
    return s


def _to_float(x):
    """
    Convert numeric-ish strings into float.

    Examples:
      "52.4%"  -> 52.4
      "+19.4%" -> 19.4
      "1,234"  -> 1234.0

    Returns None if conversion fails.
    """
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace(",", "").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------
# Table parsing: Playwright (browser-rendered)
# --------------------------------------------------------------------------------------
def _parse_simple_table_playwright(url: str):
    """
    Render a page with Playwright and scrape the first HTML <table>.

    This is used for tables that exist only after JavaScript runs.
    It reads each <td> by trying multiple strategies in priority order:

      1) Visible text (innerText)
      2) aria-label (sometimes used for accessibility)
      3) <img> alt / title (sometimes icons carry the "real" value)
      4) <a href> (rare, but can act as a fallback identifier)

    Returns:
      headers: List[str]
      rows:    List[List[str]]  (each row is a list of cell strings)
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
        # Headless Chromium is fast and consistent for scraping.
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")

        # Wait until network is idle so React has time to fetch data.
        page.goto(url, wait_until="networkidle")

        # Ensure the table exists before we attempt to scrape it.
        page.wait_for_selector("table")

        # Capture header row text
        headers = page.eval_on_selector_all(
            "table thead th",
            "ths => ths.map(th => (th.innerText || '').trim())"
        )

        # Capture body cells using a robust extraction routine.
        rows = page.evaluate(
            r"""
            () => {
              const trs = Array.from(document.querySelectorAll('table tbody tr'));
              return trs.map(tr => {
                const tds = Array.from(tr.querySelectorAll('td'));
                return tds.map(td => {
                  // 1) Visible text
                  let txt = (td.innerText || '').trim();
                  if (txt) return txt;

                  // 2) aria-label
                  const ariaSelf = (td.getAttribute('aria-label') || '').trim();
                  if (ariaSelf) return ariaSelf;

                  // 3) image alt/title
                  const img = td.querySelector('img');
                  if (img) {
                    const alt = (img.getAttribute('alt') || '').trim();
                    if (alt) return alt;
                    const title = (img.getAttribute('title') || '').trim();
                    if (title) return title;
                  }

                  // 4) href
                  const a = td.querySelector('a[href]');
                  if (a) {
                    const href = (a.getAttribute('href') || '').trim();
                    if (href) return href;
                  }

                  // If we still have nothing, return empty string.
                  return '';
                });
              });
            }
            """
        )

        browser.close()

    # Remove blank placeholder rows (common during initial rendering / virtualization).
    cleaned = []
    for r in rows:
        if not r:
            continue
        if all(str(x).strip() == "" for x in r):
            continue
        cleaned.append(r)

    return headers, cleaned


# --------------------------------------------------------------------------------------
# Table parsing: requests + bs4 fast path, with Playwright fallback
# --------------------------------------------------------------------------------------
def _parse_first_table_requests(url: str):
    """
    Attempt to scrape the first table from server-rendered HTML via requests + BeautifulSoup.

    Why this exists:
    - It's much faster than launching a browser.
    - It works when the site includes the table in its HTML.

    When it fails:
    - React sites often deliver minimal HTML and build the table later via JS.
      In that case soup.find("table") returns None, or tbody is empty.

    Behaviour:
    - If any "this looks JS-rendered" condition is detected, fall back to Playwright.
    """
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
        if tds:
            rows.append(tds)

    # If the table exists but no rows appear, it likely isn't actually populated server-side.
    if not rows:
        return _parse_simple_table_playwright(url)

    return headers, rows


# --------------------------------------------------------------------------------------
# Table parsing: ability pairs (virtualized) via Playwright scroll + accumulate
# --------------------------------------------------------------------------------------
def _parse_pairs_table_playwright_scroll(url: str, max_scroll_steps: int = 220):
    """
    Scrape the ability-pairs table which is virtualized.

    Virtualized table means:
    - Only the visible rows exist in the DOM at any given time.
    - Scrolling causes React to recycle rows (DOM nodes change content).

    Strategy:
    - Load page
    - Wait until *some* data exists
    - Read currently visible rows
    - Scroll
    - Repeat, accumulating unique rows
    - Stop once repeated scrolling yields no new rows (stale threshold)

    Returns:
      headers: List[str]
      out:     List[List[str]]  (all distinct rows discovered via scrolling)
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to scrape Windrun ability pairs.\n"
            "Install with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        ) from e

    def href_to_name(cell: str) -> str:
        """
        Convert a scraped "HREF:/path/to/ability-name" into a human-ish name.

        This is used only when a cell contains no text but contains an <a href="...">.
        """
        href = cell[5:].split("?")[0].split("#")[0].rstrip("/")
        href = href.split("/")[-1]
        return href.replace("-", " ").replace("_", " ").strip()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")

        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("table")

        # Some React pages show an empty table briefly. Poll until we see any non-empty cell.
        for _ in range(60):
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

        # Read table headers once.
        headers = page.eval_on_selector_all(
            "table thead th",
            "ths => ths.map(th => (th.innerText || '').trim())"
        )

        def read_rows():
            """
            Read the currently visible rows in the table body.

            This returns only what's currently in the DOM (a slice of the full dataset).
            """
            rows = page.evaluate(
                r"""
                () => {
                  const trs = Array.from(document.querySelectorAll('table tbody tr'));
                  return trs.map(tr => {
                    const tds = Array.from(tr.querySelectorAll('td'));
                    return tds.map(td => {
                      // 1) Visible text
                      let txt = (td.innerText || '').trim();
                      if (txt) return txt;

                      // 2) aria-label
                      const ariaSelf = (td.getAttribute('aria-label') || '').trim();
                      if (ariaSelf) return ariaSelf;

                      // 3) image alt/title
                      const img = td.querySelector('img');
                      if (img) {
                        const alt = (img.getAttribute('alt') || '').trim();
                        if (alt) return alt;
                        const title = (img.getAttribute('title') || '').trim();
                        if (title) return title;
                      }

                      // 4) href (prefix so we can recognise it later)
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
                if all((str(x).strip() == "" for x in r)):
                    continue
                cleaned.append(r)
            return cleaned

        # seen: the set of rows we've already collected (exact string tuple match)
        seen = set()
        out = []

        # stale counts how many scroll iterations produced 0 new rows.
        # Once it stays stale long enough, we assume we've hit the bottom.
        stale = 0

        # Ensure the table has focus for wheel scrolling.
        page.hover("table")

        for _ in range(max_scroll_steps):
            rows = read_rows()

            new = 0
            for r in rows:
                # Replace HREF:* cells with readable names so they don't poison uniqueness checks.
                r2 = []
                for cell in r:
                    if isinstance(cell, str) and cell.startswith("HREF:"):
                        r2.append(href_to_name(cell))
                    else:
                        r2.append(cell)

                # Defensive: ensure row width never exceeds header count.
                if headers and len(r2) > len(headers):
                    r2 = r2[: len(headers)]

                key = tuple(r2)
                if key not in seen:
                    seen.add(key)
                    out.append(r2)
                    new += 1

            # Track whether this scroll yielded anything new.
            if new == 0:
                stale += 1
            else:
                stale = 0

            # If we repeatedly stop getting new rows, we assume we're done.
            if stale >= 10:
                break

            # Scroll down and allow the virtualized table to update.
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(200)

        browser.close()

    return headers, out


# --------------------------------------------------------------------------------------
# Public API: "weights" loader
# --------------------------------------------------------------------------------------
def load_windrun_ability_weights() -> dict:
    """
    Load "ability weights" from windrun ability-high-skill.

    Output:
      { ability_norm: wr_high_float }

    Notes:
    - Windrun's column names may change over time.
    - Today we prefer WR HIGH (best proxy for "high skill winrate").
    - If WR HIGH is absent, we fall back to WR ALL, then any column containing "wr" or "win".
    """
    headers, rows = _parse_first_table_requests(ABILITY_WEIGHTS_URL)

    # Windrun headers can contain Unicode (eg "Δ") and inconsistent spacing.
    # This normaliser makes header matching stable.
    def hnorm(x: str) -> str:
        x = str(x).strip().lower()
        x = x.replace("Δ", "delta")     # treat "Δ" as the word "delta" so matching won't break
        x = re.sub(r"\s+", " ", x)      # collapse multiple spaces
        return x

    # Build a lookup map: normalised header -> index
    header_map = {hnorm(h): i for i, h in enumerate(headers)}

    def col(*names: str):
        """Return the column index of the first matching header name, else None."""
        for n in names:
            i = header_map.get(hnorm(n))
            if i is not None:
                return i
        return None

    # Ability column is typically "ABILITY".
    ability_i = col("ability", "name")
    if ability_i is None:
        ability_i = 0  # last resort fallback

    # Prefer WR HIGH, then WR ALL, then any "wr"/"win" column.
    weight_i = col("wr high", "wr all", "win %", "win rate", "wr")
    if weight_i is None:
        for i, h in enumerate(headers):
            hl = hnorm(h)
            if "wr" in hl or "win" in hl:
                weight_i = i
                break

    if weight_i is None:
        raise RuntimeError(f"Could not find a win column in headers: {headers}")

    # Build dictionary of weights keyed by normalised ability name.
    weights = {}
    for r in rows:
        # Defensive: skip malformed rows
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

    Output: List[dict] with keys:
      - a_raw, b_raw: the human-readable names from the table
      - a_norm, b_norm: normalised keys for matching
      - score: numeric "PAIR WR" (pair winrate, float)

    Notes:
    - The table is virtualized; we use scroll accumulation to get the full dataset.
    - We pick columns by header name where possible, with fallback indexes.
    """
    headers, rows = _parse_pairs_table_playwright_scroll(ABILITY_PAIRS_URL)

    headers_l = [h.strip().lower() for h in headers]

    def idx(name: str, fallback: int):
        """
        Find the index of a header by name. If the header isn't present (site changed),
        fall back to a known historical index.
        """
        name = name.strip().lower()
        return headers_l.index(name) if name in headers_l else fallback

    # Windrun's current layout:
    #   0: ABILITY 1
    #   2: ABILITY 2
    #   4: PAIR WR
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
    """
    Backwards-compatible alias for older parts of the project.
    """
    return load_windrun_ability_weights()
