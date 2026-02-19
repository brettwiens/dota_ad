# scrape_windrun_ability_pairs.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

URL = "https://api.windrun.io/ability-pairs"
TABLE_XPATH = "/html/body/div/div/div[2]/div[2]/div[2]/table"


def _find_scrollable_ancestor_js() -> str:
    return """
    (table) => {
      function isScrollable(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const overflowY = style.overflowY;
        const canScroll = (overflowY === 'auto' || overflowY === 'scroll') && (el.scrollHeight > el.clientHeight);
        return canScroll;
      }

      let el = table;
      while (el && el !== document.body) {
        if (isScrollable(el)) return el;
        el = el.parentElement;
      }
      return document.scrollingElement || document.documentElement;
    }
    """


def _scroll_until_stable(page, table_locator, max_rounds: int = 200, stable_rounds: int = 6) -> None:
    table_locator.wait_for(state="visible", timeout=30000)

    table_handle = table_locator.element_handle()
    if table_handle is None:
        raise RuntimeError("Could not get element handle for table.")

    scroll_container = page.evaluate_handle(_find_scrollable_ancestor_js(), table_handle)

    prev_count = -1
    stable = 0

    for _ in range(max_rounds):
        row_count = table_locator.locator("xpath=.//tbody/tr").count()

        if row_count == prev_count:
            stable += 1
        else:
            stable = 0
            prev_count = row_count

        if stable >= stable_rounds:
            break

        page.evaluate(
            """
            (sc) => {
              const before = sc.scrollTop;
              sc.scrollTop = sc.scrollTop + Math.max(600, sc.clientHeight * 0.9);
              if (sc.scrollTop === before) sc.scrollTop = sc.scrollHeight;
            }
            """,
            scroll_container,
        )
        page.wait_for_timeout(250)

    page.wait_for_timeout(400)


def _normalise_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _parse_win_rate_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = _normalise_text(s)
    if not t:
        return None
    t = t.replace("%", "")
    try:
        return float(t)
    except ValueError:
        return None


def get_windrun_ability_pairs(headless: bool = True) -> List[Dict[str, Any]]:
    """
    Scrapes api.windrun.io ability-pairs table and returns:

    [
        {
            "ability_1": str,
            "ability_2": str,
            "win_rate": float
        },
        ...
    ]
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        try:
            page.goto(URL, wait_until="networkidle", timeout=60000)
        except PWTimeoutError:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)

        table = page.locator(f"xpath={TABLE_XPATH}")
        table.wait_for(state="visible", timeout=30000)

        _scroll_until_stable(page, table)

        tr_loc = table.locator("xpath=.//tbody/tr")
        n = tr_loc.count()

        results: List[Dict[str, Any]] = []

        for i in range(n):
            tds = tr_loc.nth(i).locator("xpath=.//td").all_text_contents()
            tds = [_normalise_text(x) for x in tds]

            if len(tds) < 8:
                continue

            ability_1 = tds[1]
            ability_2 = tds[4]
            win_rate_raw = tds[7]

            win_rate = _parse_win_rate_to_float(win_rate_raw)

            if ability_1 and ability_2 and win_rate is not None:
                results.append(
                    {
                        "ability_1": ability_1,
                        "ability_2": ability_2,
                        "win_rate": win_rate,
                    }
                )

        browser.close()

    return results
