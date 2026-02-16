from __future__ import annotations

import json
import re
from typing import List, Tuple, Set

URL = "https://windrun.io/ability-pairs"


def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def scrape_virtual_table(max_scroll_steps: int = 200) -> Tuple[List[str], List[List[str]]]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")

        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("table")

        # Wait until the table has at least one non-empty ability cell.
        # We do this by polling for any tbody td with non-empty innerText or img alt/title.
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

        # Read headers fresh
        headers = page.eval_on_selector_all(
            "table thead th",
            "ths => ths.map(th => (th.innerText || '').trim())"
        )

        def read_rows() -> List[List[str]]:
            rows = page.evaluate(
                r"""
                () => {
                  const trs = Array.from(document.querySelectorAll('table tbody tr'));
                  return trs.map(tr => {
                    const tds = Array.from(tr.querySelectorAll('td'));
                    return tds.map(td => {
                      // 1) visible text
                      let txt = (td.innerText || '').trim();
                      if (txt) return txt;

                      // 2) aria-label
                      const ariaSelf = (td.getAttribute('aria-label') || '').trim();
                      if (ariaSelf) return ariaSelf;

                      // 3) img alt/title
                      const img = td.querySelector('img');
                      if (img) {
                        const alt = (img.getAttribute('alt') || '').trim();
                        if (alt) return alt;
                        const title = (img.getAttribute('title') || '').trim();
                        if (title) return title;
                      }

                      // 4) link href
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
            # Filter out placeholder rows that are entirely empty
            cleaned = []
            for r in rows:
                if not r:
                    continue
                if all((str(x).strip() == "" for x in r)):
                    continue
                cleaned.append(r)
            return cleaned

        def href_to_name(cell: str) -> str:
            href = cell[5:].split("?")[0].split("#")[0].rstrip("/")
            href = href.split("/")[-1]
            return href.replace("-", " ").replace("_", " ").strip()

        seen: Set[Tuple[str, ...]] = set()
        out: List[List[str]] = []

        stale = 0

        # Ensure the mouse is over the table so wheel events scroll the table region
        page.hover("table")

        for _ in range(max_scroll_steps):
            rows = read_rows()

            new = 0
            for r in rows:
                r2 = []
                for cell in r:
                    if isinstance(cell, str) and cell.startswith("HREF:"):
                        r2.append(href_to_name(cell))
                    else:
                        r2.append(cell)

                # normalise row length to header length if needed
                if headers and len(r2) > len(headers):
                    r2 = r2[: len(headers)]

                key = tuple(r2)
                if key not in seen:
                    seen.add(key)
                    out.append(r2)
                    new += 1

            if new == 0:
                stale += 1
            else:
                stale = 0

            if stale >= 10:
                break

            # Scroll a bit and let React render
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(200)

        browser.close()

    return headers, out


def main():
    headers, rows = scrape_virtual_table()

    print("\n=== WINDRUN PAIRS PROBE (SCROLL2) ===")
    print("Headers:", headers)
    print("Row count:", len(rows))

    print("\nFirst 10 rows:")
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:02d}. {r}")

    # Check the pair you care about (based on the visible columns)
    want_a = norm("Ball Lightning")
    want_b = norm("Arcane Orb")

    # find col indices by header label
    h = [x.strip().lower() for x in headers]
    a_i = h.index("ability 1") if "ability 1" in h else 0
    b_i = h.index("ability 2") if "ability 2" in h else 2

    hit = None
    for r in rows:
        if len(r) <= max(a_i, b_i):
            continue
        ra = norm(r[a_i])
        rb = norm(r[b_i])
        if {ra, rb} == {want_a, want_b}:
            hit = r
            break

    print("\nBall Lightning + Arcane Orb present?:", bool(hit))
    if hit:
        print("Row:", hit)

    with open("windrun_pairs_probe_scroll2.json", "w", encoding="utf-8") as f:
        json.dump({"headers": headers, "rows": rows}, f, indent=2)

    print("\nWrote windrun_pairs_probe_scroll2.json\n")


if __name__ == "__main__":
    main()
