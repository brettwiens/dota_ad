from playwright.sync_api import sync_playwright
import pandas as pd
from tabulate import tabulate

URL = "https://windrun.io/ability-pairs"

def scrape_and_display():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle")

        # Wait for table to render
        page.wait_for_selector("table")

        table = page.query_selector("table")

        # Extract headers
        headers = table.eval_on_selector_all(
            "thead th",
            "ths => ths.map(th => th.innerText.trim())"
        )

        # Extract rows
        rows = table.eval_on_selector_all(
            "tbody tr",
            """trs => trs.map(tr => 
                Array.from(tr.querySelectorAll('td')).map(td => td.innerText.trim())
            )"""
        )

        browser.close()

    df = pd.DataFrame(rows, columns=headers if headers else None)

    print("\n=== Windrun Ability Pairs ===\n")
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

    return df


if __name__ == "__main__":
    df = scrape_and_display()
