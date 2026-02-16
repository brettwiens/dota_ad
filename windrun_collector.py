import re
import requests
from bs4 import BeautifulSoup

ABILITY_WEIGHTS_URL = "https://api.windrun.io/ability-high-skill"
ABILITY_PAIRS_URL = "https://api.windrun.io/ability-pairs"



def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_first_table(url: str):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"No table found at {url}")

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    tbody = table.find("tbody")
    if tbody is None:
        raise RuntimeError(f"No tbody found at {url}")

    rows = []
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if tds:
            rows.append(tds)

    return headers, rows


def load_windrun_ability_weights() -> dict:
    """
    Returns: { ability_norm: weight_float }
    Pulls from https://api.windrun.io/ability-high-skill
    """
    headers, rows = _parse_first_table(ABILITY_WEIGHTS_URL)

    def find_col(candidates):
        for want in candidates:
            for i, h in enumerate(headers):
                if h.strip().lower() == want.lower():
                    return i
        return None

    ability_i = find_col(["Ability", "Name"])
    if ability_i is None:
        ability_i = 0

    weight_i = find_col(["HS Win %", "HS Win%", "Win %", "Win%", "Win Rate", "Winrate"])
    if weight_i is None:
        for i, h in enumerate(headers):
            hl = h.lower()
            if "win" in hl:
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


def load_windrun_ability_pairs():
    """
    Returns a list of dicts:
      { a_norm, b_norm, score, a_raw, b_raw }
    Pulls from https://windrun.io/ability-pairs
    """
    headers, rows = _parse_first_table(ABILITY_PAIRS_URL)

    def find_col(candidates):
        for want in candidates:
            for i, h in enumerate(headers):
                if h.strip().lower() == want.lower():
                    return i
        return None

    a_i = find_col(["Ability 1", "Ability1", "A", "First"])
    b_i = find_col(["Ability 2", "Ability2", "B", "Second"])

    if a_i is None:
        a_i = 0
    if b_i is None:
        b_i = 1

    score_i = find_col(["Win %", "Win%", "HS Win %", "HS Win%", "Score", "Synergy"])
    if score_i is None:
        for i, h in enumerate(headers):
            hl = h.lower()
            if "win" in hl or "score" in hl or "syn" in hl:
                score_i = i
                break
    if score_i is None:
        score_i = len(headers) - 1

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

        out.append({
            "a_raw": a_raw,
            "b_raw": b_raw,
            "a_norm": a,
            "b_norm": b,
            "score": score
        })

    return out


# Backwards-compatible alias (so older code does not break)
def load_windrun_ability_table():
    """
    Legacy name. Returns the same as load_windrun_ability_weights().
    """
    return load_windrun_ability_weights()
