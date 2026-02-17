"""
windrun_collector.py

Fetch Windrun datasets for Dota 2 Ability Draft.

Uses JSON-first requests against api.windrun.io endpoints.
Falls back to HTML table parsing only if JSON is not returned.

Fixes included:
- possessive artifact: nature_s_attendants -> natures_attendants
- alias mismatch: flesh_heap -> meat_shield
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any, Tuple

import requests
from bs4 import BeautifulSoup

ABILITY_WEIGHTS_URL = "https://api.windrun.io/ability-high-skill"
HERO_WEIGHTS_URL = "https://api.windrun.io/heroes"
ABILITY_PAIRS_URL = "https://api.windrun.io/ability-pairs"

UA = {"User-Agent": "Mozilla/5.0"}


# --------------------------------------------------------------------------------------
# Normalisation + parsing helpers
# --------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = str(s).strip().lower()

    # Fix inference artifact where possessive apostrophe becomes "_s_"
    s = re.sub(r"_s_(?=[a-z0-9])", "s_", s)
    s = re.sub(r"_s$", "s", s)

    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("%", "").replace(",", "").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


ALIASES: Dict[str, str] = {
    "flesh_heap": "meat_shield",
}


def norm_with_alias(name: str) -> str:
    k = _norm(name)
    return ALIASES.get(k, k)


# --------------------------------------------------------------------------------------
# Low-level fetchers
# --------------------------------------------------------------------------------------
def _fetch_json(url: str) -> Any:
    r = requests.get(url, timeout=45, headers=UA)
    r.raise_for_status()

    ctype = (r.headers.get("content-type") or "").lower()
    # Windrun APIs should be JSON
    if "application/json" in ctype or "json" in ctype:
        return r.json()

    # Sometimes servers lie about content-type. Try JSON anyway.
    try:
        return r.json()
    except Exception:
        return None


def _fetch_table_html(url: str) -> Tuple[List[str], List[List[str]]]:
    r = requests.get(url, timeout=45, headers=UA)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"No <table> found at {url}")

    thead = table.find("thead")
    headers = [th.get_text(strip=True) for th in (thead.find_all("th") if thead else table.find_all("th"))]

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    rows: List[List[str]] = []
    for tr in body_rows:
        tds = tr.find_all("td")
        if not tds:
            continue
        vals = [td.get_text(strip=True) for td in tds]
        if any(v.strip() for v in vals):
            rows.append(vals)

    return headers, rows


def _as_records(payload: Any) -> List[dict]:
    """
    Convert various JSON shapes to a list[dict].
    Handles:
      - list of dicts
      - dict with 'data' key
      - dict with 'rows' key
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("data", "rows", "results"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # sometimes the dict itself is a single record
        return [payload]

    return []


# --------------------------------------------------------------------------------------
# Public API: ability weights (WR ALL)
# --------------------------------------------------------------------------------------
def load_windrun_ability_weights() -> Dict[str, float]:
    payload = _fetch_json(ABILITY_WEIGHTS_URL)
    recs = _as_records(payload)

    # Windrun may use different field names over time.
    # We prefer WR ALL / Overall Win % / overallWin / winrateAll etc.
    candidates_name = ("ability", "name", "abilityName")
    candidates_wr = (
        "wrAll",
        "wr_all",
        "overallWin",
        "overallWinPct",
        "overall_win",
        "overall_win_pct",
        "overallWinPercent",
        "overallWinPercentage",
        "winrateAll",
        "winRateAll",
        "overall win %",
        "wr all",
    )

    out: Dict[str, float] = {}

    if recs:
        # Find actual keys present
        sample_keys = set().union(*(r.keys() for r in recs[:10]))
        name_key = next((k for k in candidates_name if k in sample_keys), None)
        wr_key = next((k for k in candidates_wr if k in sample_keys), None)

        # If we did not find exact matches, fall back to fuzzy contains
        if name_key is None:
            for k in sample_keys:
                if "ability" in k.lower() or "name" == k.lower():
                    name_key = k
                    break

        if wr_key is None:
            for k in sample_keys:
                lk = k.lower()
                if "wr" in lk and "all" in lk:
                    wr_key = k
                    break
            if wr_key is None:
                for k in sample_keys:
                    lk = k.lower()
                    if "overall" in lk and "win" in lk:
                        wr_key = k
                        break

        if name_key and wr_key:
            for r in recs:
                a_raw = r.get(name_key)
                w_raw = r.get(wr_key)
                a = norm_with_alias(a_raw)
                w = _to_float(w_raw)
                if a and w is not None:
                    out[a] = w
            return out

    # Fallback to HTML table parsing if JSON did not work
    headers, rows = _fetch_table_html(ABILITY_WEIGHTS_URL)
    headers_l = [h.strip().lower() for h in headers]

    ability_i = headers_l.index("ability") if "ability" in headers_l else 0

    weight_i = None
    for i, h in enumerate(headers_l):
        if "wr" in h and "all" in h:
            weight_i = i
            break
    if weight_i is None:
        for i, h in enumerate(headers_l):
            if "overall" in h and "win" in h:
                weight_i = i
                break
    if weight_i is None:
        raise RuntimeError(f"Could not locate WR ALL / Overall Win % in headers: {headers}")

    for r in rows:
        if len(r) <= max(ability_i, weight_i):
            continue
        a = norm_with_alias(r[ability_i])
        w = _to_float(r[weight_i])
        if a and w is not None:
            out[a] = w

    return out


# --------------------------------------------------------------------------------------
# Public API: hero weights
# --------------------------------------------------------------------------------------
def load_windrun_hero_weights() -> Dict[str, float]:
    payload = _fetch_json(HERO_WEIGHTS_URL)
    recs = _as_records(payload)

    out: Dict[str, float] = {}

    # For heroes, Windrun often has patch columns. We will choose the first numeric-ish
    # winrate field we find that looks like a percent.
    if recs:
        sample_keys = set().union(*(r.keys() for r in recs[:10]))

        hero_key = None
        for k in ("hero", "name", "heroName"):
            if k in sample_keys:
                hero_key = k
                break
        if hero_key is None:
            for k in sample_keys:
                if "hero" in k.lower() or k.lower() == "name":
                    hero_key = k
                    break

        # pick a likely winrate column
        win_key = None
        for k in sample_keys:
            lk = k.lower()
            if "win" in lk and ("rate" in lk or "pct" in lk or "%" in lk):
                win_key = k
                break

        # if none, pick first key where most values look percentish
        if win_key is None:
            def pctish(v) -> bool:
                return _to_float(v) is not None

            for k in sample_keys:
                if k == hero_key:
                    continue
                vals = [r.get(k) for r in recs[:20]]
                ok = sum(1 for v in vals if pctish(v))
                if ok >= 10:
                    win_key = k
                    break

        if hero_key and win_key:
            for r in recs:
                h_raw = r.get(hero_key)
                w_raw = r.get(win_key)
                hkey = _norm(h_raw)
                w = _to_float(w_raw)
                if hkey and w is not None:
                    out[hkey] = w
            return out

    # Fallback HTML
    headers, rows = _fetch_table_html(HERO_WEIGHTS_URL)
    headers_l = [h.strip().lower() for h in headers]
    hero_i = headers_l.index("hero") if "hero" in headers_l else 0

    # Choose first column that looks like patch (e.g. 7.38) or win%
    win_i = None
    for i, h in enumerate(headers_l):
        if i == hero_i:
            continue
        if re.fullmatch(r"\d+\.\d+[a-z]?", h):
            win_i = i
            break
    if win_i is None:
        for i, h in enumerate(headers_l):
            if i == hero_i:
                continue
            if "win" in h:
                win_i = i
                break
    if win_i is None:
        raise RuntimeError(f"Could not detect hero winrate column in headers: {headers}")

    for r in rows:
        if len(r) <= max(hero_i, win_i):
            continue
        hkey = _norm(r[hero_i])
        w = _to_float(r[win_i])
        if hkey and w is not None:
            out[hkey] = w

    return out


# --------------------------------------------------------------------------------------
# Public API: ability pairs
# --------------------------------------------------------------------------------------
def load_windrun_ability_pairs() -> List[dict]:
    payload = _fetch_json(ABILITY_PAIRS_URL)
    recs = _as_records(payload)

    # Expected fields can vary. We will detect:
    # - two ability name fields
    # - one winrate field
    out: List[dict] = []

    if recs:
        sample_keys = set().union(*(r.keys() for r in recs[:10]))

        # find ability keys
        ability_keys = []
        for k in sample_keys:
            lk = k.lower()
            if "ability" in lk and ("1" in lk or "one" in lk):
                ability_keys.append(k)
            elif "ability" in lk and ("2" in lk or "two" in lk):
                ability_keys.append(k)

        # common direct names
        for k in ("ability1", "ability_1", "ability 1", "a", "abilityOne"):
            if k in sample_keys:
                ability_keys.insert(0, k)
        for k in ("ability2", "ability_2", "ability 2", "b", "abilityTwo"):
            if k in sample_keys:
                ability_keys.append(k)

        # If still not clean, pick any two stringy columns containing "ability"
        if len(ability_keys) < 2:
            for k in sample_keys:
                if "ability" in k.lower():
                    ability_keys.append(k)
        ability_keys = list(dict.fromkeys(ability_keys))  # dedupe preserving order

        a_key = ability_keys[0] if len(ability_keys) > 0 else None
        b_key = ability_keys[1] if len(ability_keys) > 1 else None

        # find winrate key
        win_key = None
        for k in sample_keys:
            lk = k.lower()
            if "pair" in lk and ("wr" in lk or "win" in lk):
                win_key = k
                break
        if win_key is None:
            for k in sample_keys:
                lk = k.lower()
                if "wr" in lk or ("win" in lk and "rate" in lk):
                    win_key = k
                    break

        if a_key and b_key and win_key:
            for r in recs:
                a_raw = r.get(a_key)
                b_raw = r.get(b_key)
                sc_raw = r.get(win_key)

                if not a_raw or not b_raw:
                    continue

                a = norm_with_alias(a_raw)
                b = norm_with_alias(b_raw)
                sc = _to_float(sc_raw)

                if a and b:
                    out.append(
                        {
                            "a_raw": str(a_raw),
                            "b_raw": str(b_raw),
                            "a_norm": a,
                            "b_norm": b,
                            "score": sc,
                        }
                    )

            return out

    # Fallback HTML
    headers, rows = _fetch_table_html(ABILITY_PAIRS_URL)
    headers_l = [h.strip().lower() for h in headers]

    def idx(name: str, fallback: int) -> int:
        n = name.strip().lower()
        return headers_l.index(n) if n in headers_l else fallback

    a_i = idx("ability 1", 0)
    b_i = idx("ability 2", 1 if a_i == 0 else 0)

    score_i = None
    for i, h in enumerate(headers_l):
        if "pair" in h and ("wr" in h or "win" in h):
            score_i = i
            break
    if score_i is None:
        score_i = 2

    for r in rows:
        if len(r) <= max(a_i, b_i, score_i):
            continue
        a_raw = r[a_i]
        b_raw = r[b_i]
        sc_raw = r[score_i]

        a = norm_with_alias(a_raw)
        b = norm_with_alias(b_raw)
        sc = _to_float(sc_raw)

        if a and b:
            out.append(
                {
                    "a_raw": a_raw,
                    "b_raw": b_raw,
                    "a_norm": a,
                    "b_norm": b,
                    "score": sc,
                }
            )

    return out


def load_windrun_ability_table():
    return load_windrun_ability_weights()
