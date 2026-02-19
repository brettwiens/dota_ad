# windrun_collector.py
"""
windrun_collector.py

Fetch Windrun datasets for Dota 2 Ability Draft.

Design goals:
- JSON-first against api.windrun.io where possible
- HTML parsing fallbacks where JSON is not returned
- Pool-aware ability pairs (only fetch/keep pairs for the current draft pool)
- Robust to Windrun frontend changes: multiple endpoint attempts + multiple parse strategies

Fixes included:
- possessive artifact: nature_s_attendants -> natures_attendants
- alias mismatch: flesh_heap -> meat_shield
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any, Tuple, Iterable, Set

from scrape_windrun_ability_pairs import get_windrun_ability_pairs
import requests
from bs4 import BeautifulSoup

# ----------------------------
# URLs
# ----------------------------
ABILITY_WEIGHTS_URL = "https://api.windrun.io/ability-high-skill"
HERO_WEIGHTS_URL = "https://api.windrun.io/heroes"

# Static abilities map (confirmed by you)
STATIC_ABILITIES_URL = "https://api.windrun.io/api/v2/static/abilities"

# Public site pages (sometimes shells, sometimes SSR)
WINDRUN_PAIRS_PAGE_URLS = [
    "https://windrun.io/ability-pairs",
    "https://old.windrun.io/ability-pairs",
    "https://api.windrun.io/ability-pairs",
]

# Persisted query style endpoints (Windrun 3.x frontend uses these patterns, names can change)
PAIR_API_CANDIDATES = [
    # Most likely patterns
    "https://api.windrun.io/api/v2/queries/ability-pairs",
    "https://api.windrun.io/api/v2/queries/ability_pairs",
    "https://api.windrun.io/api/v2/query/ability-pairs",
    "https://api.windrun.io/api/v2/query/ability_pairs",
    "https://api.windrun.io/api/v2/persisted/ability-pairs",
    "https://api.windrun.io/api/v2/persisted/ability_pairs",
    "https://api.windrun.io/api/v2/persisted-query/ability-pairs",
    "https://api.windrun.io/api/v2/persisted-query/ability_pairs",
    # Sometimes these are under "stats" or "ability-draft"
    "https://api.windrun.io/api/v2/ability-draft/ability-pairs",
    "https://api.windrun.io/api/v2/ability-draft/ability_pairs",
    "https://api.windrun.io/api/v2/stats/ability-pairs",
    "https://api.windrun.io/api/v2/stats/ability_pairs",
]

UA = {"User-Agent": "Mozilla/5.0"}


def _browser_headers() -> Dict[str, str]:
    h = dict(UA)
    h.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-CA,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return h


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
def _fetch(url: str, *, timeout: int = 45, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    h = headers or UA
    r = requests.get(url, timeout=timeout, headers=h)
    r.raise_for_status()
    return r


def _fetch_json(url: str, *, timeout: int = 45, headers: Optional[Dict[str, str]] = None) -> Any:
    r = _fetch(url, timeout=timeout, headers=headers or UA)

    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" in ctype or "json" in ctype:
        return r.json()

    # Sometimes servers lie about content-type. Try JSON anyway.
    try:
        return r.json()
    except Exception:
        return None


def _fetch_table_html(url: str, *, timeout: int = 45, headers: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[List[str]]]:
    r = _fetch(url, timeout=timeout, headers=headers or UA)

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"No <table> found at {url}")

    thead = table.find("thead")
    headers_out = [th.get_text(strip=True) for th in (thead.find_all("th") if thead else table.find_all("th"))]

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

    return headers_out, rows


def _as_records(payload: Any) -> List[dict]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("data", "rows", "results", "items", "abilities"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        for outer_key in ("data", "result", "payload"):
            outer = payload.get(outer_key)
            if isinstance(outer, dict):
                for key in ("rows", "results", "items", "abilities"):
                    v = outer.get(key)
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, dict)]

        if any(isinstance(v, (str, int, float)) for v in payload.values()):
            return [payload]

    return []


# --------------------------------------------------------------------------------------
# Public API: ability weights (WR ALL)
# --------------------------------------------------------------------------------------
def load_windrun_ability_weights(*, timeout: int = 45) -> Dict[str, float]:
    payload = _fetch_json(ABILITY_WEIGHTS_URL, timeout=timeout)
    recs = _as_records(payload)

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
        sample_keys = set().union(*(r.keys() for r in recs[:10]))
        name_key = next((k for k in candidates_name if k in sample_keys), None)
        wr_key = next((k for k in candidates_wr if k in sample_keys), None)

        if name_key is None:
            for k in sample_keys:
                if "ability" in k.lower() or k.lower() == "name":
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
            if out:
                return out

    headers, rows = _fetch_table_html(ABILITY_WEIGHTS_URL, timeout=timeout)
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
def _parse_hero_wr_from_ssr_html(html: str) -> Dict[str, float]:
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, float] = {}

    for a in soup.find_all("a"):
        name = a.get_text(strip=True)
        if not name or len(name) < 2:
            continue

        parent_text = (a.parent.get_text(" ", strip=True) if a.parent else "")
        if "%" not in parent_text:
            continue

        m = re.search(r"(\d{1,2}\.\d{1,2})\s*%", parent_text)
        if not m:
            continue

        wr = _to_float(m.group(1))
        if wr is None:
            continue

        key = norm_with_alias(name)
        out[key] = wr

    return out


def load_windrun_hero_weights(*, timeout: int = 45) -> Dict[str, float]:
    payload = _fetch_json(HERO_WEIGHTS_URL, timeout=timeout)
    recs = _as_records(payload)

    out: Dict[str, float] = {}

    if recs:
        sample_keys = set().union(*(r.keys() for r in recs[:10]))

        hero_key = next((k for k in ("hero", "name", "heroName") if k in sample_keys), None)
        if hero_key is None:
            for k in sample_keys:
                lk = k.lower()
                if "hero" in lk or lk == "name":
                    hero_key = k
                    break

        patch_keys = [
            k for k in sample_keys
            if re.fullmatch(r"\d+\.\d+[a-z]?", str(k).strip().lower().replace("·", ".").replace("\u00b7", "."))
        ]
        win_key = patch_keys[0] if patch_keys else None

        if win_key is None:
            for k in sample_keys:
                lk = k.lower()
                if ("win" in lk and ("rate" in lk or "pct" in lk or "%" in lk)) or ("wr" in lk and "all" in lk):
                    win_key = k
                    break

        if hero_key and win_key:
            for r in recs:
                h_raw = r.get(hero_key)
                w_raw = r.get(win_key)
                hkey = norm_with_alias(h_raw)
                w = _to_float(w_raw)
                if hkey and w is not None:
                    out[hkey] = w
            if out:
                return out

    try:
        headers, rows = _fetch_table_html(HERO_WEIGHTS_URL, timeout=timeout)
        headers_l = [h.strip().lower() for h in headers]
        hero_i = headers_l.index("hero") if "hero" in headers_l else 0

        win_i = None
        for i, h in enumerate(headers_l):
            if i == hero_i:
                continue
            hh = h.replace("·", ".").replace("\u00b7", ".").strip()
            if re.fullmatch(r"\d+\.\d+[a-z]?", hh):
                win_i = i
                break

        if win_i is None:
            for i, h in enumerate(headers_l):
                if i == hero_i:
                    continue
                if "win" in h or ("wr" in h and "all" in h):
                    win_i = i
                    break

        if win_i is not None:
            out = {}
            for r in rows:
                if len(r) <= max(hero_i, win_i):
                    continue
                hkey = norm_with_alias(r[hero_i])
                w = _to_float(r[win_i])
                if hkey and w is not None:
                    out[hkey] = w
            if out:
                return out
    except Exception:
        pass

    r = _fetch(HERO_WEIGHTS_URL, timeout=timeout, headers=_browser_headers())
    out = _parse_hero_wr_from_ssr_html(r.text)
    if not out:
        raise RuntimeError("Could not parse hero winrates from Windrun heroes page (format changed).")
    return out


# --------------------------------------------------------------------------------------
# Ability static mapping
# --------------------------------------------------------------------------------------
def _build_static_ability_maps(*, timeout: int = 45) -> Tuple[Dict[str, int], Dict[int, str]]:
    payload = _fetch_json(STATIC_ABILITIES_URL, timeout=timeout)
    if not isinstance(payload, dict) or "data" not in payload or not isinstance(payload.get("data"), list):
        raise RuntimeError("Static abilities endpoint returned unexpected JSON shape (expected dict with key 'data' list).")

    norm_to_valve_id: Dict[str, int] = {}
    valve_id_to_name: Dict[int, str] = {}

    for r in payload["data"]:
        if not isinstance(r, dict):
            continue
        vid = r.get("valveId")
        nm = r.get("englishName") or r.get("shortName")
        if vid is None or nm is None:
            continue
        try:
            vid_int = int(vid)
        except Exception:
            continue
        nm_str = str(nm).strip()
        if not nm_str:
            continue

        n = norm_with_alias(nm_str)
        if not n:
            continue

        norm_to_valve_id[n] = vid_int
        valve_id_to_name[vid_int] = nm_str

    if not norm_to_valve_id:
        raise RuntimeError("Static abilities endpoint did not produce any usable (englishName, valveId) records.")

    return norm_to_valve_id, valve_id_to_name


# --------------------------------------------------------------------------------------
# Ability pairs parsing: multiple strategies (kept for future fallback)
# --------------------------------------------------------------------------------------
_PAIR_LINE_RE = re.compile(
    r"^\s*(?P<a>[^,]{2,80})\s*,\s*(?P<wr>\d{1,2}(?:\.\d{1,2})?)\s*%\s*\.\s*(?P<b>.{2,80})\s*$"
)


def _parse_pairs_from_pairs_page_html(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ; ", strip=True)

    if len(text) < 2000:
        return []

    low = text.lower()
    start = low.find("ability pairs")
    if start == -1:
        return []

    after = text[start:]
    chunks = [c.strip() for c in after.split(" ; ") if c.strip()]

    out: List[dict] = []
    for c in chunks:
        m = _PAIR_LINE_RE.match(c)
        if not m:
            continue

        a_raw = m.group("a").strip()
        b_raw = m.group("b").strip()
        wr = _to_float(m.group("wr"))

        a_norm = norm_with_alias(a_raw)
        b_norm = norm_with_alias(b_raw)

        if a_norm and b_norm:
            out.append(
                {
                    "a_raw": a_raw,
                    "b_raw": b_raw,
                    "a_norm": a_norm,
                    "b_norm": b_norm,
                    "score": wr,
                }
            )

    return out


def _try_pair_api_candidates(*, timeout: int = 45) -> List[dict]:
    def _detect_fields(records: List[dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not records:
            return None, None, None

        sample_keys = set().union(*(r.keys() for r in records[:25]))

        a_key = None
        b_key = None

        for k in ("ability1", "ability_1", "ability 1", "a", "first", "left"):
            if k in sample_keys:
                a_key = k
                break
        for k in ("ability2", "ability_2", "ability 2", "b", "second", "right"):
            if k in sample_keys:
                b_key = k
                break

        if a_key is None or b_key is None:
            for k in sample_keys:
                lk = k.lower()
                if a_key is None and "ability" in lk and ("1" in lk or "one" in lk):
                    a_key = k
                if b_key is None and "ability" in lk and ("2" in lk or "two" in lk):
                    b_key = k

        if a_key is None or b_key is None:
            candidates = []
            for k in sample_keys:
                lk = k.lower()
                if "ability" in lk or lk in ("a", "b"):
                    candidates.append(k)
            candidates = sorted(set(candidates))
            if a_key is None and candidates:
                a_key = candidates[0]
            if b_key is None and len(candidates) > 1:
                b_key = candidates[1]

        win_key = None
        for k in sample_keys:
            lk = k.lower()
            if "pair" in lk and ("wr" in lk or "win" in lk):
                win_key = k
                break
        if win_key is None:
            for k in sample_keys:
                lk = k.lower()
                if "wr" in lk or ("win" in lk and ("rate" in lk or "pct" in lk or "%" in lk)):
                    win_key = k
                    break

        return a_key, b_key, win_key

    for url in PAIR_API_CANDIDATES:
        try:
            payload = _fetch_json(url, timeout=timeout, headers=dict(UA))
            recs = _as_records(payload)
            if not recs:
                continue

            a_key, b_key, win_key = _detect_fields(recs)
            if not (a_key and b_key and win_key):
                continue

            out: List[dict] = []
            for r in recs:
                a_raw = r.get(a_key)
                b_raw = r.get(b_key)
                sc_raw = r.get(win_key)

                if not a_raw or not b_raw:
                    continue

                a_norm = norm_with_alias(a_raw)
                b_norm = norm_with_alias(b_raw)
                sc = _to_float(sc_raw)

                if a_norm and b_norm:
                    out.append(
                        {
                            "a_raw": str(a_raw),
                            "b_raw": str(b_raw),
                            "a_norm": a_norm,
                            "b_norm": b_norm,
                            "score": sc,
                        }
                    )

            if out:
                return out
        except Exception:
            continue

    return []


# --------------------------------------------------------------------------------------
# Public API: ability pairs (CACHED full download + optional in-memory filter)
# --------------------------------------------------------------------------------------
_PAIR_CACHE_FULL: Optional[List[dict]] = None


def load_windrun_ability_pairs(
    pool: Optional[Iterable[str]] = None,
    *,
    timeout: int = 60,
    headless: bool = True,
    force_reload: bool = False,
) -> List[dict]:
    """
    One-time download of the full pairs table (via Playwright), cached in memory.
    Later calls optionally filter to a pool in Python without re-hitting Windrun.

    Returns list[dict] with keys:
      a_raw, b_raw, a_norm, b_norm, score
    """
    global _PAIR_CACHE_FULL

    pool_set: Optional[Set[str]] = None
    if pool is not None:
        pool_set = {norm_with_alias(x) for x in pool if x}

    if force_reload or _PAIR_CACHE_FULL is None:
        raw_pairs = get_windrun_ability_pairs(headless=headless)

        out: List[dict] = []
        for r in raw_pairs:
            a_raw = str(r.get("ability_1") or "").strip()
            b_raw = str(r.get("ability_2") or "").strip()
            score = r.get("win_rate")

            if not a_raw or not b_raw:
                continue

            a_norm = norm_with_alias(a_raw)
            b_norm = norm_with_alias(b_raw)
            sc = _to_float(score)

            out.append(
                {
                    "a_raw": a_raw,
                    "b_raw": b_raw,
                    "a_norm": a_norm,
                    "b_norm": b_norm,
                    "score": sc,
                }
            )

        _PAIR_CACHE_FULL = out

    if pool_set is None:
        return list(_PAIR_CACHE_FULL)

    return [p for p in _PAIR_CACHE_FULL if p.get("a_norm") in pool_set and p.get("b_norm") in pool_set]


# Convenience alias (legacy name in your codebase)
def load_windrun_ability_table(*, timeout: int = 45) -> Dict[str, float]:
    return load_windrun_ability_weights(timeout=timeout)
