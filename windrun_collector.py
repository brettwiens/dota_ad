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
- typed keys:
    Internal IDs are typed, eg "hero:doom", "ability:doom_ability".
    Legacy untyped keys are also included for backward compatibility.
- collision rule (NEW):
    If multiple rows collapse to the same normalised key, keep the MAX win rate.
    This prevents Doom hero/ability style collisions from producing the wrong WR.

========================================================================================
INDEXX
  1.0 Imports
  2.0 URLs + Headers
  3.0 Normalisation + Typed Keys
  4.0 Low-level Fetchers
  5.0 Public API: Ability Weights (WR ALL)
  6.0 Public API: Hero Weights
  7.0 Static Ability Mapping
  8.0 Ability Pairs Parsing (fallback strategies)
  9.0 Public API: Ability Pairs (cached full download + optional filter)
========================================================================================
"""

from __future__ import annotations

# ======================================================================================
# TAG: 1.0 Imports
# ======================================================================================
import re
from typing import Dict, List, Optional, Any, Tuple, Iterable, Set

import requests
from bs4 import BeautifulSoup

from scrape_windrun_ability_pairs import get_windrun_ability_pairs

# ======================================================================================
# TAG: 2.0 URLs + Headers
# ======================================================================================
ABILITY_WEIGHTS_URL = "https://api.windrun.io/ability-high-skill"
HERO_WEIGHTS_URL = "https://api.windrun.io/heroes"

STATIC_ABILITIES_URL = "https://api.windrun.io/api/v2/static/abilities"

WINDRUN_PAIRS_PAGE_URLS = [
    "https://windrun.io/ability-pairs",
    "https://old.windrun.io/ability-pairs",
    "https://api.windrun.io/ability-pairs",
]

PAIR_API_CANDIDATES = [
    "https://api.windrun.io/api/v2/queries/ability-pairs",
    "https://api.windrun.io/api/v2/queries/ability_pairs",
    "https://api.windrun.io/api/v2/query/ability-pairs",
    "https://api.windrun.io/api/v2/query/ability_pairs",
    "https://api.windrun.io/api/v2/persisted/ability-pairs",
    "https://api.windrun.io/api/v2/persisted/ability_pairs",
    "https://api.windrun.io/api/v2/persisted-query/ability-pairs",
    "https://api.windrun.io/api/v2/persisted-query/ability_pairs",
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


# ======================================================================================
# TAG: 3.0 Normalisation + Typed Keys
# ======================================================================================
_ABILITY_PAREN_RE = re.compile(r"\(\s*ability\s*\)", re.IGNORECASE)


def _norm(s: str) -> str:
    s = str(s).strip().lower()

    # Fix inference artifact where possessive apostrophe becomes "_s_"
    s = re.sub(r"_s_(?=[a-z0-9])", "s_", s)
    s = re.sub(r"_s$", "s", s)

    # Preserve "(Ability)" marker before removing all parens content
    add_ability_suffix = bool(_ABILITY_PAREN_RE.search(s))

    # Remove parenthetical segments
    s = re.sub(r"\(.*?\)", "", s)

    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if add_ability_suffix and s and not s.endswith("_ability"):
        s = f"{s}_ability"

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


def norm_no_alias(name: str) -> str:
    return _norm(name)


def typed_key(kind: str, name: str, *, use_alias: bool = True) -> str:
    n = norm_with_alias(name) if use_alias else norm_no_alias(name)
    return f"{kind}:{n}"


def strip_typed_key(k: str) -> str:
    if ":" in k:
        return k.split(":", 1)[1]
    return k


# ======================================================================================
# TAG: 4.0 Low-level fetchers
# ======================================================================================
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

    try:
        return r.json()
    except Exception:
        return None


def _fetch_table_html(
    url: str, *, timeout: int = 45, headers: Optional[Dict[str, str]] = None
) -> Tuple[List[str], List[List[str]]]:
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


def _set_max(d: Dict[str, float], key: str, val: float) -> None:
    cur = d.get(key)
    if cur is None or val > cur:
        d[key] = val


# ======================================================================================
# TAG: 5.0 Public API: ability weights (WR ALL)
# ======================================================================================
def load_windrun_ability_weights(*, timeout: int = 45) -> Dict[str, float]:
    """
    Returns dict containing BOTH:
      - typed keys:  ability:<norm>
      - legacy keys: <norm>

    Collision rule:
      If multiple rows collapse to the same normalised key, keep MAX(wr).
    """
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

    def _store_ability(name_raw: Any, w: float) -> None:
        raw_norm = norm_no_alias(str(name_raw))
        alias_norm = norm_with_alias(str(name_raw))

        raw_typed = f"ability:{raw_norm}"
        alias_typed = f"ability:{alias_norm}"

        # typed keys
        _set_max(out, raw_typed, w)
        _set_max(out, alias_typed, w)

        # legacy keys
        _set_max(out, raw_norm, w)
        _set_max(out, alias_norm, w)

    # JSON first
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
                w = _to_float(w_raw)
                if not a_raw or w is None:
                    continue
                _store_ability(a_raw, w)
            if out:
                return out

    # HTML fallback
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
        a_raw = r[ability_i]
        w = _to_float(r[weight_i])
        if not a_raw or w is None:
            continue
        _store_ability(a_raw, w)

    return out


# ======================================================================================
# TAG: 6.0 Public API: hero weights
# ======================================================================================
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

        n = norm_with_alias(name)
        out[f"hero:{n}"] = wr
        out[n] = wr

    return out


def load_windrun_hero_weights(*, timeout: int = 45) -> Dict[str, float]:
    payload = _fetch_json(HERO_WEIGHTS_URL, timeout=timeout)
    recs = _as_records(payload)

    out: Dict[str, float] = {}

    def _store_hero(name_raw: Any, w: float) -> None:
        n = norm_with_alias(str(name_raw))
        out[f"hero:{n}"] = w
        out[n] = w

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
            k
            for k in sample_keys
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
                w = _to_float(w_raw)
                if not h_raw or w is None:
                    continue
                _store_hero(h_raw, w)
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
                h_raw = r[hero_i]
                w = _to_float(r[win_i])
                if not h_raw or w is None:
                    continue
                _store_hero(h_raw, w)
            if out:
                return out
    except Exception:
        pass

    r = _fetch(HERO_WEIGHTS_URL, timeout=timeout, headers=_browser_headers())
    out = _parse_hero_wr_from_ssr_html(r.text)
    if not out:
        raise RuntimeError("Could not parse hero winrates from Windrun heroes page (format changed).")
    return out


# ======================================================================================
# TAG: 7.0 Static Ability Mapping (kept)
# ======================================================================================
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


# ======================================================================================
# TAG: 8.0 Ability pairs parsing (fallback strategies)
# ======================================================================================
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
                    "a_key": f"ability:{a_norm}",
                    "b_key": f"ability:{b_norm}",
                    "score": wr,
                }
            )

    return out


def _try_pair_api_candidates(*, timeout: int = 45) -> List[dict]:
    # left as-is (not used in your current flow), kept for future fallback
    return []


# ======================================================================================
# TAG: 9.0 Public API: ability pairs (cached full download + optional filter)
# ======================================================================================
_PAIR_CACHE_FULL: Optional[List[dict]] = None


def load_windrun_ability_pairs(
    pool: Optional[Iterable[str]] = None,
    *,
    timeout: int = 60,
    headless: bool = True,
    force_reload: bool = False,
) -> List[dict]:
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
                    "a_key": f"ability:{a_norm}",
                    "b_key": f"ability:{b_norm}",
                    "score": sc,
                }
            )

        _PAIR_CACHE_FULL = out

    if pool_set is None:
        return list(_PAIR_CACHE_FULL)

    return [p for p in _PAIR_CACHE_FULL if p.get("a_norm") in pool_set and p.get("b_norm") in pool_set]


def load_windrun_ability_table(*, timeout: int = 45) -> Dict[str, float]:
    return load_windrun_ability_weights(timeout=timeout)