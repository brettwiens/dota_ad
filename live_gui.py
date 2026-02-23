# live_gui.py
from __future__ import annotations

"""
Dota AD Live Draft Assistant (Tkinter)

========================================================================================
INDEXX
  1.0 Imports + Paths
  2.0 Infer Loader
  3.0 Normalisation + Typed Keys
    3.1 base_norm
    3.2 typed_key helpers
    3.3 doom collision helpers (doom vs doom_ability)
  4.0 Draft Data Builders
    4.1 latest_image_in_folder
    4.2 build_picks (typed keys)
    4.3 ranking (weights)
  5.0 Selection State (Option A)
    5.1 state model
    5.2 apply locks and availability filters
  6.0 Ability Pairs
    6.1 resolve pair norms to local typed keys
    6.2 outstanding_pairs_resolved
    6.3 best_pair_map
  7.0 Icons
  8.0 GUI
    8.1 layout
    8.2 context menus (mark me, mark other, clear, lock)
    8.3 rendering (ranked, pairs, selected lists)
    8.4 ChatGPT explain popup (click pair)
========================================================================================
"""

# ======================================================================================
# TAG: 1.0 Imports + Paths
# ======================================================================================
import os
import re
import traceback
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import hashlib

from windrun_collector import (
    norm_with_alias as normalize_name,
    load_windrun_ability_weights,
    load_windrun_hero_weights,
    load_windrun_ability_pairs,
)

BASE_DIR = Path(r"Z:\DotaAD\dota_ad")
INFER_PATH = BASE_DIR / "infer_draft_v2.py"

SCREENSHOT_DIR = Path(
    r"C:\Program Files (x86)\Steam\userdata\59046080\760\remote\570\screenshots"
)
IMAGE_EXTS = {".jpg", ".jpeg"}

HERO_ICON_DIR = BASE_DIR / "icons" / "heroes"
ABILITY_ICON_DIR = BASE_DIR / "icons" / "abilities"

TOP_PAIRS = 60
ICON_SIZE = 36

# -------------------------
# ChatGPT cost controls
# -------------------------
CHATGPT_MODEL = "gpt-4o-mini"  # small + cheap :contentReference[oaicite:1]{index=1}
# Alternative ultra-cheap option: "gpt-4.1-nano" :contentReference[oaicite:2]{index=2}

CHATGPT_MAX_OUTPUT_TOKENS = 260  # hard cap for cost control :contentReference[oaicite:3]{index=3}

CACHE_DIR = BASE_DIR / "cache"
PAIR_EXPLAIN_CACHE_PATH = CACHE_DIR / "pair_explain_cache.json"
PAIR_EXPLAIN_CACHE_TTL_DAYS = 90  # set None/0 to never expire
PAIR_EXPLAIN_PROMPT_VERSION = "v1"  # bump to invalidate old cache entries

try:
    from PIL import Image, ImageTk

    PIL_OK = True
except Exception:
    PIL_OK = False

# Optional dependency for ChatGPT explanations:
#   pip install openai
# and set env var:
#   OPENAI_API_KEY=...
try:
    from openai import OpenAI  # type: ignore

    OPENAI_OK = True
except Exception:
    OPENAI_OK = False
    OpenAI = None  # type: ignore


# ======================================================================================
# TAG: 2.0 Infer Loader
# ======================================================================================
def load_infer_one(infer_path: Path):
    if not infer_path.exists():
        raise FileNotFoundError(f"infer file not found: {infer_path}")

    spec = importlib.util.spec_from_file_location("infer_draft_v2_local", str(infer_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for: {infer_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.infer_one


infer_one = load_infer_one(INFER_PATH)


# ======================================================================================
# TAG: 3.0 Normalisation + Typed Keys
# ======================================================================================

# TAG: 3.1 base_norm
def base_norm(name: str) -> str:
    """
    Base normalisation for display and internal keys.
    Uses windrun_collector.norm_with_alias (already handles possessive artifacts, aliases, etc).
    """
    return normalize_name(name or "")


# TAG: 3.2 typed_key helpers
def typed_key(kind: str, normed: str) -> str:
    return f"{kind}:{normed}"


def split_typed_key(k: str) -> Tuple[str, str]:
    if ":" in k:
        a, b = k.split(":", 1)
        return a, b
    return "unknown", k


def display_from_key(k: str) -> str:
    """
    Display name fallback when we only have a typed key.
    """
    _, n = split_typed_key(k)
    return n.replace("_", " ").title()


# TAG: 3.3 doom collision helpers (doom vs doom_ability)
def _variants_for_pair_norm(n: str) -> List[str]:
    """
    Equivalent norm variants so doom and doom_ability can match.
    """
    n = (n or "").strip()
    if not n:
        return []
    out = [n]
    if n.endswith("_ability"):
        out.append(n[: -len("_ability")])
    else:
        out.append(n + "_ability")
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ======================================================================================
# TAG: 4.0 Draft Data Builders
# ======================================================================================
def latest_image_in_folder(folder: Path) -> Path:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No screenshots found in: {folder}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


# TAG: 4.2 build_picks (typed keys)
def build_picks(infer_out: dict) -> List[dict]:
    """
    Produces picks with stable typed keys.

    pick fields:
      - type: hero | ultimate | ability
      - name: raw display name from model
      - base_norm: normalised (alias-aware)
      - key: typed key using base_norm (hero:doom, ultimate:doom, ability:arcane_orb, etc)
      - weight_norm: normal used for ability weight lookup (may need doom_ability handling)
    """
    picks: List[dict] = []

    def _ptype(kind: str) -> str:
        if kind == "heroes":
            return "hero"
        if kind == "ultimates":
            return "ultimate"
        return "ability"

    for kind in ("heroes", "ultimates", "abilities"):
        for item in infer_out.get(kind, []) or []:
            name = item.get("name")
            if not name or name == "unknown":
                continue

            ptype = _ptype(kind)
            bn = base_norm(name)
            k = typed_key(ptype, bn)

            wn = bn

            picks.append(
                {
                    "type": ptype,
                    "cell": item.get("cell"),
                    "name": name,
                    "base_norm": bn,
                    "key": k,
                    "weight_norm": wn,
                    "conf": item.get("conf"),
                }
            )

    return picks


# TAG: 4.3 ranking (weights)
def lookup_weight_for_pick(
    p: dict,
    ability_weights: Dict[str, float],
    hero_weights: Dict[str, float],
) -> Optional[float]:
    """
    Weights dicts may include typed keys:
      - ability:<norm>
      - hero:<norm>
    but we also fall back to legacy untyped keys.
    """
    typ = p["type"]
    bn = p["base_norm"]

    if typ == "hero":
        return hero_weights.get(f"hero:{bn}") if f"hero:{bn}" in hero_weights else hero_weights.get(bn)

    cand_norms: List[str] = []
    cand_norms.extend(_variants_for_pair_norm(p.get("weight_norm") or bn))
    cand_norms.extend(_variants_for_pair_norm(bn))

    seen = set()
    for n in cand_norms:
        if n in seen:
            continue
        seen.add(n)

        v = ability_weights.get(f"ability:{n}")
        if v is not None:
            return v

    for n in list(seen):
        v = ability_weights.get(n)
        if v is not None:
            return v

    return None


def rank_by_weight(
    picks: List[dict],
    ability_weights: Dict[str, float],
    hero_weights: Dict[str, float],
) -> List[dict]:
    out: List[dict] = []
    for p in picks:
        p2 = dict(p)
        p2["weight"] = lookup_weight_for_pick(p2, ability_weights, hero_weights)
        out.append(p2)

    def sort_key(d: dict):
        w = d.get("weight")
        c = d.get("conf")
        has_w = w is not None
        return (
            0 if has_w else 1,
            -(w if w is not None else 0),
            -(c if c is not None else 0),
        )

    out.sort(key=sort_key)
    return out


# ======================================================================================
# TAG: 5.0 Selection State (Option A)
# ======================================================================================

# status_by_key: "avail" | "me" | "other"
# locked_by_type: None or a specific key for hero/ultimate/ability lock
Status = str


def _default_status() -> Status:
    return "avail"


# TAG: 5.2 apply locks and availability filters
def compute_allowed_sets(
    all_picks: List[dict],
    status_by_key: Dict[str, Status],
    locked_by_type: Dict[str, Optional[str]],
) -> Dict[str, set]:
    """
    Applies:
      - others removed from availability
      - if locked hero, only that hero is allowed
      - if locked ultimate, only that ultimate is allowed
      - if 3 abilities selected by me, only those 3 abilities are allowed (analysis focus shifts)
    """
    all_keys = {p["key"] for p in all_picks}

    available = {k for k in all_keys if status_by_key.get(k, "avail") != "other"}
    me_selected = {k for k in available if status_by_key.get(k, "avail") == "me"}

    heroes = {k for k in available if k.startswith("hero:")}
    ults = {k for k in available if k.startswith("ultimate:")}
    abils = {k for k in available if k.startswith("ability:")}

    me_heroes = {k for k in me_selected if k.startswith("hero:")}
    me_ults = {k for k in me_selected if k.startswith("ultimate:")}
    me_abils = {k for k in me_selected if k.startswith("ability:")}

    if locked_by_type.get("hero") in heroes:
        heroes = {locked_by_type["hero"]}
    elif me_heroes:
        heroes = set(me_heroes)

    if locked_by_type.get("ultimate") in ults:
        ults = {locked_by_type["ultimate"]}
    elif me_ults:
        ults = set(me_ults)

    if len(me_abils) >= 3:
        abils = set(list(me_abils)[:3])
    elif locked_by_type.get("ability") in abils:
        abils = {locked_by_type["ability"]}

    return {
        "available": available,
        "me": me_selected,
        "heroes": heroes,
        "ultimates": ults,
        "abilities": abils,
    }


# ======================================================================================
# TAG: 6.0 Ability Pairs
# ======================================================================================

# TAG: 6.1 resolve pair norms to local typed keys
def resolve_pair_norm_to_local_key(
    pair_norm: str,
    hero_by_norm: Dict[str, str],
    ult_by_norm: Dict[str, str],
    abil_by_norm: Dict[str, str],
) -> Optional[str]:
    """
    Resolves a pair-side norm to the correct local typed key.
    Priority: ultimate, then ability, then hero.
    Uses doom variants to avoid doom hero shadowing doom ultimate.
    """
    for v in _variants_for_pair_norm(pair_norm):
        k = ult_by_norm.get(v)
        if k:
            return k
    for v in _variants_for_pair_norm(pair_norm):
        k = abil_by_norm.get(v)
        if k:
            return k
    for v in _variants_for_pair_norm(pair_norm):
        k = hero_by_norm.get(v)
        if k:
            return k
    return None


# TAG: 6.2 outstanding_pairs_resolved
def outstanding_pairs_resolved(
    allowed_keys: set,
    pairs_source: Optional[List[dict]],
    hero_by_norm: Dict[str, str],
    ult_by_norm: Dict[str, str],
    abil_by_norm: Dict[str, str],
) -> List[dict]:
    """
    Filters Windrun pairs to those fully present in allowed_keys, using robust resolution:
      - prefers ultimate over ability over hero when mapping a_norm/b_norm
      - doom vs doom_ability both map to the same pick if needed
    """
    if not pairs_source:
        return []

    out: List[dict] = []
    for p in pairs_source:
        a_norm = p.get("a_norm")
        b_norm = p.get("b_norm")
        if not a_norm or not b_norm:
            continue

        a_key = resolve_pair_norm_to_local_key(a_norm, hero_by_norm, ult_by_norm, abil_by_norm)
        b_key = resolve_pair_norm_to_local_key(b_norm, hero_by_norm, ult_by_norm, abil_by_norm)
        if not a_key or not b_key:
            continue

        if a_key in allowed_keys and b_key in allowed_keys:
            p2 = dict(p)
            p2["a_key"] = a_key
            p2["b_key"] = b_key
            out.append(p2)

    out.sort(key=lambda d: (d.get("score") is not None, d.get("score") or -1), reverse=True)
    return out


# TAG: 6.3 best_pair_map
def best_pair_map(pairs: List[dict]) -> dict:
    """
    best[k] = (partner_display, score)
    k is local typed key.
    """
    best: Dict[str, Tuple[str, Optional[float]]] = {}

    def update(k: str, partner: str, score: Optional[float]):
        cur = best.get(k)
        cur_score = cur[1] if cur else None
        if cur is None or (score or -1) > (cur_score or -1):
            best[k] = (partner, score)

    for p in pairs:
        a_key = p.get("a_key")
        b_key = p.get("b_key")
        if not a_key or not b_key:
            continue
        update(a_key, p.get("b_raw") or display_from_key(b_key), p.get("score"))
        update(b_key, p.get("a_raw") or display_from_key(a_key), p.get("score"))

    return best


# ======================================================================================
# TAG: 7.0 Icons
# ======================================================================================
class IconCache:
    def __init__(self):
        self.cache: Dict[str, object] = {}

    def _candidates(self, name: str) -> List[str]:
        base = (name or "").lower().strip()
        variants = {
            base,
            base.replace(" ", "_"),
            base.replace("_", " "),
            base.replace("-", "_"),
            normalize_name(base),
        }

        clean = set()
        for v in variants:
            v = re.sub(r"[\s_]+", "_", v).strip("_")
            clean.add(v)

        return [f"{x}.png" for x in clean]

    def get_icon(self, kind: str, name: str):
        key = f"{kind}:{name}"
        if key in self.cache:
            return self.cache[key]

        folder = HERO_ICON_DIR if kind == "hero" else ABILITY_ICON_DIR

        for fname in self._candidates(name):
            p = folder / fname
            if not p.exists():
                continue

            try:
                if PIL_OK:
                    img = Image.open(p).convert("RGBA")
                    img = img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img)
                else:
                    tk_img = tk.PhotoImage(file=str(p))

                self.cache[key] = tk_img
                return tk_img
            except Exception:
                pass

        return None


# ======================================================================================
# TAG: 8.4 ChatGPT explain popup (click pair)
# ======================================================================================
@dataclass(frozen=True)
class PairExplainRequest:
    a_name: str
    b_name: str
    a_type: str
    b_type: str
    score: Optional[float] = None

_cache_lock = threading.Lock()


def _cache_load() -> dict:
    try:
        if not PAIR_EXPLAIN_CACHE_PATH.exists():
            return {}
        with PAIR_EXPLAIN_CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _cache_save(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PAIR_EXPLAIN_CACHE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(PAIR_EXPLAIN_CACHE_PATH)


def _cache_key_for_pair(a_name: str, b_name: str, model: str) -> str:
    # Order-independent, so A+B == B+A
    a = base_norm(a_name)
    b = base_norm(b_name)
    left, right = sorted([a, b])
    raw = f"{PAIR_EXPLAIN_PROMPT_VERSION}|{model}|{left}|{right}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get_explanation(cache_key: str) -> Optional[str]:
    with _cache_lock:
        cache = _cache_load()
        entry = cache.get(cache_key)
        if not entry:
            return None

        # TTL
        ttl_days = PAIR_EXPLAIN_CACHE_TTL_DAYS
        if ttl_days and ttl_days > 0:
            ts = entry.get("ts")
            if ts is None:
                return None
            age_days = (time.time() - float(ts)) / 86400.0
            if age_days > float(ttl_days):
                # expire it
                cache.pop(cache_key, None)
                _cache_save(cache)
                return None

        text = entry.get("text")
        return text if isinstance(text, str) and text.strip() else None


def _cache_put_explanation(cache_key: str, text: str, meta: Optional[dict] = None) -> None:
    with _cache_lock:
        cache = _cache_load()
        cache[cache_key] = {
            "ts": time.time(),
            "text": text,
            "meta": meta or {},
        }
        _cache_save(cache)

class PairExplainPopup(tk.Toplevel):
    def __init__(self, master: tk.Misc, title: str):
        super().__init__(master)
        self.title(title)
        self.geometry("760x520")
        self.transient(master)
        self.grab_set()

        self.status_var = tk.StringVar(value="Thinking...")

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.status_var).pack(side="left")

        self.txt = tk.Text(self, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._set_text("Working on it...")
        self._set_readonly(True)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(fill="x")

        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")

    def _set_readonly(self, readonly: bool):
        self.txt.config(state=("disabled" if readonly else "normal"))

    def _set_text(self, text: str):
        self._set_readonly(False)
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self._set_readonly(True)

    def set_text_done(self, text: str):
        self._set_text(text)
        self.status_var.set("Done")

    def set_text_error(self, text: str):
        self._set_text(text)
        self.status_var.set("Error")


def explain_pair_via_openai(req: PairExplainRequest) -> str:
    model = CHATGPT_MODEL
    cache_key = _cache_key_for_pair(req.a_name, req.b_name, model)

    cached = _cache_get_explanation(cache_key)
    if cached:
        return cached + "\n\n(cached)"

    if not OPENAI_OK:
        return (
            "OpenAI Python SDK is not installed.\n\n"
            "Run:\n"
            "  pip install openai\n"
        )

    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set.\n\n"
            "Set it as an environment variable, for example in PowerShell:\n"
            "  setx OPENAI_API_KEY \"your_key_here\"\n"
            "Then restart your terminal/app.\n"
        )

    client = OpenAI()

    # Short prompt to keep token usage down
    prompt = (
        "Dota 2 Ability Draft analyst. Answer concisely.\n\n"
        f"Pair:\n"
        f"- A: {req.a_name} (type: {req.a_type})\n"
        f"- B: {req.b_name} (type: {req.b_type})\n"
        f"Score: {req.score}\n\n"
        "Explain what makes this pair powerful in AD.\n"
        "Format:\n"
        "1) Core interaction (2-3 bullets)\n"
        "2) How it wins fights (2-3 bullets)\n"
        "3) Execution tips (2-3 bullets)\n"
        "4) Counters (2-3 bullets)\n"
    ).strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=CHATGPT_MAX_OUTPUT_TOKENS,
    )

    text = (resp.output_text or "").strip()
    if not text:
        text = "(No response text returned.)"

    _cache_put_explanation(
        cache_key,
        text,
        meta={
            "model": model,
            "a": req.a_name,
            "b": req.b_name,
            "score": req.score,
        },
    )
    return text


# ======================================================================================
# TAG: 8.0 GUI
# ======================================================================================
class LiveDraftGUI(tk.Tk):
    # ----------------------------------------------------------------------------------
    # TAG: 8.1 Layout
    # ----------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        self.title("Dota AD Live Draft Assistant")
        self.geometry("1400x820")

        self.icon_cache = IconCache()
        self._images: List[object] = []

        self.status_by_key: Dict[str, Status] = {}
        self.locked_by_type: Dict[str, Optional[str]] = {"hero": None, "ultimate": None, "ability": None}

        self._latest_picks: List[dict] = []
        self._latest_pick_by_key: Dict[str, dict] = {}

        # Pair-click support: tag -> pair payload
        self._pair_payload_by_tag: Dict[str, dict] = {}

        self._setup_styles()

        topbar = ttk.Frame(self, padding=6)
        topbar.pack(fill="x")

        self.status = tk.StringVar(value="Starting up...")
        ttk.Label(topbar, textvariable=self.status).pack(side="left")

        ttk.Button(topbar, text="Refresh (R)", command=self.refresh).pack(side="right")
        ttk.Button(topbar, text="Quit (Q)", command=self.destroy).pack(side="right", padx=6)

        main = ttk.PanedWindow(self, orient="vertical")
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main, padding=6)
        bottom = ttk.Frame(main, padding=6)

        main.add(top, weight=4)
        main.add(bottom, weight=1)

        top_grid = ttk.Frame(top)
        top_grid.pack(fill="both", expand=True)

        for c in range(4):
            top_grid.columnconfigure(c, weight=1, uniform="top")
        top_grid.rowconfigure(1, weight=1)

        ttk.Label(top_grid, text="Heroes", style="HeroHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top_grid, text="Ultimates", style="UltimateHeader.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(top_grid, text="Abilities", style="AbilityHeader.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(top_grid, text="Outstanding Pairs", style="PairsHeader.TLabel").grid(row=0, column=3, sticky="w", padx=(8, 0))

        cols = ("rank", "weight", "pair")

        def make_tree(parent) -> Tuple[ttk.Frame, ttk.Treeview]:
            container = ttk.Frame(parent)

            t = ttk.Treeview(
                container,
                columns=cols,
                show=("tree", "headings"),
                height=22,
                style="Striped.Treeview",
            )

            xbar = ttk.Scrollbar(container, orient="horizontal", command=t.xview)
            ybar = ttk.Scrollbar(container, orient="vertical", command=t.yview)
            t.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

            t.heading("#0", text="Pick")
            t.heading("rank", text="#")
            t.heading("weight", text="WR")
            t.heading("pair", text="Best Pair")

            t.column("#0", width=140)
            t.column("rank", width=36, anchor="center")
            t.column("weight", width=70, anchor="e")
            t.column("pair", width=260)

            t.tag_configure("odd", background="#f4f4f4")
            t.tag_configure("even", background="#ffffff")

            t.tag_configure("me_bg", background="#dff0d8")       # soft green
            t.tag_configure("other_bg", background="#f2dede")    # soft red
            t.tag_configure("locked_fg", foreground="#6a0dad")   # purple

            container.rowconfigure(0, weight=1)
            container.columnconfigure(0, weight=1)

            t.grid(row=0, column=0, sticky="nsew")
            ybar.grid(row=0, column=1, sticky="ns")
            xbar.grid(row=1, column=0, columnspan=2, sticky="ew")

            return container, t

        hero_frame, self.hero_tree = make_tree(top_grid)
        ult_frame, self.ult_tree = make_tree(top_grid)
        abil_frame, self.abil_tree = make_tree(top_grid)

        hero_frame.grid(row=1, column=0, sticky="nsew")
        ult_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        abil_frame.grid(row=1, column=2, sticky="nsew", padx=(8, 0))

        pairs_frame = ttk.Frame(top_grid)
        pairs_frame.grid(row=1, column=3, sticky="nsew", padx=(8, 0))
        pairs_frame.rowconfigure(0, weight=1)
        pairs_frame.columnconfigure(0, weight=1)

        self.pairs_box = tk.Text(pairs_frame, font=("Consolas", 10), wrap="none")
        pairs_y = ttk.Scrollbar(pairs_frame, orient="vertical", command=self.pairs_box.yview)
        pairs_x = ttk.Scrollbar(pairs_frame, orient="horizontal", command=self.pairs_box.xview)
        self.pairs_box.configure(yscrollcommand=pairs_y.set, xscrollcommand=pairs_x.set)

        self.pairs_box.grid(row=0, column=0, sticky="nsew")
        pairs_y.grid(row=0, column=1, sticky="ns")
        pairs_x.grid(row=1, column=0, columnspan=2, sticky="ew")

        bottom_grid = ttk.Frame(bottom)
        bottom_grid.pack(fill="both", expand=True)
        bottom_grid.columnconfigure(0, weight=1, uniform="bottom")
        bottom_grid.columnconfigure(1, weight=1, uniform="bottom")
        bottom_grid.rowconfigure(1, weight=1)

        ttk.Label(bottom_grid, text="Selected by Me", style="SelectedMeHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(bottom_grid, text="Selected by Others", style="SelectedOtherHeader.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.sel_me_tree = self._make_selected_tree(bottom_grid)
        self.sel_other_tree = self._make_selected_tree(bottom_grid)

        self.sel_me_tree.grid(row=1, column=0, sticky="nsew")
        self.sel_other_tree.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        self.bind("<r>", lambda e: self.refresh())
        self.bind("<R>", lambda e: self.refresh())
        self.bind("<q>", lambda e: self.destroy())
        self.bind("<Q>", lambda e: self.destroy())

        self._build_context_menus()
        self._bind_tree_events(self.hero_tree)
        self._bind_tree_events(self.ult_tree)
        self._bind_tree_events(self.abil_tree)
        self._bind_selected_tree_events(self.sel_me_tree)
        self._bind_selected_tree_events(self.sel_other_tree)

        self.ability_weights: Dict[str, float] = {}
        self.hero_weights: Dict[str, float] = {}
        self.pairs_cache: List[dict] = []
        self.pairs_loaded = False
        self.pairs_loading = False

        try:
            self.status.set("Loading Windrun ability win rates...")
            self.update_idletasks()
            self.ability_weights = load_windrun_ability_weights()

            self.status.set("Loading Windrun hero win rates...")
            self.update_idletasks()
            self.hero_weights = load_windrun_hero_weights()

            self.status.set(f"Windrun loaded: abilities={len(self.ability_weights)} heroes={len(self.hero_weights)}")
        except Exception as e:
            self.status.set("Windrun load failed")
            self._render_error(e)

        self.after(150, self.refresh)

    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Striped.Treeview", rowheight=32)

        style.configure("HeroHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#8B0000")
        style.configure("UltimateHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#003366")
        style.configure("AbilityHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#0B6623")
        style.configure("PairsHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#333333")

        style.configure("SelectedMeHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground="#2e6b2e")
        style.configure("SelectedOtherHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground="#7a2a2a")

        style.map(
            "Striped.Treeview",
            background=[("selected", "#cfe8ff")],
            foreground=[("selected", "#000000")],
        )

    def _make_selected_tree(self, parent: ttk.Frame) -> ttk.Treeview:
        cols = ("key", "type", "name")
        t = ttk.Treeview(parent, columns=cols, show="headings", height=6)

        t.heading("key", text="Key")
        t.heading("type", text="Type")
        t.heading("name", text="Name")

        # Hide key column but keep the value for event handling
        t.column("key", width=0, stretch=False)
        t.column("type", width=90, anchor="w")
        t.column("name", width=320, anchor="w")

        return t

    # ----------------------------------------------------------------------------------
    # TAG: 8.2 Context menus (mark me, mark other, clear, lock)
    # ----------------------------------------------------------------------------------
    def _build_context_menus(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Mark Selected by Me", command=lambda: self._menu_apply("me"))
        self.menu.add_command(label="Mark Selected by Others", command=lambda: self._menu_apply("other"))
        self.menu.add_command(label="Clear (Available)", command=lambda: self._menu_apply("avail"))
        self.menu.add_separator()
        self.menu.add_command(label="Lock This Type to This Pick", command=self._menu_lock)
        self.menu.add_command(label="Clear Lock for This Type", command=self._menu_unlock)

        self._menu_target_tree: Optional[ttk.Treeview] = None
        self._menu_target_item: Optional[str] = None
        self._menu_target_key: Optional[str] = None

    def _bind_tree_events(self, tree: ttk.Treeview):
        tree.bind("<Button-3>", lambda e, t=tree: self._on_tree_right_click(e, t))
        tree.bind("<Double-1>", lambda e, t=tree: self._on_tree_double_click(e, t))

    def _bind_selected_tree_events(self, tree: ttk.Treeview):
        tree.bind("<Double-1>", lambda e, t=tree: self._on_selected_double_click(e, t))
        tree.bind("<Button-3>", lambda e, t=tree: self._on_selected_right_click(e, t))

    def _on_tree_right_click(self, event, tree: ttk.Treeview):
        item = tree.identify_row(event.y)

        if item:
            tree.selection_set(item)
            key = item  # iid is typed key
            self._menu_target_tree = tree
            self._menu_target_item = item
            self._menu_target_key = key
        else:
            self._menu_target_tree = tree
            self._menu_target_item = None
            self._menu_target_key = None

        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _on_tree_double_click(self, event, tree: ttk.Treeview):
        item = tree.identify_row(event.y)
        if not item:
            return

        key = item  # iid is typed key
        if not key:
            nm = tree.item(item).get("text") or ""
            bn = base_norm(nm)
            key = typed_key(
                "hero" if tree is self.hero_tree else "ultimate" if tree is self.ult_tree else "ability",
                bn,
            )

        cur = self.status_by_key.get(key, "avail")
        self.status_by_key[key] = "me" if cur != "me" else "avail"
        self.refresh()

    def _on_selected_double_click(self, event, tree: ttk.Treeview):
        item = tree.identify_row(event.y)
        if not item:
            return
        key = tree.item(item).get("values", ["", "", ""])[0]  # hidden key col
        if not key:
            return
        self.status_by_key[key] = "avail"
        self.refresh()

    def _on_selected_right_click(self, event, tree: ttk.Treeview):
        item = tree.identify_row(event.y)
        if not item:
            return
        tree.selection_set(item)
        key = tree.item(item).get("values", ["", "", ""])[0]
        if not key:
            return

        self._menu_target_tree = None
        self._menu_target_item = None
        self._menu_target_key = key
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _menu_apply(self, status: Status):
        k = self._menu_target_key
        if not k:
            return
        self.status_by_key[k] = status
        self.refresh()

    def _menu_lock(self):
        k = self._menu_target_key
        if not k:
            return
        typ, _ = split_typed_key(k)
        if typ in ("hero", "ultimate", "ability"):
            self.locked_by_type[typ] = k
        self.refresh()

    def _menu_unlock(self):
        k = self._menu_target_key
        if not k:
            return
        typ, _ = split_typed_key(k)
        if typ in self.locked_by_type:
            self.locked_by_type[typ] = None
        self.refresh()

    # ----------------------------------------------------------------------------------
    # TAG: 8.3 Rendering and refresh loop
    # ----------------------------------------------------------------------------------
    def _start_pairs_load(self, infer_out: dict):
        if self.pairs_loading or self.pairs_loaded:
            return

        if not (infer_out.get("heroes") or infer_out.get("ultimates") or infer_out.get("abilities")):
            return

        self.pairs_loading = True
        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", "Loading full ability pairs table (one-time)...\n")

        def worker():
            try:
                pairs = load_windrun_ability_pairs(timeout=60, headless=True)
                self.after(0, lambda pairs=pairs: self._pairs_loaded_ok(pairs))
            except Exception as e:
                self.after(0, lambda e=e: self._pairs_loaded_fail(e))

        threading.Thread(target=worker, daemon=True).start()

    def _pairs_loaded_ok(self, pairs: List[dict]):
        self.pairs_cache = pairs
        self.pairs_loaded = True
        self.pairs_loading = False
        self.refresh()

    def _pairs_loaded_fail(self, e: Exception):
        self.pairs_loaded = False
        self.pairs_loading = False
        self.pairs_cache = []
        print("[pairs load failed]", repr(e))
        traceback.print_exc()

        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", "Failed to load pairs:\n\n")
        self.pairs_box.insert("end", str(e) + "\n\n" + traceback.format_exc())

    def refresh(self):
        try:
            img_path = latest_image_in_folder(SCREENSHOT_DIR)
            infer_out = infer_one(str(img_path), verbose=False)

            if (not self.pairs_loading) and (not self.pairs_loaded):
                self._start_pairs_load(infer_out)

            picks = build_picks(infer_out)
            self._latest_picks = picks
            self._latest_pick_by_key = {p["key"]: p for p in picks}

            for p in picks:
                self.status_by_key.setdefault(p["key"], "avail")

            allowed = compute_allowed_sets(picks, self.status_by_key, self.locked_by_type)
            ranked = rank_by_weight(picks, self.ability_weights, self.hero_weights)

            hero_by_norm: Dict[str, str] = {}
            ult_by_norm: Dict[str, str] = {}
            abil_by_norm: Dict[str, str] = {}

            for p in picks:
                bn = p["base_norm"]
                k = p["key"]
                if p["type"] == "hero":
                    hero_by_norm[bn] = k
                elif p["type"] == "ultimate":
                    ult_by_norm[bn] = k
                    for v in _variants_for_pair_norm(bn):
                        ult_by_norm[v] = k
                else:
                    abil_by_norm[bn] = k
                    for v in _variants_for_pair_norm(bn):
                        abil_by_norm[v] = k

            allowed_pairs_keys = set(allowed["heroes"]) | set(allowed["ultimates"]) | set(allowed["abilities"])

            pairs = outstanding_pairs_resolved(
                allowed_pairs_keys,
                self.pairs_cache if self.pairs_loaded else None,
                hero_by_norm,
                ult_by_norm,
                abil_by_norm,
            )

            bmap = best_pair_map(pairs)

            self._render_ranked(ranked, bmap, allowed)
            self._render_pairs(pairs, allowed)
            self._render_selected_lists()

            extra = ""
            if self.pairs_loading:
                extra = " (pairs loading...)"
            elif not self.pairs_loaded:
                extra = " (pairs not loaded)"

            self.status.set(f"Loaded: {img_path.name}{extra}")

        except Exception as e:
            self.status.set("Error")
            self._render_error(e)

    def _render_ranked(self, ranked: List[dict], bmap: dict, allowed: dict):
        for t in (self.hero_tree, self.ult_tree, self.abil_tree):
            for r in t.get_children():
                t.delete(r)

        self._images = []

        heroes, ults, abils = [], [], []
        for i, row in enumerate(ranked, 1):
            r2 = dict(row)
            r2["rank"] = i
            if r2["type"] == "hero":
                heroes.append(r2)
            elif r2["type"] == "ultimate":
                ults.append(r2)
            else:
                abils.append(r2)

        def insert_rows(tree: ttk.Treeview, rows: List[dict], allowed_set: set):
            for idx, row in enumerate(rows, 1):
                k = row["key"]
                if k not in allowed_set:
                    continue

                name = row["name"]
                typ = row["type"]

                icon_kind = "hero" if typ == "hero" else "ability"
                icon = self.icon_cache.get_icon(icon_kind, name)
                if icon:
                    self._images.append(icon)

                w = row.get("weight")
                w_txt = "n/a" if w is None else f"{w:.2f}"

                partner = bmap.get(k)
                pair_txt = ""
                if partner:
                    p, s = partner
                    pair_txt = f"{p} ({'n/a' if s is None else f'{s:.2f}'})"

                row_tag = "odd" if (idx % 2 == 1) else "even"

                st = self.status_by_key.get(k, "avail")
                status_tag = ""
                if st == "me":
                    status_tag = "me_bg"
                elif st == "other":
                    status_tag = "other_bg"

                lock_tag = ""
                if self.locked_by_type.get(typ) == k:
                    lock_tag = "locked_fg"

                tags = tuple(t for t in (row_tag, status_tag, lock_tag) if t)

                tree.insert(
                    parent="",
                    index="end",
                    iid=k,
                    text=name,
                    image=icon if icon else "",
                    values=(f"{row['rank']:02d}", w_txt, pair_txt),
                    tags=tags,
                )

        insert_rows(self.hero_tree, heroes, allowed["heroes"])
        insert_rows(self.ult_tree, ults, allowed["ultimates"])
        insert_rows(self.abil_tree, abils, allowed["abilities"])

    def _on_pair_click(self, tag: str):
        payload = self._pair_payload_by_tag.get(tag)
        if not payload:
            return

        req = PairExplainRequest(
            a_name=payload["a_name"],
            b_name=payload["b_name"],
            a_type=payload["a_type"],
            b_type=payload["b_type"],
            score=payload.get("score"),
        )

        title = f"Why is this pair strong?"
        popup = PairExplainPopup(self, title=title)
        popup.set_text_done("Thinking...")

        def worker():
            try:
                text = explain_pair_via_openai(req)
                self.after(0, lambda: popup.set_text_done(text))
            except Exception as e:
                err = f"{e}\n\n{traceback.format_exc()}"
                self.after(0, lambda: popup.set_text_error(err))
                self.after(0, lambda: messagebox.showerror("ChatGPT error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _render_pairs(self, pairs: List[dict], allowed: dict):
        self.pairs_box.delete("1.0", "end")

        # Reset clickable mapping each render
        self._pair_payload_by_tag = {}

        self.pairs_box.tag_configure("hero", foreground="#8B0000")
        self.pairs_box.tag_configure("ultimate", foreground="#003366")
        self.pairs_box.tag_configure("ability", foreground="#0B6623")
        self.pairs_box.tag_configure("picked_bold", font=("Consolas", 10, "bold"))

        # Clickable line styling
        self.pairs_box.tag_configure("pair_clickable", underline=1)

        if self.pairs_loading:
            self.pairs_box.insert("end", "Loading ability pairs in background...\n")
            return

        if not self.pairs_loaded:
            self.pairs_box.insert("end", "(pairs not loaded)\n")
            self.pairs_box.insert(
                "end",
                "\nTip: Click-to-explain needs OpenAI SDK. Install with: pip install openai\n",
            )
            return

        if not pairs:
            self.pairs_box.insert("end", "(none)\n")
            return

        me_set = {k for k, st in self.status_by_key.items() if st == "me"}

        key_to_type: Dict[str, str] = {}
        for k in allowed["heroes"]:
            key_to_type[k] = "hero"
        for k in allowed["ultimates"]:
            key_to_type[k] = "ultimate"
        for k in allowed["abilities"]:
            key_to_type[k] = "ability"

        # Instruction line
        self.pairs_box.insert("end", "Tip: click a pair line to ask ChatGPT why it is strong.\n\n")

        for i, p in enumerate(pairs[:TOP_PAIRS], 1):
            sc_val = p.get("score")
            sc = "n/a" if sc_val is None else f"{sc_val:.2f}"

            a_key = p.get("a_key")
            b_key = p.get("b_key")
            if not a_key or not b_key:
                continue

            a_type = key_to_type.get(a_key, "ability")
            b_type = key_to_type.get(b_key, "ability")

            bold = (a_key in me_set) or (b_key in me_set)

            # Capture start index for tagging the full line as clickable
            line_start = self.pairs_box.index("end-1c")

            self.pairs_box.insert("end", f"{i:02d}. ")

            a_name = p.get("a_raw", display_from_key(a_key))
            b_name = p.get("b_raw", display_from_key(b_key))

            if bold:
                self.pairs_box.insert("end", a_name, ("picked_bold", a_type))
            else:
                self.pairs_box.insert("end", a_name, (a_type,))

            self.pairs_box.insert("end", " + ")

            if bold:
                self.pairs_box.insert("end", b_name, ("picked_bold", b_type))
            else:
                self.pairs_box.insert("end", b_name, (b_type,))

            self.pairs_box.insert("end", f"   {sc}\n")

            line_end = self.pairs_box.index("end-1c")

            # Create a unique tag per line, bind click
            tag = f"pair_{i:02d}"
            self.pairs_box.tag_add(tag, line_start, line_end)
            self.pairs_box.tag_add("pair_clickable", line_start, line_end)

            # Store payload for ChatGPT
            self._pair_payload_by_tag[tag] = {
                "a_name": a_name,
                "b_name": b_name,
                "a_type": a_type,
                "b_type": b_type,
                "score": sc_val,
            }

            # Bind click handler for this tag
            self.pairs_box.tag_bind(tag, "<Button-1>", lambda e, t=tag: self._on_pair_click(t))
            self.pairs_box.tag_bind(tag, "<Double-1>", lambda e, t=tag: self._on_pair_click(t))

    def _render_selected_lists(self):
        for t in (self.sel_me_tree, self.sel_other_tree):
            for r in t.get_children():
                t.delete(r)

        for k, st in sorted(self.status_by_key.items()):
            if st not in ("me", "other"):
                continue
            typ, _ = split_typed_key(k)
            disp = self._latest_pick_by_key.get(k, {}).get("name") or display_from_key(k)

            if st == "me":
                self.sel_me_tree.insert("", "end", values=(k, typ, disp))
            else:
                self.sel_other_tree.insert("", "end", values=(k, typ, disp))

    def _render_error(self, e: Exception):
        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", str(e) + "\n\n")
        self.pairs_box.insert("end", traceback.format_exc())


if __name__ == "__main__":
    app = LiveDraftGUI()
    app.mainloop()