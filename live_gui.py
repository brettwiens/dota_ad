from __future__ import annotations

import re
import traceback
import threading
from pathlib import Path
from typing import List, Optional, Dict

import importlib.util
import inspect
import tkinter as tk
from tkinter import ttk

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

TOP_PAIRS = 25
ICON_SIZE = 28

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


def load_infer_one(infer_path: Path):
    if not infer_path.exists():
        raise FileNotFoundError(f"infer file not found: {infer_path}")

    spec = importlib.util.spec_from_file_location("infer_draft_v2_local", str(infer_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for: {infer_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    print("USING INFER FILE:", inspect.getsourcefile(mod))
    print("infer_one signature:", inspect.signature(mod.infer_one))
    return mod.infer_one


infer_one = load_infer_one(INFER_PATH)


def latest_image_in_folder(folder: Path) -> Path:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No screenshots found in: {folder}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def build_picks(infer_out: dict) -> List[dict]:
    picks = []
    for kind in ("heroes", "ultimates", "abilities"):
        for item in infer_out.get(kind, []):
            name = item.get("name")
            if not name:
                continue

            ptype = "hero" if kind == "heroes" else "ultimate" if kind == "ultimates" else "ability"

            picks.append(
                {
                    "type": ptype,
                    "cell": item.get("cell"),
                    "name": name,
                    "name_norm": normalize_name(name),
                    "conf": item.get("conf"),
                }
            )
    return picks


def rank_by_weight(
    picks: List[dict],
    ability_weights: Dict[str, float],
    hero_weights: Dict[str, float],
) -> List[dict]:
    out = []
    for p in picks:
        p2 = dict(p)
        if p2["type"] == "hero":
            p2["weight"] = hero_weights.get(p2["name_norm"])
        else:
            p2["weight"] = ability_weights.get(p2["name_norm"])
        out.append(p2)

    def sort_key(d: dict):
        w = d.get("weight")
        c = d.get("conf")
        has_w = (w is not None)
        return (
            0 if has_w else 1,
            -(w if w is not None else 0),
            -(c if c is not None else 0),
        )

    out.sort(key=sort_key)
    return out


def available_ability_norm_set(infer_out: dict) -> set:
    vals = infer_out.get("ultimate_vec", []) + infer_out.get("ability_vec", [])
    vals = [x for x in vals if x and x != "unknown"]
    return {normalize_name(x) for x in vals}


def outstanding_pairs(infer_out: dict, pairs_source: Optional[List[dict]]) -> List[dict]:
    if not pairs_source:
        return []

    norm_set = available_ability_norm_set(infer_out)

    out = []
    for p in pairs_source:
        if p.get("a_norm") in norm_set and p.get("b_norm") in norm_set:
            out.append(p)

    out.sort(key=lambda d: (d.get("score") is not None, d.get("score") or -1), reverse=True)
    return out


def best_pair_map(pairs: List[dict]) -> dict:
    best = {}

    def update(k, partner, score):
        cur = best.get(k)
        if cur is None or (score or -1) > (cur[1] or -1):
            best[k] = (partner, score)

    for p in pairs:
        update(p["a_norm"], p["b_raw"], p.get("score"))
        update(p["b_norm"], p["a_raw"], p.get("score"))

    return best


class IconCache:
    def __init__(self):
        self.cache = {}

    def _candidates(self, name: str) -> List[str]:
        base = name.lower().strip()
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


class LiveDraftGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Dota AD Live Draft Assistant")
        self.geometry("1220x740")

        self.icon_cache = IconCache()
        self._images = []

        self._setup_styles()

        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")

        self.status = tk.StringVar(value="Starting up...")
        ttk.Label(top, textvariable=self.status).pack(side="left")

        ttk.Button(top, text="Refresh (R)", command=self.refresh).pack(side="right")
        ttk.Button(top, text="Quit (Q)", command=self.destroy).pack(side="right", padx=6)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=6)

        main.add(left, weight=3)
        main.add(right, weight=2)

        ttk.Label(left, text="Ranked Picks").pack(anchor="w")

        cols = ("rank", "weight", "conf", "cell", "pair")

        grid = ttk.Frame(left)
        grid.pack(fill="both", expand=True)

        grid.columnconfigure(0, weight=1, uniform="x")
        grid.columnconfigure(1, weight=1, uniform="x")
        grid.columnconfigure(2, weight=1, uniform="x")
        grid.rowconfigure(1, weight=1)

        ttk.Label(grid, text="Heroes").grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Ultimates").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(grid, text="Abilities").grid(row=0, column=2, sticky="w", padx=(8, 0))

        def make_tree(parent):
            t = ttk.Treeview(
                parent,
                columns=cols,
                show=("tree", "headings"),
                height=25,
                style="Striped.Treeview",
            )
            t.heading("#0", text="Pick")
            t.heading("rank", text="#")
            t.heading("weight", text="WR")
            t.heading("conf", text="Conf")
            t.heading("cell", text="Cell")
            t.heading("pair", text="Best Pair")

            t.column("#0", width=220)
            t.column("rank", width=36, anchor="center")
            t.column("weight", width=70, anchor="e")
            t.column("conf", width=60, anchor="e")
            t.column("cell", width=60)
            t.column("pair", width=180)

            t.tag_configure("odd", background="#f4f4f4")
            t.tag_configure("even", background="#ffffff")
            return t

        self.hero_tree = make_tree(grid)
        self.ult_tree = make_tree(grid)
        self.abil_tree = make_tree(grid)

        self.hero_tree.grid(row=1, column=0, sticky="nsew")
        self.ult_tree.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.abil_tree.grid(row=1, column=2, sticky="nsew", padx=(8, 0))

        ttk.Label(right, text="Outstanding Ability Pairs").pack(anchor="w")
        self.pairs_box = tk.Text(right, font=("Consolas", 10))
        self.pairs_box.pack(fill="both", expand=True)

        self.bind("<r>", lambda e: self.refresh())
        self.bind("<R>", lambda e: self.refresh())
        self.bind("<q>", lambda e: self.destroy())
        self.bind("<Q>", lambda e: self.destroy())

        self.ability_weights: Dict[str, float] = {}
        self.hero_weights: Dict[str, float] = {}
        self.pairs_cache: List[dict] = []
        self.pairs_loaded = False
        self.pairs_loading = False

        try:
            self.status.set("Loading WR ALL ability win rates...")
            self.update_idletasks()
            self.ability_weights = load_windrun_ability_weights()

            self.status.set("Loading hero win rates...")
            self.update_idletasks()
            self.hero_weights = load_windrun_hero_weights()

            print("[debug] hero_weights loaded:", len(self.hero_weights))
            print("[debug] sample hero keys:", list(self.hero_weights.keys())[:10])


            self._start_pairs_load()

            self.status.set(
                f"Windrun loaded: abilities={len(self.ability_weights)} heroes={len(self.hero_weights)}"
            )

        except Exception as e:
            self.status.set("Windrun load failed")
            self.pairs_box.delete("1.0", "end")
            self.pairs_box.insert("end", str(e) + "\n\n" + traceback.format_exc())

        self.after(150, self.refresh)

    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Striped.Treeview", rowheight=32)
        style.map(
            "Striped.Treeview",
            background=[("selected", "#cfe8ff")],
            foreground=[("selected", "#000000")],
        )

    def _start_pairs_load(self):
        if self.pairs_loading or self.pairs_loaded:
            return

        self.pairs_loading = True
        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", "Loading ability pairs in background...\n")

        def worker():
            try:
                pairs = load_windrun_ability_pairs()
                self.after(0, lambda: self._pairs_loaded_ok(pairs))
            except Exception as e:
                self.after(0, lambda: self._pairs_loaded_fail(e))

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

            picks = build_picks(infer_out)
            ranked = rank_by_weight(picks, self.ability_weights, self.hero_weights)

            pairs = outstanding_pairs(infer_out, self.pairs_cache if self.pairs_loaded else None)
            bmap = best_pair_map(pairs)

            self._render_ranked(ranked, bmap)
            self._render_pairs(pairs)

            extra = ""
            if self.pairs_loading:
                extra = " (pairs loading...)"
            elif not self.pairs_loaded:
                extra = " (pairs not loaded)"

            self.status.set(f"Loaded: {img_path.name}{extra}")

        except Exception as e:
            self.status.set("Error")
            self._render_error(e)

    def _render_ranked(self, ranked: List[dict], bmap: dict):
        for t in (self.hero_tree, self.ult_tree, self.abil_tree):
            for r in t.get_children():
                t.delete(r)

        self._images = []

        heroes, ults, abil = [], [], []
        for i, row in enumerate(ranked, 1):
            r2 = dict(row)
            r2["rank"] = i
            typ = r2.get("type")
            if typ == "hero":
                heroes.append(r2)
            elif typ == "ultimate":
                ults.append(r2)
            else:
                abil.append(r2)

        def insert_rows(tree: ttk.Treeview, rows: List[dict]):
            for idx, row in enumerate(rows, 1):
                name = row["name"]
                typ = row["type"]

                icon_kind = "hero" if typ == "hero" else "ability"
                icon = self.icon_cache.get_icon(icon_kind, name)
                if icon:
                    self._images.append(icon)

                w = row.get("weight")
                w_txt = "n/a" if w is None else f"{w:.2f}"

                c = row.get("conf")
                c_txt = "n/a" if c is None else f"{c:.3f}"

                partner = bmap.get(row["name_norm"])
                pair_txt = ""
                if partner:
                    p, s = partner
                    pair_txt = f"{p} ({'n/a' if s is None else f'{s:.2f}'})"

                tag = "odd" if (idx % 2 == 1) else "even"

                tree.insert(
                    parent="",
                    index="end",
                    text=name,
                    image=icon if icon else "",
                    values=(f"{row['rank']:02d}", w_txt, c_txt, row.get("cell", ""), pair_txt),
                    tags=(tag,),
                )

        insert_rows(self.hero_tree, heroes)
        insert_rows(self.ult_tree, ults)
        insert_rows(self.abil_tree, abil)

    def _render_pairs(self, pairs: List[dict]):
        self.pairs_box.delete("1.0", "end")

        if self.pairs_loading:
            self.pairs_box.insert("end", "Loading ability pairs in background...\n")
            return

        if not self.pairs_loaded:
            self.pairs_box.insert("end", "(pairs not loaded)\n")
            return

        if not pairs:
            self.pairs_box.insert("end", "(none)\n")
            return

        for i, p in enumerate(pairs[:TOP_PAIRS], 1):
            sc = "n/a" if p.get("score") is None else f"{p['score']:.2f}"
            self.pairs_box.insert("end", f"{i:02d}. {p['a_raw']} + {p['b_raw']}   score={sc}\n")

    def _render_error(self, e: Exception):
        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", str(e) + "\n\n")
        self.pairs_box.insert("end", traceback.format_exc())


if __name__ == "__main__":
    app = LiveDraftGUI()
    app.mainloop()
