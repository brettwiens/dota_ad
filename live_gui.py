from __future__ import annotations

"""
live_gui.py

Purpose
-------
This is the "live draft assistant" GUI for Dota 2 Ability Draft.

At a high level, the app does four things:

1) Finds your most recent Steam screenshot of the draft screen.
2) Runs your YOLO-based inference (infer_draft_v2.py) on that screenshot to detect:
     - heroes
     - ultimates
     - abilities
   and returns those detections with confidence scores.
3) Loads "how good is this ability" data from Windrun (win-rate style weights).
4) Displays everything in a simple Tkinter UI:
     - Ranked lists (Heroes / Ultimates / Abilities) with icons and weights
     - The best outstanding ability pairs (synergy / pair win rate)

Important performance idea
--------------------------
Windrun scraping can require Playwright (browser rendering) which is slower than requests.
To keep refresh quick, we preload Windrun once at startup and cache the results:

    self.weights     = load_windrun_ability_weights()
    self.pairs_cache = load_windrun_ability_pairs()

Then every refresh simply re-runs inference on the newest screenshot and filters the
already-loaded Windrun pair list.
"""

import re
import traceback
from pathlib import Path
from typing import List, Optional

import importlib.util
import inspect
import tkinter as tk
from tkinter import ttk

# We import Windrun helpers:
# - normalize_name: a stable normaliser that turns "Ball Lightning" into "ball_lightning"
# - load_windrun_ability_weights: a dict of { ability_norm: winrate_like_number }
# - load_windrun_ability_pairs: list of pair rows with a_norm/b_norm/score
from windrun_collector import (
    _norm as normalize_name,
    load_windrun_ability_weights,
    load_windrun_ability_pairs,
)

# --------------------------------------------------------------------------------------
# SETTINGS: project paths, icons, and UI display constants
# --------------------------------------------------------------------------------------

# Base folder where your project lives.
BASE_DIR = Path(r"Z:\DotaAD\dota_ad")

# Your inference script. We load it by absolute path to avoid import confusion if there
# are multiple copies of infer_draft_v2.py elsewhere on your machine.
INFER_PATH = BASE_DIR / "infer_draft_v2.py"

# Where Steam stores Dota 2 screenshots (the "latest screenshot" is assumed to be the draft).
SCREENSHOT_DIR = Path(
    r"C:\Program Files (x86)\Steam\userdata\59046080\760\remote\570\screenshots"
)

# Only these file types are considered screenshots.
IMAGE_EXTS = {".jpg", ".jpeg"}

# Icon folders used to show small images in the Treeview lists.
HERO_ICON_DIR = BASE_DIR / "icons" / "heroes"
ABILITY_ICON_DIR = BASE_DIR / "icons" / "abilities"

# How many top pairs to display in the right-hand panel.
TOP_PAIRS = 25

# Pixel size (width/height) of icon images rendered in the table.
ICON_SIZE = 28


# --------------------------------------------------------------------------------------
# PIL (Pillow) is optional
# --------------------------------------------------------------------------------------
# Why optional?
# - If Pillow is installed, we can resize icons smoothly.
# - If Pillow is not installed, Tkinter can still load PNGs directly, but without resizing.
try:
    from PIL import Image, ImageTk

    PIL_OK = True
except Exception:
    PIL_OK = False


# --------------------------------------------------------------------------------------
# LOAD INFER (ABS PATH)
# --------------------------------------------------------------------------------------
def load_infer_one(infer_path: Path):
    """
    Dynamically import infer_draft_v2.py from an absolute path and return infer_one.

    Why this method instead of "import infer_draft_v2"?
    - It guarantees we import the exact file at INFER_PATH.
    - It avoids name collisions if Python can see another module with the same name.

    Returns:
      A callable: infer_one(image_path: str, verbose: bool) -> dict
    """
    if not infer_path.exists():
        raise FileNotFoundError(f"infer file not found: {infer_path}")

    spec = importlib.util.spec_from_file_location("infer_draft_v2_local", str(infer_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for: {infer_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # Helpful debug output so you always know which infer file is running.
    print("USING INFER FILE:", inspect.getsourcefile(mod))
    print("infer_one signature:", inspect.signature(mod.infer_one))
    return mod.infer_one


# Load infer_one at import time so the GUI can call it quickly.
infer_one = load_infer_one(INFER_PATH)


# --------------------------------------------------------------------------------------
# HELPERS: screenshots, ranking logic, and pair logic
# --------------------------------------------------------------------------------------
def latest_image_in_folder(folder: Path) -> Path:
    """
    Return the most recently modified screenshot file in the Steam screenshot folder.

    This is the "live" input to the inference.
    """
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No screenshots found in: {folder}")

    # Sort newest first by modified time.
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def build_picks(infer_out: dict) -> List[dict]:
    """
    Convert infer_one(...) output into a unified list of "pick" rows.

    infer_out contains separate lists for:
      infer_out["heroes"]
      infer_out["ultimates"]
      infer_out["abilities"]

    We unify them because the ranking logic is identical: "look up windrun weight, sort".
    """
    picks = []
    for kind in ("heroes", "ultimates", "abilities"):
        for item in infer_out.get(kind, []):
            name = item.get("name")
            if not name:
                continue

            picks.append(
                {
                    # type is used to decide which Treeview to display in and which icon folder to use
                    "type": ("hero" if kind == "heroes" else "ultimate" if kind == "ultimates" else "ability"),
                    "cell": item.get("cell"),              # where it appeared on the draft grid
                    "name": name,                          # display name
                    "name_norm": normalize_name(name),     # stable key for matching weights/pairs/icons
                    "conf": item.get("conf"),              # YOLO confidence
                }
            )
    return picks


def rank_by_weight(picks: List[dict], weights: dict) -> List[dict]:
    """
    Attach Windrun weight to each pick (if available) and sort best-first.

    Sorting intent:
      - Primarily: higher Windrun weight (better winrate) first
      - Secondary: higher inference confidence first

    If an item has no weight, it sinks below weighted items.
    """
    out = []
    for p in picks:
        p2 = dict(p)
        p2["weight"] = weights.get(p["name_norm"])
        out.append(p2)

    # Sort descending by weight, then descending by confidence.
    out.sort(
        key=lambda d: (
            -1 if d.get("weight") is None else d["weight"],
            -1 if d.get("conf") is None else d["conf"],
        ),
        reverse=True,
    )
    return out


def available_ability_norm_set(infer_out: dict) -> set:
    """
    Build a set of "currently available abilities" in normalised form.

    The inference layer (infer_draft_v2.py) provides these two vectors:
      ultimate_vec: list of ultimate names on the draft screen
      ability_vec:  list of basic ability names on the draft screen

    We merge them because from a synergy perspective they are all "abilities".
    """
    vals = infer_out.get("ultimate_vec", []) + infer_out.get("ability_vec", [])
    vals = [x for x in vals if x and x != "unknown"]
    return {normalize_name(x) for x in vals}


def outstanding_pairs(infer_out: dict, pairs_source: Optional[List[dict]] = None) -> List[dict]:
    """
    Filter Windrun's full pair dataset down to only pairs where BOTH abilities are available now.

    Inputs:
      infer_out: output of infer_one(...)
      pairs_source: if provided, use this list instead of reloading windrun (caching)

    Output:
      list of pair dicts sorted best-first (highest score first)

    Why pairs_source exists:
      The Windrun pairs dataset is large and can require scrolling scrape.
      We load it once at startup and pass it in, so refresh() is fast.
    """
    norm_set = available_ability_norm_set(infer_out)
    pairs = pairs_source if pairs_source is not None else load_windrun_ability_pairs()

    out = []
    for p in pairs:
        if p.get("a_norm") in norm_set and p.get("b_norm") in norm_set:
            out.append(p)

    # Sort by score (descending). If score is None, it sinks.
    out.sort(key=lambda d: (d.get("score") is not None, d.get("score") or -1), reverse=True)
    return out


def best_pair_map(pairs: List[dict]) -> dict:
    """
    Build a quick lookup: for each ability, what is its single best partner among the
    currently outstanding pairs?

    Returns:
      dict like:
        {
          "ball_lightning": ("Arcane Orb", 70.8),
          ...
        }

    This is used to show "Best Pair" in the ranked pick lists.
    """
    best = {}

    def update(k, partner, score):
        cur = best.get(k)
        if cur is None or (score or -1) > (cur[1] or -1):
            best[k] = (partner, score)

    for p in pairs:
        update(p["a_norm"], p["b_raw"], p.get("score"))
        update(p["b_norm"], p["a_raw"], p.get("score"))

    return best


# --------------------------------------------------------------------------------------
# ICON CACHE: loads icons from disk once and reuses them
# --------------------------------------------------------------------------------------
class IconCache:
    """
    Treeview icons must remain referenced in Python or Tkinter may garbage collect them.

    This class:
      - Tries multiple filename variants for each name ("foo bar" vs "foo_bar" etc.)
      - Loads and resizes icons if Pillow is present
      - Caches PhotoImage objects so we never re-load the same file twice
    """
    def __init__(self):
        self.cache = {}

    def _candidates(self, name: str) -> List[str]:
        """
        Build a list of possible PNG filenames for a given pick name.

        This helps when:
          - your icon files use underscores but Windrun uses spaces
          - different sources use hyphens
          - the inference output slightly differs in formatting
        """
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
        """
        Return a Tkinter PhotoImage for the given icon, or None if not found.

        kind:
          - "hero" -> use HERO_ICON_DIR
          - "ability" -> use ABILITY_ICON_DIR
        """
        key = f"{kind}:{name}"
        if key in self.cache:
            return self.cache[key]

        folder = HERO_ICON_DIR if kind == "hero" else ABILITY_ICON_DIR

        for fname in self._candidates(name):
            p = folder / fname
            if not p.exists():
                continue

            try:
                # Pillow path: load, convert, resize, then wrap into PhotoImage.
                if PIL_OK:
                    img = Image.open(p).convert("RGBA")
                    img = img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img)
                else:
                    # No Pillow: Tkinter loads the PNG as-is.
                    tk_img = tk.PhotoImage(file=str(p))

                self.cache[key] = tk_img
                return tk_img
            except Exception:
                # If one candidate fails (corrupt file, etc.), try next.
                pass

        return None


# --------------------------------------------------------------------------------------
# GUI: main Tkinter application
# --------------------------------------------------------------------------------------
class LiveDraftGUI(tk.Tk):
    """
    The main application window.

    Layout:
      - Top bar: status text + Refresh/Quit buttons
      - Left side: three Treeviews (Heroes / Ultimates / Abilities)
      - Right side: a text box listing the top outstanding pairs
    """
    def __init__(self):
        super().__init__()

        self.title("Dota AD Live Draft Assistant")
        self.geometry("1220x740")

        # Icon cache and PhotoImage retention list.
        self.icon_cache = IconCache()
        self._images = []  # holds references so icons don't disappear

        # Set up styling first so widgets inherit it.
        self._setup_styles()

        # ---------------------------
        # Top status + buttons row
        # ---------------------------
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")

        self.status = tk.StringVar(value="Starting up... loading Windrun tables")
        ttk.Label(top, textvariable=self.status).pack(side="left")

        ttk.Button(top, text="Refresh (R)", command=self.refresh).pack(side="right")
        ttk.Button(top, text="Quit (Q)", command=self.destroy).pack(side="right", padx=6)

        # ---------------------------
        # Main split layout
        # ---------------------------
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=6)

        main.add(left, weight=3)
        main.add(right, weight=2)

        # ---------------------------
        # Left: ranked pick lists
        # ---------------------------
        ttk.Label(left, text="Ranked Picks").pack(anchor="w")

        # Treeview columns (besides the main "Pick" column which is #0)
        cols = ("rank", "weight", "conf", "cell", "pair")

        grid = ttk.Frame(left)
        grid.pack(fill="both", expand=True)

        # Make three equal-width columns.
        grid.columnconfigure(0, weight=1, uniform="x")
        grid.columnconfigure(1, weight=1, uniform="x")
        grid.columnconfigure(2, weight=1, uniform="x")
        grid.rowconfigure(1, weight=1)

        ttk.Label(grid, text="Heroes").grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Ultimates").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(grid, text="Abilities").grid(row=0, column=2, sticky="w", padx=(8, 0))

        def make_tree(parent):
            """
            Create a Treeview with:
              - icons in the main #0 column
              - striped rows for readability
              - fixed column widths so numbers align cleanly
            """
            t = ttk.Treeview(
                parent,
                columns=cols,
                show=("tree", "headings"),
                height=25,
                style="Striped.Treeview",
            )
            t.heading("#0", text="Pick")
            t.heading("rank", text="#")
            t.heading("weight", text="Weight")
            t.heading("conf", text="Conf")
            t.heading("cell", text="Cell")
            t.heading("pair", text="Best Pair")

            t.column("#0", width=220)
            t.column("rank", width=36, anchor="center")
            t.column("weight", width=70, anchor="e")
            t.column("conf", width=60, anchor="e")
            t.column("cell", width=60)
            t.column("pair", width=180)

            # Row striping tags.
            t.tag_configure("odd", background="#f4f4f4")
            t.tag_configure("even", background="#ffffff")
            return t

        self.hero_tree = make_tree(grid)
        self.ult_tree = make_tree(grid)
        self.abil_tree = make_tree(grid)

        self.hero_tree.grid(row=1, column=0, sticky="nsew")
        self.ult_tree.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.abil_tree.grid(row=1, column=2, sticky="nsew", padx=(8, 0))

        # ---------------------------
        # Right: outstanding pairs list
        # ---------------------------
        ttk.Label(right, text="Outstanding Ability Pairs").pack(anchor="w")
        self.pairs_box = tk.Text(right, font=("Consolas", 10))
        self.pairs_box.pack(fill="both", expand=True)

        # ---------------------------
        # Keyboard shortcuts
        # ---------------------------
        self.bind("<r>", lambda e: self.refresh())
        self.bind("<R>", lambda e: self.refresh())
        self.bind("<q>", lambda e: self.destroy())
        self.bind("<Q>", lambda e: self.destroy())

        # --------------------------------------------------------------------------------
        # PRELOAD WINDRUN ONCE (VERY IMPORTANT)
        # --------------------------------------------------------------------------------
        # Windrun scraping may launch Playwright and scroll a virtualized table.
        # Doing that every refresh would feel slow and would hammer the site.
        #
        # Instead, we load it once at startup:
        #   - ability weights: dict of ability -> winrate-like metric
        #   - ability pairs: large list of pairs with pair win rate / synergy
        #
        # Then refresh() only does:
        #   - read newest screenshot
        #   - infer
        #   - rank/filter against cached windrun data
        try:
            self.status.set("Loading Windrun ability weights...")
            self.update_idletasks()
            self.weights = load_windrun_ability_weights()

            self.status.set("Loading Windrun ability pairs...")
            self.update_idletasks()
            self.pairs_cache = load_windrun_ability_pairs()

            self.status.set(
                f"Windrun loaded: weights={len(self.weights)} pairs={len(self.pairs_cache)}"
            )
        except Exception as e:
            # If Windrun fails, we keep the GUI running but display the error.
            self.status.set("Windrun load failed")
            self.pairs_box.delete("1.0", "end")
            self.pairs_box.insert("end", str(e) + "\n\n" + traceback.format_exc())
            self.weights = {}
            self.pairs_cache = []

        # Kick off first refresh after the UI is visible.
        self.after(150, self.refresh)

    def _setup_styles(self):
        """
        UI styling: neutral theme + row height + readable selection colours.
        """
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Slightly taller rows so icons fit nicely.
        style.configure("Striped.Treeview", rowheight=32)

        # Make selection visible but not obnoxious.
        style.map(
            "Striped.Treeview",
            background=[("selected", "#cfe8ff")],
            foreground=[("selected", "#000000")],
        )

    def refresh(self):
        """
        Re-run the "live" part of the pipeline:

          1) Find newest screenshot
          2) Run infer_one on it
          3) Rank picks using cached weights
          4) Filter cached pairs down to "outstanding pairs"
          5) Render the UI

        This is designed to be fast because Windrun data is already cached.
        """
        try:
            self.status.set("Scanning...")
            self.update_idletasks()

            img_path = latest_image_in_folder(SCREENSHOT_DIR)

            # Your inference returns dicts for heroes/ults/abilities plus *_vec lists.
            infer_out = infer_one(str(img_path), verbose=False)

            picks = build_picks(infer_out)
            ranked = rank_by_weight(picks, self.weights)

            pairs = outstanding_pairs(infer_out, pairs_source=self.pairs_cache)
            bmap = best_pair_map(pairs)

            self._render_ranked(ranked, bmap)
            self._render_pairs(pairs)

            self.status.set(f"Loaded: {img_path.name}")

        except Exception as e:
            self.status.set("Error")
            self._render_error(e)

    def _render_ranked(self, ranked: List[dict], bmap: dict):
        """
        Populate the three Treeviews (heroes/ults/abilities).

        We:
          - clear existing rows
          - split ranked picks by type
          - insert into each Treeview with row striping and optional icons
        """
        for t in (self.hero_tree, self.ult_tree, self.abil_tree):
            for r in t.get_children():
                t.delete(r)

        # Clear and repopulate image references to prevent Tk from dropping icons.
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
            """
            Insert rows into a Treeview with:
              - icon in the first column
              - formatted weight/confidence numbers
              - "best pair" text based on bmap
              - alternating row shading
            """
            for idx, row in enumerate(rows, 1):
                name = row["name"]
                typ = row["type"]

                # Heroes use hero icon folder; everything else uses ability icons.
                icon_kind = "hero" if typ == "hero" else "ability"
                icon = self.icon_cache.get_icon(icon_kind, name)
                if icon:
                    self._images.append(icon)

                w = row.get("weight")
                w_txt = "n/a" if w is None else f"{w:.2f}"

                c = row.get("conf")
                c_txt = "n/a" if c is None else f"{c:.3f}"

                # bmap stores: { ability_norm: (partner_raw_name, score) }
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
        """
        Render the right-hand "Outstanding Ability Pairs" panel.

        We display only TOP_PAIRS rows for readability.
        """
        self.pairs_box.delete("1.0", "end")
        if not pairs:
            self.pairs_box.insert("end", "(none)\n")
            return

        for i, p in enumerate(pairs[:TOP_PAIRS], 1):
            sc = "n/a" if p.get("score") is None else f"{p['score']:.2f}"
            self.pairs_box.insert("end", f"{i:02d}. {p['a_raw']} + {p['b_raw']}   score={sc}\n")

    def _render_error(self, e: Exception):
        """
        Show errors in the pairs box so you can debug without opening a console.
        """
        self.pairs_box.delete("1.0", "end")
        self.pairs_box.insert("end", str(e) + "\n\n")
        self.pairs_box.insert("end", traceback.format_exc())


if __name__ == "__main__":
    # Run the GUI when executed directly.
    app = LiveDraftGUI()
    app.mainloop()
