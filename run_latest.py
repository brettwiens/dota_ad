from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from infer_draft_v2 import infer_one
from windrun_collector import load_windrun_ability_weights, load_windrun_ability_pairs


# -------------------------
# Settings
# -------------------------
SCREENSHOT_DIR = Path(r"C:\Program Files (x86)\Steam\userdata\59046080\760\remote\570\screenshots")
IMAGE_EXTS = {".jpg", ".jpeg"}

# Show heroes in the ranked list (they will not have windrun weights from the ability table)
INCLUDE_HEROES_IN_RANKING = False

# How many pairs to show
TOP_PAIRS = 20


# -------------------------
# Helpers
# -------------------------
def latest_image_in_folder(folder: Path) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Screenshot folder not found: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No jpg screenshots found in: {folder}")

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_picks(infer_out: dict) -> List[dict]:
    """
    Flatten infer_one output into a list of picks.
    """
    picks: List[dict] = []
    for kind in ("heroes", "ultimates", "abilities"):
        for item in infer_out.get(kind, []):
            name = item.get("name")
            if not name:
                continue
            picks.append(
                {
                    "type": kind[:-1] if kind != "abilities" else "ability",
                    "cell": item.get("cell"),
                    "name": name,
                    "conf": float(item.get("conf")) if item.get("conf") is not None else None,
                    "name_norm": normalize_name(name),
                }
            )
    return picks


def rank_by_weight(picks: List[dict], weights: Dict[str, Optional[float]]) -> List[dict]:
    """
    Add weight to each pick and sort by weight desc, then conf desc.
    """
    ranked = []
    for p in picks:
        w = weights.get(p["name_norm"])
        p2 = dict(p)
        p2["weight"] = w
        ranked.append(p2)

    # heroes usually have no weights from ability table, so they go bottom
    def sort_key(d):
        w = d["weight"]
        w_sort = -1.0 if w is None else float(w)
        conf = d["conf"] if d["conf"] is not None else -1.0
        return (w_sort, conf)

    ranked.sort(key=sort_key, reverse=True)
    return ranked


def available_ability_set(infer_out: dict) -> Tuple[set, set]:
    """
    Return (raw_names_set, norm_names_set) of abilities+ultimates currently available.
    """
    raw = set()
    for x in infer_out.get("ultimate_vec", []) + infer_out.get("ability_vec", []):
        if x and x != "unknown":
            raw.add(x)
    normed = {normalize_name(x) for x in raw}
    return raw, normed


def print_ranked(ranked: List[dict], image_path: Path, infer_out: dict) -> None:

    print("\n==============================")
    print("LATEST SCREENSHOT:", str(image_path))
    print("==============================\n")

    if not ranked:
        print("No picks found to rank.")
        return

    print("Ranked (by Windrun weight where available):\n")

    shown = 0
    for i, row in enumerate(ranked, 1):
        if not INCLUDE_HEROES_IN_RANKING and row["type"] == "hero":
            continue

        w = row["weight"]
        w_txt = "n/a" if w is None else f"{w:.2f}"
        conf = row["conf"]
        conf_txt = "n/a" if conf is None else f"{conf:.3f}"

        print(f"{shown+1:02d}. [{row['type']}] {row['name']:30s}  weight={w_txt:>6s}  conf={conf_txt:>6s}  ({row['cell']})")
        shown += 1

    if not INCLUDE_HEROES_IN_RANKING:
        print("\n(note: heroes omitted from ranking because windrun weights are ability-only)")

    # Also show vectors (quick scan)
    print("\nQuick vectors:")
    print("Heroes:", infer_out.get("hero_vec", []))
    print("Ultimates:", infer_out.get("ultimate_vec", []))
    print("Abilities:", infer_out.get("ability_vec", []))


def print_outstanding_pairs(infer_out: dict) -> None:
    raw_set, norm_set = available_ability_set(infer_out)

    pairs = load_windrun_ability_pairs()
    outstanding = []
    for p in pairs:
        if p["a_norm"] in norm_set and p["b_norm"] in norm_set:
            outstanding.append(p)

    outstanding.sort(key=lambda d: (d["score"] is not None, d["score"] or -1.0), reverse=True)

    print("\n=== TOP OUTSTANDING ABILITY PAIRS (available now) ===")
    if not outstanding:
        print("(none found from windrun pairs table)")
        return

    for i, p in enumerate(outstanding[:TOP_PAIRS], 1):
        sc = "n/a" if p["score"] is None else f"{p['score']:.2f}"
        print(f"{i:02d}. {p['a_raw']} + {p['b_raw']}   score={sc}")


# -------------------------
# Main
# -------------------------
import os
import time

# -------------------------
# LIVE LOOP
# -------------------------
def clear_screen():
    os.system("cls")


def run_once():
    image_path = latest_image_in_folder(SCREENSHOT_DIR)

    print(f"\nUsing screenshot: {image_path}\n")

    # Run inference
    infer_out = infer_one(str(image_path))

    # Windrun weights
    weights = load_windrun_ability_weights()

    # Rank picks
    picks = build_picks(infer_out)
    ranked = rank_by_weight(picks, weights)

    # Display results
    print_ranked(ranked, image_path, infer_out)

    # Ability pairs
    print_outstanding_pairs(infer_out)


def main():
    while True:
        clear_screen()
        print("======================================")
        print("      DOTA AD LIVE DRAFT ASSISTANT")
        print("======================================")
        print("")
        print("[R] Run scan on latest screenshot")
        print("[Q] Quit")
        print("")

        choice = input("Choice: ").strip().lower()

        if choice == "q":
            print("Good luck in your draft 👍")
            break

        if choice == "r":
            try:
                clear_screen()
                print("Running scan...\n")
                run_once()
            except Exception as e:
                print("\nERROR:", e)

            print("\nPress ENTER to return to menu...")
            input()

        else:
            print("Unknown command.")
            time.sleep(1)


if __name__ == "__main__":
    main()
