from __future__ import annotations

"""
infer_draft_v2.py

Purpose
-------
This file runs the "two stage" computer vision pipeline for Dota 2 Ability Draft:

Stage 1) Grid / icon DETECTION
  - A YOLO detector finds icon-sized bounding boxes on the draft screen.

Stage 2) Icon CLASSIFICATION
  - A hero classifier predicts which hero each hero-cell contains.
  - An ability classifier predicts which ability each ability/ultimate-cell contains.

Why we do it this way
---------------------
The draft screen contains many icons, and their exact positions are fairly stable.
We use a template of 60 known cell locations (cells.csv) to decide "where each icon should be".
Then we match each template cell to the best detector box via IoU.

This gives us:
  - stable cell naming (Hero1..Hero10, Ability1..Ability40, Ultimate1..Ultimate10, etc.)
  - a best-effort mapping even if detection is imperfect
  - consistent crops for the classifiers

Outputs
-------
The main function infer_one(...) returns a dict containing:
  - heroes / ultimates / abilities: list of predictions with confidences
  - hero_vec / ultimate_vec / ability_vec / all_picks_vec: quick lists of predicted names
  - warnings: any low-IoU cell matches that may indicate detection/template drift

It also writes:
  - a JSON file with full results
  - an overlay image showing each cell box and its predicted label

Assumptions
-----------
- The input image is exactly 1920 x 1080.
- You have trained YOLO models saved at the configured paths.
- cells.csv exists and provides cell bounding boxes in either buffered or raw coordinates.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------------------
# Base + Folder Layout
# --------------------------------------------------------------------------------------
# All project paths are built relative to BASE_DIR.
BASE_DIR = Path(r"Z:\DotaAD\dota_ad")

# --------------------------------------------------------------------------------------
# Model paths
# --------------------------------------------------------------------------------------
# DETECT_MODEL:
#   Finds "icon boxes" on the whole screenshot.
DETECT_MODEL = BASE_DIR / "grid_detector" / "best.pt"

# HERO_CLS_MODEL:
#   Takes cropped hero-cell images and predicts hero identity.
HERO_CLS_MODEL = BASE_DIR / "heroes_classifier" / "best.pt"

# ABILITY_CLS_MODEL:
#   Takes cropped ability/ultimate-cell images and predicts the ability identity.
ABILITY_CLS_MODEL = BASE_DIR / "abilities_classifier" / "best.pt"

# --------------------------------------------------------------------------------------
# Template cells CSV
# --------------------------------------------------------------------------------------
# This CSV defines where each cell is expected to be on the draft screen.
# It provides either:
#   - buffered coords: EX1/EY1/EX2/EY2
#   - raw coords: X1/Y1/X2/Y2
CELLS_CSV = BASE_DIR / "cells" / "cells.csv"

# --------------------------------------------------------------------------------------
# Output folder
# --------------------------------------------------------------------------------------
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Steam screenshot folder to monitor (most recent .jpg)
# --------------------------------------------------------------------------------------
SCREENSHOT_DIR = Path(
    r"C:\Program Files (x86)\Steam\userdata\59046080\760\remote\570\screenshots"
)
SCREENSHOT_EXTS = {".jpg", ".jpeg"}

# --------------------------------------------------------------------------------------
# Constants and tuning knobs
# --------------------------------------------------------------------------------------

# Draft screenshots must be this exact size for the template system to work.
IMG_W, IMG_H = 1920, 1080

# Detector inference settings
DET_IMGSZ = 1024    # detector input size; larger can help small icons but costs speed
DET_CONF = 0.10     # low threshold so we don't miss icons; we'll fix mistakes via template matching
DET_IOU = 0.60      # NMS threshold for detector
MAX_DETS = 140      # safety cap on number of detected boxes

# Classifier inference settings
CLS_IMGSZ = 128     # classifiers run on small crops; keep it small and fast

# Template matching threshold:
# If the best detector box for a cell has IoU below this, we still store it but warn.
MIN_IOU_MATCH = 0.30

# Template coordinate mode:
# Buffered coords are usually more forgiving if detector boxes drift slightly.
USE_BUFFERED_TEMPLATE = True

# Confidence thresholds:
# If the classifier confidence is below these thresholds, we label the pick as "unknown".
HERO_MIN_CONF = 0.30
ABILITY_MIN_CONF = 0.30


# --------------------------------------------------------------------------------------
# Basic filesystem validation
# --------------------------------------------------------------------------------------
def _require_file(p: Path, label: str) -> Path:
    """
    Fail fast with a clear message if a required file is missing.
    """
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _require_dir(p: Path, label: str) -> Path:
    """
    Fail fast with a clear message if a required directory is missing.
    """
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found (dir): {p}")
    return p


# Validate required files early so errors show up immediately when you run the script.
_require_file(DETECT_MODEL, "DETECT_MODEL")
_require_file(HERO_CLS_MODEL, "HERO_CLS_MODEL")
_require_file(ABILITY_CLS_MODEL, "ABILITY_CLS_MODEL")
_require_file(CELLS_CSV, "CELLS_CSV")
_require_dir(SCREENSHOT_DIR, "SCREENSHOT_DIR")


# --------------------------------------------------------------------------------------
# Screenshot helper
# --------------------------------------------------------------------------------------
def latest_screenshot(folder: Path) -> Path:
    """
    Return the most recently modified screenshot in the Steam screenshot folder.
    """
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SCREENSHOT_EXTS]
    if not files:
        raise FileNotFoundError(f"No jpg screenshots found in: {folder}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def clamp_xyxy(x1, y1, x2, y2, w=IMG_W, h=IMG_H):
    """
    Clamp a bounding box to image boundaries and guarantee it has non-zero area.
    """
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))

    # Ensure x2 > x1 and y2 > y1 so cropping doesn't crash.
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    return x1, y1, x2, y2


def iou_xyxy(a, b) -> float:
    """
    Compute Intersection-over-Union (IoU) between two boxes in xyxy format.

    IoU is the main scoring function used to match:
      template cell box  <->  detected icon box
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------------------
# Template loading
# --------------------------------------------------------------------------------------
def load_template_cells(csv_path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Load the expected draft grid cell boxes from a CSV.

    This loader is intentionally forgiving because spreadsheet exports can vary.
    It supports:
      - Proper CSV with headers
      - TSV saved as .csv
      - Extra whitespace in headers
      - Repeated header names (some exports do weird things)

    Expected columns (best case):
      Cell, EX1, EY1, EX2, EY2   (buffered coords)
    or:
      Cell, X1, Y1, X2, Y2       (raw coords)

    Fallback:
      If the file has 9+ columns, we can use positional indexes:
        - buffered coords at indexes [5..8]
        - raw coords at indexes [1..4]

    Returns:
      dict mapping cell name -> (x1, y1, x2, y2) in image pixel coordinates
    """
    # Detect delimiter by sampling the first chunk of the file.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    delim = "\t" if sample.count("\t") > sample.count(",") else ","

    # Read the full file.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {csv_path}")

    header = [h.strip() for h in rows[0]]

    def idx_of(name: str):
        """Case-insensitive column lookup by header name."""
        for i, h in enumerate(header):
            if h.lower() == name.lower():
                return i
        return None

    # Column positions (if present)
    cell_i = idx_of("Cell")
    has_positional_9 = len(header) >= 9

    ex1_i = idx_of("EX1")
    ey1_i = idx_of("EY1")
    ex2_i = idx_of("EX2")
    ey2_i = idx_of("EY2")

    x1_i = idx_of("X1")
    y1_i = idx_of("Y1")
    x2_i = idx_of("X2")
    y2_i = idx_of("Y2")

    cells: Dict[str, Tuple[int, int, int, int]] = {}

    for r in rows[1:]:
        # Skip empty lines.
        if not r or all(not c.strip() for c in r):
            continue

        # Ensure row is at least as long as header (helps when trailing values are missing).
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))

        # Cell name is either in "Cell" column, or in column 0 if missing.
        name = r[cell_i].strip() if cell_i is not None else r[0].strip()
        if not name:
            continue

        # Decide whether to use buffered or raw coordinates based on USE_BUFFERED_TEMPLATE.
        if USE_BUFFERED_TEMPLATE:
            if all(i is not None for i in [ex1_i, ey1_i, ex2_i, ey2_i]):
                x1, y1, x2, y2 = (
                    int(float(r[ex1_i])),
                    int(float(r[ey1_i])),
                    int(float(r[ex2_i])),
                    int(float(r[ey2_i])),
                )
            elif has_positional_9:
                x1, y1, x2, y2 = (
                    int(float(r[5])),
                    int(float(r[6])),
                    int(float(r[7])),
                    int(float(r[8])),
                )
            else:
                raise KeyError(
                    "Could not find EX1/EY1/EX2/EY2 columns and file is not in 9-column positional format."
                )
        else:
            if all(i is not None for i in [x1_i, y1_i, x2_i, y2_i]):
                x1, y1, x2, y2 = (
                    int(float(r[x1_i])),
                    int(float(r[y1_i])),
                    int(float(r[x2_i])),
                    int(float(r[y2_i])),
                )
            elif has_positional_9:
                x1, y1, x2, y2 = (
                    int(float(r[1])),
                    int(float(r[2])),
                    int(float(r[3])),
                    int(float(r[4])),
                )
            else:
                raise KeyError(
                    "Could not find X1/Y1/X2/Y2 columns and file is not in 9-column positional format."
                )

        # Clamp to valid pixel bounds and store.
        cells[name] = clamp_xyxy(x1, y1, x2, y2)

    return cells


# --------------------------------------------------------------------------------------
# Image IO helpers
# --------------------------------------------------------------------------------------
def read_image(path: str) -> np.ndarray:
    """
    Load an image from disk and confirm it is the expected resolution.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if (w, h) != (IMG_W, IMG_H):
        raise ValueError(f"Expected {IMG_W}x{IMG_H}, got {w}x{h} for {path}")
    return img


def crop(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a region from the image based on an xyxy box.
    """
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].copy()


# --------------------------------------------------------------------------------------
# Classification helper
# --------------------------------------------------------------------------------------
def classify_batch(model: YOLO, crops: List[np.ndarray], imgsz: int) -> List[Tuple[str, float]]:
    """
    Run a classifier YOLO model on many cropped images at once.

    Why batch?
    - Ultralytics can run a list of numpy arrays efficiently.
    - This is faster than calling predict(...) one crop at a time.

    Returns:
      list of (predicted_class_name, confidence)
    """
    if not crops:
        return []

    results = model.predict(crops, imgsz=imgsz, verbose=False)

    out: List[Tuple[str, float]] = []
    for r in results:
        # Ultralytics classifiers expose r.probs (class probabilities).
        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = r.names[top1]
        out.append((name, conf))

    return out


def get_crop_box(info: dict) -> Tuple[int, int, int, int]:
    """
    In this pipeline, we crop by template_box, not by detector_box.

    Why?
    - Template boxes are stable and consistent (good for classifier training and inference).
    - Detector boxes can shift slightly which may add noise.

    If you ever want to crop by detector_box instead, this is the function you would change.
    """
    return info.get("template_box")


def _cell_sort_key(s: str) -> int:
    """
    Sort helper for cell names like "Hero1", "Hero2", ..., "Hero10".
    Extracts digits and sorts numerically.
    """
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits) if digits else 0


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------
def infer_one(image_path: str, verbose: bool = True) -> dict:
    """
    Run the full draft inference pipeline on one screenshot.

    Steps
    -----
    1) Load image and template cells (expected grid positions).
    2) Run detector to find icon boxes.
    3) For each template cell:
         - find the best unused detector box by IoU
         - store match info (or warning if match is weak)
    4) Crop hero cells and classify with hero model.
    5) Crop ability+ultimate cells and classify with ability model.
    6) Build a structured output dict, save JSON + overlay image, return output.
    """
    img = read_image(image_path)

    # Load the 60 cell template mapping (cell name -> pixel box).
    template = load_template_cells(CELLS_CSV)
    if len(template) != 60 and verbose:
        print(f"WARNING: template has {len(template)} cells, expected 60")

    # -------------------------
    # Stage 1: Detection
    # -------------------------
    # The detector runs on the full screenshot and outputs icon-ish boxes.
    det_model = YOLO(str(DETECT_MODEL))
    det_res = det_model.predict(
        source=image_path,
        imgsz=DET_IMGSZ,
        conf=DET_CONF,
        iou=DET_IOU,
        max_det=MAX_DETS,
        verbose=False,
    )[0]

    if det_res.boxes is None or len(det_res.boxes) == 0:
        raise RuntimeError("Detector returned 0 boxes")

    # Convert detector tensors to plain python structures.
    det_xyxy = det_res.boxes.xyxy.cpu().numpy()
    det_conf = det_res.boxes.conf.cpu().numpy()

    det_boxes = [clamp_xyxy(*b) for b in det_xyxy]
    det_items = list(zip(det_boxes, det_conf))

    # -------------------------
    # Stage 1b: Match detector boxes to template cells
    # -------------------------
    # Each detector box should match at most one cell (hence "used").
    used = set()

    # assigned will store one record per cell: match stats, boxes, and later prediction results.
    assigned: Dict[str, Dict] = {}

    # warnings collect "something looks off" messages for later debugging.
    warnings: List[str] = []

    for cell_name, tbox in template.items():
        best_j = None
        best_iou = -1.0

        # Find the best unused detector box for this template cell by IoU.
        for j, (dbox, _) in enumerate(det_items):
            if j in used:
                continue
            i = iou_xyxy(tbox, dbox)
            if i > best_iou:
                best_iou = i
                best_j = j

        # If nothing matches well, record a warning and leave det_box empty.
        if best_j is None or best_iou < MIN_IOU_MATCH:
            warnings.append(f"Low/failed match for {cell_name}: best IoU={best_iou:.3f}")
            assigned[cell_name] = {
                "cell": cell_name,
                "match_iou": float(best_iou),
                "template_box": tbox,
                "det_box": None,
            }
        else:
            used.add(best_j)
            dbox, dconf = det_items[best_j]
            assigned[cell_name] = {
                "cell": cell_name,
                "match_iou": float(best_iou),
                "det_conf": float(dconf),
                "template_box": tbox,
                "det_box": dbox,
            }

    # -------------------------
    # Split cells into groups (heroes vs ultimates vs abilities)
    # -------------------------
    # We rely on cell naming conventions in cells.csv: "Hero1", "Ability12", "Ultimate3", etc.
    hero_cells = sorted([k for k in assigned.keys() if k.lower().startswith("hero")], key=_cell_sort_key)
    ability_cells = sorted([k for k in assigned.keys() if k.lower().startswith("ability")], key=_cell_sort_key)
    ultimate_cells = sorted([k for k in assigned.keys() if k.lower().startswith("ultimate")], key=_cell_sort_key)

    # -------------------------
    # Stage 2: Classification (heroes)
    # -------------------------
    # We crop using the template boxes (consistent framing).
    hero_crops = [crop(img, get_crop_box(assigned[c])) for c in hero_cells]

    # -------------------------
    # Stage 2: Classification (abilities + ultimates)
    # -------------------------
    # We run ultimates and abilities through the same ability classifier.
    ability_crops = [crop(img, get_crop_box(assigned[c])) for c in (ultimate_cells + ability_cells)]

    hero_model = YOLO(str(HERO_CLS_MODEL))
    ability_model = YOLO(str(ABILITY_CLS_MODEL))

    hero_preds = classify_batch(hero_model, hero_crops, CLS_IMGSZ)
    ability_preds = classify_batch(ability_model, ability_crops, CLS_IMGSZ)

    # Attach hero predictions back onto assigned records.
    for cell, (pred, conf) in zip(hero_cells, hero_preds):
        if conf < HERO_MIN_CONF:
            pred = "unknown"
        assigned[cell]["pred"] = pred
        assigned[cell]["pred_conf"] = float(conf)
        assigned[cell]["type"] = "hero"

    # Attach ability predictions back onto assigned records.
    for cell, (pred, conf) in zip((ultimate_cells + ability_cells), ability_preds):
        if conf < ABILITY_MIN_CONF:
            pred = "unknown"
        assigned[cell]["pred"] = pred
        assigned[cell]["pred_conf"] = float(conf)
        assigned[cell]["type"] = "ability"

    # -------------------------
    # Build output payload
    # -------------------------
    out = {
        "source_image": image_path,
        "ultimates": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in ultimate_cells],
        "heroes": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in hero_cells],
        "abilities": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in ability_cells],
        "warnings": warnings,
    }

    # These vectors are convenient for downstream logic (like your Windrun pair filtering).
    hero_vec = [assigned[c].get("pred") for c in hero_cells]
    ult_vec = [assigned[c].get("pred") for c in ultimate_cells]
    ability_vec = [assigned[c].get("pred") for c in ability_cells]
    all_picks_vec = hero_vec + ult_vec + ability_vec

    out["hero_vec"] = hero_vec
    out["ultimate_vec"] = ult_vec
    out["ability_vec"] = ability_vec
    out["all_picks_vec"] = all_picks_vec

    if verbose:
        print("\n=== QUICK CHECK VECTORS ===")
        print("Heroes:", hero_vec)
        print("Ultimates:", ult_vec)
        print("Abilities:", ability_vec)
        print("All picks:", all_picks_vec)

    # -------------------------
    # Save JSON output
    # -------------------------
    json_path = OUT_DIR / (Path(image_path).stem + "_draft.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # -------------------------
    # Save overlay image for debugging
    # -------------------------
    # This draws each template box and prints the predicted label above it.
    overlay = img.copy()
    for cell_name, info in assigned.items():
        x1, y1, x2, y2 = get_crop_box(info)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = cell_name
        if "pred" in info:
            label += f": {info['pred']}"

        cv2.putText(
            overlay,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
        )

    overlay_path = OUT_DIR / (Path(image_path).stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), overlay)

    if verbose:
        print("Saved:", json_path)
        print("Saved:", overlay_path)

        if warnings:
            print("Warnings (first 10):")
            for w in warnings[:10]:
                print(" -", w)

    return out


# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # If you run this file directly, it will auto-pick the newest Steam screenshot
    # and run the pipeline in verbose mode.
    img_path = latest_screenshot(SCREENSHOT_DIR)
    print("Using latest screenshot:", img_path)
    infer_one(str(img_path), verbose=True)
