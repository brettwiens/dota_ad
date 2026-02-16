import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# Base + Folder Layout (Z:\DotaAD)
# -------------------------
BASE_DIR = Path(r"Z:\DotaAD")

# Models (your new folder layout)
DETECT_MODEL = BASE_DIR / "grid_detector" / "best.pt"
HERO_CLS_MODEL = BASE_DIR / "heroes_classifier" / "best.pt"
ABILITY_CLS_MODEL = BASE_DIR / "abilities_classifier" / "best.pt"

# Data
CELLS_CSV = BASE_DIR / "cells" / "cells.csv"

# Output folder
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Steam screenshot folder to monitor (most recent .jpg)
SCREENSHOT_DIR = Path(
    r"C:\Program Files (x86)\Steam\userdata\59046080\760\remote\570\screenshots"
)
SCREENSHOT_EXTS = {".jpg", ".jpeg"}

# -------------------------
# Constants
# -------------------------
IMG_W, IMG_H = 1920, 1080

# Detector settings
DET_IMGSZ = 1024
DET_CONF = 0.10
DET_IOU = 0.60
MAX_DETS = 140

# Classifier settings
CLS_IMGSZ = 128

# Matching threshold: if a template cell cannot find a det box above this IoU, we warn
MIN_IOU_MATCH = 0.30

# Use buffered coords from CSV (EX1/EY1/EX2/EY2)
USE_BUFFERED_TEMPLATE = True

# Confidence thresholds
HERO_MIN_CONF = 0.30
ABILITY_MIN_CONF = 0.30


# -------------------------
# Helpers
# -------------------------
def _require_file(p: Path, label: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _require_dir(p: Path, label: str) -> Path:
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found (dir): {p}")
    return p


# Validate required files early (clear errors)
_require_file(DETECT_MODEL, "DETECT_MODEL")
_require_file(HERO_CLS_MODEL, "HERO_CLS_MODEL")
_require_file(ABILITY_CLS_MODEL, "ABILITY_CLS_MODEL")
_require_file(CELLS_CSV, "CELLS_CSV")
_require_dir(SCREENSHOT_DIR, "SCREENSHOT_DIR")


def latest_screenshot(folder: Path) -> Path:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SCREENSHOT_EXTS]
    if not files:
        raise FileNotFoundError(f"No jpg screenshots found in: {folder}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def clamp_xyxy(x1, y1, x2, y2, w=IMG_W, h=IMG_H):
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def iou_xyxy(a, b) -> float:
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


def load_template_cells(csv_path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Accepts:
    - Proper CSV with headers: Cell,EX1,EY1,EX2,EY2 (or X1,Y1,X2,Y2)
    - TSV saved as .csv
    - Headers with extra whitespace
    - Repeated header names (like X,Y,X,Y,EX,EY,EX,EY) by using column positions
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    delim = "\t" if sample.count("\t") > sample.count(",") else ","

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {csv_path}")

    header = [h.strip() for h in rows[0]]

    def idx_of(name: str):
        for i, h in enumerate(header):
            if h.lower() == name.lower():
                return i
        return None

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
        if not r or all(not c.strip() for c in r):
            continue

        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))

        name = r[cell_i].strip() if cell_i is not None else r[0].strip()
        if not name:
            continue

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

        cells[name] = clamp_xyxy(x1, y1, x2, y2)

    return cells


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if (w, h) != (IMG_W, IMG_H):
        raise ValueError(f"Expected {IMG_W}x{IMG_H}, got {w}x{h} for {path}")
    return img


def crop(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].copy()


def classify_batch(model: YOLO, crops: List[np.ndarray], imgsz: int) -> List[Tuple[str, float]]:
    """
    Returns list of (predicted_class_name, confidence)
    Ultralytics returns probabilities; we take top1.
    """
    if not crops:
        return []

    results = model.predict(crops, imgsz=imgsz, verbose=False)
    out: List[Tuple[str, float]] = []
    for r in results:
        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = r.names[top1]
        out.append((name, conf))
    return out


def get_crop_box(info: dict) -> Tuple[int, int, int, int]:
    return info.get("template_box")


def _cell_sort_key(s: str) -> int:
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits) if digits else 0


# -------------------------
# Main pipeline
# -------------------------
def infer_one(image_path: str) -> dict:
    img = read_image(image_path)

    template = load_template_cells(CELLS_CSV)
    if len(template) != 60:
        print(f"WARNING: template has {len(template)} cells, expected 60")

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

    det_xyxy = det_res.boxes.xyxy.cpu().numpy()
    det_conf = det_res.boxes.conf.cpu().numpy()
    det_boxes = [clamp_xyxy(*b) for b in det_xyxy]
    det_items = list(zip(det_boxes, det_conf))

    used = set()
    assigned: Dict[str, Dict] = {}
    warnings: List[str] = []

    for cell_name, tbox in template.items():
        best_j = None
        best_iou = -1.0

        for j, (dbox, _) in enumerate(det_items):
            if j in used:
                continue
            i = iou_xyxy(tbox, dbox)
            if i > best_iou:
                best_iou = i
                best_j = j

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

    hero_cells = sorted([k for k in assigned.keys() if k.lower().startswith("hero")], key=_cell_sort_key)
    ability_cells = sorted([k for k in assigned.keys() if k.lower().startswith("ability")], key=_cell_sort_key)
    ultimate_cells = sorted([k for k in assigned.keys() if k.lower().startswith("ultimate")], key=_cell_sort_key)

    hero_crops = [crop(img, get_crop_box(assigned[c])) for c in hero_cells]
    ability_crops = [crop(img, get_crop_box(assigned[c])) for c in (ultimate_cells + ability_cells)]

    hero_model = YOLO(str(HERO_CLS_MODEL))
    ability_model = YOLO(str(ABILITY_CLS_MODEL))

    hero_preds = classify_batch(hero_model, hero_crops, CLS_IMGSZ)
    ability_preds = classify_batch(ability_model, ability_crops, CLS_IMGSZ)

    for cell, (pred, conf) in zip(hero_cells, hero_preds):
        if conf < HERO_MIN_CONF:
            pred = "unknown"
        assigned[cell]["pred"] = pred
        assigned[cell]["pred_conf"] = float(conf)
        assigned[cell]["type"] = "hero"

    for cell, (pred, conf) in zip((ultimate_cells + ability_cells), ability_preds):
        if conf < ABILITY_MIN_CONF:
            pred = "unknown"
        assigned[cell]["pred"] = pred
        assigned[cell]["pred_conf"] = float(conf)
        assigned[cell]["type"] = "ability"

    out = {
        "source_image": image_path,
        "ultimates": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in ultimate_cells],
        "heroes": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in hero_cells],
        "abilities": [{"cell": c, "name": assigned[c].get("pred"), "conf": assigned[c].get("pred_conf")} for c in ability_cells],
        "warnings": warnings,
    }

    hero_vec = [assigned[c].get("pred") for c in hero_cells]
    ult_vec = [assigned[c].get("pred") for c in ultimate_cells]
    ability_vec = [assigned[c].get("pred") for c in ability_cells]
    all_picks_vec = hero_vec + ult_vec + ability_vec

    out["hero_vec"] = hero_vec
    out["ultimate_vec"] = ult_vec
    out["ability_vec"] = ability_vec
    out["all_picks_vec"] = all_picks_vec

    print("\n=== QUICK CHECK VECTORS ===")
    print("Heroes:", hero_vec)
    print("Ultimates:", ult_vec)
    print("Abilities:", ability_vec)
    print("All picks:", all_picks_vec)

    json_path = OUT_DIR / (Path(image_path).stem + "_draft.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

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

    print("Saved:", json_path)
    print("Saved:", overlay_path)

    if warnings:
        print("Warnings (first 10):")
        for w in warnings[:10]:
            print(" -", w)

    return out


if __name__ == "__main__":
    img_path = latest_screenshot(SCREENSHOT_DIR)
    print("Using latest screenshot:", img_path)
    infer_one(str(img_path))
