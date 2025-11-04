# ================================================================
# classify.py — v3.5 RefinedDetection + StrictAligned
# 改進項目：
#   1. 修正 refine/safe_crop 雙重偏移。
#   2. 提升空白檢測靈敏度，防止誤判為空欄。
#   3. 放寬整欄空白判定門檻（解決高覆蓋率但低存量問題）。
#   4. 列群組更穩定（緩衝區擴增 30%）。
# ================================================================

import os
import re
import cv2
import unicodedata
import numpy as np
from config import SCALE_FACTOR
from ocr import is_grid_blank_dynamically

DEBUG_VISUAL = True
DEBUG_DIR = "./debug_steps"


# ------------------------------------------------------------
# 安全 I/O 工具
# ------------------------------------------------------------
def safe_imwrite(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if ok:
        with open(path, "wb") as f:
            f.write(buf)
        return True
    print(f"⚠️ safe_imwrite 失敗: {path}")
    return False


def sanitize_label(label: str) -> str:
    label = unicodedata.normalize("NFKC", str(label or "UNK"))
    label = re.sub(r'[\\/:*?"<>|]', "_", label.strip())
    return label if label else "UNK"


# ------------------------------------------------------------
# 局部格線微校正（v3.5：穩定偏移版）
# ------------------------------------------------------------
def refine_box_with_local_edges(image, box, max_shift=2):
    """
    v3.6 TightAligned 專用：
      - 僅垂直方向微修正，且不會向外擴張。
      - 防止上/下邊線重新納入格子範圍。
    """
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    y1 = max(0, y - max_shift)
    y2 = min(h_img, y + h + max_shift)
    x1 = max(0, x)
    x2 = min(w_img, x + w)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return box

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    proj_y = np.sum(edges, axis=1).astype(np.float32)

    # 上下邊線偵測（僅往內收縮）
    top_idx = np.argmax(proj_y[:max_shift * 2]) if np.any(proj_y[:max_shift * 2]) else max_shift
    bottom_idx = np.argmax(proj_y[-max_shift * 2:]) if np.any(proj_y[-max_shift * 2:]) else max_shift

    dy_top = -abs(top_idx - max_shift)  # 向下收
    dy_bottom = abs(bottom_idx - max_shift)  # 向上收

    new_y = int(y + max(dy_top, -max_shift))
    new_h = int(h - dy_bottom + dy_top)
    new_y = max(0, new_y)
    new_h = max(1, min(h_img - new_y, new_h))
    return (x, new_y, w, new_h)


# ------------------------------------------------------------
# 動態空白檢測參數（提高靈敏度）
# ------------------------------------------------------------
def dynamic_page_params(gray_page):
    mean_val = np.mean(gray_page)
    std_val = np.std(gray_page)

    params = {
        "std_thresh": 18,
        "union_ink_ratio_min": 0.008,  # ↓ 更靈敏
        "persistence_min": 0.45,       # ↓ 容許筆跡較淡
        "edge_density_min": 0.004,     # ↓ 容許稀疏筆劃
        "n_cc_min": 1,
        "max_cc_area_ratio_min": 0.003
    }

    if mean_val > 180:
        for k in ("union_ink_ratio_min", "persistence_min", "edge_density_min"):
            params[k] *= 0.7
    elif mean_val < 100:
        for k in ("union_ink_ratio_min", "persistence_min", "edge_density_min"):
            params[k] *= 1.2
    if std_val < 30:
        params["std_thresh"] = max(12, std_val * 0.7)
    return params


# ------------------------------------------------------------
# 嚴格對齊裁切（修正 refine 疊加問題）
# ------------------------------------------------------------
def safe_crop(image, px, py, pw, ph, w_img, h_img,
              median_w, median_h, y_global_bias=0,
              inset_ratio=0.022 , enable_refine=True):
    """
    嚴格對齊版（v3.5 StrictAligned）：
      - refine 與 y_global_bias 不重疊。
      - inset=2%，防止裁到格線。
      - 僅垂直修正；裁切範圍穩定。
    """
    px, py, pw, ph = int(px), int(py), int(pw), int(ph)
    inset = int(min(median_w, median_h) * inset_ratio)

    if enable_refine:
        px, py, pw, ph = refine_box_with_local_edges(image, (px, py, pw, ph))
        effective_bias = 0
    else:
        effective_bias = int(y_global_bias)

    py += effective_bias

    x1, y1 = max(0, px + inset), max(0, py + inset)
    x2, y2 = min(w_img, px + pw - inset), min(h_img, py + ph - inset)

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


# ------------------------------------------------------------
# 分欄（穩定分配 + 緩衝區擴增）
# ------------------------------------------------------------
def group_boxes_by_columns(first_row_boxes, practice_boxes, tol):
    col_centers = [bx + bw / 2 for (bx, by, bw, bh) in first_row_boxes]
    columns = [[] for _ in col_centers]
    for (x, y, w, h) in practice_boxes:
        cx = x + w / 2
        j = int(np.argmin([abs(cx - c) for c in col_centers]))
        if abs(cx - col_centers[j]) <= tol * 1.3:  # 放寬 30%
            columns[j].append((x, y, w, h))
    # 若落邊界者無歸類，強制指派最近列
    for (x, y, w, h) in practice_boxes:
        cx = x + w / 2
        assigned = any((x, y, w, h) in col for col in columns)
        if not assigned:
            j = int(np.argmin([abs(cx - c) for c in col_centers]))
            columns[j].append((x, y, w, h))
    for j in range(len(columns)):
        columns[j].sort(key=lambda b: b[1])
    return columns


# ------------------------------------------------------------
# Y 偏移估算（平滑化）
# ------------------------------------------------------------
def estimate_y_global_bias(grid_boxes, grid_h_mean):
    if len(grid_boxes) < 20:
        return 0
    centers_y = np.array([y + h / 2 for (_, y, _, h) in grid_boxes])
    centers_y = np.sort(centers_y)
    valid_centers = centers_y[int(len(centers_y) * 0.1):]
    diffs = np.diff(valid_centers)
    valid = diffs[(diffs > grid_h_mean * 0.6) & (diffs < grid_h_mean * 1.4)]
    if len(valid) < 5:
        return 0
    bias = (np.median(valid) - grid_h_mean) * 0.4
    return np.clip(bias, -3, 3)


# ------------------------------------------------------------
# 主分類儲存流程（修正版）
# ------------------------------------------------------------
def process_columns_and_save(image,
                             first_row_boxes,
                             practice_boxes,
                             final_labels,
                             output_dir,
                             char_counters,
                             page_idx=0):
    total_saved, total_blank, total_addr = 0, 0, 0
    incomplete_cols = []

    gray_page = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = dynamic_page_params(gray_page)
    # ↓ 放寬整欄空白門檻
    COLUMN_MIN_RATIO = 0.08 if np.mean(gray_page) > 160 else 0.12

    h_img, w_img = image.shape[:2]
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_overlay = image.copy() if DEBUG_VISUAL else None

    # 第一列（標籤）排除
    if first_row_boxes:
        y_bottom = max([y + h for (_, y, _, h) in first_row_boxes])
        guard = int(np.median([h for (_, _, _, h) in practice_boxes]) * 0.25)
        practice_boxes = [(x, y, w, h)
                          for (x, y, w, h) in practice_boxes if y > (y_bottom - guard)]

    # 全局 Y 偏移
    all_hs = [h for (_, _, _, h) in practice_boxes] or [80]
    median_h = np.median(all_hs)
    y_bias = estimate_y_global_bias(practice_boxes, median_h)

    # 分欄
    first_row_sorted = sorted(first_row_boxes, key=lambda b: b[0])
    all_ws = [w for (_, _, w, _) in practice_boxes] or [80]
    median_w = np.median(all_ws)
    col_tol = max(10, int(median_w * 0.5))
    practice_columns = group_boxes_by_columns(first_row_sorted, practice_boxes, tol=col_tol)

    # Step1: 檢查整欄空白
    for i, label in enumerate(final_labels):
        if label == "?" or i >= len(practice_columns):
            continue
        nonblank = 0
        for (px, py, pw, ph) in practice_columns[i]:
            roi = safe_crop(image, px, py, pw, ph, w_img, h_img,
                            median_w, median_h, y_global_bias=y_bias)
            if roi is None:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            if not is_grid_blank_dynamically(enhanced, **params):
                nonblank += 1
        ratio = nonblank / max(1, len(practice_columns[i]))
        if ratio < COLUMN_MIN_RATIO:
            final_labels[i] = "?"

    # Step2: 儲存格子
    for i, label in enumerate(final_labels):
        if label == "?" or i >= len(practice_columns):
            continue
        safe_dir = os.path.join(output_dir, sanitize_label(label))
        os.makedirs(safe_dir, exist_ok=True)
        char_counters.setdefault(label, 0)
        saved = 0
        for (px, py, pw, ph) in practice_columns[i]:
            roi = safe_crop(image, px, py, pw, ph, w_img, h_img,
                            median_w, median_h, y_global_bias=y_bias)
            if roi is None:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            if not is_grid_blank_dynamically(enhanced, **params):
                char_counters[label] += 1
                fname = f"{char_counters[label]:03d}.png"
                safe_imwrite(os.path.join(safe_dir, fname), roi)
                saved += 1
                if DEBUG_VISUAL:
                    cv2.rectangle(debug_overlay, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
            else:
                total_blank += 1
        if saved > 0:
            total_saved += saved
            total_addr += len(practice_columns[i])
            if saved < 10:
                incomplete_cols.append({"char": label, "count": saved})

    if DEBUG_VISUAL:
        for (x, y, w, h) in first_row_sorted:
            cv2.rectangle(debug_overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        path = os.path.join(DEBUG_DIR, f"debug_cut_page_{page_idx:03d}.png")
        safe_imwrite(path, debug_overlay)

    return {
        "handwriting_saved": total_saved,
        "blanks_skipped": total_blank,
        "addressable_grids": total_addr,
        "incomplete_columns": incomplete_cols
    }


def print_incomplete_report(incomplete_columns, page_name=None):
    if not incomplete_columns:
        print("✔ 所有欄位均達 10 筆以上。")
        return
    print("\n" + "=" * 50)
    print("⚠️ 低存量欄位報告 (<10)")
    for log in incomplete_columns:
        msg = f"{page_name or ''} | {log['char']} : {log['count']}/10"
        print(msg)
    print("=" * 50)
