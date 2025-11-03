# ================================================================
# classify.py — 自動對齊格心＋第一列不儲存版 (v2.1)
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
# 局部格線微校正
# ------------------------------------------------------------
def refine_box_with_local_edges(image, box, search_px=3):
    x, y, w, h = box
    h_img, w_img = image.shape[:2]

    # 🔒 邊界防護，避免越界導致空白裁切
    x1 = max(0, x - search_px)
    y1 = max(0, y - search_px)
    x2 = min(w_img, x + w + search_px)
    y2 = min(h_img, y + h + search_px)

    # 若無效範圍，直接返回原 box
    if x2 <= x1 or y2 <= y1:
        return box

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return box

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    # 防呆：確保至少有 3 行邊緣供分析
    if edges.shape[0] < search_px * 2:
        return box

    proj_y = np.sum(edges, axis=1)
    top = np.argmax(proj_y[:search_px * 2])
    bottom = h + np.argmax(proj_y[-search_px * 2:])
    ny = y + top - search_px
    nh = bottom - top

    if nh < 6:
        nh = h

    # 🔧 再次邊界修正，避免負值或越界
    ny = max(0, min(h_img - nh, ny))
    return (x, ny, w, nh)


# ------------------------------------------------------------
# 安全裁切（含全頁 y 偏移修正）
# ------------------------------------------------------------
def safe_crop(image, px, py, pw, ph, w_img, h_img,
              median_w, median_h, y_global_bias=0, inset_px=4, enable_refine=True):
    """
    安全裁切：保留 detect_grid 回傳的原圖座標，不再進行縮放。
    並根據全頁 y_global_bias 與局部 refine 進行微調。
    """
    # ❌ 不再縮放座標，因 detect_grid 已經回傳原圖比例
    px, py, pw, ph = int(px), int(py), int(pw), int(ph)
    py += int(y_global_bias)

    if enable_refine:
        px, py, pw, ph = refine_box_with_local_edges(image, (px, py, pw, ph))

    inset = int(min(median_h, median_w) * 0.07)
    x1, y1 = max(0, px + inset_px), max(0, py + inset)
    x2, y2 = min(w_img, px + pw - inset_px), min(h_img, py + ph - inset)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


# ------------------------------------------------------------
# 依標籤格中心分欄
# ------------------------------------------------------------
def group_boxes_by_columns(first_row_boxes, practice_boxes, tol):
    col_centers = [bx + bw / 2 for (bx, by, bw, bh) in first_row_boxes]
    columns = [[] for _ in col_centers]
    for (x, y, w, h) in practice_boxes:
        cx = x + w / 2
        j = int(np.argmin([abs(cx - c) for c in col_centers]))
        if abs(cx - col_centers[j]) <= tol:
            columns[j].append((x, y, w, h))
    for j in range(len(columns)):
        columns[j].sort(key=lambda b: b[1])
    return columns

# ------------------------------------------------------------
# 自動估算 detect_grid 偏移量
# ------------------------------------------------------------
def estimate_y_global_bias(grid_boxes, grid_h_mean):
    """以整頁格中心的 y 分布檢測全頁偏移方向"""
    if len(grid_boxes) < 20:
        return 0
    centers_y = np.array([y + h/2 for (_, y, _, h) in grid_boxes])
    diffs = np.diff(np.sort(centers_y))
    median_gap = np.median(diffs)
    # 理想 gap ≈ mean_h * 1.05~1.15
    expected = grid_h_mean * 1.08
    bias = (expected - median_gap) * 0.6  # 加權收斂
    return np.clip(bias, -10, 10)

# ------------------------------------------------------------
# 主分類儲存流程
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
    COLUMN_MIN_RATIO = 0.3

    h_img, w_img = image.shape[:2]
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_overlay = image.copy() if DEBUG_VISUAL else None

    # ------------------- 第一列過濾 -------------------
    if first_row_boxes:
        y_bottom = max([y + h for (_, y, _, h) in first_row_boxes])
        guard = int(np.median([h for (_, _, _, h) in practice_boxes]) * 0.25)
        practice_boxes = [(x, y, w, h) for (x, y, w, h) in practice_boxes if y > (y_bottom - guard)]

    # ------------------- Y 偏移自動修正 -------------------
    all_hs = [h for (_, _, _, h) in practice_boxes] or [80]
    median_h = np.median(all_hs)
    y_bias = estimate_y_global_bias(practice_boxes, median_h)

    # ------------------- 分欄 -------------------
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
            roi = safe_crop(image, px, py, pw, ph, w_img, h_img, median_w, median_h, y_global_bias=y_bias)
            if roi is None:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if not is_grid_blank_dynamically(gray):
                nonblank += 1
        ratio = nonblank / max(1, len(practice_columns[i]))
        if ratio < COLUMN_MIN_RATIO:
            final_labels[i] = "?"

    # Step2: 儲存練習格（不含第一列）
    for i, label in enumerate(final_labels):
        if label == "?" or i >= len(practice_columns):
            continue
        safe_dir = os.path.join(output_dir, sanitize_label(label))
        os.makedirs(safe_dir, exist_ok=True)
        char_counters.setdefault(label, 0)
        saved = 0
        for (px, py, pw, ph) in practice_columns[i]:
            roi = safe_crop(image, px, py, pw, ph, w_img, h_img, median_w, median_h, y_global_bias=y_bias)
            if roi is None:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if not is_grid_blank_dynamically(gray):
                char_counters[label] += 1
                fname = f"{char_counters[label]:03d}.png"
                safe_imwrite(os.path.join(safe_dir, fname), roi)
                saved += 1
                if DEBUG_VISUAL:
                    cv2.rectangle(debug_overlay, (px, py), (px+pw, py+ph), (0,255,0), 2)
            else:
                total_blank += 1
        if saved > 0:
            total_saved += saved
            total_addr += len(practice_columns[i])
            if saved < 10:
                incomplete_cols.append({"char": label, "count": saved})

    # Step3: debug 視覺化
    if DEBUG_VISUAL:
        for (x, y, w, h) in first_row_sorted:
            cv2.rectangle(debug_overlay, (x, y), (x+w, y+h), (0,0,255), 2)
        path = os.path.join(DEBUG_DIR, f"debug_cut_page_{page_idx:03d}.png")
        safe_imwrite(path, debug_overlay)

    return {
        "handwriting_saved": total_saved,
        "blanks_skipped": total_blank,
        "addressable_grids": total_addr,
        "incomplete_columns": incomplete_cols
    }

# ------------------------------------------------------------
def print_incomplete_report(incomplete_columns, page_name=None):
    if not incomplete_columns:
        print("✔ 所有欄位均達 10 筆以上。")
        return
    print("\n" + "="*50)
    print("⚠️ 低存量欄位報告 (<10)")
    for log in incomplete_columns:
        msg = f"{page_name or ''} | {log['char']} : {log['count']}/10"
        print(msg)
    print("="*50)
