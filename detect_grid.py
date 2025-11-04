# ================================================================
# detect_grid_v4.0_CenterLock.py
# ================================================================
# 主要改進：
#   1) CenterLock：以「每列/每行的筆跡質心」重建全域網格 → 消除系統性偏移/比例誤差
#   2) 比例一致：所有通道在「縮放偵測」→「按比例回寫到原圖」完整一致
#   3) 光照補償 + 自適應二值 + ROI 擴張 + 安全峰值偵測
#   4) 三通道（Projection/Contours/Hough）擇優 + 形狀穩定性評分
#   5) BayesOpt（可選）自動微調參數，保底隨機搜尋 fallback
#   6) 可視化：debug_centerlock/overlay_*.png 與 matplotlib 顯示
#
# 用法：
#   python detect_grid_v4.0_CenterLock.py --file 003.png
#   python detect_grid_v4.0_CenterLock.py --sample 10               (批次抽樣)
#   python detect_grid_v4.0_CenterLock.py --disable-bo              (關閉 BayesOpt)
#   python detect_grid_v4.0_CenterLock.py --mincov 92               (覆蓋率門檻)
# ================================================================

import os
import cv2
import sys
import json
import math
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ------------------------------------------------
# 兼容 config.py（若無則提供預設值，方便單檔測）
# ------------------------------------------------
try:
    from config import (
        DATA_DIR,
        TARGET_NAME,
        SCALE_FACTOR,
        EXPECTED_ROWS,
        EXPECTED_COLS,
        GRIDS_PER_PAGE_THEORY,
    )
except Exception:
    print("⚠️ 找不到 config.py，使用 fallback 參數（僅供單檔測試）")
    DATA_DIR = "./data"
    TARGET_NAME = "target"
    SCALE_FACTOR = 1.25
    EXPECTED_ROWS = 9
    EXPECTED_COLS = 11
    GRIDS_PER_PAGE_THEORY = EXPECTED_ROWS * EXPECTED_COLS

# ------------------------------------------------
# 全域常數與路徑
# ------------------------------------------------
PARAM_LOG_PATH = os.path.join(DATA_DIR, "grid_params_log.json")
DEBUG_DIR = "debug_centerlock"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 預設參數（會成為 BayesOpt 的起點）
DEFAULT_PARAMS = {
    "min_area": 40000,
    "max_area": 70000,
    "min_ratio": 0.90,
    "max_ratio": 1.10,
    "cluster_thresh": 40,
}

# BayesOpt 邊界
BO_BOUNDS = {
    "min_area": (35000, 65000),
    "max_area": (40000, 85000),
    "min_ratio": (0.80, 1.00),
    "max_ratio": (1.00, 1.30),
    "cluster_thresh": (20, 60),
}

# 嘗試導入 bayesian-optimization
try:
    from bayes_opt import BayesianOptimization
    _BAYES_OPT_AVAILABLE = True
except Exception:
    _BAYES_OPT_AVAILABLE = False


# ================================================================
# 小工具
# ================================================================
def now_ts():
    return time.strftime("%H:%M:%S")


def log(s):
    print(f"[{now_ts()}] {s}")


def safe_clip_int(arr, lo, hi):
    arr = np.asarray(arr).astype(np.float32)
    return np.clip(arr, lo, hi).astype(int)


def list_to_int_tuples(lst):
    out = []
    for t in lst:
        x, y, w, h = t
        out.append((int(x), int(y), int(w), int(h)))
    return out

# =====================================
# 輔助函式
# =====================================

def smooth_projection(arr, k=5):
    """移動平均平滑化，防止 gradient 長度過短"""
    if len(arr) < k:
        return np.pad(arr, (0, k - len(arr)), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode="same")

# ================================================================
# 影像前處理
# ================================================================
def apply_illumination_correction(gray):
    """
    CLAHE + 大核模糊估算背景 + 加權扣除，平衡陰影/亮區影響
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    background = cv2.GaussianBlur(clahe_img, (81, 81), 0)
    corrected = cv2.addWeighted(clahe_img, 1.25, background, -0.25, 0)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def safe_adaptive_binary(gray_eq, block_size=23, C=10):
    """
    針對較噪的頁面安全的自適應二值化
    """
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    bin_img = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )
    return bin_img


def detect_page_roi(binary, expand_x_ratio=0.03, expand_y_ratio=0.05):
    """
    從二值影像抓最大輪廓作為頁面 ROI，並做比例擴張
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    expand_x = int(bw * expand_x_ratio)
    expand_y = int(bh * expand_y_ratio)
    sx = max(0, bx - expand_x)
    sy = max(0, by - expand_y)
    sw = min(binary.shape[1] - sx, bw + 2 * expand_x)
    sh = min(binary.shape[0] - sy, bh + 2 * expand_y)
    return sx, sy, sw, sh


# ================================================================
# 安全峰值偵測（投影）
# ================================================================
def robust_find_peaks(proj, dist, thr_ratio=0.3 ):
    """
    對投影曲線做：長度檢查→梯度→門檻→距離約束
    """
    if proj is None:
        return []

    proj = np.array(proj, dtype=np.float32).flatten()
    if proj.size < 4:
        return []

    # 避免全零或極低對比
    if np.max(proj) <= 1:
        return []

    # 安全梯度
    try:
        grad = np.gradient(proj)
    except Exception as e:
        log(f"⚠️ robust_find_peaks() 梯度失敗: {e}")
        return []

    t = np.max(proj) * thr_ratio
    out = []
    for i in range(2, len(proj) - 2):
        if proj[i] > t and grad[i - 1] > 0 and grad[i + 1] < 0:
            if not out or (i - out[-1]) > dist:
                out.append(i)
    return out


# ================================================================
# Hough 線微調（柔性貼合）
# ================================================================
def refine_lines_with_hough(binary, x_coords, y_coords, max_shift=7, roi_offset=(0, 0)):
    """
    以 HoughLinesP 擷取近似水平/垂直的線群，對候選座標做「就近貼合」。
    不調整步距，只做方向與位置的小幅修正。
    """
    edges = cv2.Canny(binary, 70, 160)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=90,
        minLineLength=int(min(binary.shape) * 0.45),
        maxLineGap=60
    )
    if lines is None:
        return x_coords, y_coords

    horiz, vert = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if ang < 10 or ang > 170:
            horiz.append(int((y1 + y2) / 2))
        elif 80 < ang < 100:
            vert.append(int((x1 + x2) / 2))

    horiz = sorted(horiz)
    vert = sorted(vert)

    def snap(base, candidates):
        base = list(base)
        if not candidates:
            return base
        out = []
        for c in base:
            near = min(candidates, key=lambda d: abs(d - c))
            out.append(int(near) if abs(near - c) <= max_shift else int(c))
        return out

    x_aligned = snap(x_coords, vert)
    y_aligned = snap(y_coords, horiz)

    # 若 ROI 有偏移，這裡不直接加回去，因為主調用端（建盒）會帶入 ROI 原點
    return x_aligned, y_aligned


# ================================================================
# CenterLock：以筆跡質心重建全域網格（內建 SafeShrink）
# ================================================================
def centerlock_reconstruct(boxes, binary_full, rows=EXPECTED_ROWS, cols=EXPECTED_COLS, shrink_ratio=0.06 ):
    """
    CenterLock
    --------------------------------------------------
    改進要點：
    - 根據筆跡質心計算全域偏移（修正整頁左上/右下漂移）
    - 維持行列比例穩定
    - 局部筆跡分佈仍平滑對齊（不破壞整體方陣）
    """
    if not boxes:
        return []

    H, W = binary_full.shape[:2]
    centers_x = np.zeros((rows, cols), dtype=np.float32)
    centers_y = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=bool)

    # === 計算各格筆跡質心 (Center of Ink)
    k = 0
    for i in range(rows):
        for j in range(cols):
            if k >= len(boxes):
                break
            x, y, w, h = boxes[k]
            k += 1
            roi = binary_full[y:y + h, x:x + w]
            if roi.size <= 0:
                continue
            m = cv2.moments(roi)
            if m["m00"] > 0:
                cx = x + (m["m10"] / m["m00"])
                cy = y + (m["m01"] / m["m00"])
                centers_x[i, j] = cx
                centers_y[i, j] = cy
                valid[i, j] = True
            else:
                centers_x[i, j] =  x + w/2 
                centers_y[i, j] =  y + h/2

    # === 平均行/列中心
    mean_y_per_row = np.array(
        [np.nanmean(centers_y[i, :]) if np.any(valid[i, :]) else np.nan for i in range(rows)]
    )
    mean_x_per_col = np.array(
        [np.nanmean(centers_x[:, j]) if np.any(valid[:, j]) else np.nan for j in range(cols)]
    )

    # === NaN 補值
    def nan_fill(v):
        idx = np.arange(len(v))
        good = ~np.isnan(v)
        if not np.any(good):
            return np.linspace(0, len(v) - 1, len(v))
        return np.interp(idx, idx[good], v[good])
    mean_y_per_row = nan_fill(mean_y_per_row)
    mean_x_per_col = nan_fill(mean_x_per_col)

    # === 平滑處理（消除局部鋸齒）
    mean_y_per_row = cv2.GaussianBlur(mean_y_per_row.reshape(-1, 1), (5, 1), 0).flatten()
    mean_x_per_col = cv2.GaussianBlur(mean_x_per_col.reshape(-1, 1), (5, 1), 0).flatten()

    # === 全域格距
    dy_global = np.median(np.diff(mean_y_per_row)) if rows > 1 else 0
    dx_global = np.median(np.diff(mean_x_per_col)) if cols > 1 else 0

    # === 局部偏移曲線（行列方向）
    adj_y = mean_y_per_row - np.linspace(mean_y_per_row[0], mean_y_per_row[-1], rows)
    adj_x = mean_x_per_col - np.linspace(mean_x_per_col[0], mean_x_per_col[-1], cols)
    adj_y = cv2.GaussianBlur(adj_y.reshape(-1, 1), (5, 1), 0).flatten()
    adj_x = cv2.GaussianBlur(adj_x.reshape(-1, 1), (5, 1), 0).flatten()

    # === 全域筆跡偏移 (Global Bias Correction)
    #     以所有筆跡中心的平均，對齊整頁格中心
    all_valid_x = centers_x[valid]
    all_valid_y = centers_y[valid]
    if len(all_valid_x) > 10:
        ink_center_x = np.mean(all_valid_x)
        ink_center_y = np.mean(all_valid_y)
        grid_center_x = np.mean([b[0] + b[2] / 3 for b in boxes])
        grid_center_y = np.mean([b[1] + b[3] / 3 for b in boxes])
        global_shift_x = (ink_center_x - grid_center_x) * 0.80
        global_shift_y = (ink_center_y - grid_center_y) * 0.80
    else:
        global_shift_x = 0
        global_shift_y = 0

    # === 邊界重建（含全域偏移 + 邊界安全鎖）
    y_start = max(0, mean_y_per_row[0] - dy_global / 2 + global_shift_y)
    y_end   = min(H - 1, mean_y_per_row[-1] + dy_global / 2 + global_shift_y)
    x_start = max(0, mean_x_per_col[0] - dx_global / 2 + global_shift_x)
    x_end   = min(W - 1, mean_x_per_col[-1] + dx_global / 2 + global_shift_x)

    y_edges = np.linspace(y_start, y_end, rows + 1)
    x_edges = np.linspace(x_start, x_end, cols + 1)

    # === 局部筆跡平滑修正
    max_shift_ratio = 0.60 
    for i in range(1, rows):
        shift = np.clip(adj_y[i - 1] * 0.30 , -dy_global * max_shift_ratio, dy_global * max_shift_ratio)
        y_edges[i] += shift
    for j in range(1, cols):
        shift = np.clip(adj_x[j - 1] * 0.30 , -dx_global * max_shift_ratio, dx_global * max_shift_ratio)
        x_edges[j] += shift

    # === 生成格框
    out = []
    for i in range(rows):
        for j in range(cols):
            y1, y2 = y_edges[i], y_edges[i + 1]
            x1, x2 = x_edges[j], x_edges[j + 1]
            box_w, box_h = (x2 - x1), (y2 - y1)
            shrink_w = box_w * shrink_ratio
            shrink_h = box_h * shrink_ratio
            x1 += shrink_w; x2 -= shrink_w
            y1 += shrink_h; y2 -= shrink_h

            x1 = max(0, min(W - 2, x1))
            y1 = max(0, min(H - 2, y1))
            x2 = max(x1 + 1, min(W - 1, x2))
            y2 = max(y1 + 1, min(H - 1, y2))
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return out



# ================================================================
# 由座標網格建盒（內建 SafeShrink）
# ================================================================
def build_boxes_from_coords(x_coords, y_coords, roi_xy, scale_wh=(1.0, 1.0), shrink_ratio = 0.06 ):
    """
    x_coords/y_coords：ROI 內部邊界座標（長度 = cols+1 / rows+1）
    roi_xy: (x,y) ROI 左上角
    scale_wh: (w_ratio, h_ratio) 回寫到原圖的比例
    shrink_ratio: 四周裁切比例（預設 8%）
    """
    rx, ry = roi_xy
    w_ratio, h_ratio = scale_wh
    boxes = []
    for i in range(EXPECTED_ROWS):
        for j in range(EXPECTED_COLS):
            y1, y2 = y_coords[i], y_coords[i + 1]
            x1, x2 = x_coords[j], x_coords[j + 1]

            X1 = (rx + x1) * w_ratio
            Y1 = (ry + y1) * h_ratio
            W = (x2 - x1) * w_ratio
            H = (y2 - y1) * h_ratio

            # === 🔹 加入 shrink（減少邊線）===
            shrink_w = W * shrink_ratio
            shrink_h = H * shrink_ratio
            X1 += shrink_w
            Y1 += shrink_h
            W -= 2 * shrink_w
            H -= 2 * shrink_h

            boxes.append((int(X1), int(Y1), int(W), int(H)))
    return boxes


# ================================================================
# 三通道：Projection
# ================================================================
def find_grid_boxes_by_projection(image_scaled, params, orig_size=None):
    """
    在「縮放後影像」上執行，最後按比例回寫原圖座標。
    """
    Hs, Ws = image_scaled.shape[:2]
    if orig_size is None:
        Ho, Wo = Hs, Ws
    else:
        Ho, Wo = orig_size

    w_ratio = Wo / Ws
    h_ratio = Ho / Hs

    gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    gray_eq = apply_illumination_correction(gray)
    binary = safe_adaptive_binary(gray_eq, block_size=21, C=8)

    roi_rect = detect_page_roi(binary, expand_x_ratio=0.03, expand_y_ratio=0.05)
    if roi_rect is None:
        return []
    rx, ry, rw, rh = roi_rect
    roi = binary[ry:ry + rh, rx:rx + rw].copy()

    # 投影
    horz_proj = cv2.GaussianBlur(np.sum(roi, axis=1).astype(np.float32), (1, 11), 0)
    vert_proj = cv2.GaussianBlur(np.sum(roi, axis=0).astype(np.float32), (11, 1), 0)

    avg_side = np.sqrt((params["min_area"] + params["max_area"]) / 2.0) ** 0.5
    # 調整 dist 使峰間距合理
    dist_y = int(max(6, min(rh // (EXPECTED_ROWS + 1), 100)))
    dist_x = int(max(6, min(rw // (EXPECTED_COLS + 1), 100)))

    y_peaks = robust_find_peaks(horz_proj, dist=dist_y, thr_ratio=0.35)
    x_peaks = robust_find_peaks(vert_proj, dist=dist_x, thr_ratio=0.35)

    # 均勻化（長度 = rows+1/cols+1）
    def regularize(coords, expect, length):
        coords = sorted(set(coords))
        if len(coords) < 2:
            return np.linspace(0, length - 1, expect + 1, dtype=int).tolist()
        start, end = coords[0], coords[-1]
        step = (end - start) / expect
        reg = [int(start + i * step) for i in range(expect + 1)]
        reg = safe_clip_int(reg, 0, length - 1).tolist()
        return reg

    y_coords = regularize(y_peaks, EXPECTED_ROWS, rh)
    x_coords = regularize(x_peaks, EXPECTED_COLS, rw)

    # Hough 微調（在 ROI 內做）
    x_coords_ref, y_coords_ref = refine_lines_with_hough(roi, x_coords, y_coords, max_shift=7)
    x_coords = x_coords_ref
    y_coords = y_coords_ref

    # 先以投影座標建盒（縮放後 → 原圖）
    boxes = build_boxes_from_coords(x_coords, y_coords, (rx, ry), (w_ratio, h_ratio))

    # 在「原圖尺度」下做 CenterLock 重建（binary_full 必須是原圖尺寸）

    gray_o = cv2.cvtColor(cv2.resize(image_scaled, (Wo, Ho)), cv2.COLOR_BGR2GRAY)
    gray_o_eq = apply_illumination_correction(gray_o)

    binary_full = safe_adaptive_binary(gray_o_eq, block_size=21, C=8)

    boxes = centerlock_reconstruct(boxes, binary_full, rows=EXPECTED_ROWS, cols=EXPECTED_COLS)

    log(f"📏 CenterLock-Proj 完成: {len(boxes)} 格")
    return list_to_int_tuples(boxes)


# ================================================================
# 三通道：Contours
# ================================================================
def find_grid_boxes_by_contours(image_scaled, params, orig_size=None):
    Hs, Ws = image_scaled.shape[:2]
    if orig_size is None:
        Ho, Wo = Hs, Ws
    else:
        Ho, Wo = orig_size

    w_ratio = Wo / Ws
    h_ratio = Ho / Hs

    gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 27, 23
    )
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    eroded = cv2.erode(thresh, erode_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
    fixed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(fixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / h if h > 0 else 0
        if (params["min_area"] < area < params["max_area"]) and (params["min_ratio"] < ratio < params["max_ratio"]):
            candidates.append((x, y, w, h))

    if not candidates:
        return []

    # 用候選 boxes 據此推估網格邊界（簡易：取所有 x/y 起訖的唯一集合）
    xs = sorted(set([b[0] for b in candidates] + [b[0] + b[2] for b in candidates]))
    ys = sorted(set([b[1] for b in candidates] + [b[1] + b[3] for b in candidates]))

    # 若過多，取等距 subsample 到 rows+1 / cols+1
    def resample_to(arr, target):
        if len(arr) < 2:
            return np.linspace(0, (Hs if arr is ys else Ws) - 1, target, dtype=int).tolist()
        idxs = np.linspace(0, len(arr) - 1, target, dtype=int)
        return [int(arr[i]) for i in idxs]

    x_coords = resample_to(xs, EXPECTED_COLS + 1)
    y_coords = resample_to(ys, EXPECTED_ROWS + 1)

    # 回寫到原圖座標
    boxes = build_boxes_from_coords(x_coords, y_coords, (0, 0), (w_ratio, h_ratio))

    # CenterLock 重建
    gray_o = cv2.cvtColor(cv2.resize(image_scaled, (Wo, Ho)), cv2.COLOR_BGR2GRAY)
    gray_o_eq = apply_illumination_correction(gray_o)
    binary_full = safe_adaptive_binary(gray_o_eq, block_size=21, C=8)
    boxes = centerlock_reconstruct(boxes, binary_full, rows=EXPECTED_ROWS, cols=EXPECTED_COLS)

    log(f"📏 CenterLock-Contours 完成: {len(boxes)} 格")
    return list_to_int_tuples(boxes)


# ================================================================
# 三通道：Hough
# ================================================================
def find_grid_boxes_by_hough(image_scaled, params, orig_size=None):
    Hs, Ws = image_scaled.shape[:2]
    if orig_size is None:
        Ho, Wo = Hs, Ws
    else:
        Ho, Wo = orig_size

    w_ratio = Wo / Ws
    h_ratio = Ho / Hs

    gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(denoised, 25, 80, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=110,
        minLineLength=int(image_scaled.shape[0] // 4),
        maxLineGap=50,
    )
    if lines is None:
        return []

    # 聚類線位以得到邊界候選
    def cluster_lines(positions, threshold=params["cluster_thresh"]):
        if not positions:
            return []
        positions = sorted(positions)
        clusters = []
        group = [positions[0]]
        for p in positions[1:]:
            if abs(p - group[-1]) < threshold:
                group.append(p)
            else:
                clusters.append(int(np.mean(group)))
                group = [p]
        clusters.append(int(np.mean(group)))
        return clusters

    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) if (x2 - x1) != 0 else 90
        if ang < 15 or ang > 165:
            horiz.append(int((y1 + y2) / 2))
        elif 75 < ang < 105:
            vert.append(int((x1 + x2) / 2))

    h_lines = cluster_lines(horiz)
    v_lines = cluster_lines(vert)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # 取得 rows+1 / cols+1 個邊界
    def to_edges(lines, target, length):
        lines = sorted(lines)
        if len(lines) >= target:
            idx = np.linspace(0, len(lines) - 1, target, dtype=int)
            out = [lines[i] for i in idx]
        else:
            out = np.linspace(0, length - 1, target, dtype=int).tolist()
        return out

    y_coords = to_edges(h_lines, EXPECTED_ROWS + 1, Hs)
    x_coords = to_edges(v_lines, EXPECTED_COLS + 1, Ws)

    # 回寫到原圖座標
    boxes = build_boxes_from_coords(x_coords, y_coords, (0, 0), (w_ratio, h_ratio))

    # CenterLock 重建
    gray_o = cv2.cvtColor(cv2.resize(image_scaled, (Wo, Ho)), cv2.COLOR_BGR2GRAY)
    gray_o_eq = apply_illumination_correction(gray_o)
    binary_full = safe_adaptive_binary(gray_o_eq, block_size=21, C=8)
    boxes = centerlock_reconstruct(boxes, binary_full, rows=EXPECTED_ROWS, cols=EXPECTED_COLS)

    log(f"📏 CenterLock-Hough 完成: {len(boxes)} 格")
    return list_to_int_tuples(boxes)


# ================================================================
# 品質評估與三通道擇優
# ================================================================
def evaluate_grid_boxes(grid_boxes, expected_count=GRIDS_PER_PAGE_THEORY):
    """
    分數 = 70% 格數接近度 + 30% 形狀穩定（寬/高 變異係數越小越好）
    """
    if not grid_boxes:
        return -1.0
    n = len(grid_boxes)
    score_count = max(0.0, 1.0 - abs(n - expected_count) / max(1.0, expected_count))
    widths = [w for (_, _, w, h) in grid_boxes]
    heights = [h for (_, _, w, h) in grid_boxes]
    if not widths or not heights or np.mean(widths) <= 1e-6 or np.mean(heights) <= 1e-6:
        return score_count
    w_cv = np.std(widths) / (np.mean(widths) + 1e-6)
    h_cv = np.std(heights) / (np.mean(heights) + 1e-6)
    score_shape = max(0.0, 1.0 - (w_cv + h_cv))
    return 0.7 * score_count + 0.3 * score_shape


def find_grid_boxes_with_params(image_scaled, params, original_size=None):
    """
    在「縮放後」影像上跑 3 通道，統一以 original_size 比例回寫。
    """
    best_boxes, best_method, best_score = [], "None", -999.0
    methods = {
        "Projection": find_grid_boxes_by_projection,
        "Contours": find_grid_boxes_by_contours,
        "Hough": find_grid_boxes_by_hough,
    }

    for name, func in methods.items():
        try:
            boxes = func(image_scaled, params, orig_size=original_size)
        except Exception as e:
            log(f"⚠️ {name} 失敗: {e}")
            boxes = []
        score = evaluate_grid_boxes(boxes, expected_count=GRIDS_PER_PAGE_THEORY)
        log(f"🔎 {name}: boxes={len(boxes)}, score={score:.3f}")
        if score > best_score:
            best_boxes, best_method, best_score = boxes, name, score

    # 防呆：若格數異常，回退 Contours
    if len(best_boxes) < 50 or len(best_boxes) > 120:
        log(f"⚠️ {best_method} 偵測格數異常 ({len(best_boxes)}), 回退 Contours")
        try:
            best_boxes = find_grid_boxes_by_contours(image_scaled, params, orig_size=original_size)
            best_method = "Contours (fallback)"
            best_score = evaluate_grid_boxes(best_boxes)
        except Exception as e:
            log(f"❌ Contours fallback 失敗: {e}")
            best_boxes = []

    return best_boxes, best_method, best_score


# ================================================================
# BayesOpt 自動調參（可選）
# ================================================================
def optimize_params(image_scaled, expected_grids):
    """
    調參目標：格數接近 + 形狀穩定
    """
    from random import uniform

    def eval_with(p):
        boxes, _, _ = find_grid_boxes_with_params(image_scaled, p, original_size=None)
        n = len(boxes)
        score = -abs(n - expected_grids) / max(1.0, expected_grids)
        if n > 5:
            ws = [b[2] for b in boxes]
            hs = [b[3] for b in boxes]
            w_cv = np.std(ws) / (np.mean(ws) + 1e-6)
            h_cv = np.std(hs) / (np.mean(hs) + 1e-6)
            score -= (w_cv + h_cv) * 0.3
        return score

    if _BAYES_OPT_AVAILABLE:
        try:
            def black_box(min_area, max_area, min_ratio, max_ratio, cluster_thresh):
                p = {
                    "min_area": float(min_area),
                    "max_area": float(max_area),
                    "min_ratio": float(min_ratio),
                    "max_ratio": float(max_ratio),
                    "cluster_thresh": float(cluster_thresh),
                }
                return float(eval_with(p))

            bo = BayesianOptimization(f=black_box, pbounds=BO_BOUNDS, random_state=42, verbose=0)
            bo.maximize(init_points=4, n_iter=10)
            tuned = {k: float(v) for k, v in bo.max["params"].items()}
            log(f"🔧 Bayesian 最佳參數: {tuned}")
            return tuned
        except Exception as e:
            log(f"⚠️ Bayesian 調參失敗，改用隨機搜尋: {e}")

    # fallback：隨機搜尋
    best_p, best_s = None, -999
    for _ in range(10):
        p = {
            "min_area": uniform(45000, 70000),
            "max_area": uniform(50000, 80000),
            "min_ratio": uniform(0.8, 1.0),
            "max_ratio": uniform(1.0, 1.3),
            "cluster_thresh": uniform(15, 60),
        }
        s = eval_with(p)
        if s > best_s:
            best_s, best_p = s, p
    log(f"🎯 隨機搜尋最優參數: {best_p}")
    return best_p or DEFAULT_PARAMS


# ================================================================
# 封裝主入口
# ================================================================
def find_grid_boxes(image, expected_grids=GRIDS_PER_PAGE_THEORY, mincov=90.0, enable_bo=True):
    """
    image: 可為 ndarray 或 {page_key, image} 結構
    回傳：boxes（原圖座標）
    """
    if isinstance(image, SimpleNamespace):
        page_key = getattr(image, "page_key", "inline_image")
        img = getattr(image, "image", None)
    elif isinstance(image, dict):
        page_key = image.get("page_key", "inline_image")
        img = image.get("image", None)
    else:
        page_key = "inline_image"
        img = image

    if img is None:
        log("❌ find_grid_boxes: invalid image input")
        return []

    orig_h, orig_w = img.shape[:2]
    scaled_img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    log(f"🔍 Scale={SCALE_FACTOR:.2f} | original={orig_w}x{orig_h} | scaled={scaled_img.shape[1]}x{scaled_img.shape[0]}")

    # 載入/準備參數 cache
    cache = {}
    if os.path.exists(PARAM_LOG_PATH):
        try:
            with open(PARAM_LOG_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            log(f"⚠️ 參數快取讀取失敗: {e}")

    params = cache.get(page_key, DEFAULT_PARAMS.copy())

    boxes, method, score = find_grid_boxes_with_params(scaled_img, params, original_size=(orig_h, orig_w))
    coverage = len(boxes) / max(1.0, expected_grids) * 100.0
    abnormal = (len(boxes) < 80 or len(boxes) > 110 or coverage < mincov)

    if enable_bo and abnormal:
        log(f"⚠️ [{page_key}] 偵測異常 (n={len(boxes)}, cov={coverage:.2f}%) → 啟動調參")
        tuned = optimize_params(scaled_img, expected_grids)
        cache[page_key] = tuned
        try:
            with open(PARAM_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"⚠️ 寫入參數快取失敗: {e}")
        boxes, method, score = find_grid_boxes_with_params(scaled_img, tuned, original_size=(orig_h, orig_w))
        coverage = len(boxes) / max(1.0, expected_grids) * 100.0
        if len(boxes) < 80 or len(boxes) > 110:
            # 最後再嘗試單通道 Hough 當保底
            log("⚠️ 仍異常，Hough 保底")
            try:
                boxes = find_grid_boxes_by_hough(scaled_img, tuned, orig_size=(orig_h, orig_w))
                method = "Hough (fallback)"
            except Exception as e:
                log(f"❌ Hough fallback 失敗: {e}")
    log(f"✅ [{page_key}] 使用 {method} | 共 {len(boxes)} 格 | 覆蓋率 {coverage:.2f}%")
    return boxes


# ================================================================
# 視覺化
# ================================================================
def visualize_overlay(img, boxes, title="overlay", save_path=None, show=True, line_th=2):
    overlay = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), line_th)

    if save_path:
        cv2.imwrite(save_path, overlay)
        log(f"🟢 overlay 儲存於: {save_path}")

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()


# ================================================================
# 單張/批次 測試
# ================================================================
def test_single_image(img_path, expected_grids=GRIDS_PER_PAGE_THEORY, visualize=True, mincov=90.0, enable_bo=True):
    img = cv2.imread(img_path)
    if img is None:
        log(f"❌ 無法讀取影像: {img_path}")
        return None

    boxes = find_grid_boxes(
        image={"page_key": os.path.basename(img_path), "image": img},
        expected_grids=expected_grids,
        mincov=mincov,
        enable_bo=enable_bo,
    )
    coverage = len(boxes) / max(1.0, expected_grids) * 100.0
    result = {
        "path": img_path,
        "image": img,
        "boxes": boxes,
        "detected": len(boxes),
        "coverage": coverage,
        "method": "auto",
    }

    if visualize:
        out_path = os.path.join(DEBUG_DIR, f"overlay_{os.path.basename(img_path)}")
        visualize_overlay(img, boxes, title=f"{os.path.basename(img_path)} — {len(boxes)} grids ({coverage:.2f}%)",
                          save_path=out_path, show=True)
    return result


def test_batch_images(dir_path, png_list, sample=5, mincov=90.0, enable_bo=True, visualize=False):
    sel = random.sample(png_list, min(sample, len(png_list)))
    results = []
    for name in sel:
        p = os.path.join(dir_path, name)
        r = test_single_image(p, GRIDS_PER_PAGE_THEORY, visualize=visualize,
                              mincov=mincov, enable_bo=enable_bo)
        if r:
            results.append(r)
    if not results:
        log("⚠️ 無有效測試結果")
        return

    avg_cov = float(np.mean([r["coverage"] for r in results]))
    worst = min(results, key=lambda r: r["coverage"])
    log(f"📈 平均格子覆蓋率: {avg_cov:.2f}%")
    log(f"📉 最差頁面: {os.path.basename(worst['path'])} ({worst['coverage']:.2f}%)")


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="CenterLock v4.0 格子偵測（單張/批次 + 自動調參）")
    parser.add_argument("--file", type=str, default=None, help="指定單張檔名（優先）")
    parser.add_argument("--sample", type=int, default=1, help="批次抽樣張數（>1 啟用批次）")
    parser.add_argument("--mincov", type=float, default=90.0, help="覆蓋率門檻（低於則視為異常）")
    parser.add_argument("--disable-bo", action="store_true", help="關閉 Bayesian Optimization")
    parser.add_argument("--no-show", action="store_true", help="不顯示 matplotlib 視窗（仍輸出 overlay）")
    args = parser.parse_args()

    target_dir = os.path.join(DATA_DIR, TARGET_NAME)
    if not os.path.isdir(target_dir):
        log(f"❌ 找不到資料夾 {target_dir}")
        sys.exit(1)

    pngs = [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
    if not pngs:
        log(f"⚠️ {target_dir} 下沒有 PNG 檔")
        sys.exit(1)

    # 單張優先
    if args.file:
        img_path = os.path.join(target_dir, args.file)
        log(f"\n🎯 單檔格子偵測測試：{args.file}")
        res = test_single_image(
            img_path,
            GRIDS_PER_PAGE_THEORY,
            visualize=not args.no_show,
            mincov=args.mincov,
            enable_bo=not args.disable_bo,
        )
        if res:
            log(f"✅ 最終採用 auto，格數 {res['detected']}，覆蓋率 {res['coverage']:.2f}%")
        return

    # 無指定檔名 → 依 sample 走單張 / 批次
    if args.sample <= 1:
        chosen = random.choice(pngs)
        chosen = "012.png"  # for debug

        img_path = os.path.join(target_dir, chosen)
        log(f"\n🎯 單檔格子偵測測試：{chosen}")
        res = test_single_image(
            img_path,
            GRIDS_PER_PAGE_THEORY,
            visualize=not args.no_show,
            mincov=args.mincov,
            enable_bo=not args.disable_bo,
        )
        if res:
            log(f"✅ 最終採用 auto，格數 {res['detected']}，覆蓋率 {res['coverage']:.2f}%")
    else:
        log(f"\n📊 批次格子偵測測試模式：隨機抽樣 {args.sample} 張 PNG")
        test_batch_images(
            target_dir,
            png_list=pngs,
            sample=args.sample,
            mincov=args.mincov,
            enable_bo=not args.disable_bo,
            visualize=False,
        )


if __name__ == "__main__":
    main()
