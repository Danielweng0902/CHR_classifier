# ================================================================
# detect_grid.py — 格子偵測模組 (Bayesian Optimization + Log + Fallback + 精準回縮)
# ================================================================

import cv2
import numpy as np
import math
import os
import random
import argparse
import matplotlib.pyplot as plt
import json
from types import SimpleNamespace
from config import DATA_DIR, TARGET_NAME, SCALE_FACTOR, EXPECTED_ROWS, EXPECTED_COLS, GRIDS_PER_PAGE_THEORY

# ------------------------------------------------
# Bayesian Optimization 檢查
# ------------------------------------------------
try:
    from bayes_opt import BayesianOptimization
    _BAYES_OPT_AVAILABLE = True
except ImportError:
    _BAYES_OPT_AVAILABLE = False
    print("⚠️ 未安裝 'bayesian-optimization'，將使用隨機搜尋 fallback 模式。")

# ------------------------------------------------
# 全域設定
# ------------------------------------------------
PARAM_LOG_PATH = os.path.join(DATA_DIR, "grid_params_log.json")

DEFAULT_PARAMS = {
    "min_area": 35000,
    "max_area": 65000,
    "min_ratio": 0.85,
    "max_ratio": 1.15,
    "cluster_thresh": 40,
}

BO_BOUNDS = {
    "min_area": (25000, 70000),
    "max_area": (60000, 90000),
    "min_ratio": (0.7, 0.95),
    "max_ratio": (1.05, 1.3),
    "cluster_thresh": (20, 60),
}

# ================================================================
# 評估格子偵測品質
# ================================================================
def evaluate_grid_boxes(grid_boxes, expected_count=99, tol=15):
    if not grid_boxes:
        return -1
    count = len(grid_boxes)
    score_count = max(0, 1 - abs(count - expected_count) / expected_count)
    widths = [w for (_, _, w, h) in grid_boxes]
    heights = [h for (_, _, w, h) in grid_boxes]
    if not widths or not heights:
        return score_count
    w_cv = np.std(widths) / (np.mean(widths) + 1e-6)
    h_cv = np.std(heights) / (np.mean(heights) + 1e-6)
    score_shape = max(0, 1 - (w_cv + h_cv))
    return 0.7 * score_count + 0.3 * score_shape

# ================================================================
# 三通道格子偵測法
# ================================================================
def find_grid_boxes_by_contours(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 27, 23
    )
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    eroded = cv2.erode(thresh, erode_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
    fixed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(fixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    grid_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / h if h > 0 else 0
        if (
            params["min_area"] < area < params["max_area"]
            and params["min_ratio"] < ratio < params["max_ratio"]
        ):
            grid_boxes.append((x, y, w, h))
    return grid_boxes

def find_grid_boxes_by_hough(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(denoised, 25, 80, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=110,
                             minLineLength=int(image.shape[0] // 4),
                             maxLineGap=50)
    if lines is None:
        return []

    def cluster_lines(lines, threshold=params["cluster_thresh"]):
        if not lines:
            return []
        lines = sorted(lines)
        clusters, cluster = [], [lines[0]]
        for pos in lines[1:]:
            if abs(pos - cluster[-1]) < threshold:
                cluster.append(pos)
            else:
                clusters.append(int(np.mean(cluster)))
                cluster = [pos]
        clusters.append(int(np.mean(cluster)))
        return clusters

    horz, vert = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) if x2 - x1 != 0 else 90
        if angle < 15 or angle > 165:
            horz.append(y1)
        elif 75 < angle < 105:
            vert.append(x1)
    h_lines, v_lines = cluster_lines(horz), cluster_lines(vert)
    boxes = []
    if len(h_lines) > 1 and len(v_lines) > 1:
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i + 1]
                x1, x2 = v_lines[j], v_lines[j + 1]
                w, h = x2 - x1, y2 - y1
                if w > 50 and h > 50:
                    boxes.append((x1, y1, w, h))
    return boxes

def find_grid_boxes_by_projection(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   21, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < image.shape[1] * 0.3 or h < image.shape[0] * 0.3:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    roi = binary[y:y + h, x:x + w]

    horz_proj = np.sum(roi, axis=1).astype(np.float32)
    vert_proj = np.sum(roi, axis=0).astype(np.float32)
    horz_proj = cv2.GaussianBlur(horz_proj.reshape(-1, 1), (1, 9), 0).flatten()
    vert_proj = cv2.GaussianBlur(vert_proj.reshape(-1, 1), (1, 9), 0).flatten()

    def find_peaks(projection, min_dist, threshold_ratio=0.5):
        threshold = np.max(projection) * threshold_ratio
        peaks = []
        for i in range(1, len(projection) - 1):
            if projection[i] > threshold and projection[i] > projection[i - 1] and projection[i] > projection[i + 1]:
                if all(abs(i - p) >= min_dist for p in peaks):
                    peaks.append(i)
        return peaks

    avg_side = (params["min_area"] ** 0.5 + params["max_area"] ** 0.5) / 2
    y_coords = find_peaks(horz_proj, int(avg_side * 1.0))
    x_coords = find_peaks(vert_proj, int(avg_side * 1.0))

    def regularize_lines(coords, expected, axis_len):
        coords = sorted(coords)
        if len(coords) < expected + 1 and len(coords) >= 2:
            mean_gap = np.median(np.diff(coords))
            while len(coords) < expected + 1:
                coords.append(coords[-1] + mean_gap)
        elif len(coords) > expected + 1:
            coords = np.linspace(coords[0], coords[-1], expected + 1).astype(int).tolist()
        coords = [max(0, min(axis_len - 1, int(c))) for c in coords]
        return coords

    y_coords = regularize_lines(y_coords, EXPECTED_ROWS, roi.shape[0])
    x_coords = regularize_lines(x_coords, EXPECTED_COLS, roi.shape[1])
    y_start, y_end = min(y_coords), max(y_coords)
    x_start, x_end = min(x_coords), max(x_coords)
    y_coords = np.linspace(y_start, y_end, EXPECTED_ROWS + 1).astype(int)
    x_coords = np.linspace(x_start, x_end, EXPECTED_COLS + 1).astype(int)

    boxes = []
    for i in range(EXPECTED_ROWS):
        for j in range(EXPECTED_COLS):
            y1, y2 = y_coords[i] + y, y_coords[i + 1] + y
            x1, x2 = x_coords[j] + x, x_coords[j + 1] + x
            w_box, h_box = x2 - x1, y2 - y1
            if w_box > 20 and h_box > 20:
                boxes.append((x1, y1, w_box, h_box))
    print(f"📏 Projection 外框對齊: rows={len(y_coords)-1}, cols={len(x_coords)-1}, ROI=({x},{y},{w},{h})")
    return boxes

# ================================================================
# ✅ 三通道比較：選出最佳結果
# ================================================================
def find_grid_boxes_with_params(image, params):
    best_boxes, best_score, best_method = [], -1, ""
    for method, func in {
        "Contours": find_grid_boxes_by_contours,
        "Hough": find_grid_boxes_by_hough,
        "Projection": find_grid_boxes_by_projection,
    }.items():
        boxes = func(image, params)
        boxes = [(x, y, w, h) for (x, y, w, h) in boxes if 40 < w < image.shape[1] / 2 and 40 < h < image.shape[0] / 2]
        if len(boxes) > 5:
            ws = np.array([b[2] for b in boxes])
            hs = np.array([b[3] for b in boxes])
            mw, mh = np.median(ws), np.median(hs)
            boxes = [b for b in boxes if 0.5 * mw < b[2] < 1.5 * mw and 0.5 * mh < b[3] < 1.5 * mh]
        score = evaluate_grid_boxes(boxes)
        if score > best_score:
            best_boxes, best_score, best_method = boxes, score, method
    if len(best_boxes) > 120 or len(best_boxes) < 80:
        print(f"⚠️ {best_method} 偵測格數異常 ({len(best_boxes)})，回退至 Contours")
        best_boxes = find_grid_boxes_by_contours(image, params)
        best_method = "Contours (fallback)"
    return best_boxes, best_method, best_score

# ================================================================
# 主入口：find_grid_boxes()
# ================================================================
def find_grid_boxes(image, expected_grids=GRIDS_PER_PAGE_THEORY, mincov=90.0, enable_bo=True):
    # --- 支援 SimpleNamespace 或 dict 輸入 ---
    if isinstance(image, SimpleNamespace):
        page_key = getattr(image, "page_key", "inline_image")
        image = getattr(image, "image", None)
    elif isinstance(image, dict):
        page_key = image.get("page_key", "inline_image")
        image = image.get("image", None)
    else:
        page_key = "inline_image"

    if image is None:
        print("❌ find_grid_boxes: invalid image input")
        return []

    # --- 原圖與縮放版本 ---
    orig_h, orig_w = image.shape[:2]
    scaled_image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # --- 嘗試從快取讀取上次最佳參數 ---
    cache = {}
    if os.path.exists(PARAM_LOG_PATH):
        try:
            with open(PARAM_LOG_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"⚠️ 參數快取讀取失敗: {e}")
            cache = {}

    params = cache.get(page_key, DEFAULT_PARAMS.copy())
    boxes, method, score = find_grid_boxes_with_params(scaled_image, params)
    coverage = len(boxes) / expected_grids * 100.0
    abnormal = (len(boxes) < 80 or len(boxes) > 110 or coverage < mincov)

    # --- 自動調參 ---
    if enable_bo and abnormal:
        print(f"⚠️ [{page_key}] 偵測異常 → 啟動 Bayesian Optimization ...")
        tuned = optimize_params(scaled_image, expected_grids)
        cache[page_key] = tuned
        with open(PARAM_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        boxes, method, score = find_grid_boxes_with_params(scaled_image, tuned)
        coverage = len(boxes) / expected_grids * 100.0
        if len(boxes) < 80 or len(boxes) > 110:
            boxes = find_grid_boxes_by_hough(scaled_image, tuned)
            method = "Hough (fallback)"
        print(f"🎯 [{page_key}] 最佳化後覆蓋率: {coverage:.2f}%")

    # -----------------------------------------------------------
    # 最終輸出座標統一為「放大座標」，讓 classify 自動回縮
    # -----------------------------------------------------------

    # 取得原圖與放大圖尺寸
    # image: 已在前方解包成 np.ndarray，因此這裡直接取 shape
    h_img, w_img = image.shape[:2]
    scaled_h, scaled_w = scaled_image.shape[:2]

    # 檢查放大倍率（若 detect_grid 內確實有放大）
    scale_factor = scaled_w / w_img if abs(scaled_w - w_img) > 10 else 1.0
    if scale_factor != 1.0:
        print(f"🔍 detect_grid 放大倍率偵測: ×{scale_factor:.2f}")
    else:
        print("🔍 detect_grid 未使用放大倍率。")

    # -----------------------------------------------------------
    # 統一輸出為「原圖座標系統」，確保與 classify 一致
    # -----------------------------------------------------------

    boxes_out = []
    for (x, y, w, h) in boxes:
        if SCALE_FACTOR != 1.0:
            boxes_out.append((
                int(round(x / SCALE_FACTOR)),
                int(round(y / SCALE_FACTOR)),
                int(round(w / SCALE_FACTOR)),
                int(round(h / SCALE_FACTOR))
            ))
        else:
            boxes_out.append((int(x), int(y), int(w), int(h)))

    print(f"✅ [{page_key}] 使用 {method} | 共 {len(boxes_out)} 格 | 覆蓋率 {coverage:.2f}%")
    return boxes_out




# ================================================================
# Bayesian Optimization
# ================================================================
def optimize_params(image, expected_grids=99, init_points=5, n_iter=25):
    if not _BAYES_OPT_AVAILABLE:
        return DEFAULT_PARAMS.copy()

    def objective(min_area, max_area, min_ratio, max_ratio, cluster_thresh):
        params = {
            "min_area": int(min_area),
            "max_area": int(max_area),
            "min_ratio": float(min_ratio),
            "max_ratio": float(max_ratio),
            "cluster_thresh": int(cluster_thresh),
        }
        boxes, _, _ = find_grid_boxes_with_params(image, params)
        score = len(boxes) / expected_grids * 100.0
        if len(boxes) < 80 or len(boxes) > 110:
            score *= 0.2
        return score

    optimizer = BayesianOptimization(f=objective, pbounds=BO_BOUNDS,
                                     random_state=42, allow_duplicate_points=True)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best = optimizer.max["params"]
    for k in best:
        if "area" in k or "thresh" in k:
            best[k] = int(best[k])
    return best

# ================================================================
# 單圖測試函式（新增，支援 mincov 與 enable_bo）
# ================================================================
def test_single_image(img_path, expected_grids, visualize=True, mincov=90.0, enable_bo=True):
    """單張影像格子偵測測試（與 main.py 相容）"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 無法讀取影像: {img_path}")
        return None

    boxes = find_grid_boxes(
        image={"page_key": os.path.basename(img_path), "image": img},
        expected_grids=expected_grids,
        mincov=mincov,
        enable_bo=enable_bo
    )
    coverage = len(boxes) / expected_grids * 100.0
    result = {
        "path": img_path,
        "image": img,
        "boxes": boxes,
        "detected": len(boxes),
        "coverage": coverage,
        "method": "auto"
    }

    if visualize:
        overlay = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_dir = "./debug_steps"
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"overlay_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, overlay)
        print(f"🟢 overlay 輸出: {out_path}")
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"{os.path.basename(img_path)} — {len(boxes)} grids ({coverage:.2f}%)")
        plt.axis("off")
        plt.show()

    return result

# ================================================================
# CLI 主入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="格子偵測單張/批次測試（含自動調參）")
    parser.add_argument("--sample", type=int, default=1, help="抽樣張數（預設=1）")
    parser.add_argument("--mincov", type=float, default=90.0, help="覆蓋率門檻（低於則自動調參）")
    parser.add_argument("--disable-bo", action="store_true", help="關閉 Bayesian Optimization")
    args = parser.parse_args()

    target_dir = os.path.join(DATA_DIR, TARGET_NAME)
    if not os.path.isdir(target_dir):
        print(f"❌ 找不到資料夾 {target_dir}")
        exit(1)

    pngs = [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
    if not pngs:
        print(f"⚠️ {target_dir} 下沒有 PNG 檔")
        exit(1)

    if args.sample <= 1:
        chosen = random.choice(pngs)
        img_path = os.path.join(target_dir, chosen)
        print(f"\n🎯 單檔格子偵測測試：{chosen}")
        res = test_single_image(img_path, GRIDS_PER_PAGE_THEORY, visualize=True,
                                mincov=args.mincov, enable_bo=not args.disable_bo)
        if res:
            print(f"✅ 最終採用 {res['method']}，共 {res['detected']} 格，覆蓋率 {res['coverage']:.2f}%")
    else:
        print(f"\n📊 批次格子偵測測試模式：隨機抽樣 {args.sample} 張 PNG")
        selected = random.sample(pngs, min(args.sample, len(pngs)))
        results = []
        for name in selected:
            path = os.path.join(target_dir, name)
            res = test_single_image(path, GRIDS_PER_PAGE_THEORY, visualize=False,
                                    mincov=args.mincov, enable_bo=not args.disable_bo)
            if res:
                results.append(res)
        if not results:
            print("⚠️ 無有效測試結果")
            exit(0)
        avg_cov = np.mean([r["coverage"] for r in results])
        worst = min(results, key=lambda r: r["coverage"])
        print(f"\n📈 平均格子覆蓋率: {avg_cov:.2f}%")
        print(f"📉 最差頁面: {os.path.basename(worst['path'])} ({worst['coverage']:.2f}%)")
        show_debug_image(worst)
