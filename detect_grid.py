# ================================================================
# detect_grid.py — 格子偵測模組 (Bayesian Optimization + Log + Fallback)
# ================================================================

import cv2
import numpy as np
import math
import os
import random
import argparse
import matplotlib.pyplot as plt
import json
from config import DATA_DIR, TARGET_NAME

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

# 預設參數（直接取你原版本）
DEFAULT_PARAMS = {
    "min_area": 35000,
    "max_area": 65000,
    "min_ratio": 0.85,
    "max_ratio": 1.15,
    "cluster_thresh": 40,
}

# 搜尋空間（Bayesian Optimization 用）
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
        lines = sorted(lines)
        clusters, cluster = [], [lines[0]]
        for pos in lines[1:]:
            if pos - cluster[-1] < threshold:
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
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    horz_proj, vert_proj = np.sum(binary, axis=1), np.sum(binary, axis=0)
    def find_peaks(projection, min_dist, threshold_ratio=0.3):
        threshold = np.max(projection) * threshold_ratio
        peaks = []
        for i in range(1, len(projection) - 1):
            if (
                projection[i] > threshold
                and projection[i] > projection[i - 1]
                and projection[i] > projection[i + 1]
            ):
                if all(abs(i - p) >= min_dist for p in peaks):
                    peaks.append(i)
        return peaks
    avg_side = (params["min_area"] ** 0.5 + params["max_area"] ** 0.5) / 2
    y_coords = find_peaks(horz_proj, int(avg_side * 0.8))
    x_coords = find_peaks(vert_proj, int(avg_side * 0.8))
    boxes = []
    if len(y_coords) > 1 and len(x_coords) > 1:
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                y1, y2 = y_coords[i], y_coords[i + 1]
                x1, x2 = x_coords[j], x_coords[j + 1]
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ================================================================
# 統一呼叫入口
# ================================================================
def find_grid_boxes_with_params(image, params):
    best_boxes, best_score, best_method = [], -1, ""
    for method, func in {
        "Contours": find_grid_boxes_by_contours,
        "Hough": find_grid_boxes_by_hough,
        "Projection": find_grid_boxes_by_projection,
    }.items():
        boxes = func(image, params)
        score = evaluate_grid_boxes(boxes)
        if score > best_score:
            best_boxes, best_score, best_method = boxes, score, method
    return best_boxes, best_method, best_score


# ================================================================
# Bayesian Optimization 自動調參
# ================================================================
def optimize_params(image, expected_grids=90, init_points=8, n_iter=20):
    def objective(min_area, max_area, min_ratio, max_ratio, cluster_thresh):
        params = {
            "min_area": int(min_area),
            "max_area": int(max_area),
            "min_ratio": float(min_ratio),
            "max_ratio": float(max_ratio),
            "cluster_thresh": int(cluster_thresh),
        }
        boxes, _, _ = find_grid_boxes_with_params(image, params)
        return len(boxes) / expected_grids * 100.0

    if _BAYES_OPT_AVAILABLE:
        optimizer = BayesianOptimization(f=objective, pbounds=BO_BOUNDS, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best = optimizer.max["params"]
        for k in best:
            if "area" in k or "thresh" in k:
                best[k] = int(best[k])
        return best
    else:
        best_cov, best_params = -1, None
        for _ in range(40):
            params = {
                k: random.uniform(lo, hi) if isinstance(lo, float) else random.randint(int(lo), int(hi))
                for k, (lo, hi) in BO_BOUNDS.items()
            }
            boxes, _, _ = find_grid_boxes_with_params(image, params)
            cov = len(boxes) / expected_grids * 100.0
            if cov > best_cov:
                best_cov, best_params = cov, params
        return best_params


# ================================================================
# 單檔測試（含自動調參）
# ================================================================
def test_single_image(img_path, expected_grids=90, visualize=False, mincov=90.0, enable_bo=True):
    """對單張圖執行格子偵測，覆蓋率不足時自動調參"""
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 無法讀取 {img_path}")
        return None

    SCALE_FACTOR = 1.5
    image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    page_key = os.path.basename(img_path)
    cache = {}
    if os.path.exists(PARAM_LOG_PATH):
        with open(PARAM_LOG_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    params = cache.get(page_key, DEFAULT_PARAMS.copy())

    boxes, method, score = find_grid_boxes_with_params(image, params)
    coverage = len(boxes) / expected_grids * 100.0

    if enable_bo and coverage < mincov:
        print(f"⚠️ 覆蓋率 {coverage:.2f}% 低於 {mincov}% → 啟動 Bayesian Optimization")
        tuned = optimize_params(image, expected_grids)
        cache[page_key] = tuned
        with open(PARAM_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        boxes, method, score = find_grid_boxes_with_params(image, tuned)
        coverage = len(boxes) / expected_grids * 100.0
        print(f"🎯 最佳化後覆蓋率: {coverage:.2f}%")

    result = {
        "path": img_path,
        "image": image,
        "boxes": boxes,
        "method": method,
        "score": score,
        "coverage": coverage,
        "detected": len(boxes),
    }
    if visualize:
        show_debug_image(result)
    return result


# ================================================================
# 顯示結果圖
# ================================================================
def show_debug_image(result):
    debug_img = result["image"].copy()
    for (x, y, w, h) in result["boxes"]:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(result['path'])} | {result['detected']} grids ({result['coverage']:.1f}%) [{result['method']}]")
    plt.axis("off")
    plt.show()


# ================================================================
# CLI 主入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="格子偵測單張/批次測試（含自動調參）")
    parser.add_argument("--sample", type=int, default=1, help="抽樣張數（預設=1）")
    parser.add_argument("--mincov", type=float, default=90.0, help="覆蓋率門檻（低於則自動調參）")
    parser.add_argument("--disable-bo", action="store_true", help="關閉 Bayesian Optimization")
    args = parser.parse_args()

    GRIDS_PER_PAGE_THEORY = 9 * 10
    target_dir = os.path.join(DATA_DIR, TARGET_NAME)
    if not os.path.isdir(target_dir):
        print(f"❌ 找不到資料夾 {target_dir}")
        exit(1)

    pngs = [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
    if not pngs:
        print(f"⚠️ {target_dir} 下沒有 PNG 檔")
        exit(1)

    # 單張模式
    if args.sample <= 1:
        chosen = random.choice(pngs)
        img_path = os.path.join(target_dir, chosen)
        print(f"\n🎯 單檔格子偵測測試：{chosen}")
        result = test_single_image(img_path, GRIDS_PER_PAGE_THEORY, visualize=True, mincov=args.mincov, enable_bo=not args.disable_bo)
        print(f"✅ 最終採用 {result['method']}，共 {result['detected']} 格，覆蓋率 {result['coverage']:.2f}%")

    # 批次模式
    else:
        print(f"\n📊 批次格子偵測測試模式：隨機抽樣 {args.sample} 張 PNG")
        selected = random.sample(pngs, min(args.sample, len(pngs)))
        results = []
        for name in selected:
            path = os.path.join(target_dir, name)
            res = test_single_image(path, GRIDS_PER_PAGE_THEORY, visualize=False, mincov=args.mincov, enable_bo=not args.disable_bo)
            if res:
                results.append(res)

        if not results:
            print("⚠️ 無有效測試結果")
            exit(0)

        avg_cov = np.mean([r["coverage"] for r in results])
        worst = min(results, key=lambda r: r["coverage"])
        print(f"\n📈 平均格子覆蓋率: {avg_cov:.2f}%")
        print(f"📉 最差頁面: {os.path.basename(worst['path'])} ({worst['coverage']:.2f}%)")
        print(f"偵測方法: {worst['method']} | 偵測格數: {worst['detected']}")
        show_debug_image(worst)
