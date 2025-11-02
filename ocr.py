# ================================================================
# ocr.py — OCR 與空白檢測模組 (含隨機抽樣統計模式)
# ================================================================

import cv2
import pytesseract
import numpy as np
import os
import random
import argparse
import matplotlib.pyplot as plt
from config import DATA_DIR, TARGET_NAME, TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ------------------------------------------------
# OCR 前處理：ROI 增強
# ------------------------------------------------
def prepare_roi_for_ocr(full_img, box):
    x, y, w, h = box
    roi = full_img[y:y+h, x:x+w]
    m = int(min(h, w) * 0.12)
    if m > 0 and h > 2*m and w > 2*m:
        roi = roi[m:h-m, m:w-m]

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 50, 50)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 27, 10)
    b3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 27, 10)
    return [b1, b2, b3]


# ------------------------------------------------
# 單格 OCR：取多版本最佳字元與信心度
# ------------------------------------------------
def ocr_char_and_conf(img_bin):
    cfg = "--oem 3 --psm 8"
    candidates = []

    if not isinstance(img_bin, list):
        img_bin = [img_bin]

    for img in img_bin:
        try:
            data = pytesseract.image_to_data(
                img, lang='chi_tra', config=cfg, output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for i, c in enumerate(data['conf'])
                     if int(c) > -1 and data['text'][i].strip()]
            text = "".join(t for t in data['text'] if t.strip())
            char = "".join(c for c in text if '\u4e00' <= c <= '\u9fff')
            final_char = char[0] if char else ""
            mean_conf = float(np.mean(confs)) if confs else 0.0
            candidates.append((final_char, mean_conf))
        except Exception as e:
            print(f"⚠️ OCR 錯誤: {e}")
            continue

    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda x: x[1])


# ------------------------------------------------
# 空白檢測輔助函式
# ------------------------------------------------
def _persistence_mask(gray, ksizes=(25, 41), min_keep=2):
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ams = []
    for k in ksizes:
        k = max(15, k | 1)
        am = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, k, 10)
        ams.append(am)
    stack = [otsu] + ams
    union = np.zeros_like(otsu)
    votes = np.zeros_like(otsu, dtype=np.uint8)
    for m in stack:
        union = cv2.bitwise_or(union, m)
        votes = cv2.add(votes, (m > 0).astype(np.uint8))
    keep = (votes >= min_keep).astype(np.uint8) * 255
    inter = cv2.bitwise_and(union, keep)
    inter_cnt = int(cv2.countNonZero(inter))
    union_cnt = int(cv2.countNonZero(union))
    persistence_ratio = (inter_cnt / union_cnt) if union_cnt > 0 else 0.0
    union_ratio = (union_cnt / gray.size) if gray.size else 0.0
    return persistence_ratio, union_ratio


def _stroke_stats(gray):
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA]
             for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 20]
    n_cc = len(areas)
    max_cc = max(areas) if areas else 0
    max_cc_area_ratio = max_cc / float(gray.size)
    return edge_density, n_cc, max_cc_area_ratio


def is_grid_blank_dynamically(gray,
                              std_thresh=25,
                              union_ink_ratio_min=0.020,
                              persistence_min=0.60,
                              edge_density_min=0.010,
                              n_cc_min=1,
                              max_cc_area_ratio_min=0.004):
    if gray is None or gray.size == 0:
        return True
    if np.std(gray) < std_thresh:
        return True
    persis, union_ratio = _persistence_mask(gray)
    if union_ratio < union_ink_ratio_min or persis < persistence_min:
        return True
    edge_density, n_cc, max_cc_area_ratio = _stroke_stats(gray)
    if (edge_density < edge_density_min or
        n_cc < n_cc_min or
        max_cc_area_ratio < max_cc_area_ratio_min):
        return True
    return False


# ================================================================
# 單檔 / 批次測試模式
# ================================================================
def run_single_sample(target_dir):
    """隨機抽樣一張 PNG 並顯示 OCR 結果"""
    pngs = [f for f in os.listdir(target_dir) if f.lower().endswith('.png')]
    if not pngs:
        print(f"⚠️ {target_dir} 無 PNG 可測試")
        return
    chosen = random.choice(pngs)
    img_path = os.path.join(target_dir, chosen)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    roi_w, roi_h = w // 10, h // 12
    x = random.randint(0, max(1, w - roi_w))
    y = random.randint(0, max(1, h - roi_h))
    box = (x, y, roi_w, roi_h)
    roi_list = prepare_roi_for_ocr(img, box)
    char, conf = ocr_char_and_conf(roi_list)

    gray = cv2.cvtColor(img[y:y+roi_h, x:x+roi_w], cv2.COLOR_BGR2GRAY)
    blank_flag = is_grid_blank_dynamically(gray)
    persis, union_ratio = _persistence_mask(gray)
    edge_density, n_cc, max_cc_area_ratio = _stroke_stats(gray)

    print(f"\n🎯 單張 OCR 測試: {chosen}")
    print(f"OCR 辨識: '{char}' | 信心度: {conf:.1f}% | 空白格: {blank_flag}")
    print(f"筆畫統計: 持久度={persis:.3f}, 墨跡率={union_ratio:.3f}, 邊緣密度={edge_density:.3f}, CC數={n_cc}")

    # 顯示紅框與結果
    cv2.rectangle(img, (x, y), (x + roi_w, y + roi_h), (0, 0, 255), 4)
    cv2.putText(img, f"{char} ({conf:.1f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR Result: '{char}' ({conf:.1f}%)")
    plt.axis("off")
    plt.show()


def run_batch_sample(target_dir, sample_count):
    """隨機抽樣多張圖，統計平均信心度與空白率"""
    pngs = [f for f in os.listdir(target_dir) if f.lower().endswith('.png')]
    if not pngs:
        print(f"⚠️ {target_dir} 無 PNG 可測試")
        return

    total_conf, valid_cnt, blank_cnt = 0, 0, 0
    for i in range(sample_count):
        chosen = random.choice(pngs)
        img_path = os.path.join(target_dir, chosen)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        roi_w, roi_h = w // 10, h // 12
        x = random.randint(0, max(1, w - roi_w))
        y = random.randint(0, max(1, h - roi_h))
        roi_list = prepare_roi_for_ocr(img, (x, y, roi_w, roi_h))
        char, conf = ocr_char_and_conf(roi_list)
        gray = cv2.cvtColor(img[y:y+roi_h, x:x+roi_w], cv2.COLOR_BGR2GRAY)
        if is_grid_blank_dynamically(gray):
            blank_cnt += 1
        else:
            valid_cnt += 1
            total_conf += conf

    avg_conf = total_conf / max(valid_cnt, 1)
    blank_rate = blank_cnt / sample_count * 100
    print("\n📊 批次 OCR 測試統計結果")
    print("-" * 50)
    print(f"抽樣數量: {sample_count}")
    print(f"平均信心度: {avg_conf:.2f}%")
    print(f"空白格比例: {blank_rate:.2f}%")
    print(f"有效格數: {valid_cnt}, 空白格數: {blank_cnt}")


# ================================================================
# 入口點
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR 單圖 / 批次測試模式")
    parser.add_argument("--sample", type=int, default=1,
                        help="指定抽樣張數 (預設=1，即單張模式)")
    args = parser.parse_args()

    target_dir = os.path.join(DATA_DIR, TARGET_NAME)
    if not os.path.isdir(target_dir):
        print(f"❌ 找不到資料夾 {target_dir}")
        exit(1)

    if args.sample <= 1:
        run_single_sample(target_dir)
    else:
        run_batch_sample(target_dir, args.sample)
