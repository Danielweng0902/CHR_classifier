# ================================================================
# ocr.py — OCR 與空白檢測模組（修正版：強化白名單識字 + 放寬空白門檻）
# ================================================================

import cv2
import pytesseract
import numpy as np
import os
import random
import argparse
import matplotlib.pyplot as plt
import unicodedata
from config import DATA_DIR, TARGET_NAME, TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ------------------------------------------------
# OCR 前處理：ROI 增強（強化版）
# ------------------------------------------------
def prepare_roi_for_ocr(full_img, box, enlarge=1.5):
    """
    ROI 增強：
      1. 內縮邊框去除格線。
      2. 放大影像提升辨識。
      3. 加強對比與反相處理。
    """
    x, y, w, h = box
    roi = full_img[y:y+h, x:x+w]

    # === Step1: 內縮 10~15% 去格線 ===
    m = int(min(h, w) * 0.15)
    if m > 0 and h > 2*m and w > 2*m:
        roi = roi[m:h-m, m:w-m]

    # === Step2: 放大提升辨識效果 ===
    roi = cv2.resize(roi, None, fx=enlarge, fy=enlarge, interpolation=cv2.INTER_CUBIC)

    # === Step3: 灰階 + CLAHE 對比強化 ===
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # === Step4: 三種不同二值化版本 ===
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 27, 10)
    b3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 27, 10)

    # === Step5: 若圖像太暗則反相處理 ===
    mean_val = np.mean(g)
    if mean_val < 110:
        b1, b2, b3 = [cv2.bitwise_not(x) for x in (b1, b2, b3)]

    return [b1, b2, b3]


# ------------------------------------------------
# 單格 OCR：多版本取最佳字元與信心度
# ------------------------------------------------
def ocr_char_and_conf(img_bin):
    """
    多版本容錯 OCR 函式：
    - 自動處理 pytesseract 輸出中混有 int/str 類型的 conf。
    - 嘗試三種二值化版本取最高信心結果。
    """
    cfg = "--oem 3 --psm 8"
    candidates = []

    if not isinstance(img_bin, list):
        img_bin = [img_bin]

    for img in img_bin:
        try:
            data = pytesseract.image_to_data(
                img, lang='chi_tra', config=cfg, output_type=pytesseract.Output.DICT
            )

            # --- 安全轉換 conf ---
            confs = []
            texts = []
            for i in range(len(data['text'])):
                text = str(data['text'][i]).strip()
                c = data['conf'][i]
                # 容錯轉換：可能是 str、float 或 int
                try:
                    c_val = float(c)
                except (ValueError, TypeError):
                    c_val = -1
                if c_val > -1 and text:
                    confs.append(c_val)
                    texts.append(text)

            if not texts:
                continue

            text_joined = "".join(texts)
            # 僅保留中文字（\u4e00-\u9fff）
            char = "".join(c for c in text_joined if '\u4e00' <= c <= '\u9fff')
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
# 空白檢測輔助函式（同原版）
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


# ------------------------------------------------
# 空白偵測（放寬版）
# ------------------------------------------------
def is_grid_blank_dynamically(gray,
                              std_thresh=18,                # 原25 → 放寬
                              union_ink_ratio_min=0.010,    # 原0.02 → 放寬
                              persistence_min=0.50,          # 原0.6 → 放寬
                              edge_density_min=0.006,        # 原0.01 → 放寬
                              n_cc_min=1,
                              max_cc_area_ratio_min=0.003):  # 原0.004 → 放寬
    """
    動態空白偵測（放寬版）：
      - 降低筆畫密度門檻，避免淡字被當空白。
      - 仍保留多特徵組合。
    """
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
def run_single_sample(target_dir, whitelist=None):
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
    char, conf = ocr_char_and_conf(roi_list, whitelist)

    gray = cv2.cvtColor(img[y:y+roi_h, x:x+roi_w], cv2.COLOR_BGR2GRAY)
    blank_flag = is_grid_blank_dynamically(gray)
    persis, union_ratio = _persistence_mask(gray)
    edge_density, n_cc, max_cc_area_ratio = _stroke_stats(gray)

    print(f"\n🎯 單張 OCR 測試: {chosen}")
    print(f"OCR 辨識: '{char}' | 信心度: {conf:.1f}% | 空白格: {blank_flag}")
    print(f"筆畫統計: 持久度={persis:.3f}, 墨跡率={union_ratio:.3f}, 邊緣密度={edge_density:.3f}, CC數={n_cc}")

    cv2.rectangle(img, (x, y), (x + roi_w, y + roi_h), (0, 0, 255), 4)
    cv2.putText(img, f"{char} ({conf:.1f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR Result: '{char}' ({conf:.1f}%)")
    plt.axis("off")
    plt.show()


def run_batch_sample(target_dir, sample_count, whitelist=None):
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
        char, conf = ocr_char_and_conf(roi_list, whitelist)
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
    parser.add_argument("--sample", type=int, default=1, help="抽樣張數 (預設=1)")
    parser.add_argument("--whitelist", type=str, default="", help="白名單字元（可選）")
    args = parser.parse_args()

    target_dir = os.path.join(DATA_DIR, TARGET_NAME)
    if not os.path.isdir(target_dir):
        print(f"❌ 找不到資料夾 {target_dir}")
        exit(1)

    wl = args.whitelist if args.whitelist else None
    if args.sample <= 1:
        run_single_sample(target_dir, wl)
    else:
        run_batch_sample(target_dir, args.sample, wl)
