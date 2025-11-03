# ================================================================
# main.py — Compatible with classify.py v2.1 (no duplicate rescale)
# ================================================================

import os
import cv2
import sys
import numpy as np
import pytesseract
import shutil
import unicodedata
from types import SimpleNamespace

# === 匯入模組 ===
from config import *
from detect_grid import find_grid_boxes
from ocr import prepare_roi_for_ocr, ocr_char_and_conf, is_grid_blank_dynamically
from whitelist import WhitelistManager
from classify import process_columns_and_save, print_incomplete_report
from report import generate_final_report
import preprocess_pages  # ✅ PDF → PNG 自動化


# ================================================================
# 1️⃣ 初始化
# ================================================================
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
ensure_dirs()

target_dir = os.path.join(DATA_DIR, TARGET_NAME)
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

png_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
if not png_files:
    print(f"\n⚙️ 偵測到 {TARGET_NAME} 下沒有 PNG，執行 PDF 預處理階段...")
    preprocess_pages.run_preprocessing()
    png_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
    if not png_files:
        print("❌ 預處理後仍未產生 PNG，終止。")
        sys.exit(1)
else:
    print(f"✔ 偵測到現有 {len(png_files)} 張頁面圖像。")

# 重設輸出資料夾
for d in [OUTPUT_DIR, DEBUG_DIR]:
    if os.path.isdir(d):
        shutil.rmtree(d)
ensure_dirs()


# ================================================================
# 2️⃣ 準備白名單
# ================================================================
wl = WhitelistManager(WHITELIST_FILE)
choice = input("是否啟用白名單推斷？(Enter=是 / n=否): ").strip().lower()
if choice != "n":
    wl.activate()
    first_char = input("請輸入第一個字元 (可留空): ").strip()
    wl.set_anchor(first_char)
    print("→ 白名單功能已啟用。")
else:
    print("→ 白名單功能已停用。")


# ================================================================
# 3️⃣ 主流程
# ================================================================
char_counters = {}
incomplete_columns_log = []

total_pages_processed = 0
total_grids_found = 0
total_label_boxes_found = 0
total_practice_grids_found = 0
total_labels_recognized = 0
total_handwriting_saved = 0
total_blanks_skipped = 0
total_addressable_grids = 0

png_files.sort()
print(f"\n📂 開始處理 {TARGET_NAME}，共 {len(png_files)} 頁")

for idx, fname in enumerate(png_files, start=1):
    page_path = os.path.join(target_dir, fname)
    img = cv2.imread(page_path)
    if img is None:
        print(f"⚠️ 無法讀取 {fname}，跳過。")
        continue

    total_pages_processed += 1
    print(f"\n--- 分析頁面 {fname} ---")

    # === Step 1: 格子偵測 ===
    img_obj = SimpleNamespace(page_key=fname, image=img)
    grid_boxes = find_grid_boxes(
        image=img_obj,
        expected_grids=GRIDS_PER_PAGE_THEORY,
        mincov=90.0,
        enable_bo=True
    )

    if len(grid_boxes) < 9:
        print(f"⚠️ 格子過少 ({len(grid_boxes)})，跳過此頁。")
        continue

    grid_boxes.sort(key=lambda b: (b[1], b[0]))
    COL_COUNT = EXPECTED_COLS
    first_row_boxes = grid_boxes[:COL_COUNT]
    practice_boxes = grid_boxes[COL_COUNT:]

    total_grids_found += len(grid_boxes)
    total_label_boxes_found += len(first_row_boxes)
    total_practice_grids_found += len(practice_boxes)
    print(f"  -> 標籤格 {len(first_row_boxes)} | 練習格 {len(practice_boxes)}")

    # === Step 2: OCR 標籤 ===
    ocr_results = []
    for (x, y, w, h) in first_row_boxes:
        if w <= 0 or h <= 0:
            ocr_results.append(None); continue
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            ocr_results.append(None); continue

        roi_raw = prepare_roi_for_ocr(img, (x, y, w, h))
        if isinstance(roi_raw, list):
            roi_list = [r for r in roi_raw if r is not None and r.size > 0]
        elif isinstance(roi_raw, np.ndarray):
            roi_list = [roi_raw]
        else:
            roi_list = []
        if not roi_list:
            ocr_results.append(None); continue

        ch, conf = ocr_char_and_conf(roi_list)
        if not ch or conf < 45:
            gray_label = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            _, alt_bin = cv2.threshold(gray_label, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ch_alt, conf_alt = ocr_char_and_conf([alt_bin])
            if conf_alt > conf:
                ch, conf = ch_alt, conf_alt

        gray_label = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        if is_grid_blank_dynamically(gray_label):
            ch = None
        ocr_results.append(ch if ch else None)

    readable = [c if c else '?' for c in ocr_results]
    print(f"  -> 初步 OCR 結果: [{' '.join(readable)}]")

    # === Step 3: 白名單推斷 ===
    final_labels = wl.resolve_labels(
        ocr_results, page_idx=idx - 1, global_offset=wl.global_offset
    )
    total_labels_recognized += sum(1 for l in final_labels if l != '?')
    print(f"  -> 推斷結果: [{' '.join(final_labels)}]")

    # === Step 3.5: '?' 欄位救援 (_UNK)
    def majority_nonblank_ratio(boxes, img, thr=0.35):
        if not boxes: return False
        h_img, w_img = img.shape[:2]
        nonblank = 0
        for (px, py, pw, ph) in boxes:
            gray = cv2.cvtColor(img[int(py):int(py+ph), int(px):int(px+pw)], cv2.COLOR_BGR2GRAY)
            if not is_grid_blank_dynamically(gray):
                nonblank += 1
        return (nonblank / max(1, len(boxes))) >= thr

    # 分欄方式需與 classify 相容
    first_row_sorted = sorted(first_row_boxes, key=lambda b: b[0])
    all_ws = [w for (_, _, w, _) in practice_boxes] or [80]
    col_tol = max(12, int(np.median(all_ws) * 0.6))

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

    practice_columns = group_boxes_by_columns(first_row_sorted, practice_boxes, tol=col_tol)

    for i, lab in enumerate(final_labels):
        if lab == "?" and i < len(practice_columns):
            if majority_nonblank_ratio(practice_columns[i], img):
                final_labels[i] = "_UNK"

    print(f"  -> 修正後標籤: [{' '.join(final_labels)}]")

    # === Step 4: 儲存練習格（不含第一列） ===
    stats = process_columns_and_save(
        image=img,
        first_row_boxes=first_row_boxes,   # 用於分欄，不儲存
        practice_boxes=practice_boxes,     # 真正切割來源
        final_labels=final_labels,
        output_dir=OUTPUT_DIR,
        char_counters=char_counters,
        page_idx=idx
    )

    total_handwriting_saved += stats["handwriting_saved"]
    total_blanks_skipped += stats["blanks_skipped"]
    total_addressable_grids += stats["addressable_grids"]
    incomplete_columns_log.extend([{"page": fname, **log} for log in stats["incomplete_columns"]])
    print_incomplete_report(stats["incomplete_columns"], page_name=fname)


# ================================================================
# 4️⃣ 最終報告統計
# ================================================================
final_stats = {
    "total_grids_found": total_grids_found,
    "total_label_boxes_found": total_label_boxes_found,
    "total_practice_grids_found": total_practice_grids_found,
    "total_labels_recognized": total_labels_recognized,
    "total_handwriting_saved": total_handwriting_saved,
    "total_blanks_skipped": total_blanks_skipped,
    "total_addressable_grids": total_addressable_grids,
    "incomplete_columns_log": incomplete_columns_log
}

generate_final_report(
    final_stats,
    pages_processed=total_pages_processed,
    grids_per_page_theory=GRIDS_PER_PAGE_THEORY
)
