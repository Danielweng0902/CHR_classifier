# ================================================================
# main.py — Chinese Handwriting Recognition Pipeline (Modular)
# ================================================================
# 功能總覽：
#   1. 初始化環境與資料夾
#   2. 自動重新命名頁面圖片
#   3. 格子偵測（三通道）
#   4. 標籤 OCR 與白名單推斷
#   5. 字跡切割與分類儲存
#   6. 統計與最終報告
# ================================================================

import os
import cv2
import sys
import pytesseract
import shutil
import unicodedata

# === 匯入模組 ===
from config import *
from detect_grids import find_grid_boxes
from ocr import prepare_roi_for_ocr, ocr_char_and_conf, is_label_blank_ultra_strict
from whitelist import WhitelistManager
from classify import process_columns_and_save, print_incomplete_report
from report import generate_final_report

# ================================================================
# 1️⃣ 初始化環境設定
# ================================================================
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
ensure_dirs()

# 指定要處理的資料夾
if TARGET_NAME:
    target_dirs = [os.path.join(DATA_DIR, TARGET_NAME)]
    print(f"✔ 僅處理指定的子資料夾: {TARGET_NAME}")
else:
    target_dirs = [os.path.join(DATA_DIR, d)
                   for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"✔ 未指定 target_name，將遍歷 data/ 下 {len(target_dirs)} 個子資料夾")

if not target_dirs:
    print("❌ 錯誤: 沒有找到可用的資料夾")
    sys.exit(1)

# 清空舊的輸出資料夾
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
if os.path.isdir(DEBUG_DIR):
    shutil.rmtree(DEBUG_DIR)
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
else:
    print("→ 白名單功能已停用。")

# ================================================================
# 3️⃣ 主處理流程
# ================================================================
char_counters = {}
incomplete_columns_log = []

# 全域統計變數
total_pages_processed = 0
total_grids_found = 0
total_label_boxes_found = 0
total_practice_grids_found = 0
total_labels_recognized = 0
total_handwriting_saved = 0
total_blanks_skipped = 0
total_addressable_grids = 0

# ------------------------------------------------
# 遍歷各資料夾頁面
# ------------------------------------------------
for tdir in target_dirs:
    image_files = [f for f in os.listdir(tdir) if f.lower().endswith(('.jpg', '.png'))]
    image_files.sort()

    if not image_files:
        print(f"⚠️ {tdir} 中沒有找到圖片，跳過。")
        continue

    print(f"\n📂 正在處理資料夾: {os.path.basename(tdir)}，共 {len(image_files)} 張")

    for idx, fname in enumerate(image_files, start=1):
        page_path = os.path.join(tdir, fname)
        img = cv2.imread(page_path)
        if img is None:
            print(f"  ⚠️ 無法讀取 {fname}，跳過。")
            continue

        total_pages_processed += 1
        page_name = os.path.basename(fname)
        print(f"\n--- 分析頁面 {page_name} ---")

        # === Step 1: 放大影像 + 格子偵測 ===
        img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        grid_boxes = find_grid_boxes(img)
        if len(grid_boxes) < 9:
            print(f"  ⚠️ 格子過少 ({len(grid_boxes)})，跳過此頁。")
            continue

        grid_boxes.sort(key=lambda b: (b[1], b[0]))
        COL_COUNT = 9
        first_row_boxes = grid_boxes[:COL_COUNT]
        practice_boxes = grid_boxes[COL_COUNT:]

        total_grids_found += len(grid_boxes)
        total_label_boxes_found += len(first_row_boxes)
        total_practice_grids_found += len(practice_boxes)

        print(f"  -> 標籤格 {len(first_row_boxes)} | 練習格 {len(practice_boxes)}")

        # === Step 2: OCR 標籤 ===
        ocr_results = []
        for label_box in first_row_boxes:
            roi_bin = prepare_roi_for_ocr(img, label_box)
            ch, conf = ocr_char_and_conf(roi_bin)

            # 信心度低或空白判為 None
            if ch and conf < 45:
                ch = None

            x, y, w, h = label_box
            gray_label = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            if is_label_blank_ultra_strict(gray_label):
                ch = None

            ocr_results.append(ch)

        readable = [c if c else '?' for c in ocr_results]
        print(f"  -> 初步 OCR 結果: [{' '.join(readable)}]")

        # === Step 3: 白名單推斷 ===
        final_labels = wl.resolve_labels(ocr_results, page_idx=idx - 1, global_offset=wl.global_offset)
        total_labels_recognized += sum(1 for l in final_labels if l != '?')
        print(f"  -> 推斷結果: [{' '.join(final_labels)}]")

        # === Step 4: 儲存字跡 ===
        stats = process_columns_and_save(
            image=img,
            first_row_boxes=first_row_boxes,
            practice_boxes=practice_boxes,
            final_labels=final_labels,
            output_dir=OUTPUT_DIR,
            char_counters=char_counters
        )

        # 更新全域統計
        total_handwriting_saved += stats["handwriting_saved"]
        total_blanks_skipped += stats["blanks_skipped"]
        total_addressable_grids += stats["addressable_grids"]
        incomplete_columns_log.extend([
            {"page": page_name, **log} for log in stats["incomplete_columns"]
        ])

        print_incomplete_report(stats["incomplete_columns"], page_name=page_name)

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

generate_final_report(final_stats, pages_processed=total_pages_processed,
                      grids_per_page_theory=GRIDS_PER_PAGE_THEORY)
