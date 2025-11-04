# ================================================================
# main.py — v3.3 AutoWhitelist + PrecisionCrop (自動白名單 & 精準裁切版)
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
# 2️⃣ 白名單管理（自動載入記憶狀態）
# ================================================================
wl = WhitelistManager(WHITELIST_FILE)

if wl.text and wl.enabled:
    print(f"🧠 自動載入白名單設定：已啟用（錨點={wl.global_offset}）")
    wl.activate()  # 確保同步
else:
    choice = input("是否啟用白名單推斷？(Enter=是 / n=否): ").strip().lower()
    if choice != "n":
        wl.activate()
        first_char = input("請輸入第一個字元 (可留空): ").strip()
        wl.set_anchor(first_char)
        print("→ 白名單功能已啟用並已記憶。")
    else:
        print("→ 白名單功能已停用。")
        wl.deactivate()

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

# ================================================================
# OCR 投票救援函式
# ================================================================
def column_vote_ocr(img, col_boxes, allow_set=None, sample_k=6, min_conf=45):
    votes = {}
    take = col_boxes[:max(1, min(sample_k, len(col_boxes)))]
    for (px, py, pw, ph) in take:
        # 保證不越界
        py, ph = int(py), int(ph)
        if py + ph > img.shape[0]:
            ph = img.shape[0] - py
        roi = img[py:py+ph, int(px):int(px+pw)]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if is_grid_blank_dynamically(gray):
            continue
        ch1, conf1 = ocr_char_and_conf([gray])
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ch2, conf2 = ocr_char_and_conf([inv])
        ch, conf = (ch1, conf1) if conf1 >= conf2 else (ch2, conf2)
        if not ch or conf < min_conf:
            continue
        if allow_set and ch not in allow_set:
            continue
        votes[ch] = votes.get(ch, 0) + 1
    return max(votes.items(), key=lambda kv: kv[1])[0] if votes else None

# ================================================================
# 主迴圈
# ================================================================
for idx, fname in enumerate(png_files, start=1):
    page_path = os.path.join(target_dir, fname)
    img = cv2.imread(page_path)
    if img is None:
        print(f"⚠️ 無法讀取 {fname}，跳過。")
        continue

    total_pages_processed += 1
    print(f"\n--- 分析頁面 {fname} ---")

    # === Step 1: 格子偵測（確保為原圖座標） ===
    img_obj = SimpleNamespace(page_key=fname, image=img)
    grid_boxes = find_grid_boxes(
        image=img_obj,
        expected_grids=GRIDS_PER_PAGE_THEORY,
        mincov=90.0,
        enable_bo=True
    )
    # 確保所有座標合法
    grid_boxes = [(max(0, x), max(0, y), min(img.shape[1]-x, w), min(img.shape[0]-y, h))
                  for (x, y, w, h) in grid_boxes]

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
    wl._save_state()  # ✅ 即時儲存啟用狀態與 anchor
    total_labels_recognized += sum(1 for l in final_labels if l != '?')
    print(f"  -> 推斷結果: [{' '.join(final_labels)}]")

    # === Step 4: '_UNK' 投票救援 ===

    def majority_nonblank_ratio(boxes, img, thr=0.3):
        """
        安全檢測該欄是否主要為非空格（支援 CenterLock 格子偏移情況）
        - 自動邊界 clip，防止 ROI 超界導致 cv2.cvtColor 報錯
        - ROI.size == 0 會自動略過
        """
        if not boxes:
            return False

        H, W = img.shape[:2]
        nonblank_count = 0
        total_valid = 0

        for (px, py, pw, ph) in boxes:
            # --- 安全邊界修正 ---
            x1, y1 = int(px), int(py)
            x2, y2 = int(px + pw), int(py + ph)
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            total_valid += 1
            if not is_grid_blank_dynamically(gray):
                nonblank_count += 1

        if total_valid == 0:
            return False

        ratio = nonblank_count / total_valid
        return ratio >= thr

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
    rescued, unk_total = 0, 0
    allow_set = set(list(wl.text)) if wl.text else None

    for i, lab in enumerate(final_labels):
        if i >= len(practice_columns): continue
        if lab == "?":
            if majority_nonblank_ratio(practice_columns[i], img):
                voted = column_vote_ocr(img, practice_columns[i], allow_set=allow_set)
                if voted:
                    final_labels[i] = voted
                    rescued += 1
                else:
                    final_labels[i] = "_UNK"
                    unk_total += 1
        elif lab == "_UNK":
            voted = column_vote_ocr(img, practice_columns[i], allow_set=allow_set)
            if voted:
                final_labels[i] = voted
                rescued += 1

    print(f"  -> 修正後標籤: [{' '.join(final_labels)}]")
    print(f"  ✅ 投票救回 {rescued} 欄，剩餘 _UNK: {unk_total}")

    # === Step 5: 儲存練習格（確保座標安全） ===
    stats = process_columns_and_save(
        image=img,
        first_row_boxes=first_row_boxes,
        practice_boxes=practice_boxes,
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
# 4️⃣ 最終報告
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
