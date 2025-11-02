# ================================================================
# classify.py — 字跡切割、儲存與統計模組
# ================================================================

import os
import cv2
import numpy as np
from ocr import is_grid_blank_dynamically


def process_columns_and_save(image,
                             first_row_boxes,
                             practice_boxes,
                             final_labels,
                             output_dir,
                             char_counters):
    """
    對整頁的標籤欄與練習格進行分類儲存。
    - 若標籤為 '?'，整欄跳過。
    - 若非空白格，裁切儲存圖片。
    回傳:
        stats = {
            "handwriting_saved": int,
            "blanks_skipped": int,
            "addressable_grids": int,
            "incomplete_columns": list[dict(page, char, count)]
        }
    """

    total_handwriting_saved = 0
    total_blanks_skipped = 0
    total_addressable_grids = 0
    incomplete_columns = []

    COLUMN_MIN_RATIO = 0.3  # 若整欄非空白格比例 < 此值，視為無效欄

    # === Step 1: 整欄一致性審核 ===
    for i, char_label in enumerate(final_labels):
        if char_label == "?":
            continue

        lx = first_row_boxes[i][0]
        column_boxes = [b for b in practice_boxes if abs(b[0] - lx) < 50]

        nonblank_cnt = 0
        for (px, py, pw, ph) in column_boxes:
            roi = image[py:py + ph, px:px + pw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if not is_grid_blank_dynamically(gray):
                nonblank_cnt += 1

        ratio = nonblank_cnt / len(column_boxes) if column_boxes else 0
        if ratio < COLUMN_MIN_RATIO:
            final_labels[i] = "?"

    # === Step 2: 計算可定址格子總數 ===
    for i, char_label in enumerate(final_labels):
        if char_label != "?":
            lx = first_row_boxes[i][0]
            count = len([b for b in practice_boxes if abs(b[0] - lx) < 50])
            total_addressable_grids += count

    # === Step 3: 裁切與儲存 ===
    for i, char_label in enumerate(final_labels):
        if char_label == "?":
            continue

        lx = first_row_boxes[i][0]
        column_boxes = [b for b in practice_boxes if abs(b[0] - lx) < 50]
        os.makedirs(os.path.join(output_dir, char_label), exist_ok=True)

        if char_label not in char_counters:
            char_counters[char_label] = 0

        saved_count = 0
        for (px, py, pw, ph) in column_boxes:
            roi = image[py:py + ph, px:px + pw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if not is_grid_blank_dynamically(gray):
                char_counters[char_label] += 1
                filename = f"{char_counters[char_label]:03d}.png"
                path = os.path.join(output_dir, char_label, filename)

                success, buffer = cv2.imencode(".png", roi)
                if success:
                    with open(path, "wb") as f:
                        f.write(buffer)
                    saved_count += 1
            else:
                total_blanks_skipped += 1

        if saved_count > 0:
            total_handwriting_saved += saved_count
            if saved_count < 10:
                incomplete_columns.append({
                    "char": char_label,
                    "count": saved_count
                })

    # === Step 4: 回傳統計資料 ===
    stats = {
        "handwriting_saved": total_handwriting_saved,
        "blanks_skipped": total_blanks_skipped,
        "addressable_grids": total_addressable_grids,
        "incomplete_columns": incomplete_columns
    }
    return stats


# ------------------------------------------------
# 統一的低存量報告輸出
# ------------------------------------------------
def print_incomplete_report(incomplete_columns, page_name=None):
    """
    輸出低存量欄位報告 (<10 筆)
    """
    if not incomplete_columns:
        print("✔ 所有欄位均達 10 筆以上。")
        return

    print("\n" + "=" * 50)
    print("--- ⚠️ 低存量欄位報告 (儲存數量 < 10) ---")
    print("=" * 50)

    for log in incomplete_columns:
        char = log["char"]
        count = log["count"]
        if page_name:
            print(f"頁面: {page_name:<15} | 字元: '{char}' | 儲存數: {count}/10")
        else:
            print(f"字元: '{char}' | 儲存數: {count}/10")

    total_missing = sum(10 - log["count"] for log in incomplete_columns)
    print("-" * 20)
    print(f"→ 總共缺少 {total_missing} 個字跡。")
    print("=" * 50)
