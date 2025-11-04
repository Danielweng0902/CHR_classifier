# ================================================================
# report.py — 成果統計與最終報告模組
# ================================================================

import math

def generate_final_report(stats, pages_processed, grids_per_page_theory=90):
    """
    生成並輸出最終成果統計報告。

    Parameters:
        stats (dict): 統計資料，例如：
            {
                "total_pages_processed": int,
                "total_grids_found": int,
                "total_label_boxes_found": int,
                "total_practice_grids_found": int,
                "total_labels_recognized": int,
                "total_handwriting_saved": int,
                "total_blanks_skipped": int,
                "total_addressable_grids": int,
                "incomplete_columns_log": list[dict(page, char, count)]
            }
        pages_processed (int): 成功處理的頁數。
        grids_per_page_theory (int): 每頁理論格數（預設 9×10 = 90）。
    """
    total_pages_processed = pages_processed
    total_grids_found = stats.get("total_grids_found", 0)
    total_label_boxes_found = stats.get("total_label_boxes_found", 0)
    total_practice_grids_found = stats.get("total_practice_grids_found", 0)
    total_labels_recognized = stats.get("total_labels_recognized", 0)
    total_handwriting_saved = stats.get("total_handwriting_saved", 0)
    total_blanks_skipped = stats.get("total_blanks_skipped", 0)
    total_addressable_grids = stats.get("total_addressable_grids", 0)
    incomplete_columns_log = stats.get("incomplete_columns_log", [])

    print("\n" + "=" * 60)
    print("📊 最終成果統計報告")
    print("=" * 60)

    if total_pages_processed == 0:
        print("❌ 沒有處理任何頁面，無法產生報告。")
        return

    print(f"總處理頁數: {total_pages_processed} 頁")
    print(f"總偵測格子數: {total_grids_found} (含標籤 + 練習格)")
    print(f"  ├─ 標籤列格子: {total_label_boxes_found}")
    print(f"  └─ 練習格數量: {total_practice_grids_found}")
    print("-" * 50)

    unresolved_labels = total_label_boxes_found - total_labels_recognized
    print(f"成功辨識的標籤欄位數: {total_labels_recognized}")
    print(f"未能辨識的標籤欄位數: {unresolved_labels}")
    print("-" * 50)

    unaddressable = total_practice_grids_found - total_addressable_grids
    print(f"可定址練習格總數 (有標籤): {total_addressable_grids}")
    print(f"不可定址練習格總數 (標籤未知): {unaddressable}")
    print("-" * 50)

    total_non_blank_addressable = max(total_addressable_grids - total_blanks_skipped, 0)
    print(f"實際有字跡的格子: {total_non_blank_addressable}")
    print(f"跳過的空白格: {total_blanks_skipped}")
    print(f"成功儲存字跡: {total_handwriting_saved}")
    print("-" * 50)

    # --- 儲存率 ---
    if total_non_blank_addressable > 0:
        storage_rate = (total_handwriting_saved / total_non_blank_addressable) * 100
        print(f"✅ 字跡儲存率 (已儲存 / 有字跡): {storage_rate:.2f}%")
    else:
        print("⚠ 無法計算儲存率：無可用格子")

    # --- 資料產出率 ---
    theoretical_total = grids_per_page_theory * total_pages_processed
    denominator = theoretical_total - total_blanks_skipped - unaddressable
    if denominator > 0:
        yield_rate = (total_handwriting_saved / denominator) * 100
        print(f"✅ 資料產出率 (已儲存 / (理論 - 空白 - 未知)): {yield_rate:.2f}%")
    else:
        print("⚠ 無法計算產出率：分母為 0 或負數")

    # --- 低存量欄位報告 ---
    if incomplete_columns_log:
        print_low_stock_report(incomplete_columns_log)
    else:
        print("\n✔ 所有字元均達 10 筆以上，無低存量警告。")

    print("=" * 60)
    print("📘 報告生成完成。")
    print("=" * 60)


# ------------------------------------------------
# 子函式：低存量報告
# ------------------------------------------------
def print_low_stock_report(incomplete_columns_log):
    """
    顯示低存量欄位 (儲存數量 < 10)
    """
    print("\n" + "=" * 60)
    print("⚠️ 低存量欄位報告 (儲存數量 < 10)")
    print("=" * 60)

    sorted_log = sorted(incomplete_columns_log, key=lambda x: (x.get("page", ""), x["char"]))
    for log in sorted_log:
        page = log.get("page", "N/A")
        char = log["char"]
        count = log["count"]
        print(f"頁面: {page:<15} | 字元: '{char}' | 儲存數量: {count}/10")

    total_missing = sum(10 - log["count"] for log in incomplete_columns_log)
    print("-" * 20)
    print(f"→ 總缺少字跡數: {total_missing}")
    print("=" * 60)
