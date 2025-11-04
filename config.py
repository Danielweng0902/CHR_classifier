# ================================================================
# config.py — 統一環境與參數設定
# ================================================================

import os

EXPECTED_ROWS = 11   # 預期的列數（橫向）
EXPECTED_COLS = 9    # 預期的欄數（縱向）
GRIDS_PER_PAGE_THEORY = EXPECTED_ROWS * EXPECTED_COLS


# === 1. 基本環境設定 ===
TARGET_NAME = "251103"   # 可手動指定；若為 "" 則自動處理 data/ 下所有子資料夾

# === 2. 路徑設定 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join("E:\\datasets", TARGET_NAME)
DEBUG_DIR = os.path.join(SCRIPT_DIR, "debug_steps")
WHITELIST_FILE = os.path.join(SCRIPT_DIR, "whitelist.txt")

# OCR / Tesseract 設定
# ================================================================
# Windows 預設安裝位置
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === 4. 預設參數（可統一管理） ===
GRID_EXPECTED_COUNT = 99        # 理論格子數 (9x11 或 9x10)
GRID_TOLERANCE = 15             # 格子數允許誤差
SCALE_FACTOR = 1.3             # 頁面放大倍率（建議 1.2～2.0）

# === 5. 目錄初始化輔助函式 ===
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print("✔ Config: 輸出與除錯資料夾已確認建立。")
