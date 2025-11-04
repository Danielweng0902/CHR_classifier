# ===================================================================
# preprocess_pages.py
#
# 核心功能 (Core Features):
#   - 自動將 PDF (補習班生字練習簿等) 轉換為逐頁 PNG。
#   - 方向校正：
#       * 多重檢查 (標籤列 OCR → 全圖 OCR → 空白率 → Tesseract OSD)。
#       * 自動判斷是否需要 180° 旋轉，避免倒置。
#   - Debug 輔助：會輸出帶有旋轉資訊的圖片到 debug_steps/。
#
# 運行流程 (Execution Flow):
#   1. 初始化環境：
#        - 確認輸入 PDF 是否存在。
#        - 建立輸出資料夾 data/... 和 debug_steps/。
#        - 若已存在舊檔案則清空。
#
#   2. PDF 轉換：
#        - 使用 pdf2image 將 PDF 每頁轉為 PIL Image。
#        - 轉為 OpenCV 格式 (BGR) 以利處理。
#
#   3. 頁面校正：
#        - correct_orientation(): 多步驟串聯判斷，決定是否需要旋轉。
#        - correct_skew(): (已移除，保留介面)。
#
#   4. 儲存：
#        - 每頁輸出兩份：
#            * 校正後 PNG → data/cramschool_merged/
#            * 附註旋轉角度的 debug 圖片 → debug_steps/
#
#   5. 完成：
#        - 輸出處理狀態與完成訊息。
# ===================================================================
import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import shutil
from config import DATA_DIR, TARGET_NAME

# ------------------------------------------------
# 全域設定
# ------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.07.0\Library\bin"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = os.path.join(SCRIPT_DIR, "pdf", f"{TARGET_NAME}.pdf")
PAGES_DIR = os.path.join(DATA_DIR, TARGET_NAME)
DEBUG_DIR = os.path.join(SCRIPT_DIR, "debug_steps", TARGET_NAME)

# ================================================================
# 保守方向校正 (v5 串聯法)
# ================================================================
def correct_orientation(image):
    print("    -> 執行方向校正 (標籤列→OCR→空白率→OSD)...")
    rotated_180 = cv2.flip(image, -1)
    h, w = image.shape[:2]
    small = cv2.resize(image, None, fx=0.5, fy=0.5)
    small_rot = cv2.flip(small, -1)

    def avg_conf(img):
        data = pytesseract.image_to_data(
            img, lang='chi_tra', config='--psm 6',
            output_type=pytesseract.Output.DICT
        )
        confs = [float(c) for c in data['conf'] if c != '-1']
        return np.mean(confs) if confs else 0.0

    try:
        # Step 1: 標籤列比對
        row_h = h // 15
        top, bottom = image[:row_h, :], image[-row_h:, :]
        top_conf, bot_conf = avg_conf(top), avg_conf(bottom)
        print(f"      [標籤列] 上={top_conf:.2f}, 下={bot_conf:.2f}")
        if top_conf >= bot_conf:
            return image, 0
        print("        -> 底部較清楚，繼續檢查...")

        # Step 2: OCR 全圖驗證
        conf_norm, conf_rot = avg_conf(small), avg_conf(small_rot)
        print(f"      [OCR] 正常={conf_norm:.2f}, 旋轉={conf_rot:.2f}")
        if conf_rot <= conf_norm:
            return image, 0
        print("        -> OCR 檢查支持旋轉，進入空白率分析...")

        # Step 3: 空白率檢查
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        top_blank = cv2.countNonZero(bin_img[:h//4, :]) / bin_img[:h//4, :].size
        bot_blank = cv2.countNonZero(bin_img[3*h//4:, :]) / bin_img[3*h//4:, :].size
        print(f"      [空白率] 上={top_blank:.2f}, 下={bot_blank:.2f}")
        if not (bot_blank < top_blank):
            return image, 0

        # Step 4: OSD 驗證
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation, conf = osd.get("rotate", 0), osd.get("confidence", 0)
        print(f"      [OSD] rotation={rotation}, conf={conf:.1f}")
        if conf >= 40 and rotation == 180:
            print("        -> OSD 確認為倒置，旋轉 180°。")
            return rotated_180, 180
        return image, 0

    except Exception as e:
        print(f"    -> 校正錯誤: {e}，回傳原圖。")
        return image, 0


def correct_skew(image):
    """保留介面 (已停用)"""
    return image

# ================================================================
# 主流程：PDF → PNG
# ================================================================
def run_preprocessing():
    print(f"\n📘 [Preprocess] 處理目標: {TARGET_NAME}")
    print(f"🔍 尋找 PDF: {PDF_FILE}")

    if not os.path.exists(PDF_FILE):
        print(f"❌ 找不到 PDF 檔案: {PDF_FILE}")
        return

    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"✔ 建立輸出資料夾: {PAGES_DIR}")
    print(f"✔ 建立 debug 資料夾: {DEBUG_DIR}")

    # 若已有舊檔案則清空
    if os.listdir(PAGES_DIR) or os.listdir(DEBUG_DIR):
        print("🧹 清空舊的輸出資料夾...")
        shutil.rmtree(PAGES_DIR)
        shutil.rmtree(DEBUG_DIR)
        os.makedirs(PAGES_DIR)
        os.makedirs(DEBUG_DIR)

    try:
        pages = convert_from_path(PDF_FILE, dpi=300, poppler_path=POPPLER_PATH)
        print(f"  -> 成功載入 {len(pages)} 頁。")
        print("  -> 開始方向校正與輸出 PNG...")

        for i, page_pil in enumerate(pages):
            page_num = i + 1
            print(f"\n  -- 處理第 {page_num} 頁 --")
            img = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
            oriented, rot_angle = correct_orientation(img)
            final = correct_skew(oriented)

            # Debug 圖片
            annotated = final.copy()
            cv2.putText(annotated, f"Rotation: {rot_angle} deg", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10, cv2.LINE_AA)
            debug_out = os.path.join(DEBUG_DIR, f"page_{page_num:03d}_annotated.png")
            cv2.imwrite(debug_out, annotated)

            # 儲存最終 PNG
            final_path = os.path.join(PAGES_DIR, f"page_{page_num:03d}.png")
            is_ok, buffer = cv2.imencode(".png", final)
            if is_ok:
                with open(final_path, "wb") as f:
                    f.write(buffer)
            print(f"    -> 已輸出 {os.path.basename(final_path)}")

    except Exception as e:
        print(f"❌ PDF 預處理失敗: {e}")
        return

    print(f"\n✅ 預處理完成，輸出至 {PAGES_DIR}")

# ================================================================
# 直接執行模式
# ================================================================
if __name__ == "__main__":
    run_preprocessing()