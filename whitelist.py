# ================================================================
# whitelist.py — 白名單與智慧推斷模組
# ================================================================

import unicodedata
import numpy as np

class WhitelistManager:
    """
    處理 OCR 標籤結果的智慧推斷模組：
      - 支援白名單匹配、全域錨點、模式記憶。
      - 適用於手寫字練習簿標籤列推斷。
    """

    def __init__(self, whitelist_path):
        self.text = ""
        self.enabled = False
        self.global_offset = None
        self.pattern_db = []
        self.last_known_anchor_index = -1
        self._load(whitelist_path)

    # ------------------------------------------------
    # 載入白名單檔案
    # ------------------------------------------------
    def _load(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.text = "".join(f.read().split())
                print(f"✔ 成功載入白名單，共 {len(self.text)} 個字元。")
        except FileNotFoundError:
            print(f"❌ 找不到白名單檔案 '{path}'，已停用白名單功能。")
            self.text = ""

    # ------------------------------------------------
    # 啟用與初始化
    # ------------------------------------------------
    def activate(self):
        """啟用白名單模式"""
        if not self.text:
            print("⚠ 無法啟用白名單：尚未載入內容。")
            return False
        self.enabled = True
        print("→ 白名單推斷功能已啟用。")
        return True

    def set_anchor(self, first_char):
        """
        設定全域錨點字元，用於跨頁序列推斷。
        """
        if not self.text:
            print("⚠ 尚未載入白名單。")
            return
        if not first_char:
            return

        normalized = unicodedata.normalize("NFKC", first_char)
        if normalized in self.text:
            self.global_offset = self.text.find(normalized)
            print(f"✔ 已設定全域錨點 '{normalized}' (index={self.global_offset})。")
        else:
            print(f"⚠ 起始字 '{first_char}' 不在白名單中，忽略。")

    # ------------------------------------------------
    # 主推斷函式
    # ------------------------------------------------
    def resolve_labels(self, ocr_results, page_idx=0, global_offset=None):
        """
        根據 OCR 結果與既有模式記憶推斷最終標籤列。
        輸入: ocr_results = [ '我', '?', '天', ... ]
        回傳: final_labels = [ '我', '是', '天', ... ]
        """
        if not self.enabled or not self.text:
            return [c if c else '?' for c in ocr_results]

        final_labels = ["?"] * len(ocr_results)
        is_determined = False

        # --- 層級 1：使用者指定的全域錨點 ---
        if page_idx == 0 and global_offset is not None:
            for i in range(len(ocr_results)):
                inferred_idx = global_offset + i
                if 0 <= inferred_idx < len(self.text):
                    final_labels[i] = self.text[inferred_idx]
            self.last_known_anchor_index = global_offset
            print(f"  -> [層級 1] 採用全域錨點推斷。")
            return final_labels

        # --- 層級 2：套用已記憶模式 ---
        if self.pattern_db:
            for pattern in self.pattern_db:
                if len(pattern) != len(ocr_results):
                    continue
                matches = sum(1 for i in range(len(ocr_results))
                              if ocr_results[i] and ocr_results[i] == pattern[i])
                if matches >= len(ocr_results) // 2:
                    final_labels = pattern
                    self._update_anchor_from_pattern(pattern)
                    print(f"  -> [層級 2] 模式匹配成功。")
                    return final_labels

        # --- 層級 3：智慧錨點推斷 ---
        final_labels = self._infer_from_anchors(ocr_results)
        if any(label != '?' for label in final_labels):
            if final_labels not in self.pattern_db:
                self.pattern_db.append(final_labels)
                print(f"  -> [層級 3] 新模式已學習: [{' '.join(final_labels)}]")
        return final_labels

    # ------------------------------------------------
    # 內部函式：根據單點錨或序列偏移推斷
    # ------------------------------------------------
    def _infer_from_anchors(self, ocr_results):
        anchors = []
        for i, ch in enumerate(ocr_results):
            if not ch:
                continue
            normalized = unicodedata.normalize("NFKC", ch)
            if normalized in self.text:
                idx = self.text.find(normalized)
                anchors.append({'pos': i, 'idx': idx})

        if not anchors:
            return [c if c else '?' for c in ocr_results]

        # 嘗試序列配對 (滑動視窗匹配)
        MIN_MATCH_COUNT = 3
        best_score, best_offset = -1, None
        for offset in range(len(self.text) - len(ocr_results) + 1):
            score = 0
            for i in range(len(ocr_results)):
                if ocr_results[i] and ocr_results[i] == self.text[offset + i]:
                    score += 1
            if score > best_score:
                best_score, best_offset = score, offset

        if best_score >= MIN_MATCH_COUNT:
            return self._apply_offset(best_offset, len(ocr_results))

        # 否則採用中位數偏移校正
        offsets = [a['idx'] - a['pos'] for a in anchors]
        if offsets:
            median_offset = int(np.median(offsets))
            return self._apply_offset(median_offset, len(ocr_results))

        return [c if c else '?' for c in ocr_results]

    # ------------------------------------------------
    # 套用偏移量生成標籤序列
    # ------------------------------------------------
    def _apply_offset(self, offset, length):
        labels = ["?"] * length
        for i in range(length):
            idx = i + offset
            if 0 <= idx < len(self.text):
                labels[i] = self.text[idx]
        self.last_known_anchor_index = offset
        print(f"     -> 偏移量: {offset} 已套用。")
        return labels

    # ------------------------------------------------
    # 更新 anchor index
    # ------------------------------------------------
    def _update_anchor_from_pattern(self, pattern):
        try:
            first_valid = next(c for c in pattern if c != '?')
            self.last_known_anchor_index = self.text.find(first_valid)
        except StopIteration:
            pass
