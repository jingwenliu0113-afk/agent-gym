# 037 — 公開後的作品集呈現：英文 README、繁中 README.zh-TW.md、封存段落的位置與 CI

日期：2026-10-30。使用者在 repo 公開（2026-10-30，比照 inspection-robot 的規則）時決定：公開後做作品集呈現，README 的布局照
inspection-robot（「照 robot 移到 zh-TW」），CI 跑 ruff 與完整 pytest。**只改呈現，不改研究**：任何資料、結果、結論、Gate、封存或凍結的
內容都不變。

## 決定

1. **README 的布局**
   - `README.md` 改為完整的英文版（作品集呈現）：開頭的 60 秒摘要（問題、方法、主要結果、驗證範圍），之後是示範、結果、架構、
     工程重點、快速開始、研究紀律、限制、授權。英文版是手寫的翻譯與摘要，不含程式產生的段落。
   - 原本的繁體中文 README **原樣**移到 `README.zh-TW.md`（`git mv`），只在開頭加上語言連結，並把指令中的 `--readme README.md` /
     `--update-readme README.md` 改為 `README.zh-TW.md`。程式產生並由測試比對的段落（synthetic 示範、第一輪 `results`、Study 2、
     Haiku 跨模型比較、Study 3 的三個段落）都留在 `README.zh-TW.md`，內容逐 byte 不變。
   - 英文 README 中的數字由測試對照已 commit 的 `summary.json`（`tests/test_readme_claims.py`），避免兩種語言不同步。

2. **第一輪封存段落的位置**（`docs/results/round1-seal.json` 不改）
   - 封存紀錄的鍵 `README.md#results` 是封存時的名稱，保留不變；它指的段落現在位於 `README.zh-TW.md`。`round1_seal` 改從
     `README.zh-TW.md` 讀這個段落，sha256 不變（測試確認目前的段落仍與封存值相同）。
   - `results_report` 的封存保護改為同時拒絕 `README.zh-TW.md`（封存段落所在）與 `README.md`（英文版不含程式產生的段落）。
   - 封存的 `docs/results/README.md` 中「重現」一節的 `--readme README.md` 是封存當時的指令，不修改；目前的寫法見 `README.zh-TW.md`。

3. **報告工具與測試**：`demo_synthetic`、`study2_report`、`study3_report` 的段落一律寫進 `README.zh-TW.md`（由 `--readme` /
   `--update-readme` 指定；說明與測試同步）。讀取 repo README 的測試改讀 `README.zh-TW.md`。

4. **CI**（GitHub Actions，`.github/workflows/ci.yml`）：push 與 pull request 時在固定版本的 runner 上執行 `ruff check` 與完整
   `pytest`（keyless；不需要 GPU、API key 或 `runs/`，需要原始 run 目錄的測試依原本的規則跳過）。CI 只做檢查，不產生或修改任何結果。

## 不改的部分

- 任何結果檔、封存紀錄（`round1-seal.json`）、凍結紀錄（`034-freeze-record.json`、`036-freeze-record.json`）、決策的研究內容、
  分析與統計程式的行為。
- 中文 README 的內容（除了上述的語言連結與指令中的檔名）。
