# 038 — README 判定的附條件，與 Study 3 並列段落事先措辭的更正

日期：2026-11-02。**只改呈現，不改研究**：資料、結果檔、判定、Gate、封存（`docs/results/round1-seal.json`）與凍結紀錄
（`034-freeze-record.json`、`036-freeze-record.json`）都不變；分析與報告程式，以及它們產生、由測試比對的 README 段落與結果頁也不變。
本紀錄是 README 附條件的出處，也是並列段落措辭的正式更正。

## 背景

1. 英文 `README.md`（decision 037）寫 Study 3「分級對兩個 agent 都有效」、Study 2「以驟降為主（4 / 4）」，但沒有附上下面的條件
   （只在 Haiku 的 RQ1 提到 epic-closeout H3 用完 turn，沒有用在分級）。條件 1–3 的數字取自已 commit 的結果檔，或由其中的計數與
   比例直接算出；條件 4 由結果檔的計數計算。英文版又寫「下面所有數字都由 `tests/test_readme_claims.py` 檢查」，但這個測試只以子字串
   比對主要的數字（18 個）。
2. decision 036「RQ1-X」（`036-study3-proposal.md:114`）事先寫定，並列表必須註明：「兩個 agent 的失敗來自不同的題（預期本機主要在
   H1 / H2，Haiku 主要在 H3），所以 cliff 比例的差異不能歸因於 agent。」這句由程式產生：`experiments/study3_report.py:63` 的
   `COMPARE_NOTE`（`tests/test_study3_report.py:393` 鎖定原文；`experiments/study3_analysis.py:1104` 另有同義的 note）。它出現在：
   - `README.zh-TW.md` 的 study3-compare 段落（粗體；由 `study3_report --readme` 寫入的程式產生段落）；
   - `docs/results/study3/README.md:152`、`docs/results/study3-haiku/README.md:133`（結果頁，測試以 summary.json 逐 byte 比對）。
   正式結果與這句不完全一致：前半說得太絕對，括號中的預期只有一半成立（見下面「決定」2）。decision 036「實作紀錄」
   （`036-study3-proposal.md:855-857`）已記下差異，並依 2026-10-29 的決定沿用原文（`350d8c2`）。README 在其他段落寫了重疊的數字
   （`README.zh-TW.md`「結論的分類」；`README.md` 寫 partly different tasks），但沒有在原句旁更正；結果頁只有原文。

## 決定

### 1. 判定的附條件（`README.md` 的「Caveats」，At a glance 與結果表指向它；`README.zh-TW.md`「結論的分類」同步）

事先登記的判定（標籤、Wilson 區間、檢定與門檻）都不變；附條件只說明怎麼讀這些判定。數字的出處：

1. **本機模型的 agent_error（Study 3 分級）**
   - 288 個 eval episode 中 67 個是 agent_error，H1 / H2 / H3 = 6 / 23 / 38（`docs/results/study3/summary.json` 的
     `analysis.baseline.levels.*.agent_error` 與 `n_episodes`；結果頁 `docs/results/study3/README.md:28-32`）。
   - 定義：模型自己的輸出無法成為合法的 turn（例如輸出被截斷；`agents/baseline.py:112-113` 的 `AdapterError`）。依 036 §3「context 與
     agent_error」記為 agent_error（`036-study3-proposal.md:233`），在 baseline 中計為未成功（`036-study3-proposal.md:240`）。dev 的
     13 個都是輸出截斷（`036-study3-proposal.md:747`）；eval 的 67 個在結果檔中只有計數。
   - 事先登記的敏感度分析（`036-study3-proposal.md:244`，只報告）排除它們之後，H1 − H3 從 0.312 降為 0.222；每級 32 題中，H2 只剩
     30 題、H3 只剩 25 題還有 episode（`analysis.baseline.rq2_without_agent_error` 的 `point` 與 `n_tasks`；結果頁 `:39`，不做檢定）。
2. **turn 上限（Study 3 分級；Haiku 的 RQ1）**
   - Haiku 以 max_turns 結束的 episode 只出現在 H3（`docs/results/study3-haiku/summary.json` 的
     `analysis.baseline.levels.H3.max_turns_rate` = 0.25，× 96 = 24；H1、H2 為 0），也只出現在 epic-closeout
     （`analysis.baseline.families.epic-closeout.max_turns_rate` = 1/3，× 72 = 24；其他題型為 0）。
   - epic-closeout 每級 8 題（`n_tasks` 24 ÷ 3 級）、R = 3，所以 epic-closeout H3 的 24 個 episode 全部以 max_turns 結束
     （上限 20，`036-freeze-record.json`）。
   - 這是 eval 之前登記的已知限制：逐一讀取的解法在 epic-closeout H3 超過可用的 turn（036 §3 (c)，`036-study3-proposal.md:222`；
     兩份結果頁「已知限制」第 2 點）。
   - 其中失敗的 12 個（`analysis.baseline.families.epic-closeout.outcomes.failure`）占 Haiku H3 的 28 個失敗
     （`analysis.baseline.levels.H3.outcomes.failure`）中的 12 個：epic-closeout 失敗的 source 只有 H3 的 4 條
     （`analysis.sources.n_failed_by_family`、`analysis.rq1.max_turns_sources`），這 4 條就是 4 條無法判定的曲線
     （結果頁 `docs/results/study3-haiku/README.md:71`）。
   - 本機模型也有 episode 以 max_turns 結束：H2 2 個、H3 12 個，H1 沒有（`docs/results/study3/summary.json` 的
     `analysis.baseline.levels.*.max_turns_rate` 0.021 / 0.125，× 96；結果頁 `docs/results/study3/README.md:31-32`）。這些 episode 有
     trajectory，不是 agent_error（agent_error 沒有 trajectory，`experiments/run_baseline.py:420-424`），附條件 1 的敏感度分析只排除
     agent_error（`experiments/study3_analysis.py:514`），所以仍包含它們。
3. **Haiku 的 13 / 15 只來自 2 個題型（Study 3 RQ1）**
   - 15 條可判定的曲線是 sync-repair 10、scoped-triage 5（cliff 8、5）；epic-closeout 的 4 條都無法判定，verified-close 沒有失敗的
     source（`analysis.curves` 的 `family` 與 `class`；結果頁 `docs/results/study3-haiku/README.md:128-131`）。
   - Wilson 區間把 15 條曲線當成獨立。task 層級的相依由「每題一條 source」處理（規則 B，`036-study3-proposal.md:46`；
     `SPEC.md:106-109`）；題型層級沒有調整。
4. **Study 2 的 4 / 4 對區間方法不穩健（Study 2 RQ1）**
   - 判定依事先登記的 Wilson 95% CI [0.510, 1.000]（034 §2，`034-study2-proposal.md:52-61`；`docs/results/study2/summary.json` 的
     `analysis.sweep.rq1`）。
   - 改用精確的 Clopper–Pearson 95% 區間：k = n 時下限是 (0.05 / 2)^(1/n)，n = 4 為 0.025^(1/4) = 0.398，區間 [0.398, 1.000]
     包含 0.5，依同一條三分規則就是「無法判斷」。
   - 這是事後的穩健性附註，不是事先登記的分析，**不改變判定**：把正式結論改寫成「證據不足」等於看過結果之後改變結論規則
     （034「結案」，`034-study2-proposal.md:657-659`）。

`README.md` 的說明改為「所有數字取自或算自已 commit 的結果檔與 Study 3 的凍結紀錄，主要數字由 `tests/test_readme_claims.py` 對照
結果檔檢查」。`README.md` 中附條件 1–4 與下面更正（重疊的 source 與曲線）的主要數字也加進這個測試（Clopper–Pearson 的下限以 k = n
時的公式計算）。這個測試只讀 `README.md`：`README.zh-TW.md` 的對應文字與本紀錄中的數字（例如下面逐級別的失敗 source 數）不在測試內。

### 2. 並列段落事先措辭的更正

- 「兩個 agent 的失敗來自不同的題」**說得太絕對，只部分成立**（兩份 summary.json 的 `analysis.sources.failed`、`analysis.curves` 的
  `task_id` 與 `class`）：
  - 失敗的 source 部分重疊：Haiku 的 19 條中 13 條本機也失敗；本機的 53 條中也是這 13 條，本機獨有 40 條、Haiku 獨有 6 條。
  - 兩個 agent 都有曲線的題 8 題（同 036「實作紀錄」），這 8 條在本機都無法判定；所以 cliff 比例背後的可判定曲線（本機 3 / 5 的 5 條、
    Haiku 13 / 15 的 15 條）沒有共同的題。
- 括號中的預期只有一半成立：「預期本機主要在 H1 / H2」**不成立**，本機失敗的 source 是 H1 16、H2 19、H3 18；「Haiku 主要在 H3」
  **成立**，H3 14 / 19（`analysis.sources.by_level.*.n_failed`；`README.zh-TW.md` 並列段落的分母表；同 `036-study3-proposal.md:856-857`）。
- 結論「cliff 比例的差異不能歸因於 agent」與不合併、不做 agent 之間檢定的規則（decision 009 §4）**不變**。理由寫得更精確：兩個 agent
  的失敗 source 部分重疊、但不相同，可判定的曲線沒有共同的題，而且各自從自己的 trajectory 重跑（同 `README.zh-TW.md`「結論的分類」）。
- **不改**：decision 036 的登記原文、`COMPARE_NOTE` 與 `study3_analysis` 的 note、`README.zh-TW.md` 中程式產生的 study3-compare 段落、
  兩份結果頁。提到本紀錄的位置：
  - `README.zh-TW.md`：study3-compare 段落之後（段落外）的更正與「限制」（有連結）；狀態列、「結論的分類」與 decision 的索引（文字）；
  - `README.md`：Study 3 結果中說明兩個 agent 失敗的題部分重疊的那一點，以及「Caveats」的開頭（有連結）；
  - `NEXT.md`「目前位置」。

## 為什麼

- 事先寫定措辭，是為了不在看過結果之後改寫報告。但這句的事實前提說得太絕對、括號中的預期又有一半不成立，它仍出現在 README
  （粗體）與兩份結果頁，讀者會把它讀成結果。另立紀錄更正，原文與更正都看得到，也不動任何由測試逐 byte 比對、或重產報告時會被覆蓋的輸出。
- 英文 README 是公開的入口。附條件 1–3 的數字取自或直接算自結果檔，4 由結果檔的計數計算；在英文版寫出來，不改變任何判定。

## 放棄的替代方案

- **直接改結果頁或 README 的 study3-compare 段落**：結果頁與 summary.json 逐 byte 比對的測試會失敗；README 的段落在下一次重產報告時會被
  覆蓋；兩者也都不符合 decision 037 對程式產生內容的處理（逐 byte 不變）。
- **改 `COMPARE_NOTE`**：等於事後改寫登記的報告措辭，還要重產並重新 commit 結果頁。
- **把 Study 2 的判定改成「無法判斷」，或改用 Clopper–Pearson 判定**：改變事先登記的結論規則（034 §2）。
- **在英文 README 拿掉「valid」**：分級依事先登記的檢定成立，拿掉會弱化真的陳述；改為附上條件。

## 對實驗的影響

沒有。不重跑任何實驗、不重算任何正式分析，也不改任何判定、門檻、資料切分或 Gate；附條件中的換算與 Clopper–Pearson 的下限只用已
commit 的計數。只改 README 的呈現、新增本紀錄、在 `NEXT.md`「目前位置」加一段，並擴充 `tests/test_readme_claims.py`。
