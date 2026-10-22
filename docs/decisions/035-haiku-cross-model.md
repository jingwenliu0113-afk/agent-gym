# 035 — 跨模型比較：Claude Haiku 5.5 跑 Study 2 的 eval（付費 API）

日期：2026-10-22　狀態：採用（使用者 2026-10-22 授權：模型 Claude Haiku 5.5、總預算約 70 美元、不做試跑、直接開始）

## 背景

- Study 2（decision 034）已完成並結案：本機模型 `qwen3.5:9b-q4_K_M` 在 eval 36 題 × R = 3 上 101 success、6 failure、1 agent_error；
  RQ1 判定「以驟降為主」，但只有 4 條可判定的失敗曲線（證據很弱）。
- 使用者授權以 Claude API 做跨模型比較，預算約 70 美元。依離線估算（`experiments.cost_estimate` 的方法套用到 Study 2 的題目），
  Opus 5.5 連 eval baseline 都可能超過預算，Sonnet 5.5 只夠 baseline；Haiku 5.5 是唯一能在預算內跑完 baseline 與 sweep 的模型。
- 使用者知道 Haiku 的能力未經實測，決定不做試跑、直接以 Haiku 5.5 開始。

## 1. 範圍：不變的部分

- 第一輪的封存、Core Gate（`gradual`，不進 Advanced）與 Study 2 的結論都不變；Study 2 的正式輸出（`docs/results/study2/`、README 的
  study2 段落）不修改。
- 這是一輪正式比較，**只有一輪**：不依結果調整題目、門檻、規則或分析，不換模型重跑，也不追加 rollout。
- 題目：Study 2 的 eval 36 題（`docs/decisions/034-freeze-record.json` 的 task hash，測試持續比對）。**不跑 dev**（使用者決定不試跑）。

## 2. Agent（凍結）

與 decision 009 的 Anthropic agent 完全相同，只把模型換成 `claude-haiku-5-5`（`agents.baseline.haiku_agent_config()`；測試確認兩者只差 model）：

| 項目 | 值 |
|---|---|
| prompt | `baseline/v1`（system 與 developer prompt 與本機 agent 相同） |
| tools | Issue Tracker 的 5 個 tools（`issue_tracker/v1`） |
| 模型 | `claude-haiku-5-5`，`max_tokens` 16000 |
| thinking | `{"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "error"}}`，beta `thinking-binding-controls-2026-08-01` |
| effort | `medium`（明確設定；也是 Haiku 5.5 的預設） |
| tool_choice | `auto`、`disable_parallel_tool_use: true` |
| 其他 | prompt caching `auto`；不送 sampling 參數；沒有 fallbacks（Haiku 5.5 沒有 server-side fallback）；SDK 重試 4 次、逾時 600 秒 |
| episode | max_turns 20；每個 episode 最多 2 次嘗試，錯誤分類同 decision 009（模型輸出的錯誤不重試） |

- 與本機 agent 的差異是模型、adapter（Anthropic Messages API vs. Ollama）與 thinking 的方式；比較的是兩個 **agent**，不只是兩個模型。
- `refusal`（安全分類器拒絕）照 decision 009 記為 refusal，計為未成功，也可以成為 source 規則 B 的失敗 source。
- **沒有試跑**：設定沒有先在 dev 上驗證。若 API 以 400 拒絕 request（harness 錯誤，不計費），就停止；只修正 adapter 與 API 的相容問題
  （不改 prompt、模型或參數），記在本 decision 後重新開始。
  - adapter 的修正依 decision 009 要更新 adapter 版本，等於新的 agent 設定（config_id 不同）：程式不會合併新舊設定的目錄，
    所以要以新設定**重新跑完整的 eval baseline（108 個 episode）與 sweep**；修正前的目錄只保留為中止的紀錄，花費仍計入總預算。
  - 沒有改變設定的中斷（例如網路或 API 的暫時錯誤）：baseline 以新的目錄重跑，採計規則同 Study 2（每個 (task, rollout_index)
    取最早結束者）；sweep 以 `--resume-from` 接續。依 decision 034 凍結的完整性規則，列入中斷的 baseline 目錄時，
    baseline 會標示為「資料不完整」（分級檢定仍照常計算），這一點照實報告，不修改規則。
- 付費的 Study 2 run 只接受凍結的 Haiku agent（config_id 等於 `haiku_agent_config()`），程式在每個入口檢查。

## 3. 費用與預算（凍結）

- 價格（官方價格頁，2026-10-22 核對）：prompt ≤ 100,000 tokens 時，input $0.10、5 分鐘 cache 寫入 $0.125、1 小時 cache 寫入 $0.20、
  cache 讀取 $0.01、output $0.50（每百萬 tokens）；prompt 超過 100,000 tokens 時為 $0.50、$0.625、$1、$0.05、$2.50。
  prompt 的長度包含 cache 讀寫的 tokens。程式依每個 request 的 prompt 長度選擇費率（`agents.baseline.PRICING`）。
- **帳戶層級的計價條件**：API 在 `usage` 回報 `inference_geo`、`service_tier`、`speed`。`inference_geo` 為空或 `global` → 標準價；
  `us`（US-only 推論，工作區的資料落地設定）→ × 1.1；其他值、非 standard 的 service tier 或 fast 模式 → 成本未知 → 停止（decision 014）。
- **總帳**：`~/.agent-gym/study2-haiku-billing.jsonl`（`experiments/billing.py`）。放在任何 checkout 之外：同一台機器的所有 checkout
  與 worktree 共用，不會被 `git clean` 刪除。每次呼叫**之前**寫一行預留（這次呼叫的最壞成本），呼叫之後寫結算（實際成本）或作廢
  （沒有 response、不計費）；每行寫入後 fsync。已花費 = 已結算 + 沒有結算也沒有作廢的預留（以最壞成本計：程序在呼叫途中被中斷時
  保守地計入）。最後一行寫到一半、格式不符 → 拒絕任何付費 run，人工檢查。
- **總帳被重設的檢查**：`runs/` 中這個 agent 的目錄的 manifest 合計大於總帳 → 拒絕（總帳被刪除或換了檔案）。
- **總上限 70 美元**：eval baseline、sweep 與所有中止或中斷的嘗試合計。
  - eval baseline：所有 baseline 嘗試合計 ≤ 15 美元（離線估算 0.5–8.6 美元）；`--max-cost-usd 15`，程式扣除總帳中先前 baseline 的花費。
  - sweep：`--total-cost-usd 70`；這一次的上限 = 70 − 1（安全邊際）− 總帳的全部花費。
  - 一般停止：每個 episode / rerun 開始前檢查上述上限。
  - **保險絲**：每次呼叫前，開始時的總花費 + 這次 run 已結算 + 這次呼叫的最壞成本 > 70，或這次 run 已有無法計價的呼叫（以最壞成本
    計入）→ 在呼叫之前停止。最壞成本 = 1.1 ×（prompt 估計 × 最貴的 input 類費率 + 16000 × 最貴的 output 費率），prompt 以每 2 個字元
    一個 token 估計（偏高）。所以合計不會超過 70 美元。被打斷的那一次嘗試不寫 manifest 列，它已計費的呼叫在總帳上。
    - sweep：等同預算停止（寫 summary、標示因預算停止）。
    - baseline：不寫 summary（與中斷相同）；分析由 manifest 重建，依 decision 034 的完整性規則標示為資料不完整。
  - 連線中斷或逾時（沒有 response、沒有 HTTP status）：伺服器可能已經計費 → 以這次呼叫的最壞成本結算；有 HTTP status 的錯誤回應不計費 → 作廢。
  - 總帳上有無法計價的呼叫 → 不再開始任何付費 run（decision 014）。
  - 殘留風險：SDK 在同一次呼叫內部的自動重試（連線中斷或逾時後重送），若伺服器對前幾次也計費，只有最後一次有對應的結算；
    只在網路錯誤時發生。
- **一次只跑一個付費程序**：總帳旁的檔案鎖（`fcntl.flock`，路徑先正規化），拿不到就拒絕，不等待。程序鎖、累計上限與保險絲都在
  `run_split` / `run_sweeps` 及其內部函式中檢查，呼叫端不能指定其他總帳；每個 run 用一個只能使用一次、由 `paid_session` 發出的 session。
- 付費 run 的目錄必須是 repo 的 `runs/` 的直接子目錄（從 repo 根目錄執行、`--run-name` 不含 `/`），總帳的核對才看得到它。
- 帳戶層級的計價條件也套用在 decision 009 的 Opus 付費路徑（第一輪的 Optional 付費 run，從未正式執行）的成本計算上。
- 時間：sweep 約 1.7 萬次呼叫、循序執行，預計 10–30 小時。

## 4. 協定（與 Study 2 相同，只有 sweep 的順序不同）

- eval baseline R = 3（108 個 episode）；source 規則 B；plan (2, 10, 48, 0.3)、判定 `bonferroni95`；主要規則「排除 max_turns 後仍合格」；
  分析程式 `experiments.study2_analysis` 不變（B = 10,000、seed 0）。RQ1–RQ3 的定義、分類與結論規則都與 decision 034 相同。
- 108 個 episode 全部成功 → 天花板，不跑 sweep（程式檢查：此時付費 sweep 的 CLI 拒絕執行）。
- **sweep 的順序**：先跑失敗的 source（依 pool 順序），再跑其他 source（依 pool 順序）。各 source 的 sweep 互相獨立，順序不影響任何估計值；
  在預算有上限時，這樣能先完成 RQ1 的母體。
  - 「失敗的 source」= source trajectory 的 `grade().success` 為 False，也就是 decision 034 §2 的 RQ1 母體。規則 B 選到的 refusal
    若最終狀態仍通過評分，排在其他 source 之中，也不屬於 RQ1 的母體（與 Study 2 相同）。
  - 失敗優先的順序只接受於付費的 Study 2 sweep；第一輪與本機 Study 2 的核對仍只接受 pool 順序。
- 預算用完 → 停止並標示資料不完整；RQ2 只用 sweep 全部完成的 seed 區塊（decision 034 §8）。

## 5. 跨模型比較（事先登記：只做描述）

- 兩個 agent 各自的 Study 2 分析並列：各級別 success、成功但不安全、分級檢定、RQ1 的判定與 k / n、行為性與機械性 cliff 的條數。
- **不做模型之間的假設檢定**（只有一組 36 題、兩個 agent）；不合併兩個 agent 的資料；不以 Haiku 的結果改寫 Study 2 的結論。
- 結論的分類沿用結案時的三類：已證實的結論、未通過的檢定、探索性發現。

## 6. 輸出

- run 目錄（gitignored）：`runs/haiku-study2-eval-<日期>`、`runs/haiku-study2-sweep-<日期>-<n>`。
- 報告：`docs/results/study2-haiku/` 與 README 的 `<!-- study2-haiku:start -->` 段落（`experiments.study2_report`）。
  報告屬於哪個 agent 依 baseline 目錄 run.json 的 config_id 判定（不是凍結的本機 agent 也不是 Haiku agent → 拒絕）；
  報告程式拒絕把一個 agent 的報告寫進另一個 agent 的目錄或其子目錄。報告中的花費是列出的目錄的 manifest 合計；全部付費呼叫以總帳為準。
- credential 完全由 Anthropic SDK 解析；程式不讀取、不保存任何 key，由使用者自行在環境中設定。

## 為什麼

- 使用者的預算下，Haiku 5.5 是唯一能跑完整協定（baseline 與 sweep）的模型；完整協定才能比較「從哪一步開始救不回來」。
- agent 除了模型以外都沿用 decision 009，協定除了 sweep 順序以外都沿用 decision 034，所以差異只來自 agent，結果可以並列。
- 費用、上限與停止條件在第一次付費呼叫之前寫下並由程式檢查。

## 放棄的替代方案

- **先試跑 dev**：使用者決定不做。
- **Opus 5.5 / Sonnet 5.5**：預算內做不了 sweep。
- **維持 Study 2 的 pool 順序**：預算用完時，失敗的 source 可能散落在沒跑到的 seed 區塊，主要的 RQ1 會不完整。
- **只 sweep 失敗的 source**：會改變 RQ2 與分類的母體，與 Study 2 不能並列。

## 對實驗的影響

- 新增 Haiku 的 agent 設定、付費 Study 2 的檢查（只限 eval、只限 Haiku、上限由程式計算）、sweep 的順序選項與報告的輸出位置。
- 第一輪、Study 2 的程式行為、輸出與封存都不變（測試比對）。

## 付費前審查（2026-10-22；任何付費呼叫之前）

三個角度的對抗式審查（預算安全、Haiku 5.5 的 API 相容性、協定與既有行為）確認的問題，都已修正並加上測試：

- **預算可能超過 70 美元**（blocker）：原本的上限只依列出的目錄計算；重新開始的 baseline、漏列的目錄、adapter 修正後的新設定、
  中斷而沒有寫進 manifest 的嘗試，花費都可能漏算。→ 改為總帳（§3），所有付費呼叫都寫進同一份帳，上限一律以總帳計算。
- 第二輪審查（blocker）：總帳原本在 checkout 的 `runs/` 中（其他 worktree 看到的是空帳）→ 移到 `~/.agent-gym/`，並核對 manifest；
  Python API 可以指定其他總帳或直接呼叫內部函式 → 總帳不能指定，檢查放進內部函式；0.5 美元的邊際不足以涵蓋最壞的一個 episode
  → 每次呼叫前的預留與保險絲，邊際改為 1 美元。其他：呼叫途中被中斷的花費（預留）、凍結設定的檢查、中斷重跑後的天花板判定改用採計的
  episode、報告的歸屬改用 run.json 的設定、拒絕寫進另一個 agent 的子目錄、`sources` 型別錯誤時沿用原本的檢查。
- 第三輪審查（blocker）：保險絲原本不計入無法計價的呼叫 → session 自己記帳，出現無法計價的呼叫後下一次呼叫前就停止；
  （major）sweep 中保險絲留下的 harness 中止列會讓目錄無法分析 → 保險絲改為 BaseException，不留下中止列，等同預算停止（測試以
  嚴格核對確認）；其他：session 只能使用一次且必須由 `paid_session` 發出、run 目錄必須在 `runs/` 之下、總帳核對改為遞迴掃描、
  天花板判定只看採計的 episode 是否全部成功、誤用 session 的檢查在建立目錄之前。
- 第四輪審查（blocker）：baseline 在 episode 中途觸發保險絲時寫出的 summary 與 manifest 不一致，目錄無法讀取 → 不寫 summary，
  當成中斷（測試以 `load_runs` 與 source 選擇確認可讀）；（minor）連線中斷或逾時原本被作廢 → 以最壞成本結算。
- **兩個付費程序同時執行**（major）：各自以為還有整份預算。→ 程序鎖。
- **帳戶層級的加價沒有計入**（blocker）：工作區預設 US-only 推論時，實際費用 × 1.1，而帳上照標準價。→ 依 `usage` 回報的條件計價（§3）。
- **重啟程序與程式不符**（major）：adapter 修正會產生新的設定，不能沿用先前的 episode。→ §2 改寫。
- 次要：天花板時程式仍會跑 sweep（→ 由程式拒絕）；「失敗的 source」的定義寫明（§4）；失敗優先的順序只接受於付費的 Study 2；
  多餘的 `--total-cost-usd` 在任何副作用之前就拒絕。

## 執行紀錄

- **2026-10-23，第一次啟動沒有 credential**：`runs/haiku-study2-eval-20261023` 在第一次呼叫時，Anthropic SDK 在送出 request **之前**
  以 `TypeError`（找不到認證方式）停止。沒有任何 request 送到 API、沒有任何 episode、沒有花費：總帳只有一筆預留與作廢（$0），manifest
  只有一列 0 次計費呼叫的 harness 中止。這個目錄**不列入 eval baseline 的分析**（它不含任何模型輸出；列入只會把 baseline 標成中斷，
  不改變任何數字），保留在 `runs/` 作為紀錄。這個處理在任何 Haiku 結果之前決定。正式的 eval baseline 改用新的目錄名稱重新開始。
- **2026-10-23，正式的 eval baseline**（`runs/haiku-study2-eval-20261023-b`，使用者在自己的終端機執行）：108 個 episode、956 次呼叫，
  沒有中止的嘗試、沒有因預算停止；總帳 $0.117（與 summary 相同），沒有無法計價的呼叫；956 次回應的模型都是 `claude-haiku-5-5`。
  - **108 / 108 success**，三個級別都是 1.000；全部安全，沒有 forbidden action、constraint violation 或 tool 錯誤；平均 turn L1 5.6、L2 9.0、L3 12.0。
  - 依 §4，完全天花板 → **不跑 sweep**（程式也會拒絕）。RQ1 對 Haiku 不適用；分級檢定無效（L1 − L3 = 0）。
  - 報告：`docs/results/study2-haiku/` 與 README 的 study2-haiku 段落（`experiments.study2_report`）；分析：`runs/haiku-study2-analysis-20261023`。
  - 跨模型比較（§5，只做描述）：同一組 36 題 × 3，本機模型 101 / 108、Haiku 108 / 108。不做模型之間的檢定；不改寫 Study 2 的結論。
  - 總花費 $0.117，總預算 70 美元中其餘的沒有使用。這一輪比較完成，沒有後續的付費 run。
