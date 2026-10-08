# 018 — sweep 續跑、summary 全欄核對、目錄識別（第八輪審查後；在任何 LLM 結果之前）

日期：2026-10-08　狀態：採用（補足 decision 012 §1 / 017 的執行與核對細節；不改主要 metric、資料切分、模型或 SweepPlan）

## 背景

第八輪對抗式審查（decision 017 的實作）確認：

1. **中斷後重跑的 sweep 沒有可用的流程**：012 §1 規定「重跑必須用新目錄，報告時列出所有目錄」。但 run_sweep 的 CLI 一律重跑全部 source，
   017 §5.1 又規定同一個 source 在兩個目錄都完成 → ValueError。結果只要中斷時已有任何 source 完成，「中斷目錄 + 重跑目錄」就無法一起分析；
   只分析重跑目錄則會漏掉第一個目錄已計費的成本與中止嘗試，而且已完成的 source 會被重複付費。
2. analyze_baseline 只核對 summary 的少數欄位（017 §4.3 寫的是「每個欄位」）：從完整 run 刪掉失敗的 rollout 不會被發現，主指標從 0.933 變成 1.0。
3. analyze_sweep 沒有核對 summary.sources 等於 select_sources（017 §5.1）；run_sweeps 以子集呼叫時，只有少數 family 完成也會報出 cliff 比例。
4. 其他：同一個目錄以不同大小寫的路徑傳兩次（macOS 不分大小寫）不會被去重；symlink loop 漏出 RuntimeError；不存在或沒有 manifest 的 baseline
   目錄被當成中斷的 run；未完成 sweep 的 manifest 列完全不驗證；response_model 不只一種時 sweep 仍報時間窗；sweep 報告沒有 multi-tool turn（009 §4）。

## 決策

### 1. sweep 續跑（實作 012 §1 的「重跑用新目錄」）

- `run_sweeps(..., skip=...)`：sources 必須**剛好涵蓋全部 eval task**（`select_sources` 的形狀，否則 ValueError）。skip 中的 source 不在這個目錄執行，
  summary 新增 `skipped`；skip 只能包含有 source 的 task。
- CLI `--resume-from DIR [DIR ...]`：先以 analyze_sweep 的完整核對讀這些目錄，plan 與 criterion 必須與這次相同，否則不執行。
  已完成的 source 列入 skip，只執行其餘的 source。已完成的 source 因此不會重複付費。
- sweep 格式升為 `agent-gym/sweep-run/v3`（多了 skipped；目前還沒有任何真實資料）。
- run_sweeps 自己的彙總改以 **eval pool 的全部 family** 判定是否做 bootstrap（012 §1）：任一 family 在這個目錄沒有完成的 sweep → None。
  正式的彙總仍由 analyze_sweep 跨目錄重算。
- analyze_sweep：每個目錄的 skipped 都會列出。skipped 的 source 若沒有在任何一起分析的目錄中完成，就屬於 not_completed，報告標示資料不完整。
  只列 skipped 不會導致錯誤，但會被看見。這個目錄的 manifest 不得有 skipped source 的列。

### 2. sweep summary 的語意核對（補 017 §5.1）

- `sources` 必須等於 eval pool 的全部 task（依 pool 順序），`missing_sources` 必須等於其中沒有 source 的 task。
- 待執行 = sources − missing − skipped（依 pool 順序）。未因預算停止時，completed 必須等於待執行；因預算停止時，completed 必須是待執行的前綴。
- 有 summary 的目錄不得有 harness 中止。尚未用完嘗試的 infrastructure 中止，只能出現在因預算停止的目錄（重試被預算擋下）。
- 所有目錄、所有 rollout（含未完成的 sweep）的 manifest 列都要驗證：attempt 必須是 1..n 連續且不重複，只有 infrastructure 中止可以重試；
  checkpoint_turn 必須在 source 的合法範圍內；checkpoint_id 必須等於 `make_checkpoint(source, k)` 的 id。
- `config_id` 必須是 str 且與 source 相同。

### 3. baseline summary 全欄核對（落實 017 §4.3）

- 頂層 key 必須剛好是 run_split 寫出的欄位。split = eval，n_tasks = eval pool 大小，rollouts 為 int >= 1，config_id 與 manifest 相同，
  stopped_by_budget 為 bool。
- results 的 (task_id, rollout_index) 必須等於 run_split 的排程（rollout_index 外層、eval pool 順序內層）：未因預算停止時是全部
  rollouts × 15，因預算停止時是排程的前綴。
- 每一列的 key 剛好是 EpisodeResult 的欄位，family 等於 eval pool。完成者的 trajectory_safe、forbidden 計數、cost_usd、model_turns 等於
  trajectory 的 grade() / usage()；未完成者這些欄位為 None。
- spent_usd、model_calls、unknown_cost_calls 必須與 manifest 的 cost_usd、billed_calls、unknown_cost_calls 加總相符（成本容許浮點加總順序造成的誤差）。

### 4. 目錄識別與存在

- 「同一個目錄」以檔案系統識別（`os.stat` 的 device + inode）判定，不比較路徑字串。macOS 不分大小寫時，以大小寫不同的路徑傳兩次也會被拒絕。
  stat 失敗（不存在、symlink loop 等）→ ValueError。
- baseline 與 sweep 的每個輸入目錄都必須存在且有 `manifest.jsonl`。中斷時還沒有任何嘗試寫入 manifest 的目錄，沒有可報告的內容
  （那次嘗試已計費的 call 本來就不在任何紀錄中，見 NEXT 已知限制），不要傳入分析。

### 5. 報告

- response_model 不只一種時，sweep 的時間窗也不報（與 017 §4.5 的 baseline 一致）。
- sweep 報告列出 multi-tool turn（009 §4）：source 的（所有 turn）與 sweep rollout 的（k 之後的 turn）分開列。
- sweep 中止嘗試的部分安全指標標示「計數含 checkpoint 前綴（source 自己的步驟）」。數值不改：manifest 的部分軌跡本來就含前綴（decision 009）。

## 為什麼

- 付費 sweep 是最貴的一步。被中斷後必須能只補跑缺的部分，而且所有目錄的成本都要進報告。
- summary 是 run 自己寫的摘要，分析不能因為它被改動或截斷就算出不同的主指標。
- 路徑字串不能代表目錄身分（不分大小寫的檔案系統、symlink）。

## 放棄的替代方案

- 重複完成時自動挑一個目錄（例如最早的）：等於在結果上挑選；017 §5.1 已規定不挑選。
- 續跑時把 skipped 的 source 結果複製到新目錄：同一份資料會出現在兩處，成本與稽核都會混亂。
- 只在分析時容許子集 summary：run_sweeps 自己的彙總仍會在少數 family 上做 bootstrap，誤導讀者。

## 對實驗的影響

- 付費 sweep 中斷後：`run_sweep ... --resume-from runs/<舊目錄>`，接著 `analyze_sweep runs/<舊目錄> runs/<新目錄> --baseline-run ...`。
- 不影響任何估計方法或判定規則。
