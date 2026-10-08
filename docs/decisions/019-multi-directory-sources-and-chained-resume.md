# 019 — 多目錄 eval baseline 的 source、sweep 開始紀錄、鏈式續跑（第九輪審查後；在任何 LLM 結果之前）

日期：2026-10-08　狀態：採用（補足 decision 012 §3 與 018 §1；不改主要 metric、資料切分、模型、SweepPlan 或 criterion）

## 背景

第九輪對抗式審查（decision 018 的實作）確認兩個缺陷：

1. **鏈式續跑會重複付費**：`--resume-from` 只把給定目錄中「完成的 sweep」列入 skip。給定目錄自己略過的 source（在更早的目錄完成）
   不會被擋下，於是被重新 sweep、再付一次費。之後「所有目錄一起分析」因同一個 source 在兩個目錄完成而 ValueError（017 §5.1 不挑選）；
   只分析較新的目錄則看起來完整，卻漏掉最早目錄的成本。若中間的續跑目錄被中斷（沒有 summary），連它略過了哪些 source 都沒有紀錄。
2. **eval baseline 跨多個目錄時，source 沒有預先登記的規則**：中斷或因預算停止後以新目錄重跑（009、012 §1），016 A1 規定所有目錄都計入，
   所以同一個 task 可能有兩條 rollout_index 0。012 §3 沒有規定取哪一條，`select_sources` / `run_sweep` / `analyze_sweep` 也只接受一個
   baseline 目錄（argparse 給兩次時靜默保留最後一個）。結果是使用者必須在看過結果後挑選目錄，這正是 012 §3 禁止的事。

同一輪中審查判定「不是缺陷」、但成本很低而一併處理的：沒有 summary 的目錄無從核對 plan；續跑時已沒有要跑的 source 仍會建立無法分析的
目錄；NEXT 的續跑指令沒有寫 `--run-name`；續跑時沒有顯示先前已花的成本；README 的測試時間、extra 與 decision 範圍過時。

## 決策

### 1. 多目錄 eval baseline 的 source（補 012 §3，看結果前固定）

- eval baseline = **所有** eval baseline 目錄，與 analyze_baseline 使用的清單相同（中斷的目錄也要列入）。`select_sources`、
  `run_sweep --baseline-run`、`analyze_sweep --baseline-run` 都接受多個目錄，讀取與核對方式與 analyze_baseline 相同（`load_runs`）。
- 每個 eval task 的 rollout_index 0 = 所有目錄中**最早結束**的那一條：比較該 rollout 最後一次嘗試的 manifest utc_time（換算成 UTC）。
  interrupted 的 rollout（沒有結束，017 §4.1）不算。時間完全相同 → ValueError（無法決定）。
- 那一條完成（success / failure / refusal）→ 它就是 source；agent_error / aborted → 該 task 沒有 source。依 012 §3，不得以其他 rollout
  補上，其他目錄的 rollout_index 0 也一樣。
- 只有一個目錄時，結果與原本相同。
- CLI 的 `--baseline-run` 可給多個值，也可重複出現，所有值都會保留。

### 2. sweep 開始紀錄 `run.json`（sweep 格式升為 `agent-gym/sweep-run/v4`）

- run_sweeps 在第一次嘗試之前寫入 `run.json`：format、plan、criterion、config_id、skipped（pool 順序），以及 sources（物件：
  每個 eval task → source 的 trajectory_id，沒有 source 時為 null）。
- analyze_sweep：每個目錄都必須有 v4 的 `run.json`。
  - plan / criterion 取自 run.json，所以沒有 summary 的目錄也會參與「所有目錄的 plan 與 criterion 相同」的核對。
  - summary 的 plan、criterion、config_id、skipped 必須與 run.json 相同。
  - run.json 的 sources 必須等於以這次給定的 baseline 目錄重新選出的 source。如果換了 baseline 目錄的組合而改變了 source → ValueError。
  - skipped 取自 run.json，所以被中斷的續跑目錄也會列出它略過的 source；manifest 不得有這些 source 的列。
- 只有 run.json、沒有 manifest 的目錄（第一次嘗試前就中斷），仍然不要傳入分析（018 §4）。

### 3. 續跑必須列出所有先前的 sweep 目錄

- `--resume-from` 給的任何目錄所略過的 source（run.json 的 skipped），都必須在給定的目錄中有完成的 sweep；否則拒絕執行：
  不建立目錄、不發出任何 call，並提示把更早的目錄也一起傳入。
- 續跑前印出先前目錄已花的成本，以及成本無法量測的已計費 call 數。`--max-cost-usd` 是**這一次執行**的上限，不含先前目錄的花費；
  跨目錄的總花費由使用者依印出的金額決定（012 §2：每一階段的預算由使用者決定）。
- 沒有任何要執行的 source（全部已完成或缺漏）時：CLI 印出訊息後結束，不建立目錄；run_sweeps 直接 ValueError。這樣的目錄沒有 manifest，
  本來就無法分析。

### 4. 報告與文件

- sweep 分析列出 baseline 目錄（`baseline_runs`）；分析格式升為 `agent-gym/sweep-analysis/v3`。
- NEXT 與 run_sweep 的續跑指令寫成完整形式：`--run-name <新名稱> --resume-from runs/<第一個> [runs/<第二個> ...]`，要列出所有先前的
  sweep 目錄。
- README：補上測試時間、只有 `uv sync --extra llm` 之後才會跑 SDK 契約測試，以及 decisions 範圍 001–019。

## 為什麼

- 取最早結束的 rollout 0：重跑只能補上「rollout 0 還沒結束」的 task，永遠不能取代已經存在的 source；規則只依時間，與結果無關。
- run.json 在任何花費之前寫入，所以每個有計費的目錄都記錄了自己的設定與略過清單，中斷也不會遺失。
- 付費 sweep 是最貴的一步，同一個 source 不能因為漏給一個目錄就再付一次費。

## 放棄的替代方案

- source 取最晚的目錄（以重跑覆蓋），或由使用者指定目錄：等於允許「重跑到結果滿意為止」或事後挑選。
- 續跑時自動掃描 `runs/` 找出所有先前目錄：依賴目錄命名與位置，可能把無關的目錄算進來。
- 跨目錄累計的成本上限：先前的花費已經發生；每一次付費執行前由使用者決定這一次的預算比較直接，而且先前花費會印出。
- baseline 也做 skip 式續跑：016 A1 已規定多個目錄全部計入；這不在本次範圍內。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。
- 指令：`run_sweep --baseline-run runs/<eval> [runs/<eval-rerun> ...]`；`analyze_sweep ... --baseline-run` 要給同一組目錄。
- 目前還沒有任何真實的 sweep 資料，所以格式升版不影響既有結果。
