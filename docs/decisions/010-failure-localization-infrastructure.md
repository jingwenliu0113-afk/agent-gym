# 010 — SPEC §6 failure localization infrastructure 與 synthetic policy 驗證

日期：2026-10-08　狀態：採用（infrastructure；在任何真正的 LLM run 之前）

## 背景

SPEC §6：從 trajectory checkpoint 重新 rollout，估計 P(success | checkpoint)；先少量掃描，再對疑似變化處加密取樣；統計以 task 為
cluster，使用 task-level bootstrap，不把同一 task 的多次 rollout 當獨立樣本。使用者指示：先用 synthetic policy 驗證
P(success | checkpoint) 與 bootstrap 流程，不跑真正的 LLM、不擴題庫、不宣稱統計上的泛化。

## 決策

### 1. Synthetic policy（`agents/synthetic.py`）：已知真實 recoverability 的測試替身

- 依 task 的 oracle 行動；每個 turn 以固定機率 q 改為發出一個**不可逆的錯誤**：在「受保護 issue」上留言
  （受保護 issue = success_constraints 中 `issue_unchanged` 涵蓋 comments 的 issue 裡 id 最小者；comment 無法刪除，
  constraint 永遠失敗）。錯誤之後照常繼續 oracle（success 已不可能）。
- **對 conversation 無狀態**（decision 006）：oracle 進度由 conversation 中非錯誤的 assistant tool call 數決定；
  每次決策的隨機數 = `sha256(rollout_seed, canonical_json(messages))` 映射到 [0, 1)，因此同 seed、同 conversation 必得同結果。
  rollout_seed 是取樣隨機性的替身，不屬於 agent config（同一 agent 的不同 rollout 共用 config_id）。
- **解析解**：設 oracle 長度 n、checkpoint 前綴中已完成 r 個 oracle call：
  - 前綴中已有錯誤 → P(success | k) = 0；
  - 否則 P(success | k) = (1 − q)^(n − r + 1)（剩餘 n − r 個 oracle call 加上結束 turn，每個 turn 都可能犯錯）。
  成功的續跑恰好使用 n − r + 1 個 turn，只要 max_turns ≥ n + 1 就不受 max_turns 影響。
- 因此一條「在第 m 個 turn 犯錯」的 source trajectory，其真實曲線在 k = m 與 k = m + 1 之間從 (1 − q)^(n − m + 1) 掉到 0：
  這是可驗證的 **recoverability cliff**；沒有犯錯的 source 則是平滑上升的曲線（gradual，無 cliff）。

### 2. Rollout 與結果分類（`replay/localization.py`）

- 每個 rollout = `rerun(make_checkpoint(source, k), policy_factory(...))`，一律經由 `replay.aborts.run_logged` 執行（manifest，decision 009）。
- `policy_factory` 產生的每個 policy 必須與 source 的 agent 有相同 config_id（否則不叫 rerun）。
- 結果分類（decision 009 預先登記）：success、failure、refusal（最後一個 turn 的 stop_reason == "refusal"）、
  agent_error（agent_output 中止）、aborted（infrastructure 中止且已無可用嘗試）。另記 `termination == "max_turns"` 與
  stopped_without_action（final_answer、未成功、整個續跑沒有任何 mutation）。
- **主估計** p̂(k) = successes / n（refusal、agent_error、aborted 都計為非 success）；敏感度：p̂_completed = successes /（success + failure）。
  每個 checkpoint 同時報告 Wilson 95% CI、max_turns 終止比例、各類計數。
- k = 0 也以 rerun 估計；source trajectory 本身不計入任何 P(success | k)。
- 排程以 round-robin 交錯各 checkpoint（不是一個 checkpoint 跑完再下一個），降低時間效應；manifest 的 utc_time 用於分時間窗報告。

### 3. 掃描協定（coarse → refine）

- 合法 checkpoint：k ∈ [0, len(turns) − 1]（decision 006：k < max_turns，且前綴每個 turn 都有 tool call）。
- **Coarse**：k ∈ {0, s, 2s, …} ∪ {K}（K = 最後一個合法 checkpoint），每點 `n_coarse` 個 rollout。
- **Refine**：相鄰 coarse 點 (a, b) 若 p̂(a) − p̂(b) ≥ `drop_threshold`，則 a 與 b 之間所有 checkpoint（含兩端）補到 `n_fine` 個 rollout。
- **Cliff 判定**（per source）：refine 後相鄰 checkpoint 的最大下降 Δ* = max_k [p̂(k) − p̂(k+1)]，位置 k*；
  若 Δ* ≥ `drop_threshold` 且兩點的 Wilson CI 不重疊，標記為 cliff，否則為 gradual。
- 參數（`SweepPlan`）在看到任何真實結果之前，依 synthetic 模擬的偵測力選定並寫入下一份 decision log；本 decision 先固定
  synthetic 驗證用的預設值：`coarse_stride = 2`、`n_coarse = 8`、`n_fine = 24`、`drop_threshold = 0.3`。

### 4. 統計（`replay/stats.py`）

- 單一 checkpoint：同一 checkpoint 的 rollout 在給定 checkpoint 下獨立同分佈 → Wilson score interval。
- 跨 task 的彙總（例如 Δ*、cliff 比例）：**family 分層的 task-level（cluster）bootstrap**（decision 005）：每個 family 內有放回重抽
  相同數量的 task，統計量為 task 等權平均，B = 10000，percentile 95% CI。每個 task 只貢獻一個值；同一 task 的多次 rollout
  不視為獨立樣本（rollout 的雜訊已包含在該 task 的值中，由 cluster 重抽一併反映）。
- bootstrap 以固定 seed 的 `random.Random` 實作，結果可重現。

### 5. Synthetic 驗證（測試）

- **估計正確性**：在 dev split + smoke tasks 上，以 synthetic policy 產生 source trajectory，對全部 checkpoint 估計，
  Wilson 95% CI 涵蓋解析解的比例在固定 seed 下 ≥ 0.85。
- **Cliff 定位**：在第 m 個 turn 犯錯的 source，refine 後 k* = m，且被標為 cliff；沒有犯錯的 source 不被標為 cliff。
- **Bootstrap**：對已知真值的模擬 task 集合重複實驗，95% CI 涵蓋真值的比例接近名目值（測試門檻 ≥ 0.88）；分層重抽保持每個 family 的 task 數。
- **流程完整性**：所有 rollout 都出現在 manifest；infrastructure 中止會重試並記錄；agent_output 中止計為非 success。

## 為什麼

- 先用解析解已知的 policy 驗證，才能分辨「估計流程錯了」與「agent 真的有 cliff」；否則第一批 LLM 結果無從檢驗。
- comment 不可撤回，是本 domain 中最乾淨的不可逆動作，讓 cliff 的位置與大小都可精確計算。
- Wilson 而非常態近似：小樣本與接近 0 / 1 的機率下涵蓋率較好。
- cluster bootstrap 對應 EXPERIMENT_RULES「不把 10 次同 task rollout 當 10 個獨立 task」。

## 放棄的替代方案

- 直接用 LLM 驗證流程：成本高、沒有真值、無法區分流程錯誤與模型行為。
- 兩階段 bootstrap（task 內再重抽 rollout）：在 task 可交換的假設下，單階段 cluster bootstrap 已反映 task 內雜訊；兩階段會重複計入。
- 以 CI 不重疊作為 refine 觸發條件：coarse 樣本小、CI 很寬，會漏掉真正的變化；refine 用點估計門檻、cliff 判定才要求 CI 不重疊。

## 對實驗的影響

- 真正的 LLM sweep 必須使用本協定與屆時寫入 decision log 的 SweepPlan；source 的選取規則（每個 eval task 第一條完整 source，不挑選）
  在第一次 LLM run 前另行固定。
- 成本取決於 SweepPlan 與軌跡長度；付費 sweep 前須估算並經使用者同意。
- 15 題 eval 的結果只能當 infrastructure validation / preliminary observation，不得宣稱統計上的泛化；也不得依此擴充題庫。
