# 013 — failure localization 協定的修正與澄清（在任何真實結果之前）

日期：2026-10-08　狀態：採用（修正 decision 010 / 011；此時沒有任何 LLM 結果，只有 synthetic 模擬）

## 背景

§6 測試撰寫後的 critic 審查（對 decision 010 契約與實作逐條比對）找出幾處「規格與程式不一致」或「規格沒有寫清楚」的地方。
依 CLAUDE.md，先記錄差異再處理。這些修正都發生在任何真實 rollout 之前；decision 011 的偵測力模擬依修正後的規則重跑。

## 決策

### 1. 門檻比較用精確有理數

`p̂(a) − p̂(b) >= drop_threshold` 以 `Fraction(successes, n)` 計算下降，`drop_threshold` 取其十進位表示（0.3 即 3/10）。
浮點減法會把剛好等於門檻的下降算成略小（例如 7/10 − 4/10 = 0.29999999999999993），使 decision 010 的「含等號」失效，
也破壞「同大小取 k 較小者」。`largest_drop` 回傳精確下降最接近的 float；Wilson CI 不重疊仍以 float 比較。

### 2. Refine 反覆進行，直到沒有「未加密的大下降」（修正 decision 010 §3）

差異：decision 010 §3 把 Δ* 寫成連續 k 的 max_k [p̂(k) − p̂(k + 1)]，程式則對「相鄰的已估計 checkpoint」計算，而且 refine 只做一次。
當一個 refine 區間的端點在加密後改變了估計，它與旁邊**未加密**的 coarse 點之間可能出現 >= 門檻的下降，但中間的 k 從未估計，
cliff 的位置無法確定（兩種寫法會給出不同結論）。

修正：
- 待加密區間 = 依 k 排序的相鄰已估計點 (a, b) 中，精確下降 >= drop_threshold，且 [a, b] 內有任何 checkpoint 的 rollout 數 < n_fine
  （包括從未估計的中間點）者。
- coarse 之後反覆：取所有待加密區間，把區間內（含兩端）所有 checkpoint 的目標數提高到 n_fine（從既有數量接續，round-robin），
  直到沒有待加密區間。每一輪都使總 rollout 數增加且有上限 n_fine × L，因此一定結束。
- 結果：任何 >= 門檻的相鄰下降都落在連續的 (k, k + 1) 上、兩端都有 n_fine 個 rollout，因此 Δ* 的位置明確；
  `detect_cliff`（相鄰已估計點）與 decision 010 §3 的連續 k 定義在 cliff 判定上一致。
- `SweepResult.refined_intervals` = 所有輪次標記過的區間（去重、依 a 遞增）。
- 排程由單一函式 `replay.localization.run_protocol` 決定，`sweep()` 與偵測力模擬（`experiments/sweep_power.simulate_sweep`）共用，
  避免兩份實作不一致。

### 3. Δ* 不截斷

只上升的曲線 Δ* 為負（例如 −0.5）。單一 source 報告原值；跨 task 彙總（decision 012 的平均最大下降）也用原值，不在 0 截斷。

### 4. `checkpoint_truth` 只對 synthetic policy 自己產生的軌跡有定義

- 前綴每個 turn 必須剛好一個 tool call，且是錯誤 call 或下一個 oracle call（on-policy）；否則 ValueError。
  理由：一個同時包含錯誤 call 與 oracle call 的 turn 實際上已經留下不可逆的留言，但「剛好一個 call 才算錯誤」的計數規則會給出 > 0 的
  「真值」；off-policy 的前綴也沒有解析解。
- 0 <= k < len(turns)（與 valid checkpoint 一致）；trajectory 的 policy 必須是 synthetic；傳入的 mistake 必須與
  `model_config["mistake"]` 逐 byte 相同、mistake_rate 必須與 `model_config["mistake_rate"]` 數值相等；否則 ValueError
  （避免以錯誤的 q 靜默算出錯誤真值）。
- `synthetic_agent_config` 以 `float(q)` 保存 mistake_rate，使 q = 0 與 q = 0.0 是同一個 agent（同一個 config_id）。

### 5. 其他澄清

- `stopped_without_action` 的「未成功」指 `grade().success` 為 False（grader 判為成功的 refusal 不算）。
- decision 010 §5 的「在第 m 個 turn 犯錯 → k* = m 且標為 cliff」只在 cliff 前的真值遠高於門檻時成立（測試使用 q = 0.05）；
  真值本身低於門檻的錯誤（例如 q = 0.25、n = 7、m = 0，真值約 0.10）不會、也不應被標為 cliff。
- `sweep()` 中 `run_logged` 的事前檢查錯誤（例如 manifest 已有相同的嘗試）原樣往外拋，不執行任何 rollout。

### 6. bootstrap 的 estimand（澄清 decision 005，不改協定）

decision 005 已寫明正式統計的母體 = eval split 的 15 個 instances（task 固定，只有 rollout 雜訊是隨機的）。在這個 estimand 下，
family 分層 task-level percentile bootstrap 偏保守（模擬涵蓋率 0.90 – 1.00，依抽到的 task 而定）。若改問「同一題型的其他 instance」
（superpopulation），在 5 個 family × 3 題的設計下涵蓋率只有約 0.86，低於名目 0.95 與 decision 010 §5 的 0.88 門檻
（每層只有 3 題時 percentile bootstrap 低估變異）。這與「15 題 eval 不得宣稱統計上的泛化」一致；測試以 strict xfail 記錄這個已知限制。

## 對實驗的影響

- decision 011 的偵測力模擬依第 1、2 點重跑，結果更新在 011。
- 這些修正改變了 refine 的協定；在任何真實 sweep 之前完成，不涉及看結果後調整。
