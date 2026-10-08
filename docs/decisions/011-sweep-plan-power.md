# 011 — 真正 LLM sweep 的 SweepPlan：以 synthetic 偵測力模擬選定

日期：2026-10-08　狀態：選擇規則先登記（「決策」各節寫於執行模擬之前）；**結果：沒有合格的 plan，待使用者在文末選項中決定**

## 背景

decision 010 第 3 點：SweepPlan（coarse_stride、n_coarse、n_fine、drop_threshold）必須在看到任何真實結果之前，依 synthetic 模擬的
偵測力選定並寫入下一份 decision log。010 的 `DEFAULT_SYNTHETIC_PLAN = (2, 8, 24, 0.3)` 只用於流程驗證，不是 LLM sweep 的 plan。
本 decision 不涉及任何 LLM 呼叫或真實結果。

## 決策

### 1. 模擬方法（`experiments/sweep_power.py`）

- 以已知的 P(success | k) 曲線取代 agent：checkpoint k 的每個 rollout ~ Bernoulli(curve[k])。
- 排程（coarse 點、refine 區間）、每點估計（Wilson CI）與 cliff 判定一律呼叫 `replay.localization` 中 `sweep()` 使用的同一組函式
  （decision 013 之後為 `run_protocol`、`estimate_checkpoint`、`detect_cliff`），因此模擬與真正的 sweep 使用同一套規則。
- 每個 (scenario, plan) 重複 `reps = 400` 次，`random.Random(seed)` 固定 seed；結果可逐 byte 重現。

### 2. 情境（預先登記）

L = 合法 checkpoint 數（= source trajectory 的 turn 數）。baseline 的 max_turns = 20，本題庫 oracle 長度 1–7，預期 L 多落在 3–10。

- **cliff**：curve[k] = p_pre（k <= m）、p_post（k > m）。(p_pre, p_post) ∈ {(0.9, 0.1), (0.8, 0.2), (0.7, 0.2), (0.6, 0.1)}
  （下降 0.5–0.8），L ∈ {4, 8, 12}，m ∈ {0, ⌊(L − 1)/2⌋, L − 2}。
- **gradual**：curve 由 0.9 線性降到 0.1，L ∈ {8, 12}（每步下降 < 0.15）。
- **flat**：curve 恆為 0.5 或 0.9，L ∈ {4, 8, 12}。

### 3. 候選 plan

coarse_stride ∈ {1, 2, 3} × n_coarse ∈ {4, 6, 8, 10} × n_fine ∈ {16, 24, 32}（n_fine >= n_coarse）× drop_threshold ∈ {0.25, 0.3, 0.4}。

### 4. 選擇規則（機械化，`select_plan`）

一個 plan 合格 ⇔ 同時滿足：
1. 每個 cliff 情境：detect_rate（is_cliff 且 k* = m 的比例）>= 0.80；
2. 每個 gradual / flat 情境：flag_rate（is_cliff 的比例）<= 0.05。

合格者中選「所有情境中最大的平均 rollout 數」最小者；同分依序比較「所有情境平均 rollout 數的平均」、再依
(coarse_stride, n_coarse, n_fine, drop_threshold) 的字典序取最小。**沒有任何合格 plan → 不自行放寬門檻，記為 blocker 交給使用者決定。**

## 為什麼

- 偵測力與成本要在看到結果前一起固定；否則可以事後挑一個「剛好看得到 cliff」的 plan。
- 以最壞情境的 rollout 數為成本指標：付費 sweep 的預算必須涵蓋最長的 trajectory。
- 用 Bernoulli 曲線而不是 SyntheticPolicy：可直接控制下降大小與位置，且快到能掃完整個候選網格；判定函式與 sweep() 共用。

## 放棄的替代方案

- 直接沿用 DEFAULT_SYNTHETIC_PLAN：它是為流程驗證挑的，沒有經過偵測力與成本的比較。
- 以 LLM 試跑結果調 plan：違反「看結果前固定」。

## 對實驗的影響

- 真正的 LLM sweep 使用本 decision 選出的 plan；改動必須另立 decision log。
- 這些情境是假設；真正 trajectory 的曲線形狀未知，偵測力只對「與情境相近的曲線」成立，報告時必須說明。

## 結果（2026-10-08，`uv run python -m experiments.sweep_power`，reps = 400、seed = 0）

第一次執行後，decision 013 修正了門檻比較（精確有理數）與 refine 協定（反覆加密）；**以下數字是依修正後的程式重跑的結果**
（第一次執行的數字已作廢；結論相同）。完整輸出寫到 gitignored 的 `experiments/outputs/sweep_power.md`，可由上面的指令逐 byte 重現。

**依預先登記的規則，108 個候選 plan 沒有任何一個合格**（`select_plan` 回傳 None）。依第 4 點，不自行放寬門檻，交由使用者決定。

診斷（不改變上面的結論）：
- 失敗主要來自 **gradual 情境的誤判**：0.9 → 0.1 線性下降、L = 8（每步降 0.11）時，refine 後相鄰點很多，最大下降的 95% Wilson CI
  不重疊的機率約 0.14 – 0.17（多重比較）。cliff 偵測率 >= 0.80 的 plan 在 gradual 上的誤判都超過 0.05。
- 降低 n 雖能壓低誤判，但 (0.7, 0.2)、(0.6, 0.1) 這類下降 0.5 的 cliff 偵測率會掉到 0.5 – 0.7。

摘錄（min cliff detect / max non-cliff flag / 所有情境中最大的平均 rollout 數）：

| plan (stride, n_coarse, n_fine, threshold) | min cliff detect | max non-cliff flag | max mean rollouts |
|---|---|---|---|
| (2, 8, 24, 0.3)（decision 010 的流程驗證預設） | 0.640 | 0.150 | 127.6 |
| (2, 8, 32, 0.3) | 0.757 | 0.140 | 157.7 |
| (2, 10, 32, 0.3) | 0.833 | 0.138 | 199.6 |
| (1, 10, 32, 0.3) | 0.858 | 0.168 | 216.1 |

### 探索（未採用，只供使用者決定時參考；`criterion="bonferroni-exploratory"`）

把 cliff 判定的 CI 改為 **Bonferroni 校正**（z = Φ⁻¹(1 − 0.025 / m)，m = 相鄰已估計點的對數；其餘規則與排程相同），同樣的情境、reps、seed：

- 原候選網格（n_fine <= 32）：誤判降到 <= 0.02，但 cliff 偵測率最多約 0.53，仍無合格者。
- 另外試 n_fine ∈ {48, 64}（超出預先登記的網格）：16 個中有 11 個兩個門檻都達到；依第 4 點的成本規則選出
  **(2, 10, 48, 0.3)**：偵測 0.830、誤判 0.030、最大平均 rollout 273.9。

### 給使用者的選項與成本（15 個 eval source；成本 = 最大平均 rollout 數 × 平均每個 rollout 的估算成本，低 / 高假設見 decision 012）

| 選項 | 判定與 plan | cliff 偵測 / gradual 誤判 | 估算總成本（低 – 高） |
|---|---|---|---|
| A | 改用 Bonferroni 校正的 CI（修改 decision 010 的 cliff 判定）+ (2, 10, 48, 0.3) | 0.83 / 0.03 | $213 – $3,241 |
| B | 維持 decision 010 判定 + (2, 10, 32, 0.3)，報告時明列 gradual 曲線約 14% 會被誤判為 cliff | 0.83 / 0.14 | $155 – $2,362 |
| C | 維持判定 + (2, 8, 24, 0.3)，偵測率約 0.64、誤判約 0.15 | 0.64 / 0.15 | $99 – $1,510 |

建議 A：Gate 判定（decision 012）依賴「cliff vs gradual」的區分，gradual 被誤判為 cliff 會直接誤導是否進 Advanced；
A 是唯一兩個門檻都達到的設定。選定後必須另立 decision log，把判定正式實作到 `replay.localization`（目前只存在於模擬程式）並測試，
再以正式實作重跑本模擬確認。成本的高估情境假設每次 call 4000 個 output tokens、turn 數為 oracle 的 2 倍且不計 prompt caching，
dev sanity 之後會以實測值重估。
