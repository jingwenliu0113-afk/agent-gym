# NEXT

## 目前位置

Core：Day 1 完成；Week 1 除「#5 baseline 真正跑完整 task」以外都完成；#5 與 SPEC §6 中**不需要 API key 的部分全部完成**。
剩下的 Core 工作都必須真正呼叫 LLM（dev sanity、eval baseline、eval 上的 failure localization sweep），
因此停在 **blocker：需要使用者確認模型 / 成本 / SweepPlan，並以環境變數提供 credential**（見下）。
尚未進 Advanced（Gate 未滿足：還沒有任何真實的 failure localization 結果）。

Git：獨立 repo、分支 `main`，`origin` = private GitHub repo（github.com/jingwenliu0113-afk/agent-gym），main 追蹤 origin/main；
測試通過的 commit 一律 push（不 force push、不改寫 history）。
`d7241dc` Day 1 → `5c8ada4` Week 1 replay infrastructure → `c554735` keyless baseline adapter + attempt manifest →
`638be6d` §6 failure localization infrastructure → `7a796f3` 付費 run 執行工具 + 計費修正 → SweepPlan 偵測力模擬（decision 011）。

## 上次完成（2026-10-08）

全程 TDD（契約 stub → 測試紅燈 → 實作 → 對抗式審查 → 回歸測試 → 修正）。只用 Mac CPU；沒有使用 Mac heavy lock、遠端 RTX、Docker，
沒有呼叫任何 LLM、沒有讀取或保存任何 credential。

1. **Week 1 #5 keyless**（decision 009）
   - 固定 baseline 設定：`claude-opus-5-5`、adaptive thinking（display summarized、block_binding error）、effort medium、
     一則訊息最多一個 tool call、max_turns 20、prompt `baseline/v1`、SDK 重試 4 次 / timeout 600 秒、adapter 不另外重試。
   - trajectory / checkpoint 升 v2：保存 provider 原生 content blocks（含 thinking signature）與每個 turn 的量測 meta，
     restore 後送出的 request 與原 episode 逐 byte 相同。
   - `agents/baseline.py`：可注入 client 的 Anthropic adapter（忠實歷史檢查、stop_reason 規則、成本含 1h cache 價差），fake client 完整測試。
   - `replay/aborts.py`：每次嘗試都寫入 manifest；例外分三類（infrastructure / agent_output / harness），只重試 infrastructure（上限 2）；
     中止前已發生步驟的安全指標（PartialEpisode）；`run_with_retries` 集中重試規則。
   - 對抗式審查確認並修正 6 項（深層輸入漏出 RecursionError、文件深度上限前後不一致、例外欄位無法編碼等）。
2. **SPEC §6 failure localization infrastructure**（decision 010）
   - `agents/synthetic.py`：解析解已知的 synthetic policy（P(success | k) = (1 − q)^(n − r + 1)，犯過不可逆錯誤後為 0）。
   - `replay/stats.py`：Wilson CI、family 分層 task-level bootstrap（B = 10000）。
   - `replay/localization.py`：coarse → **反覆** refine（`run_protocol`，sweep 與模擬共用）、round-robin、P(success | checkpoint) 估計、
     cliff 判定（門檻以精確有理數比較），全部經 manifest。
   - synthetic 驗證：Wilson CI 對解析解的涵蓋率 >= 0.85、cliff 定位到 k* = m、沒有犯錯的 source 不被標為 cliff。
   - 測試 critic 與對抗式審查後的協定修正（decision 013，在任何真實結果之前）：refine 反覆到沒有「未加密的大下降」、Δ* 不截斷、
     `checkpoint_truth` 只對 on-policy 軌跡有定義、bootstrap 的 estimand = 15 個固定 eval instance（superpopulation 涵蓋率約 0.86，
     以 strict xfail 記錄）。
3. **SweepPlan 偵測力模擬**（decision 011，`experiments/sweep_power.py`，依 decision 013 修正後重跑）：依預先登記的規則，
   108 個候選 plan **沒有任何一個合格**（gradual 曲線約 14–17% 被誤判為 cliff）。已整理 A / B / C 三個選項與成本，**待使用者決定**
   （建議 A：Bonferroni 校正 + (2, 10, 48, 0.3)，偵測 0.83 / 誤判 0.03）。
4. **付費 run 的執行工具**（decision 012，提案）：`experiments/run_baseline.py`（dry-run、成本上限、不覆寫既有 run）、
   `experiments/run_sweep.py`（依 decision 012 選 source、跨 source 共用預算、family 分層 bootstrap 彙總 cliff 比例與最大下降）、
   `experiments/cost_estimate.py`（以 dry-run 的實際 request 大小估算成本）。dev / eval 的 dry-run 全部成功且安全。
5. **計費修正**（decision 014，審查確認的高嚴重度問題）：API 已回傳但被 adapter 拒絕的回應（例如 max_tokens 截斷）原本不記帳，
   成本上限可被繞過。現在一律記帳；manifest 升 `attempt/v2`，每列記錄該次嘗試的已計費 call 數與成本；有上限時模型必須在 PRICING。

測試：`uv run pytest` → 8260 passed、1 xfailed（已知限制的特徵測試），約 70 秒；`ruff check` 通過。

## 下一步（需要使用者先決定，依序）

1. **使用者確認**（decision 012「待使用者確認的事項」）：
   - 模型 `claude-opus-5-5` 與 dev sanity 上限 $10；
   - 在環境變數提供 `ANTHROPIC_API_KEY`（程式不讀取、不保存；由 SDK 解析），確認帳號可用 `thinking-binding-controls-2026-08-01` beta；
   - eval rollout 數 R（提案 3）；SweepPlan 選項（decision 011：A / B / C）；Gate 判定的操作化（decision 012 第 4 節）。
   - 若選 A：另立 decision log，把 Bonferroni 判定正式實作到 `replay.localization`（目前只在模擬程式中，標為探索）並測試，再重跑模擬確認。
2. dev sanity：`uv run --extra llm python -m experiments.run_baseline --split dev --confirm-paid-run --max-cost-usd 10`；
   以實測 tokens / call 重新估算 eval 與 sweep 的成本，再請使用者確認預算。
3. eval baseline（R rollouts）→ 以 decision 012 的規則選 source → 依選定的 SweepPlan 實作 / 測試（若選 A，先另立 decision log 並實作
   Bonferroni 判定、重跑 decision 011 模擬）→ eval sweep：
   `uv run --extra llm python -m experiments.run_sweep --baseline-run runs/<eval-run> --plan S,NC,NF,T --confirm-paid-run --max-cost-usd X`
   （plan 沒有預設值；`--dry-run` 已驗證）。
4. 有真實結果後：recoverability curve、trajectory safety、「最終成功但中途有 forbidden action」案例、Gate 判定，再寫 README 首屏。

## Blocked / 需要使用者

- **所有真正呼叫 LLM 的步驟**：需要上面第 1 點的確認與 credential（依指示本 session 未要求、未處理任何 key）。
- **SweepPlan**：decision 011 依預先登記規則無合格 plan，依規則不自行放寬，等使用者選擇。

## 分析前必須注意的已知限制（decision 003 / 004 / 005 / 007 / 009 / 011）

- 嘗試但失敗的危險操作不計入 `forbidden_action_count`，改由 `forbidden_attempt_count` 記錄（decision 008）；兩者單位不同
  （call vs Change），比較時用 step 集合。無法解讀的 call 不計入 attempt。
- scope 內的反覆修改不計為 unnecessary。
- `to_values` 型 forbid rule 只看變更後的值：挑選「最終成功但中途有 forbidden action」展示案例前，必須人工確認被標記的不是「改回原狀」。
- generator v1 的背景 labels 可能與指令語義衝突（eval 8/15 的 target 帶 `wontfix` 或 `needs-info`）：只能描述，不得事後分層當主要結論。
- `forbidden_action_count = 0` 不代表全面安全：部分 forbid rule 是低觸發率安全網。
- Week 1 eval 只有 15 題，結果只能當基礎設施驗證與初步觀察，不得宣稱統計上的泛化；不得因此擴充題庫。
- smoke tasks（handwritten/v2）只用於回歸測試與 sanity run，不得計入任何統計。
- instruction 以「名字 (id)」引用 user（SPEC 只有 5 個 tools，沒有 user 查詢 tool）。
- eval 中 `gen-escalate-label-s1` 與 `-s2` 的 instruction 文字相同（初始狀態不同）。
- manifest 不支援並行寫入；付費 run 被手動中斷時，進行中的嘗試不會進 manifest，重跑必須用新目錄並在報告列出所有目錄。
- 偵測力模擬的情境是假設；真實曲線形狀未知。
