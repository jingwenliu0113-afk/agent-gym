# NEXT

## 目前位置

Core：Day 1 完成；Week 1 除「#5 baseline 真正跑完整 task」以外都完成；#5 與 SPEC §6 中**不需要 API key 的部分全部完成**。
需要真正呼叫 LLM 的步驟（dev sanity、eval baseline、eval 上的 failure localization sweep）依使用者決策 **deferred / blocked-by-paid-API**；
keyless Core infrastructure（含分析工具、驟降判定選項、README 示範）已完成；第七輪對抗式審查找出的 keyless 缺口
（已登記但漏實作的規則、只信任 summary 的分析）已依 decision 017 補齊；第八～十二輪審查與付費流程模擬找出的缺口已依 decision 018 – 024 補齊。
**所有 keyless Core 工作已完成；剩下的都在下方「Blocked / 需要使用者」**（付費 LLM run 與研究設計決策）。
尚未進 Advanced（Gate 未滿足：還沒有任何真實的 failure localization 結果）。

Git：獨立 repo、分支 `main`，`origin` = private GitHub repo（github.com/jingwenliu0113-afk/agent-gym），main 追蹤 origin/main；
測試通過的 commit 一律 push（不 force push、不改寫 history）。
`d7241dc` Day 1 → `5c8ada4` Week 1 replay infrastructure → `c554735` keyless baseline adapter + attempt manifest →
`638be6d` §6 failure localization infrastructure → `7a796f3` 付費 run 執行工具 + 計費修正 → SweepPlan 偵測力模擬（decision 011）→
`b8c600f` 預先登記的分析工具 + README 示範（decision 015 / 016）→ `9baae38`…`63bde0b` 第七輪審查修正（decision 017）→
第八輪審查修正（decision 018）→ `7a94d24` 第九輪（decision 019）→ `724f3b9` 第十輪（decision 020）→ `f750931` 第十一輪（decision 021）→ `88bdb02` 第十二輪（decision 022）→ `9cf7f8f` 付費流程模擬（decision 023）→ 023 的審查（decision 024）。

## 上次完成（2026-10-08 / 10-09）

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

6. **預先登記的分析工具與 README 示範**（decision 015 / 016）：驟降判定可選擇；`analyze_baseline` / `analyze_sweep`；
   README 首屏以 synthetic policy（明確標示不是 LLM 結果）示範 Task → actions → state changes → grader → replay。
7. **第七輪對抗式審查後的修正**（decision 017，在任何 LLM 結果之前）：
   - 實作 decision 006：伴隨 max_turns 比例上升的最大下降不認定為 cliff（`assess_drop`、`SweepResult.max_turns_confounded`、
     sweep 格式 v2）。
   - 實作 decision 009：sweep 的每個完成 rerun trajectory 存成 `rollouts/<task_id>/k<k>-r<i>.json`（response.model 因此可查）；
     response.model 不只一種時分析不報任何 pooled / 分組估計（取代 016 A8 的「只標示」）。
   - analyze_baseline：補 decision 003 的 outcome_correct、tool errors、steps、tokens、latency，decision 008 / 009 的被擋下嘗試與
     multi-tool turn；每個中止嘗試都列出（None = 未知，不當 0）；可讀中斷的 run（沒有 summary，未完成的重試記為 interrupted）；
     summary 與 manifest / trajectory 逐欄交叉核對；同一目錄不可重複。
   - analyze_sweep：可讀多個目錄（中斷後重跑）；每個完成的 sweep 以 manifest 與存檔的 rollout 重算核對（分類、排程、估計、判定）；
     缺漏以 eval pool 為準（baseline 因預算停止時不再誤報完整的 cliff 比例）；時間窗、response.model、中止嘗試、各類計數。
   - source 選取改用同一個嚴格 loader、涵蓋全部 15 個 eval task；manifest / strict JSON loader 的例外一律為 ValueError
     （目錄當檔案、超大 int、沒有時區的 utc_time、`1e400`）；README 與示範文字的兩處不實描述已修正。
   - 審查判定不成立而未處理的兩項：Gate「收尾」分支（多數 gradual 且彼此不同）的機械化判定——其操作化屬於 decision 012 §4
     待使用者確認的部分；README 加一個 scripted 的「最終成功但不安全」示範——不是 SPEC 要求的 Core 項目。

8. **第八輪對抗式審查後的修正**（decision 018）：
   - sweep 續跑：`run_sweep ... --resume-from runs/<舊目錄>` 只執行尚未完成的 source（不重複付費），summary 記錄 skipped
     （sweep 格式 v3）；`run_sweeps` 的 sources 必須涵蓋全部 eval task，自己的彙總以 eval pool 的全部 family 判定。
   - sweep summary 的語意核對（sources = 全部 eval task、completed / skipped / 預算停止的關係），以及所有 manifest 列（含未完成的 sweep）
     的嘗試序列與 checkpoint 身分核對。
   - baseline summary 全欄核對（排程、計數、成本、每列的安全 / 成本欄位）：刪掉失敗的 rollout 不再能改變主指標。
   - 目錄以 device + inode 識別（macOS 不分大小寫、symlink）；不存在、symlink loop、沒有 manifest 的目錄一律 ValueError。
   - response_model 不只一種時 sweep 也不報時間窗；sweep 報告列出 multi-tool turn；部分安全指標標示含前綴；run_baseline 的
     說明文字移除不存在的 `--estimate`。
   - 審查判定不成立而未處理：sweep 部分安全指標改成只算 rerun 自己的步驟（數值語意不變，只加註）；中止嘗試保存完整部分軌跡
     （decision 006 / 009 已登記只記計數）；以 dev run 實測 tokens 重估成本的專用 CLI（屬於付費步驟 a 的一部分，現有 API 已足夠）；
     極端數值（1e308 美元、10^400 tokens）的加總溢位。

9. **第九輪對抗式審查後的修正**（decision 019）：
   - 鏈式續跑：`--resume-from` 必須列出所有先前的 sweep 目錄；給定目錄略過、卻沒有在任何給定目錄完成的 source → 拒絕執行（不建立
     目錄、沒有任何 call）。原本只給最近的目錄時，更早目錄已完成的 source 會被重新 sweep、再付一次費。
   - 多目錄 eval baseline：source 取所有目錄中最早結束的 rollout_index 0（interrupted 不算；它是 agent_error / aborted → 沒有
     source，不以其他目錄補上）；`run_sweep` / `analyze_sweep` 的 `--baseline-run` 接受多個目錄（可重複給）。
   - sweep 開始時寫入 `run.json`（plan、criterion、config_id、skipped、sources；sweep 格式 v4）：沒有 summary 的目錄也能核對 plan，
     被中斷的續跑目錄也有略過清單；分析時以給定的 baseline 目錄重新選出的 source 必須與它相同。分析報告列出 baseline 目錄（分析格式 v3）。
   - 沒有任何要執行的 source 時不建立目錄；續跑前印出先前目錄已花的成本（`--max-cost-usd` 是這一次執行的上限）。
   - 審查判定不成立而未處理：decision 011 的選項成本與 012 的上限依不同的 L 假設，數字不直接可比（兩者都是暫定值，dev sanity 後
     重估）；跨目錄累計的成本上限（改為印出先前花費）。

10. **第十輪對抗式審查後的修正**（decision 020）：
   - `--resume-from` 可重複給（原本重複給時只保留最後一個，依順序可能略過一個目錄而重複付費）。
   - 續跑的血緣檢查：run.json 新增 `run_id` 與 `resumed_from`（sweep 格式 v5）。續跑前在相關目錄的上層目錄中，找出從給定目錄續跑過、
     有 manifest、卻沒有被傳入的後續目錄 → 拒絕執行。原本漏給「較晚」的目錄（例如續跑被中斷後重下同一個指令）時會重複付費。
   - analyze_sweep 驗證 run_id / resumed_from，同一個 run_id 出現在兩個目錄（複製的目錄）→ ValueError。
   - rollout 0 的同時間檢查只看最早的時間，與傳入順序無關；目標目錄已存在時提示以 `--resume-from` 續跑。

11. **第十一輪對抗式審查後的修正**（decision 021）：
   - dry-run 與付費輸出分開：baseline 也在開始時寫入 `run.json`（split、rollouts、config_id、`dry_run`；baseline 格式 v2），sweep 的
     run.json 加上 `dry_run`（格式 v6）。dry-run 與付費的 baseline 不得合併（原本較早的 dry-run 預演會成為「最早結束」的 source）；
     付費 sweep 必須用付費 baseline；續跑不得跨模式；後代檢查只看同一種模式；分析不得混合，dry-run 報告開頭標示「不是研究結果」
     （分析格式：baseline v3、sweep v4）。原本 dry-run 預演會擋下付費續跑，並誘導把假結果併入。
   - 祖先目錄必須齊全：續跑與分析時，給定目錄 resumed_from 中的每個 run_id 都必須在給定的目錄中（原本沒有完成任何 sweep 的祖先目錄
     可以被漏掉，它的成本與中止嘗試會消失）。取代 018 / 019 對「只分析部分目錄」的容許。
   - 審查判定不成立而未處理：`--run-name` 含路徑分隔時後代目錄可能在其他位置（020 已記錄的限制）；runs/ 內的備份複本會先被要求
     傳入、再被判定為複本（fail closed）；B.2 沒有可照打的指令（R 待使用者決定）；decision 004 / 007 / 009 缺少範本段落標題。

12. **第十二輪對抗式審查後的修正**（decision 022）：analyze_baseline / analyze_sweep 拒絕寫入已存在且不是空的 `--out`
   （原本沿用 dry-run 分析的目錄時，會留下沒有標示的 dry-run 曲線 SVG）；dry-run 分析的 SVG 標題加上「[dry-run]」。
   審查判定不成立而未處理：以 Python 直接呼叫 `analyze(load_rollouts(...))` 時沒有模式資訊（021 §1 已記錄的限制）。

13. **依 NEXT B.1–B.4 原文模擬付費流程後的修正**（decision 023）：
   - 續跑接續做到一半的 source：原本因預算停止或 Ctrl-C 中斷時，正在跑的 source 已付費的 rollout 會在續跑時全部重做（模擬中每次
     重複付費 $0.28 – $0.53；上限低於單一 source 的成本時永遠無法完成）。現在續跑重放同一個排程，已有最終結果的 rollout 直接採用、
     只執行缺的部分（`sweep(prior=...)`、`run_sweeps(carried=...)`，sweep 格式 v7）；analyze_sweep 以「自己的列 + 祖先目錄的最終結果」
     核對完成的 sweep，重複執行已有最終結果的 rollout → ValueError。
   - trajectory 先寫入（atomic）再在 manifest 記「completed」：原本在兩者之間中斷或存檔失敗會讓 baseline 目錄無法讀取；存檔失敗
     現在是 harness 中止（照常記帳）。

14. **decision 023 審查後的修正**（decision 024；含 65 條隨機續跑鏈的模擬，其餘的曲線都與沒有中斷的排程完全相同）：
   - sweep 目錄在第一次嘗試之前就建立空的 manifest：原本完全靠接續完成的 sweep（中斷落在最後一個 source 記帳之後、sweeps 檔寫入
     之前）沒有 manifest，無法分析也無法再續跑，那條付費曲線永遠進不了報告。後代檢查改為只看有紀錄的目錄。
   - run.json、summary.json、sweeps/<task>.json 都以 atomic 方式寫入（`replay.atomic.write_atomic`）：原本磁碟已滿時留下截斷的
     檔案，目錄因此無法讀取。讀取 JSON 失敗的錯誤訊息包含檔案路徑。
   - 審查判定不成立而未處理：以 Python API 繞過 CLI 時，completing 目錄的祖先以外的目錄可能有重複的最終結果（列為 partial、成本照計）。
   - 驗證：審查留下的隨機續跑鏈模擬（3 種 plan、65 條鏈、每條最多 80 次續跑，混合預算停止、Ctrl-C（call 中、重試中、存檔前後、記帳後、sweep 完成後）與不完美的假 client）在 024 之後全部通過：每條曲線都與沒有中斷的排程完全相同，沒有任何 rollout 在已有最終結果後再執行，成本、partial_sweeps 與中止嘗試的計數都正確（024 之前有 8 條失敗）。

測試：`uv run pytest` → 8699 passed、1 xfailed（已知限制的特徵測試；需 `uv sync --extra llm`，否則 SDK 契約測試會 skip），約 4–6 分鐘；`ruff check` 通過。

## 使用者決策（2026-10-08 / 10-09）

- 需要 `ANTHROPIC_API_KEY` 或其他付費 API 的正式 LLM run（baseline / eval / failure-localization sweep）**全部延後**；
  不要求 API key、不執行任何付費 API call。Claude Code Desktop 自身的 token 可自由用於開發、測試、review、commit、push。
- 不得把 Claude Code Desktop 的內部模型當成正式 benchmark provider（除非使用者明確修改實驗設計）。
- 只因需要真實 LLM API 而卡住的步驟標記 `deferred / blocked-by-paid-API`；其他 keyless Core 工作照常推進。
  所有 keyless Core infrastructure 完成、只剩 paid runs 時，才一次性回報需要的 paid runs、預估成本與順序。
- （2026-10-09）使用者離開期間：需要使用者決策、批准、key、credential、sudo、人工標註、修改 frozen protocol、進 Advanced / Optional 的
  事項不等待，記到下方「Blocked / 需要使用者」後跳過；只有所有可做的 Core 都完成或全部 blocked 時才停。
- （2026-10-09 晚）Mac 使用自動電力模式：只暫停會產生正式 performance-sensitive 結果的 Mac 實驗（正式 Mac performance benchmark、
  正式 incident collection 先 deferred）；一般 heavy 工作、correctness、資料處理、開發測試、gpu-server 工作照常。本 project 目前唯一
  performance-sensitive 的數字是正式 LLM run 的 `latency_s`（SPEC §4 的 latency 指標），而正式 run 已因付費 API deferred；
  dry-run、測試、審查不產生正式結果，不受影響。電力模式期間若要跑正式 run，必須先恢復一般電力設定或延後（見 Blocked B）。

## 下一步

### keyless Core（2026-10-09 全部完成）

1. 驟降判定可選擇（decision 015：`--criterion wilson95 | bonferroni95`，sweep 必須明確指定）。
2. eval baseline 分析：`uv run python -m experiments.analyze_baseline runs/<eval-run> [runs/<rerun> ...] --out runs/<name>`
   （decision 016 A / 017 §4：主指標與敏感度、family 分層 bootstrap、decision 003 指標、安全指標分開、每個中止嘗試的部分安全指標、
   被擋下的嘗試、multi-tool turn、時間窗、response.model 不得合併、中斷的目錄、「最終成功但中途有 forbidden action」案例、成本）。
3. sweep 分析：`uv run python -m experiments.analyze_sweep runs/<sweep-run> [runs/<rerun> ...] --baseline-run runs/<eval-run> [runs/<eval-rerun> ...] --out runs/<name>`
   （decision 016 B / 017 §5 / 019：每個完成的 sweep 重算核對、每個 source 的 recoverability curve 表格 + SVG、max_turns 混淆、
   cliff turn 的 (tool, 類別) 模式、Gate 評估（提案規則；資料不完整時標示）、時間窗、response.model、中止嘗試、成本）。
4. README 首屏：以可重現的 synthetic 示範（`experiments/demo_synthetic.py`，明確標示不是 LLM 結果）展示
   Task → Agent actions → state changes → grader → replay from checkpoint；測試確保 README 與程式輸出同步。

## Blocked / 需要使用者

以下全部需要使用者決定或付費 API；keyless Core 已沒有其他可做的工作。程式不讀取、不保存任何 API key（由 SDK 從環境變數解析）。

### A. 研究設計決策（paid runs 之前）

1. 模型 `claude-opus-5-5`（decision 009）與各階段預算；credential 由環境變數提供，帳號必須可用 `thinking-binding-controls-2026-08-01` beta。
2. eval rollout 數 R（提案 R = 3）。
3. SweepPlan / 驟降判定（decision 011；依預先登記規則**沒有合格 plan**，不自行放寬）。選定後另立 decision log 記錄：
   - A：`--criterion bonferroni95` + (2, 10, 48, 0.3)：cliff 偵測 0.83、gradual 誤判 0.03；估算 $213 – $3,241（唯一兩個門檻都達到的設定）
   - B：`--criterion wilson95` + (2, 10, 32, 0.3)：0.83 / 0.14；$155 – $2,362
   - C：`--criterion wilson95` + (2, 8, 24, 0.3)：0.64 / 0.15；$99 – $1,510
4. Gate 判定的操作化（decision 012 §4 提案，尚未確認），包含「收尾」分支（多數 source 為 gradual 且彼此不同 → 正式報告「失敗是逐步累積」）
   中「彼此不同」的定義；目前分析只機械化判定進 Advanced 的條件。

### B. 付費 LLM runs（依序；每一步都需要使用者先確認預算並提供 credential）

前提：Mac 不在自動電力模式（`latency_s` 是 performance-sensitive 指標；2026-10-09 晚的電力模式期間不跑正式 run）。
成本都是 dry-run request 大小推得的估計（`uv run python -m experiments.cost_estimate`；低 / 高情境見 decision 012），dev sanity 後以實測重估。

1. dev sanity（估計 $0.42 – $6.78）：`uv run --extra llm python -m experiments.run_baseline --split dev --confirm-paid-run --max-cost-usd 10`；
   以實測 tokens / call 重估 eval 與 sweep 的成本，再請使用者確認預算。
2. eval baseline（每輪 15 題估計 $1.16 – $18.51；R = 3 約 $3.5 – $56）：
   `uv run --extra llm python -m experiments.run_baseline --split eval --rollouts <R> --confirm-paid-run --max-cost-usd <X>`
   （R 由 A.2 決定；`--rollouts` 預設為 1）。被中斷或因預算停止時以新目錄（`--run-name <新名稱>`）重跑；dry-run 的目錄不能一起傳入
   （decision 021 §1）；之後所有指令的 baseline
   都要列出**全部**目錄（analyze_baseline 的位置參數、run_sweep / analyze_sweep 的 `--baseline-run`；decision 016 A1 / 019 §1）。
3. 以 decision 012 / 019 的規則選 source → eval sweep（成本見 A.3）：
   `uv run --extra llm python -m experiments.run_sweep --baseline-run runs/<eval-run> [runs/<eval-rerun> ...] --plan S,NC,NF,T --criterion C --confirm-paid-run --max-cost-usd X`
   （plan 與 criterion 都沒有預設值；`--dry-run` 已驗證）。被中斷或因預算停止時續跑：
   `... --run-name <新名稱> --resume-from runs/<第一個 sweep> [runs/<第二個 sweep> ...]`（列出**所有**先前的 sweep 目錄，較早與較晚的
   都要；`--max-cost-usd` 是這一次的上限，先前花費會印出；做到一半的 source 會接續、已有結果的 rollout 不再執行），分析時把所有
   sweep 目錄一起傳給 analyze_sweep（decision 018 – 023）。
4. 以真實結果跑 `analyze_baseline` / `analyze_sweep`，把 recoverability curve、trajectory safety、「最終成功但中途有
   forbidden action」案例、Gate 判定放進 README 的「結果」段（README 目前只有 synthetic 示範）。
5. Gate 判定（需要使用者確認 A.4 之後才能判斷是否進 Advanced）。

### C. 環境（系統層，需要使用者處理）

- 2026-10-09：`/usr/bin/git`（xcrun shim）載入 `/Applications/Xcode.app/.../libxcrun.dylib` 失敗（"incompatible architecture
  (have 'arm64', need 'arm64e')"），所以直接打 `git` 的指令都會失敗。修正需要系統層操作（例如更新 / 重新安裝 Xcode，或
  `sudo xcode-select --switch ...`），依規則不自行處理。
  - 暫時的繞過方式（不改任何系統設定）：直接用 `/Applications/Xcode.app/Contents/Developer/usr/bin/git`。它沿用原本已授權的
    keychain helper，所有 commit 都已用它 push 到 origin。
  - 不要用 `DEVELOPER_DIR=/Library/Developer/CommandLineTools git push`：Command Line Tools 的 osxkeychain helper 會停在
    keychain 授權（已中止，沒有處理任何 credential）。

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
- manifest 不支援並行寫入；付費 run 被手動中斷時，進行中的嘗試不會進 manifest（那次嘗試中已計費的 call 也不會出現在任何
  成本加總中），重跑必須用新目錄。分析工具可直接同時讀中斷的目錄與重跑的目錄（decision 017）；sweep 以 `--resume-from` 續跑
  （decision 018 / 019 / 020，必須列出所有先前的 sweep 目錄）。續跑前的後代檢查只看相關目錄的上層目錄（不遞迴）：
  搬到其他位置的後續 sweep 目錄偵測不到。沒有 `--resume-from` 的新 sweep 不做這個檢查（無法分辨刻意重新開始與忘了續跑）。
- 每次分析都要用新的 `--out`（或空目錄；decision 022）。
- dry-run 標記由 CLI 的 `--dry-run` 決定（decision 021 §1）；直接以 Python 呼叫 `run_split` / `run_sweeps` 時預設 `dry_run=False`。
  dry-run 的輸出永遠不能與付費輸出一起分析或互相續跑；預演續跑要用 dry-run 鏈。sweep 目錄一定有 manifest（可能是空的，decision 024 §1），可以照常傳入；baseline 目錄中斷在第一次嘗試之前時沒有 manifest，沒有可分析的內容，不要傳入分析工具。
- sweep 會為每個完成的 rerun 存一個 trajectory 檔（含前綴，約為 source 的大小）；付費 sweep 的 `runs/` 可能達數百 MB（gitignored）。
- max_turns 混淆判定只看最大下降：被混淆時不改看其他下降（decision 017 §1，刻意保守）。
- 偵測力模擬的情境是假設；真實曲線形狀未知。
