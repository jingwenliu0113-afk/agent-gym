# NEXT

## 目前位置

Core / SPEC「Day 1」完成；「Week 1」中不需要 API key 的項目已完成（#1–#4、#6、#7、#8）。
剩下 **Week 1 #5 baseline agent**（真正呼叫 LLM 需要使用者在環境變數提供 `ANTHROPIC_API_KEY`）。
尚未進 Advanced（Gate 未滿足：還沒有任何 failure localization 結果）。

Git：獨立 repo，分支 `main`。`d7241dc chore: establish core baseline`（Day 1）；
Week 1 generator / Recorder / checkpoint-replay / Docker 規則 / forbidden attempt 指標以 `feat: complete week 1 replay infrastructure` commit。

## 上次完成（2026-10-08）

Week 1（TDD：契約 stub → 4 角度契約審查 → 5 個 agent 先寫測試、確認紅燈 → 實作轉綠 → 4 角度對抗式驗證）：

1. **#4 parameterized task generator**：`tasks/generator.py`（decision 005）
   - 5 個題型 × seeds {0,1,2,3} = 20 instances；dev = seed 0（5 題，含 noop 分支），eval = seeds 1–3（15 題）。
   - 每個 instance 附 oracle 與具名 negatives（錯誤解），生成時自我檢查「正解成功且安全、每個錯誤解都被判出」。
   - 決定性：字串 seed、只用 `rng.random()`、跨 PYTHONHASHSEED 一致；20 題的 golden hash 已在任何 agent run 前錄製。
   - 統計協定已預先登記：eval 等權、family 分層 task-level bootstrap（B=10000、95% percentile CI）。
2. **#6 Recorder**：`agents/policy.py`、`replay/trajectory.py`、`replay/runner.py`（decision 006）
   - 無狀態 policy 介面 + `ScriptedPolicy`；trajectory 保存每個 turn 的 model message、tool calls、env 紀錄（含 state hash）、usage，
     以及 task、prompts、prompt version、model config、final snapshot。
   - TOOL_SPECS 全文已納入 `issue_tracker/v1` 並以 golden hash 釘住。
3. **#8 checkpoint / restore / rerun**：`replay/checkpoint.py`（decision 006）
   - checkpoint 自帶前綴並在載入時重放驗證（解決 decision 004 的前綴錯配）；`verify_lineage` 以來源 trajectory 與呼叫端的 checkpoint 為錨點。
   - 已驗證 SPEC §5 核心 invariant：save / load 後 restore 的 conversation 與原 episode 中 policy 實際收到的逐 byte 相同、
     env 一致、決定性 rerun 逐 byte 重現（smoke 5 題 + generator 20 題的每個 checkpoint）。
4. 對抗式驗證修正（decision 007）：超大 int 的 `OverflowError`、深層巢狀 JSON 的 `RecursionError`、tuple 被 canonical_json 掩蓋、
   錯誤訊息格式化超大 int。並記錄 3 項已知限制（見下）。
5. MACHINE_POLICY.md 新增 Docker 規則（Core 不使用 Docker Desktop）。
6. **新增 safety 指標 `forbidden_attempt_count`**（decision 008，grader 升 `agent-gym/grader/v2`）：從 trajectory 的每筆 tool call
   推導 attempted change，被 tool 拒絕或沒有造成 mutation 的危險嘗試也會記錄；`forbidden_action_count` 與 `trajectory_safe` 語意不變。

測試：`uv run pytest` → 6100 passed（約 15 秒）。資源：全程只用 Mac；未使用 Mac heavy lock、遠端 RTX、Docker 或任何 API key。

## 下一步（建議依序）

1. **Week 1 #5 baseline agent（先做不需要 key 的部分）**。開始前先寫 decision log，在任何 eval run 之前固定：
   prompt 內容與 `prompt_version`、model 與 sampling 參數、`max_turns`（含理由）、是否允許一則訊息多個 tool call、
   暫時性錯誤的重試上限、例外中止的紀錄方式（decision 006）。
   - Anthropic adapter 以可注入的 client 實作 `Policy` 介面；不得啟用需要原樣回傳 provider block 的功能（例如 extended thinking）。
   - 用假的 client 做單元測試（不需要 key）；真正呼叫 LLM 的 dev sanity run 等使用者提供環境變數。
2. **SPEC §6 failure localization 工具**（不需要 key）：rerun sweep（先少量掃描、再對疑似變化處加密取樣）、
   P(success | checkpoint) 估計、family 分層 task-level bootstrap、每個 checkpoint 同時報告 `termination == "max_turns"` 的比例。
   開始前寫 decision log；可考慮用「已知真實 recoverability 的合成 policy」先驗證估計流程（需在 decision log 說明）。
3. 有真實結果後，再寫 README 首屏（Task → Agent actions → state changes → grader → replay from checkpoint）。

## Blocked / 需要使用者

- **真正呼叫 LLM 的 baseline run**：需要使用者在環境變數提供 `ANTHROPIC_API_KEY`（依指示本 session 未要求、未處理）。
  上面的第 1、2 步在 Mac 上都能先完成，不受影響。

## 分析前必須注意的已知限制（decision 003 / 004 / 005 / 007）

- 嘗試但失敗的危險操作不計入 `forbidden_action_count`，改由 `forbidden_attempt_count` 記錄（decision 008）；兩者單位不同
  （call vs Change），比較時用 step 集合。無法解讀的 call（非 dict args、issue_id 型別錯誤、未知 tool、spec 外參數）不計入 attempt。
- scope 內的反覆修改不計為 unnecessary。
- `to_values` 型 forbid rule 只看變更後的值：挑選「最終成功但中途有 forbidden action」展示案例前，必須人工確認被標記的不是「改回原狀」。
- generator v1 的背景 labels 可能與指令語義衝突（eval 8/15 的 target 帶 `wontfix` 或 `needs-info`）：只能描述，不得事後分層當主要結論。
- `forbidden_action_count = 0` 不代表全面安全：部分 forbid rule 是低觸發率安全網。
- Week 1 eval 只有 15 題，結果只能當基礎設施驗證與初步觀察。
- smoke tasks（handwritten/v2）只用於回歸測試與 sanity run，不得計入任何統計。
- instruction 以「名字 (id)」引用 user（SPEC 只有 5 個 tools，沒有 user 查詢 tool）。
