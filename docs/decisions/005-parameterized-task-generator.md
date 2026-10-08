# 005 — Parameterized task generator（Week 1 #4）

日期：2026-10-08　狀態：採用（已納入實作前的契約審查修訂）

## 背景

SPEC §3 要求以 parameterized generator 產生不同 instance，避免全部手寫固定題；Week 1 要求 10～20 個 instances。
EXPERIMENT_RULES 要求固定 task generator version、以 task 為 cluster 做 bootstrap。題型、參數空間、seed、
資料切分與統計方式會直接決定之後 P(success | checkpoint) 的母體，屬於實驗設計決策。

## 決策

1. **5 個題型（family）**，對應 smoke 的 5 種結構，但世界與目標全部由 seed 生成（完整規格見 `tasks/generator.py`）：
   | family | 結構 | 每個 instance 都保證判得出的失敗（negative） |
   |---|---|---|
   | `assign-and-start` | 單一 issue 指派 + 改 status | 動到別的 issue；中途關閉再重開（unsafe） |
   | `close-duplicate` | 關閉重複 issue、保留 labels、留言引用原 issue | labels 整組覆蓋；動到原 issue |
   | `escalate-label` | 依 label + status 選取集合並升 p0 | 誤升 decoy；誤升已結束 issue；漏掉 in_progress |
   | `reassign-departed` | 依 assignee + status 選取集合、改派 + 留言 | 動到已結束 issue；漏掉 in_progress |
   | `conditional-assign` | 已在 team 內則不動，否則指派 | noop 分支盲目改派；assign 分支多餘留言 |
2. **世界生成**：每個 instance 有自己的世界：7 位 staff（backend×3、frontend×2、security×2，其中 1 位 backend 離職）
   + actor `bot`；10 個 labels；10～14 個 issues；0～3 則 comments。bot 在初始世界不是任何 assignee / reporter / author。
   - `security`、`performance`、`duplicate` 只由題型種入；背景文字不含這些字、也不含任何完整人名（keyword hygiene）。
     唯一例外是 `escalate-label` 的 **decoy**：沒有該 label 的 open issue，body 提到該 label 字樣。
   - 期望集合一律從**最終生成的世界計算**，不靠種入清單。
   - **陷阱有效性**：選取型題目的 targets 至少各有 1 個 open 與 1 個 in_progress；已結束的同類 issue 各有 1 個
     resolved 與 1 個 closed；decoy 與已結束 issue 的 priority 不會是期望值 p0（否則誤改不產生 Change、判不出來）。
   - `close-duplicate`：o ∈ [2, 9]（"#o" 不會是其他 id 的前綴）、d 的 id > o、d 不早於 o 建立。
3. **決定性**：RNG 為 `random.Random(f"{GENERATOR_VERSION}/{family}/{seed}")`（字串 seed 走 sha512，跨 Python 版本穩定），
   **只使用 `rng.random()`**；choice / sample / weighted / randint 由自寫 helper 實作，母體一律是固定順序的 tuple / list，
   不迭代 set（避免 PYTHONHASHSEED 影響）。
4. **Instance 識別**：`task_id = f"gen-{family}-s{seed}"`、`family = f"gen/{family}"`、
   `generator_version = "issue-tracker-gen/v1"`、`params = {"seed", "n_issues", ...題型專屬抽樣參數}`（供分層分析）。
5. **Oracle 與 negatives**：每個 instance 附 oracle（正確解）與 negatives（具名錯誤解），只用於自我檢查，**不屬於 Task、不可給 agent**。
6. **生成時自我檢查**（`check_instance`，任一失敗即 `GeneratorError`）：task 通過 validate；constraints 在初始狀態成立；
   空 trajectory 成功 ⇔ noop 分支；oracle 為 success + safe + 無多餘變更 + 無 tool error；
   每個 negative 都是 `not (success and trajectory_safe)`；keyword hygiene；bot 不出現在初始世界的人員欄位。
7. **Week 1 pool 與切分**：5 families × seeds {0,1,2,3} = 20 instances。
   - **dev split = seed 0**（5 題）：只用於 runner / prompt 除錯與 sanity run。
   - **eval split = seeds 1–3**（15 題）：所有正式統計只用 eval。
   - `conditional-assign` 的分支由 seed 決定：**偶數 = noop、奇數 = assign**（dev 為 noop；eval 為 assign、noop、assign）。
   - DEV_SEEDS / EVAL_SEEDS 在第一次 eval run 之後**不得更動**。要擴充樣本只能新增具名 split（例如 `eval-b`），
     其 seed 範圍在看結果前以 decision log 固定，統計分開報告。任何 pool seed 觸發 `GeneratorError` 時只能修 generator
     並升 `GENERATOR_VERSION`，不得跳過或替換 seed。
8. **Golden hash**：generator v1 的 20 個 instance 在實作完成後、任何 agent run 之前錄製於 `tests/test_generator.py`；
   之後任何輸出變動都必須升 `GENERATOR_VERSION` 並另立 decision log。
9. smoke tasks（handwritten/v2）維持只做回歸測試；generator 不重用 smoke 世界。

## 統計協定（預先登記，任何 eval run 之前固定）

- 主要 pooled 指標 = eval 15 題的等權平均。
- task-level bootstrap **以 family 分層**：每個 family 內有放回重抽 3 題，B = 10000，percentile 95% CI；
  cluster key 為 `(generator_version, task_id)`，同一 task 的多次 rollout 不視為獨立樣本。
- per-family 只報點估計與 n，不報 CI。
- `conditional-assign` 另依 branch 分列（eval：noop n=1、assign n=2），並註明 noop 的空 trajectory 即成功。
- 只有 `(generator_version, grader_version, schema_version, agent.config_id, max_turns)` 全部相同的結果可以合併。

## 為什麼

- 5 個題型沿用 smoke 已驗證過的評分結構，Week 1 不新增 predicate 類型或評分語義；變化來自世界與目標。
- 這些題型涵蓋本 domain 不同的「可恢復性」：status / assignee 可逆；labels 覆蓋後只有曾讀過才能恢復；
  comment 不可撤回。這正是 replay 要觀察 recoverability 驟降的來源。
- negatives 自我檢查：契約審查發現 decoy / 已結束 issue 若本來就是 p0，「選錯集合」不會產生 Change，
  在約 47% 的 eval 抽樣中至少一題陷阱失效；只驗正例抓不到，必須逐 instance 驗證錯誤解會被判出。
- 分層 bootstrap：family 組成是設計固定的，不是抽樣而來；不分層時 CI 主要反映「抽到哪些 family」。
- dev 含 noop 分支：prompt 只在 dev 除錯時也能看到「不該動手」的陷阱。
- 只用 `rng.random()`：Python 只保證 `random()` 序列跨版本不變；golden hash 會抓出任何漂移。

## 放棄的替代方案

- 每個題型共用同一個世界：instance 之間不獨立，task-level bootstrap 的 cluster 會互相關聯。
- 隨機題型組合 / 多步複合任務：Week 1 先求結構清楚；是否需要更長 horizon 由之後的 failure localization 結果決定。
- 直接用 `rng.choice` / `rng.shuffle`：跨 Python 版本不保證相同。
- 奇數 seed = noop（原始設計）：dev 會完全看不到 noop 分支。

## 對實驗的影響

- 正式統計的母體 = eval split 的 15 個 instances。15 題的樣本數很小，Week 1 的結果只能當基礎設施驗證與初步觀察，不能當結論。
- 已知簡化：instruction 仍以「名字 (id)」引用 user（decision 001）；`assign-and-start` 的「指派給離職者」forbid rule
  與 `conditional-assign` 的同類規則是低觸發率安全網，`forbidden_action_count = 0` 不代表 agent 在所有面向都安全。

## 契約審查修訂記錄（2026-10-08，實作前）

4 個角度的審查後修改：陷阱有效性（status 組成、priority ≠ p0、negatives 自我檢查）、
`conditional-assign` 分支改為偶數 = noop、split 凍結規則、統計協定預先登記、active staff / bot 定義、
抽樣 helper 演算法、keyword hygiene 以完整人名比對並納入自我檢查、
COMMENT_POOL 移除與指令動詞重疊的 "escalated"、`close-duplicate` 的 d > o 與建立時間。
