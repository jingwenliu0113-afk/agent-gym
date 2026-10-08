# 007 — Week 1 對抗式驗證後的修訂與已知限制

日期：2026-10-08　狀態：採用（在任何 agent run 之前）

## 背景

Week 1（generator、recorder、checkpoint / replay）實作完成、6032 個測試全綠後，以 4 個角度（replay invariant、
lineage / 竄改、generator 量測有效性、輸入驗證與效能）做對抗式探測，每個 finding 由獨立 verifier 重寫重現腳本並對照契約判定。
9 個 finding 中 6 個確認（合併後為 4 個缺陷），3 個駁回。全部修正都先寫回歸測試、確認紅燈後才修
（`tests/test_adversarial_regressions.py` 的 Week 1 段落）。

## 決策（修正）

1. **`Usage` 拒絕超出 float 範圍的 int**：原本 `math.isfinite(10**400)` 會拋出 `OverflowError`，穿透
   `Usage.from_dict`、`PolicyStep`、`Trajectory.load`、`Checkpoint.from_dict` 的例外契約。改以精確的 int / float 比較判斷，
   超出 `±sys.float_info.max` 即 `ValueError`。
2. **trajectory / checkpoint 文件的 JSON 結構規則（Trajectory 規則 10、Checkpoint 規則 6）**：
   容器只能是 `list` / `dict`（不含 tuple）、key 為 str、葉節點為 None / str / bool / int / float，
   巢狀深度不超過 `MAX_DOCUMENT_DEPTH = 256`。以迭代方式檢查，且 `from_dict` / `load` 在任何深拷貝之前先檢查。
   `load_strict_json` 遇到 `RecursionError` 改拋 `ValueError`。
   - 原本約 500 層以上的巢狀 JSON 會在 deepcopy / json 解析時漏出 `RecursionError`；
   - 原本 tuple 經 canonical_json 後與 list 無法區分，含 tuple 的 checkpoint 能通過 verify 與 checkpoint_id，
     卻在 restore / rerun 時才以 `SnapshotError` 失敗。
3. **Checkpoint 規則 6：通過 verify 的 checkpoint 一定能 restore**（比照 Trajectory 規則 7，實際 restore 一次 env_snapshot）。
4. **錯誤訊息不再格式化可能是超大整數的值**（turn index、step 編號）。

以上只改變不合法輸入的例外型別與拒絕時機，不改變任何合法輸入的語義；格式與版本字串不變。
新的深度上限只拒絕約 256 層以上的文件：正常錄製的 trajectory 中 args 已有 64 層上限、model_config 已有 128 層上限，
generator 的 task params 只有一層。`Task.params` 本身原本沒有深度上限，這是新收緊的規則（只在 trajectory / checkpoint 層）。

## 已知限制（駁回但必須記錄）

- **背景 labels 可能與指令的語義線索衝突**：generator v1 的背景 labels 取自含 `wontfix`、`needs-info` 的集合，
  因此 target issue 可能帶 `wontfix`。主 session 實算：20 題中有 12 題（eval 8/15）的 target 帶 `wontfix` 或 `needs-info`；
  其中帶 `wontfix` 的是 `gen-assign-and-start-s0`、`-s1` 與 `gen-escalate-label-s3`，其餘為較常見的 `needs-info`。
  指令本身明確（選取規則只看 label 與 status），grader 判定正確；agent 因此遲疑屬於真實的指令遵循失敗。
  但這是**未受控的背景干擾**：它比較可能造成「從一開始就失敗」的平坦曲線，而不是 recoverability 驟降。
  分析失敗案例時須註明；**不得**在看過結果後才依此事後分層並當成主要結論。若要控制，必須升 `GENERATOR_VERSION` 並另立 decision log。
- **`to_values` 型 forbid rule 只看變更後的值**：例如「任何 issue status → closed」也會命中「把誤開的 closed issue 改回 closed」，
  而原本的誤改（重開）不在 forbidden 集合內，只會記為 unnecessary mutation 或 constraint violation。
  這符合 decision 003 的定義（`change.after ∈ to_values`），但挑選 README 的「最終成功但中途有 forbidden action」
  展示案例之前，**必須人工確認被標記的那一步不是「改回原狀」**。
- **policy 的 agent_config 必須在生命週期內不變**（decision 006）。rerun 只在呼叫第一次 `next_message` 前讀取並比對；
  episode 中途才改變 agent_config 的 policy 違反介面前提，不在偵測範圍內。LLM adapter（Week 1 #5）必須遵守。

## 對實驗的影響

- 不影響 generator v1 的 golden hash、smoke handwritten/v2、grader / schema / trajectory / checkpoint 的版本字串。
- 新增兩項分析前檢查（寫入 NEXT.md）：展示案例需人工確認 forbidden 步驟的性質；背景 label 干擾只能做描述，不能事後分層下結論。
