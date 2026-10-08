# 002 — Environment state hash、snapshot / restore 與逐步紀錄

日期：2026-10-08　狀態：採用（Day 1）

## 背景

SPEC §2 要求 reset / snapshot / restore；SPEC §5 要求每一步保存 state hash before / after，
並定義核心 invariant：restore 後 agent 與 environment 看到的 observable state 必須與原 checkpoint 一致。
hash 涵蓋什麼、snapshot 包含什麼、紀錄用什麼粒度描述變更，會影響之後 failure localization 能否成立。

## 決策

1. **Canonical JSON**：`sort_keys`、`separators=(",", ":")`、`ensure_ascii=False`、`allow_nan=False`；
   hash = `"sha256:" + sha256(utf-8 bytes)`。
2. **Hash 範圍**：`{"schema_version", "actor_id", "state": state.to_dict()}`。
   **`step` 不納入 hash**。
3. **Snapshot 格式**（`agent-gym/env-snapshot/v1`）：
   `{format, schema_version, actor_id, step, state, state_hash}`，純 JSON。
   - snapshot **只含 environment 狀態**，不含 tool call records；trajectory 屬於 Recorder（Week 1）。
   - restore 時重新計算 hash，與 `state_hash` 不一致即 raise `SnapshotError`。
   - restore 後 `step` 從 snapshot 接續，records 清空。
4. **Change 模型以 issue 為中心**：`Change(issue_id, field, before, after)`；
   `field` 為可變 issue 欄位之一，或 `"comments"`（新增 comment，after 為 comment dict）。
   users / labels / reporter_id / created_at 的變化、既有 comment 被改或刪，tools 不可能產生，
   一律視為資料損毀（raise）。
5. **ToolCallRecord** 含 `step, tool, args, ok, result, error, state_hash_before, state_hash_after, changes`，
   **不含任何時間**；可由 record 重建 agent 當時看到的 observation。
6. 失敗的 call 與 read-only call 的 changes 一定為空，且 hash before == after。

## 為什麼

- step 不進 hash：step 不影響任何 tool observation；若納入，read-only call 也會改變 hash，
  「state 是否改變」就無法直接從 hash 判斷，也無法辨識不同 trajectory 走到同一狀態。
- actor_id 進 hash：它影響 add_comment 的結果，屬於 observable state。
- snapshot 不含 records：environment 與 trajectory 職責分離，避免之後 Recorder checkpoint
  （需含 conversation、prompt、model config）重複存兩份 tool 紀錄而不一致。
- restore 驗 hash：checkpoint 損毀或被手動改動時立即失敗，而不是默默產生錯誤的 replay。
- Issue-centric Change：5 個 tools 只會改 issue 欄位或新增 comment；用同一個模型就能統一
  表達 forbidden 規則與 mutation scope，不需要通用 diff framework。
- Record 不含時間：trajectory 本身才能 deterministic、可逐 byte 比對；latency 屬非決定性量測，
  由 agent runner 另行傳給 grader。

## 放棄的替代方案

- 用 `pickle` / `deepcopy` 做 snapshot：無法跨 process 保存、無法人工檢查、版本脆弱。
- hash 涵蓋 step 或 records：見上。
- 泛用 JSON diff（JSON Patch 等）：表達力過剩，forbidden / scope 規則會變複雜。
- 在 snapshot 內一起存 records：與 Recorder 職責重疊（Week 1 再定 checkpoint 格式）。

## 對實驗的影響

- 「restore 後 observable state 一致」在 env 層可被測試精確驗證：hash 相同、所有 read tool 結果相同、
  相同後續 call 序列產生相同 observation / hash / changes。
- Grader 以「在新 env 上重放全部 tool call」驗證 trajectory 完整性（prefix + rerun 接續後仍須一致；見 003）。
- 改動 hash 範圍或 snapshot 格式必須升級 `SNAPSHOT_FORMAT` 並另立 decision log。

## 契約審查後修訂（2026-10-08，實作前）

- **call_tool 前置條件**：name 必須是 str；args 必須是 JSON 值（可 canonical 化、可 UTF-8 編碼、不含 tuple）。
  違反時視為 runner 的錯誤：raise、不記錄、step 與 state 不變。args 是 JSON 值但不是 dict 時，
  照常記錄為 `invalid_arguments`（這是 agent 行為）。
- **提交原子性**：tool 在 state 拷貝上執行，hash 與 diff 都成功才提交；非 ToolError 例外直接往外拋且 env 不變。
- **restore 驗證清單與原子性**：key 集合、format、schema_version、step（非 bool、>= 0）、state 可解析
  且為 canonical、actor_id 存在、hash 相符；任一失敗 raise `SnapshotError`（SchemaError 也包裝），
  env 完全不變。未初始化的 env 可直接 restore。
- **reset** 對 dict 與 `IssueTrackerState` 一律以 from_dict 驗證並正規化；失敗時 env 不變。
- **Step 語義**：`record.step` = call 前的 `env.step`；`snapshot["step"]` = 下一筆 record 的 step；
  從 checkpoint rerun 的完整 trajectory = `original[:snap["step"]] + env.records`。
- **Aliasing**：`records`、`current_state()`、`snapshot()`、observation 都回傳深拷貝；restore 不修改傳入的 snapshot。
- **canonical_json**：dict key 必須是 str（自行遞迴檢查，避免 `{1:..}` 與 `{"1":..}` 碰撞），NaN/Inf → ValueError，
  其他非 JSON 型別 → TypeError。
- step 不受 hash 保護；竄改 step 只有在被跳過的區段有淨狀態變化時才會被 grader 擋下（**已於 decision 004 更正**：淨變化為零的區段無法偵測，前綴來源綁定交給 Week 1 Recorder checkpoint）。
