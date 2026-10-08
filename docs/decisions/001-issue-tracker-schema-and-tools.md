# 001 — Issue Tracker schema 與 5 個 tools

日期：2026-10-08　狀態：採用（Day 1）

## 背景

SPEC §1 只規定 entity 名稱（issues / users / comments / labels）與 5 個 tool 名稱
（search_issues / get_issue / update_issue / add_comment / assign_issue），沒有規定欄位、
參數語義與錯誤行為。這些會直接決定 trajectory 長相、哪些錯誤可能發生、grader 能看到什麼，
屬於資料格式與評測協定層級的決策。

## 決策

1. **Entity 欄位**（`apps/issue_tracker/schema.py`）
   - User：`id, name, team, active`。`active=False` 代表已離職 / 停用。
   - Label：`name`（主鍵）, `description`。
   - Issue：`id(int), title, body, status, priority, assignee_id, reporter_id, labels, created_at`。
     - status ∈ {open, in_progress, resolved, closed}；priority ∈ {p0, p1, p2, p3}。
     - labels 一律存成排序、去重的 tuple。
     - `created_at` 是種子資料給定的 ISO 日期字串，tool 不會修改；**沒有 updated_at**。
   - Comment：`id(int), issue_id, author_id, body`。
   - next comment id **由資料推導**（max id + 1），不另存 counter。
2. **Tool 語義**（`apps/issue_tracker/tools.py`）
   - `update_issue` 的 `labels` 是**整組取代**，不是 add/remove。
   - `assign_issue(assignee_id=null)` 代表取消指派；可指派給 `active=False` 的 user。
   - `search_issues` 結果依 id 遞增，回傳 `total`（limit 前的匹配數）；`assignee_id="unassigned"` 查未指派。
   - `add_comment` 的 author 固定是 env 的 actor（agent 代表的 user）。
3. **錯誤**：只有三種 code：`unknown_tool`、`invalid_arguments`、`not_found`。
   參數以 `TOOL_SPECS` 的 input_schema 嚴格驗證（bool 不算 integer、`"3"` 不算 issue_id、多餘參數算錯）。
   錯誤一律不改 state（先驗證再套用）。
4. **業務規則不在 tool 層阻擋**：指派給離職者、關閉不該關的 issue、移除 label 等，
   tool 都允許；是否「危險」由 task 的 forbidden 規則在 grader 判定。
5. **使用者以 id 引用**：SPEC 只給 5 個 tools，沒有 user 查詢 tool；task instruction 需要的
   user 會以「名字 (id)」形式寫明。

## 為什麼

- labels 整組取代：真實 API 常見，且會自然產生「不小心丟掉既有 label」這種
  **成功了但破壞 constraint** 的失敗，正是本研究要觀察的 unrecoverable / unsafe mutation。
- 業務規則交給 grader：若 env 直接擋掉危險操作，就永遠觀察不到「最後成功但中途有 forbidden action」，
  與 SPEC §4「最後成功不代表 trajectory 安全」的研究目標衝突。
- 沒有 updated_at / wall clock：Core 必須 deterministic；邏輯時間戳也會讓 read-only call 以外的
  hash 與 diff 多出雜訊。
- 推導 next comment id：snapshot / restore 時 counter 與資料不可能不一致。
- 嚴格參數驗證：讓 tool error 是可明確計數的事件，而不是被寬鬆解析默默吃掉。

## 放棄的替代方案

- `add_labels` / `remove_labels`：較安全但減少可觀察的破壞性錯誤，也偏離 SPEC 的 5 個 tool。
- 加 `list_users` / `search_users` tool：會超出 SPEC 指定的 5 個 tools。若 Week 1 baseline 顯示
  缺 user 查詢造成大量與研究無關的失敗，再另立 decision log 討論。
- 在 env 內建權限 / side-effect policy：屬 SPEC 的 Optional（side-effect safety policy），Core 不做。
- 寬鬆參數（接受 `"3"`、`"#3"`）：會模糊 tool error 指標。

## 對實驗的影響

- tool error 指標只會來自三種明確 code。
- label 整組取代讓「保留既有 labels」成為 smoke task 中的典型 constraint。
- 所有 smoke task 的 instruction 會帶 user id；這降低了 user 識別難度，是已知的簡化。
- 之後若改動任一 tool 語義，必須升級 `SCHEMA_VERSION`（目前 `issue_tracker/v1`）並另立 decision log，
  不同版本的結果不可直接合併比較。

## 契約審查後修訂（2026-10-08，實作前）

4 位獨立審查者（SPEC 符合度 / replay 準備度 / 量測有效性 / 可測試性）審查後，補充以下規定：

- **錯誤判定順序固定**：unknown_tool → args 非 dict → 多餘參數（依名稱排序）→ 缺 required（依 spec 順序）
  → 逐欄位（依 properties 宣告順序）type → enum → minLength → min/max → items → update_issue 至少一欄 → not_found。
  一次只回報第一個錯誤。理由：錯誤訊息就是 agent 的 observation，若依 dict 插入順序決定，
  record 經 canonical JSON 存檔再重放會得到不同 observation，破壞 replay invariant。
- **錯誤訊息模板固定**（列於 `tools.py` docstring），屬於 `issue_tracker/v1` 的一部分。
- JSON Schema 子集的適用範圍：minLength 只作用於 str、min/max 只作用於 int、`integer` 判定為 `type(v) is int`。
- `labels: []` 合法（清空 labels）；`body: ""` 合法；`title: ""` 不合法；`assignee_id: ""` 不合法。
- 寫入與現值相同（含 assign 同一人、None → None）視為成功且不產生 Change。
- search_issues 的 query 以 `str.casefold()` 子字串比對，不做 Unicode 正規化；label 篩選大小寫敏感。
- user id 不可為保留字 `"unassigned"`。
- tool 回傳值必須是 JSON-native（不含 tuple），確保 record 經 JSON 往返後完全相等。
