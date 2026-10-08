# 004 — Day 1 對抗式驗證後的修訂

日期：2026-10-08　狀態：採用（Day 1；在任何 agent run 之前）

## 背景

Day 1 實作完成、3749 個測試全綠後，以 4 個角度（replay / grader 量測有效性 / 輸入驗證 / SPEC 與 smoke 語義）
對實作做對抗式探測。每個 finding 都由獨立 verifier 重寫重現腳本、對照契約判定。
16 個 finding 中 9 個確認為真實缺陷（其中 2 組重複），7 個駁回（現象成立但契約未規定，或屬設計取捨）。
本文件記錄因此改動的資料格式、驗證規則、評測協定與已知限制。

## 決策

1. **State schema 收緊**（`apps/issue_tracker/schema.py`）
   - 所有字串欄位必須可 UTF-8 編碼（不可含 lone surrogate），否則 `SchemaError`。
   - id / issue_id 必須在 int64 範圍 `[-2**63, 2**63)`。Task 中引用的 issue id 同樣適用。
   - 理由：合法 JSON（`"\ud800"`）就能產生 lone surrogate；原本這種 state 能通過驗證、reset 成功，
     但之後 hash / snapshot / 每次 call_tool 都失敗，env 永久卡死。超大 int 會讓錯誤訊息的格式化本身失敗。
2. **call_tool 前置條件補強**（`env/environment.py`）
   - tool 名稱必須可 UTF-8 編碼（否則 `ValueError`，不記錄）。
   - args 巢狀深度上限 `MAX_ARGS_DEPTH = 64`（否則 `ValueError`，不記錄）；原本約 500 層就會在 deepcopy 漏出 `RecursionError`。
   - restore 計算 hash 失敗時包成 `SnapshotError`（防禦性處理）。
3. **錯誤訊息不再對任意值做 repr / str**：無法字串化的超大整數在 tool 錯誤訊息中顯示為
   `<int too large to display>`；env / task 的錯誤訊息只顯示非字串值的型別名。
4. **`Usage` 建構時驗證**（`grader/schema.py`）：整數欄位 `type(x) is int` 或 None；latency_s / cost_usd
   為有限數字或 None。`grade()` 的 usage 不是 `Usage` 時 `TypeError`。理由：NaN / inf 會讓
   `GradeReport.to_dict()` 無法通過 canonical_json，報告無法持久化。
5. **Grader integrity 改為逐 byte 比對**（`grader/grader.py`）
   - 比較 `canonical_json(record.to_dict())`，不再用 Python `==`（原本 `3 == 3.0`、`True == 1` 會讓型別被竄改的 record 通過）。
   - 不在 integrity 前 deepcopy 輸入（原本不可複製的偽造輸入會漏出 `TypeError`）；所有指標改用**重放產生的 records** 計算，
     報告內容因此不可能沿用偽造的值。
   - `evaluate_predicate` 的 issue_id 必須是 int64 int，否則 `TaskError`（原本 `True` / `1.0` 會被靜默當成 #1）。
6. **Smoke tasks 升版 `handwritten/v1` → `handwritten/v2`**
   - 唯一內容變更：`smoke-escalate-security` 的指令由
     "Do not change any other field of those issues, and do not modify any other issue." 改為
     "Do not change any other field of those issues, do not comment on them, and do not modify any other issue."
   - 理由：constraint 涵蓋 comments（decision 003），但指令原本沒有明講，是 5 題中唯一措辭不一致的題目；
     習慣留稽核留言的 agent 會因措辭歧義而非研究相關原因失敗。
   - 種子世界不變（world hash 相同）；5 題的 content hash 因 generator_version 改變而全部更新（`tests/test_smoke_tasks.py`）。
   - v1 從未用於任何 agent run，**不是看到結果後修改題目**。

## 已知限制（更正先前過度的保證）

- **step 竄改 / 前綴錯配只能部分偵測**：step 不在 state hash 內。從 checkpoint rerun 時，若呼叫端用錯誤的 `snap["step"]`
  切前綴，或刪掉一段**淨狀態變化為零**的區段（read-only call、失敗的 call、改了又改回）並重編 step，
  拼接結果與一條可真實執行的 trajectory 逐 byte 相同，只拿 records 的 grader 原理上無法分辨；
  forbidden action（例如關閉後又重開）會從報告消失。decision 002 / 003 中「由 grader 擋下」的說法已更正。
  `tests/test_adversarial_regressions.py::test_records_only_grading_cannot_detect_removed_net_zero_prefix` 固定此性質。
- **處理方式**：前綴來源綁定屬於 Week 1 Recorder checkpoint：checkpoint 必須一起保存 snapshot 與前綴 records，
  載入時驗證 `len(prefix) == snapshot["step"]`、前綴 step 連續、`prefix[-1].state_hash_after == snapshot["state_hash"]`
  （step 為 0 時比初始 hash）。rerun helper 由 checkpoint 組裝前綴，不讓呼叫端自行切 `original[:k]`。

## 放棄的替代方案

- 把 step 納入 state hash：會讓 read-only call 也改變 hash，違反 decision 002 的設計理由。
- 逐筆 record 的 hash chain：沒有可信錨點時可整條重算，擋不住刻意竄改，且屬過早的通用機制。
- 在 tool spec 為 issue_id 加 minimum / maximum：會改變 agent 看到的 spec，屬實驗設計變更，沒必要。
- `Usage` 要求非負數：契約未規定，暫不收緊。

## 駁回但記錄的觀察（不改程式）

- `env.records` 每次存取都深拷貝全部紀錄；逐步讀取 `records[-1]` 會是 O(n²)。Week 1 runner 應使用 `call_tool`
  回傳的 observation，只在結束時讀一次 records。
- 經 canonical_json 持久化再重建的 observation，dict 的 key 順序與 live observation 不同（內容相等）。
  Week 1 runner 呈現給 agent 時應一律以 canonical_json 序列化，確保 restore / replay 前後 agent 看到的文字逐 byte 相同。

## 對實驗的影響

- 任何有效輸入的評分語義都沒有改變；grader 版本字串維持 `agent-gym/grader/v1`，schema 維持 `issue_tracker/v1`
  （新規則只拒絕兩類輸入：先前就無法正常運作的（lone surrogate、超過 4300 位數的整數），以及超出 int64 的 id；
  現有任何 task / 測試資料都沒有後者）。
- smoke tasks 只用於回歸測試，`handwritten/v2` 之後若再改內容仍須升版並另立 decision log。
