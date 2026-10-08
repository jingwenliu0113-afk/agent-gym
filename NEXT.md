# NEXT

## 目前位置

Core / SPEC「Day 1」**已完成**。下一個 milestone 是 SPEC「Week 1」。尚未進 Advanced（Gate 未滿足：還沒有任何 failure localization 結果）。

## 上次完成（2026-10-08）

Day 1 六項全部完成：

1. Issue Tracker schema — `apps/issue_tracker/schema.py`（decision 001）
2. 5 個 tools — `apps/issue_tracker/tools.py`（錯誤判定順序與訊息模板固定，是 agent 的 observation）
3. Task schema 與 predicate DSL — `tasks/schema.py`（decision 003）
4. Grader schema 與 `grade()` — `grader/`：success 與 trajectory_safe 分開回報；integrity 以重放逐 byte 驗證
5. reset / snapshot / restore — `env/environment.py`（decision 002）：核心 invariant 有隨機化測試
6. 5 個手寫 smoke tasks — `tasks/smoke.py`（`handwritten/v2`，附 oracle 解；**只用於回歸測試，不得計入任何統計**）

流程（TDD）：

- 先寫契約 stub 與 decision log，經 4 個角度的契約審查。
- 由 7 個 agent 先寫測試。紅燈階段 3677 個 failed，全部是 `NotImplementedError`。
- 實作後轉為綠燈。
- 從 4 個角度做對抗式驗證，9 個確認缺陷都已修正（decision 004），smoke 升版 v2。

測試：`uv run pytest` → 3836 passed（約 3 秒）。

資源：全程只用 Mac；沒有使用 Mac heavy lock，也沒有使用遠端 RTX（MACHINE_POLICY：Core 不需要）。

## 下一步（Week 1，建議依序）

1. **Week 1 #4：parameterized task generator（10～20 個 instances）**。開始前先寫 decision log，涵蓋題型、參數空間與資料切分。
   - 集合選取題要自動產生補集 `issue_unchanged` constraint（decision 003 撰寫規則）。
   - 每個 instance 都要通過與 smoke 相同的 invariant：constraints 在初始狀態成立、空 trajectory 不成功、oracle 解為 success + safe + 無多餘變更。
   - 固定 `generator_version` 與 seed；generator 同時產生 oracle 解，但不得交給 agent。
2. **Week 1 #6：Recorder**。完整 checkpoint 包含 env snapshot、前綴 records、conversation、system / developer prompt、prompt version、model config。
   **必須做 decision 004 的前綴來源綁定**：`len(prefix) == snapshot["step"]`，且 `prefix[-1].state_hash_after == snapshot["state_hash"]`；rerun 的前綴由 checkpoint 組裝。
3. **Week 1 #8：從 checkpoint restore 並 rerun**。先用 scripted agent 驗證整條路徑。
4. **Week 1 #5：baseline agent**（ReAct 風格 tool use）。runner 注意事項：
   - observation 一律以 `canonical_json` 呈現給 agent；
   - 使用 `call_tool` 的回傳值，不要逐步讀 `env.records`（會是 O(n²)）；
   - 量測 token / latency / cost / model_turns，填入 `Usage`。

## Blocked / 需要使用者決定

- **baseline agent 的真實 LLM 呼叫需要 API 存取**，例如在 shell 設定 `ANTHROPIC_API_KEY`。依 CLAUDE.md，Claude 不會自行尋找或要求 credential，需由使用者設定。在此之前，runner、recorder、replay 都可以先用 scripted / mock agent 在 Mac 上完成與測試，這部分不 blocked。
- **本資料夾不是 git repo**（`git status` 失敗）。本 session 沒有自行 `git init`，請使用者決定是否初始化，以及 remote 怎麼設定。

## 已知限制（詳見 decision 003 / 004）

- 嘗試但失敗的危險操作不計入 forbidden；scope 內的反覆修改不計為 unnecessary。
- 只拿 records 的 grader 無法偵測「淨變化為零的前綴被刪除」→ 由 Week 1 Recorder 處理。
- comment 內容只以 `contains` 關鍵字檢查，不評語意品質。
- smoke 指令中的 user 一律附 id；SPEC 只有 5 個 tools，沒有 user 查詢 tool。
