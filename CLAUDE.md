# Claude Code 工作規則：Agent Gym + Replay

你目前只負責 **agent-gym**。

## 開始任何工作前

1. 完整閱讀 `SPEC.md`。
2. 閱讀 `NEXT.md`。
3. 閱讀 `MACHINE_POLICY.md`。
4. 閱讀 `docs/EXPERIMENT_RULES.md`。
5. 檢查 `git status`。
6. 確認目前正在推進哪個 Core milestone。
7. 若需要遠端 RTX 5070 Ti，先依 `MACHINE_POLICY.md` 判斷是否真的必要。
8. 遠端不可用時，只把該計算步驟標記 blocked，繼續 Mac 上可完成的工作。
9. 除非 Gate 已明確通過，否則不得進 Advanced。
10. 除非使用者明確要求，否則不得實作 Optional。

## 專案邊界

- Core 只有 Issue Tracker 一個 domain。
- 不接 Gmail、真實 filesystem、真實 production API。
- 不做 LangSmith / Langfuse 替代品。
- replay 的價值必須對應一個沒有 replay 很難回答的研究問題。


## 機器與資源衝突

- 必須遵守 `MACHINE_POLICY.md`。
- 正式 Heavy workload / benchmark 所需 lock 拿不到時，不得搶占。
- 不得刪除別的 session 建立的 lock。
- 遠端桌機不可用時，只把該步驟標記 blocked，繼續 Mac 可做的工作。
- 不得自行要求或尋找任何 credential。

## 共通執行邊界

- 不得自行擴張研究問題。
- 不得因為「未來可能整合」而預先建立通用 framework。
- 不得跳過 Core 直接實作 Advanced 或 Optional。
- 不得只回報成功案例。
- 不得看到結果後偷偷修改指標、測試題、資料切分或 benchmark protocol。
- 任何會改變實驗設計的決策都必須留下 decision log。
- 每次 session 結束前更新 `NEXT.md`。
- 如果發現 SPEC 與程式現況不一致，先記錄差異，再處理，不要默默改規格。

## 公開與授權

- GitHub remote：`origin`，**Public**（使用者 2026-10-29 核准），https://github.com/jingwenliu0113-afk/agent-gym。
- **授權：BSD-3-Clause**（原創程式碼，`LICENSE`、`pyproject.toml` 一致）。不得加入 GPL / AGPL 依賴；加入第三方內容時保留原授權並另外列出。
- 本 repo 的 commit email 已設為 GitHub noreply（repo-local git config）；不得改寫 Git 歷史、不得 force push。
- 不得 commit credential、`runs/`、實驗產物與大型檔案；也不得寫入遠端主機的帳號、IP、私有主機名稱或所在地點。
