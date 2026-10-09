# 023 — 續跑接續做到一半的 source、trajectory 先寫入再記帳（最後收斂檢查後；在任何 LLM 結果之前）

日期：2026-10-09　狀態：採用（補足 decision 012 §1、018 §1、017 §4.1；不改 SweepPlan、排程、估計方法、判定規則、主要 metric、資料切分或模型）

## 背景

最後一輪收斂檢查依 NEXT B.1–B.4 原文模擬付費流程（假的付費 client），確認兩個缺陷：

1. **續跑會把做到一半的 source 從頭重做**（高嚴重度）。018 §1 只把「完成的 sweep」列入 skip。預算檢查在每次嘗試前進行，
   所以因預算停止（或 Ctrl-C）幾乎都發生在某個 source 的中途。續跑時，這個 source 在舊目錄已付費的 rollout 全部重做、再付一次費。
   - 模擬中，每次續跑重複付費 $0.28 – $0.53。
   - `--max-cost-usd` 低於單一 source 的 sweep 成本時，每次續跑都花滿上限、重做同一批 rollout，永遠無法完成（`exit 0`，沒有任何進展）。
   - 依 decision 011 選項 A 的估算，每個 source 約 $14 – $216，每次預算停止平均浪費約半個 source。
   - 這違反 018 的理由「被中斷後必須能只補跑缺的部分」，以及 NEXT 寫的「不重複付費」。
2. **trajectory 在 manifest 記帳之後才寫入**（低嚴重度）。run_logged 先把「completed」列寫入並 fsync，之後才存 trajectory 檔。
   如果在兩者之間中斷，或存檔失敗（例如磁碟已滿），manifest 會說完成、檔案卻不存在。
   - 這個 baseline 目錄因此無法讀取，所有後續指令都會失敗。
   - 唯一的出路是不傳入它，但它的已計費成本與 rollout 就會從報告中消失。
   - 這違反 017 §4.1 與 016 A1。

## 決策

### 1. 續跑接續做到一半的 source（sweep 格式升為 `agent-gym/sweep-run/v7`）

- **最終結果的定義**：rollout (k, i) 的最後一次嘗試屬於下列之一，即為最終結果：
  - completed（trajectory 已存檔並通過核對）；
  - agent_output（agent_error）；
  - infrastructure 且已達 MAX_EPISODE_ATTEMPTS（aborted）。

  其他情況（infrastructure 但還沒用完嘗試、harness 中止）都不是最終結果，續跑時會在新目錄從第 1 次嘗試重新執行。
  那些嘗試的成本仍計入它們自己的目錄。
- **續跑**：對每個尚未完成的 source，CLI 從所有 `--resume-from` 目錄收集它的最終結果，交給 `run_sweeps(carried=...)`。
  `sweep(..., prior=...)` 依原本的 run_protocol 重放排程：
  - 已有最終結果的 (k, i) 直接採用舊結果，不再執行，也不寫新的 manifest 列或 rollout 檔；
  - 其餘照常執行。
- **前綴條件**：carried 必須剛好是重放排程的前綴（中斷前依排程順序執行，本來就是前綴）。如果排程要求一個 carried 以外的 (k, i)，
  而 carried 還有沒用到的項目，就在執行任何新的 rollout 之前 ValueError；排程結束時 carried 還有剩，也一樣 ValueError。
- **為什麼排程不變**：重放時，已有的結果與原本相同，所以 coarse → refine 的每個決定都與沒有中斷時相同。
  這只是把同一個預先登記的排程分到幾個目錄執行；plan、criterion、估計與判定都不變。
- **analyze_sweep**：目錄 D 完成某個 source 的 sweep 時，這條曲線的 rollout 必須剛好是「D 自己的列」加上「D 的祖先目錄（resumed_from）
  中該 source 的最終結果」。每個 rollout 以它最終結果所在的目錄核對（manifest、rollout 檔、parent、分類），其他規則都與 017 §5.2 相同。
  - 同一個 (k, i) 的最終結果出現在兩個目錄，或祖先已有最終結果而 D 又執行了它（重複付費）→ ValueError。
  - 祖先目錄的列被接續進曲線之後，就不再列為「部分完成」。partial_sweeps 只列各目錄中沒有進入任何曲線的 (k, i) 數。

### 2. trajectory 先寫入，再記帳

- `run_logged(..., save_path=None)` / `run_with_retries(..., save_path=None)`：完成且通過 `_check_completed` 後，先把 trajectory 寫到
  save_path，再附加「completed」列。存檔失敗（例如 OSError）屬於 harness 中止：照常寫入中止列後往外拋。
  如果在存檔時中斷（KeyboardInterrupt），manifest 不會有這次嘗試，與 NEXT 已記錄的「進行中的嘗試不進 manifest」相同。
- `Trajectory.save` 改為 atomic：先寫入同目錄的暫存檔並 fsync，再以 os.replace 換成目標檔，最後盡量 fsync 目錄。
  失敗時刪除暫存檔，目標檔不會只寫了一半。輸出的 bytes 不變。
- run_baseline 與 sweep 的存檔都改由 run_logged 負責（save_path）。

## 為什麼

- 付費 sweep 是最貴的一步；已付費、已有最終結果的 rollout 不能因為中斷而作廢。
- manifest 是成本與完成狀態的依據；它說完成的 rollout，必須能在磁碟上找到。

## 放棄的替代方案

- 只在剩餘預算不足以完成下一個 source 時拒絕開始：需要在執行中估計單一 source 的成本，估計值不可靠；而且 Ctrl-C 仍然會作廢做到一半的 source。
- 讀取時把「completed 但沒有檔案」視為 interrupted：會掩蓋寫入順序的問題，而且那次嘗試的 trajectory 永遠遺失。
- 續跑時把舊目錄的 rollout 檔複製到新目錄：同一份資料出現在兩處（018 已放棄）。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。一個 source 的 rollout 可能分散在數次執行（不同時間）；時間窗與 response_model 的規則
  （009 §4、017 §4.5）照常適用。
- 目前還沒有任何真實資料，格式升版不影響既有結果。
