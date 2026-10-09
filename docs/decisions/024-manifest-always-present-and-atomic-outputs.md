# 024 — sweep 目錄一定有 manifest、所有輸出檔 atomic 寫入（decision 023 審查後；在任何 LLM 結果之前）

日期：2026-10-09　狀態：採用（補足 decision 023 §1 / §2、018 §4；不改排程、估計方法、判定規則、主要 metric、資料切分或模型）

## 背景

decision 023 的對抗式審查（含隨機的續跑鏈模擬：65 條鏈、每條 6–80 次續跑、各種中斷點）確認兩個缺陷：

1. **完全靠接續完成的 sweep 沒有 manifest**（中嚴重度）。觸發條件是：中斷剛好落在「最後一個 source 的最後一個 rollout 已記帳」與
   「它的 `sweeps/<task>.json` 寫入」之間。這時續跑不需要任何新的 call 就能完成這個 source，而 manifest 只會在第一次嘗試時建立，
   所以新目錄只有 run.json、summary.json 與 sweeps 檔。後果：
   - 018 §4 規定沒有 manifest 的目錄不得傳入分析，所以這條付費曲線永遠進不了報告，Gate 的資料也一直不完整；
   - 每次再續跑都 `exit 0`、沒有任何進展。

   隨機模擬中有 7 條鏈遇到這個情況，其餘的曲線都與沒有中斷的排程完全相同。
2. **summary.json 與 sweeps/<task>.json 不是 atomic 寫入**（低嚴重度）。磁碟已滿時會留下截斷的檔案，baseline 與 sweep 目錄
   因此無法被任何文件記載的指令讀取，錯誤訊息（JSONDecodeError）也沒有指出是哪個檔案。唯一的出路是不傳入那個目錄，
   它的成本就會從報告中消失。這與 023 §2 修正的是同一類問題。

同一輪判定「不是缺陷」：以 Python API 繞過 CLI 時，completing 目錄的祖先以外的目錄可能有重複的最終結果。CLI 不會產生這種情況，
也不會讓任何數字出錯（列為 partial、成本照計）。

## 決策

1. **run_sweeps 在寫入 run.json 之後、第一次嘗試之前，建立空的 `manifest.jsonl`。** 因此每個 sweep 目錄都有 manifest；
   完全靠接續完成的目錄也能分析與續跑。
2. **020 §3 的後代檢查改為只看「有紀錄」的目錄**：manifest 至少有一列，或至少有一個 `sweeps/*.json`。
   第一次嘗試之前就中斷的目錄只有空的 manifest、沒有任何內容，仍不需要傳入；傳入也無妨，因為它沒有任何列。
3. **所有會被後續指令讀取的輸出都以 atomic 方式寫入**：同目錄暫存檔、fsync、os.replace，再盡量 fsync 目錄。
   適用範圍是 baseline 與 sweep 的 run.json、summary.json，以及 sweeps/<task>.json。共用 `replay.atomic.write_atomic`，
   Trajectory.save 也改用它。寫入失敗時目標檔不存在（或維持原狀），目錄就會被讀成中斷的 run 或尚未完成的 source，
   這兩種情況續跑與分析本來就能處理。
4. **讀取 JSON 失敗的錯誤訊息一律包含檔案路徑。**

## 為什麼

- 023 讓「完成 sweep 卻沒有任何新嘗試」成為合法情況，018 §4「沒有 manifest = 沒有內容」的前提因此不再成立。
- 已計費的目錄不能因為一次寫入失敗就變成無法讀取。

## 放棄的替代方案

- 分析時接受沒有 manifest 但有 sweeps 檔的目錄：會讓「沒有 manifest」有兩種意義，核對更複雜。建立空的 manifest 比較單純。
- 讀取時把截斷的 JSON 當成不存在：會掩蓋真正的損毀（018 §2：不從被改動或截斷的 summary 計算）。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。
- NEXT 的說明改為：sweep 目錄一定有 manifest（可能是空的）；baseline 目錄中斷在第一次嘗試之前時仍沒有 manifest，不要傳入。
