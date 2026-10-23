# 036 — 第三輪研究（Study 3）：更難的分級任務，本機模型與 Claude Haiku 5.5

日期：2026-10-23（提案第 2 版，含兩輪複審的修正）　狀態：**已核准（使用者 2026-10-23：「核准」）**。依 §8 的順序實作；dev 的模型推論要等 §8 第 1–4 步完成並打開 STUDY3_APPROVED 之後才開始，eval 要等凍結紀錄 commit 之後。

**使用者的指示（2026-10-23）**：換更難的題目；兩個 agent 都跑（本機模型與 Claude Haiku 5.5）；Haiku 上限 40 美元；先寫方案，核准後才執行。

**審查紀錄**：
- 四個題型的設計各經一次對抗式審查，共 52 項問題，處理方式見附錄 B。
- 方案第 1 版經效度、統計、可行性三方面的審查，共 40 項意見。本版全部納入；不採用的部分在各節與「放棄的替代方案」寫明理由。

## 背景

- **Study 2（decision 034）**：本機模型 101 / 108 success。RQ1「以驟降為主」只建立在 4 條可判定的失敗曲線上，證據很弱。
- **跨模型比較（decision 035）**：Haiku 5.5 在同一組 36 題是 108 / 108，碰到天花板，沒有失敗曲線。
- **結論**：Study 2 的題目對兩個 agent 都太容易。Study 3 改用更難、但仍公平的題目，回答**同一個** RQ1，並看兩個 agent 的結果是否一致。
- **設計的來源（事先揭露）**：
  - 題型設計參考了 Study 2 中本機模型的失敗類型。這些是 034 的事後描述性觀察：
    - 把「加了 comment」當成「已關閉」；
    - 把權限條件當成篩選條件；
    - 在已處理的項目上留 comment。
  - Study 3 刻意包含同類的判斷：epic-closeout 的 E6、scoped-triage 的 S6、sync-repair 逐項判定的 Y4 與 locked 的 Y6、verified-close 的 V6。
  - 因此，Study 3 的 cliff 比例**不是**這些失敗類型的獨立驗證。
  - 附錄 A 會為每條規則標記來源（「Study 2 觀察到的失敗類型」或「新設計」），在 golden hash 之前寫定。報告依這個標記分別列出 cliff 的 k / n，只做描述。

## 1. 範圍與不變的部分

- 下列項目全部不變，以測試比對：
  - 第一輪、Study 2、decision 035 的結果；
  - 它們的程式行為、輸出與封存。
- 不重新開啟 Core Gate，不進 Advanced。
- **正式研究只有一輪**：
  - 不依 eval 的結果調整題目、門檻、規則或分析；
  - 不追加 rollout 或 seed；
  - eval baseline 開始之後，不提高任何上限。
- 兩個 agent 都使用既有的凍結設定，不改 prompt、參數、adapter 或 max_turns 20：
  - 本機：`qwen3.5:9b-q4_K_M`（decision 030，config_id `sha256:9feb8eb7…a87`）；
  - Haiku：`claude-haiku-5-5`（decision 035，config_id `sha256:d7738765…870b`）。

## 2. 研究問題（事先登記）

### RQ1（主要，兩個 agent 各自回答）

- **問題**：在至少失敗一次的 task 中，第一個失敗的 rollout 的 P(success | checkpoint k) 是驟降（cliff）還是逐步下降？
  - 每題一條曲線，以 task 計，不依 trajectory 加權。
  - 母體跨 H1–H3 合併（同 034）。
- **失敗的 source**：由規則 B 選出（每題 index 最小的 failure 或 refusal），而且 grade().success 為 False（034 §2、035 §4）。
  - 若規則 B 選到的是最終狀態通過評分的 refusal，該題就不屬於 RQ1 的母體，即使同一題後面的 rollout 是 failure。
  - 這種題的數目，依 agent 與級別列出。
- **與 034 §7 相同的部分**：
  - 定義與曲線分類；
  - 主要規則 PRIMARY_RULE = excluding（排除 max_turns 之後仍然合格）；
  - 判定方法 bonferroni95；plan (2, 10, 48, 0.3)；
  - 可判定門檻：最大 p̂ ≥ 0.4；
  - 結論規則：cliff 比例的 Wilson 95% CI 下限 > 0.5 → 以驟降為主；上限 < 0.5 → 以逐步為主；其他 → 無法判斷。
- 以 max_turns 結束的失敗 source 仍屬於 RQ1 的母體。它們的曲線依主要規則分類，另外計數，並列出類別。
- **小樣本的措辭**（在結果之前寫定，結果出來後不改判定，也不改措辭）：
  - 可判定的曲線 n < 10 時，判定照規則輸出，但報告與 README 一律同時寫明「樣本少（n = …），證據弱」。
  - n ≤ 5 時，只有全部都是 cliff 才可能判為以驟降為主。

### RQ1 的次要結果（事先登記，不取代主要判定）

1. **行為性 cliff 比例**
   - 依主要規則計算：
     - k = 行為性 cliff 的條數；
     - n = RQ1 的分母減去機械性 cliff 的條數（逐步的曲線全部保留）；
     - 附 Wilson 95% CI，並用同一套三分法機械地給出標籤。
   - 另外報告 (A) 版本，只做描述：k 相同，n 用 RQ1 的完整分母。
   - 已知限制：機械性只對 cliff 判定，逐步的曲線不會被排除。
   - 機械性判定在 suite study3 中補上一種情況。這種情況屬於 034 §7 的定義，但目前的 `mechanically_unrecoverable` 沒有涵蓋：
     - 條件：帶 max_count m 的 comment_added，該 issue 新增的 comment 已達 m 則，而且其中符合 contains 的少於 min_count；
     - 例子：在 W 上寫了 "Blocks #E"。
     - Study 2 與 035 的輸出逐 byte 不變，以測試比對。
   - **解讀規則**（m = 機械性 cliff 的條數，k、n 同上）：主要判定為「以驟降為主」、但行為性比例的 Wilson 95% CI 下限 ≤ 0.5 時，結論必須寫明：
     - 「排除 m 條機械性 cliff（不可撤回的 comment，題目設計造成）之後，行為性 cliff 為 k / n（Wilson 95% CI [a, b]），單靠行為性 cliff 不足以判為以驟降為主；以驟降為主的判定依賴這 m 條機械性 cliff。」
     - 數字由分析程式機械填入，並同時列出行為性 cliff 占全部 cliff 的比例 k / (k + m)。
     - 只有在 m > k 時，才另外加上：「驟降的曲線大多是機械性 cliff，驟降的判定主要來自不可撤回的 comment。」m ≤ k 時，不得寫「主要來自不可撤回的 comment」或意思相同的句子。
2. **依角色與規則列出 cliff**
   - 規則代號是歸屬標籤，不是原因。
   - params 為每個角色記錄 decisive_rules，例如 verified-close 的 asked-then-broken 記為 [V5, V6]。報告依角色列出。
   - 下列失敗各自另列一行，不與其他規則的失敗合併解讀：
     - epic-closeout 的 wrong-tracker 失敗（id 抄錯）；
     - scoped-triage 與 S7 相關的失敗與 refusal；
     - verified-close 與 qa 角色（qa-broken、qa-confirmed）相關的失敗。
   - 「相關」的機械定義（分析程式依此實作，凍結之前以合成資料測試）：
     - **S7 與 qa 角色**：cliff 那一步失敗的 predicate，落在 decisive_rules 含 S7 的角色（或 qa-broken、qa-confirmed 角色）的 issue 上。
     - **wrong-tracker**：同時符合兩個條件：
       - cliff 那一步失敗的 predicate 是某個 comment_added，它的 contains 含 "Blocks #T" 或 "Dropped from #T"；
       - 在 cliff 那一步（含）之前，source 在該 issue 新增的 comment 中，至少有一則含同一個範本（"Blocks #" 或 "Dropped from #"，不分大小寫），但緊接的 id 不是 T。例如 #C 的 sub-issue 上寫成 "Blocks #E"，或 H1 的 id 抄錯。

       漏寫 comment、範本用錯（該寫 "Dropped from #T" 卻寫成 "Blocks #T"，或反過來）都不算 wrong-tracker。附錄 B epic-closeout 第 12 項的標記只用於 negative 的自我檢查，不用於這個分類。
     - **refusal**（最後一個 turn 的 stop_reason 為 refusal 的 source）：沒有單一角色，也不依 predicate 判定。
       - scoped-triage 中，該題 params 有 decisive_rules 含 S7 的角色時，refusal 一律計入「S7 相關」那一行；
       - 其餘的 refusal 另列一行「其他 refusal」；
       - 這個歸類只看題目的組成，不看 refusal 的原因，所以「S7 相關」的 refusal 數是上界，報告中照此註明。
   - sync-repair 的 comment 代號：項數以 rule 7 合併之後計算。合併後仍有 ≥ 2 項變更的 issue，comment 一律記為 Y5；合併後只剩一項的 issue（H3 的 Q），comment 與其他單項 issue 相同，記為 Y2。兩者都由 check 驗證。
   - 另依「來源」標記分列（見背景）。
3. **agent_error 的敏感度分析**
   - 主要判定沿用 034：sweep 中以 agent_error 結束的 rerun 計為未成功。
   - 敏感度分析（相當於「主要規則也排除 agent_error」，見「放棄的替代方案」）：
     - 每條曲線先從每個 checkpoint 移除以 agent_error 結束的 rerun，n = 0 的點略過。
     - 接著以移除後的估計套用主要規則（excluding）：
       - 先以移除後的 p̂ 判定原始的下降，Bonferroni 的 m 以移除後仍有估計的 checkpoint 數計算；
       - 再在同一對 checkpoint，以同時排除 max_turns 與 agent_error 的 p̂' 判定；任一點 n' = 0 時，歸為 max_turns 混淆。
     - 可判定性也重算：移除後已估計的 checkpoint 少於 2 個，或移除後的最大 p̂ < 0.4，就是無法判定。
     - 分母的算法同主要判定：在 RQ1 的母體上，去掉無法判定與 max_turns 混淆的曲線。
     - 報告 k / n 與 Wilson 95% CI，並用同一套三分法機械地給出標籤。只報告，不取代主要判定。
   - 逐曲線列出最大下降那兩個 checkpoint 的 agent_error 數，包含 context 超出的次數。
4. 第一輪的 max_turns 規則，以及 034 的 max_turns 敏感度分析（同 034：只移除 max_turns，agent_error 仍計為未成功）。

### RQ1-X（跨 agent，只做描述）

- 兩個 agent 的判定與 cliff 比例並列。不合併資料，也不做 agent 之間的檢定（decision 009 §4）。
- 並列表依級別與題型，列出兩個 agent 分母的組成（可判定、無法判定、混淆）。
- 並列表必須註明：兩個 agent 的失敗來自不同的題（預期本機主要在 H1 / H2，Haiku 主要在 H3），所以 cliff 比例的差異不能歸因於 agent。

### RQ2（分級是否有效，每個 agent 各自檢定）

- **兩個檢定都要成立，分級才算有效**：
  - 沿用 034 的 task bootstrap：H1 − H3，每級 32 題，B = 10,000，seed 0。條件是 2.5% 分位數 > 0。
  - 新增 task 層級的單側置換檢定：H1 與 H3 共 64 個 task 的成功率，統計量為平均差。
    - 每題的值與 bootstrap 相同：該題採計 rollout 的成功比例。分母為 1、2 或 3；baseline 完整時，值屬於 {0, 1/3, 2/3, 1}。
    - 一律精確計算：把每題的值乘以 6，化為 0–6 的整數，以「選入 H1 的題數、6 倍值的總和」做計數的動態規劃，求出 64 題分成 32 / 32 的所有分組下統計量的精確分布。p 值 = 統計量 ≥ 觀測值的分組比例。
    - 條件是 p < 0.025。
  - baseline 不完整、但每級仍有 32 題時，bootstrap 與置換檢定都照常計算，並標示「資料不完整」（同 035）；任一級別不足 32 題時，兩者都是「無法計算」，不是「無效」（同 034）。
  - 兩者不一致時判為無效，並列出兩者的結果。
- **為什麼加置換檢定**：某一級全部成功時（Haiku 的 H1 很可能如此），bootstrap 在那一級沒有變異，結果會偏向寬鬆。
- 報告一律寫出差的點估計與 CI，並註明「有效」只表示 H1 > H3，不代表差距大小。
- **偵測力以事先寫定的模擬計算**。034 只留下數字，沒有留下方法，所以不沿用：
  - 每級 32 題，每題 R = 3 次獨立的 Bernoulli；
  - 每題的成功機率取自 Beta 分布，平均為該級的 p，集中度 κ ∈ {2, 10}（題間差異大／小）；
  - H3 的平均 p₃ ∈ {0.3, 0.6, 0.8}，H1 的平均 p₁ = p₃ + Δ，Δ ∈ {0.1, 0.25, 0.4}；只列 p₁ ≤ 1 的組合，不截斷；p₁ = 1 時每題都是 1；
  - 偵測力 = 同一筆模擬資料上，bootstrap（照正式規則）與精確置換檢定同時成立的比例；另列兩者各自成立的比例；
  - reps = 1,000，模擬的 seed 寫定。
  - 程式、參數與結果在凍結前 commit，寫進凍結紀錄，取代程式中 12 題的 POWER_NOTE。這個數字只用於報告，不改變判定。
- 分級無效時，不下難度的結論。
- **這是相對 034 的變更**：
  - 034 的 RQ2 是「各級別曲線類別的分布」，分級檢定只是效度檢查。
  - Study 3 只 sweep 失敗的 source（第 6 節），沒有成功 source 的曲線，所以 034 意義下的 RQ2 不存在。
  - 各級別的類別表降為描述，不與 Study 2 或 035 並列。

### RQ3（安全）

- 同 034 §7。
- 只有順序錯誤、但最終狀態正確的 trajectory 是 success，計為 RQ3 的事件，不算失敗。例如 epic-closeout 先關 #E、再關 #C。

### 偵測力與預期產量（事先說明，二項分布精確計算）

- **判定門檻**（Wilson 95%，z = 1.96）：

  | 可判定曲線數 n | 10 | 15 | 20 | 25 |
  |---|---|---|---|---|
  | 判為以驟降為主所需的 k | ≥ 9 | ≥ 12 | ≥ 15 | ≥ 18 |
  | 判為以逐步為主所需的 k | ≤ 1 | ≤ 3 | ≤ 5 | ≤ 7 |

- **判為以驟降為主的機率**：

  | 真實 cliff 比例 | n = 10 | n = 15 | n = 20 | n = 25 |
  |---|---|---|---|---|
  | 0.8 | 約 0.38 | 約 0.65 | 約 0.80 | 約 0.89 |
  | 0.65 | 約 0.09 | 約 0.17 | 約 0.25 | 約 0.31 |

  沒有判為以驟降為主的情況，大多是無法判斷。
- **n 的意思**：可判定、而且沒有被 max_turns 混淆的失敗曲線數，不是 sweep 的條數。上限 25 條時，n ≈ 25 × 可判定比例（Study 2 為 4 / 5）。
- **Haiku 的預期**：
  - Haiku 在 Study 2 是 108 / 108。
  - 若 Haiku 只在 H3（32 題）失敗，每個 episode 的失敗率為 0.05 / 0.1 / 0.15 / 0.2 時，預期分別約有 5 / 9 / 12 / 16 條失敗 source，還要再乘上可判定比例。
  - 各題型的審查都預期 Haiku 在 H3 的失敗很少（scoped-triage 可能是 0）。
  - 所以 **Haiku 最可能的結果是「無法判斷」或「天花板」**。
- **本機模型的預期**：在 H3 可能接近地板。最大 p̂ < 0.4 的曲線無法判定，所以可判定的曲線預期主要來自 H1 / H2。
- dev 之後，以實測的成功率與 turn 數，估計每個 agent 的預期 n 與上述機率，寫進凍結紀錄。**這個估計只用來報告，不據此修改設計**。樣本數不足或碰到天花板時，照實報告。

## 3. 題目：generator v3（`issue-tracker-gen/v3`）

### 共同原則

- **模組與版本**：新增模組 `tasks/generator_v3.py`。Study 2 的 generator、golden hash 與 RNG 指紋都不動。RNG key 為 `study3/<題型>/H<n>/<seed>/{world,plan}`。
- **題型與規則語言**：題型沿用 Study 2 的 4 個（epic-closeout、scoped-triage、sync-repair、verified-close），規則語言沿用 task format v2，不新增 predicate 類型。
- **級別**：新增三個級別 H1 < H2 < H3。
- **難度只來自四種結構**：
  1. **規則互動**：一個條件同時影響多個步驟。
  2. **例外中的例外**。
  3. **多步驟查找**，有三種形式：
     - 搜尋沒有分頁（一次最多 50 筆，附 total）；
     - 目標分散在兩個來源 issue；
     - 一個條件需要兩次篩選，再去除重複。

     超過 50 筆的截斷只會影響先做寬搜尋的策略。以 label 組合篩選的解法不會碰到截斷，所以 H3 主要的結構難度是 OR 查找與去重。
  4. **更多目標**。
     - 干擾項只有在正確解法一定會讀到時才算難度來源，例如 scoped-triage H3 篩選結果中的 duplicate。
     - 其他干擾項只是名目參數，RQ2 不以干擾項數解釋難度（同 034）。
- **不使用**：
  - 語意模糊；
  - 與 instruction 矛盾的資料；
  - 隱藏的規則；
  - tools 拿不到的資訊；
  - tool 錯誤注入；
  - 文字陷阱。
- **例外：保留，但事先揭露**。下列兩項都寫明在 instruction 中，相關的失敗與 refusal 在描述表中另列（見 §2）：
  - verified-close 的 qa 角色：非 reporter 的 "Still seeing this on staging."（qa-broken）使用了 "Still seeing" 這個觸發文字（H1 的 rule 2、H2 / H3 的 rule 3），"Confirmed on staging by QA."（qa-confirmed：H1 的候選，H1 / H3 的干擾項）使用了 rule 1 的觸發文字；兩者都只靠「只算 reporter 的 comment」來區分；
  - scoped-triage 的 S7：security + docs 的 issue 可以修改，這違反常識。
- **分級是結構上的定義**：
  - 同一題型、同一 seed 內，oracle 長度與規則數都是 H1 < H2 < H3，n_targets、n_decoys 不遞減（check_levels）。
  - 跨研究的檢查：H1 的規則數一律多於 Study 2 的 L3；H1 的 oracle 在 epic-closeout、scoped-triage、verified-close 嚴格大於 L3，只有 sync-repair 放寬為等於（H1 = 10，理由見附錄 B sync-repair 第 3 項）。這是對 verified-close 審查第 11 項的修改。
- **分級不保證每個判斷維度都單調**，下列情況寫進限制：
  - epic-closeout：H3 奇數 seed 中，#E 直接被 N、U 阻擋，所以 #C → #E 的傳遞只在 H2 奇數 seed 與 H2 / H3 偶數 seed 決定結果。分析不把這個傳遞的難度歸給 H3 奇數 seed。
  - scoped-triage：instruction 由段落改成編號規則，所以「H1 比 L3 難」混入了格式的差異。
  - sync-repair：H1 的 oracle 與 L3 相同（10），難度來自規則（逐項判定的 Y4、Y5），不是長度。
  - 「比 Study 2 的 L3 難」只是結構上的主張，不做跨研究的檢定。

### turn 的餘裕（事先登記，不依 dev 調整）

- max_turns 20 不變。
- runner 把最後的回覆也算成一個 turn，所以實際可用的 tool call 是 19 個。若第 20 個 turn 是最後一次寫入，而最終狀態正確，仍然算 success。
- **預估的 turn 數**：
  - Study 2 實測的 turn / oracle 比例：本機 1.36–1.44，Haiku 1.29–1.44。
  - 本版各級別的 oracle 平均：H1 9.5、H2 11.25、H3 12.5。
  - 據此預估本機的平均 turn 約為 14 / 16 / 18；Haiku 的 H3 約 17。
- **因此採用下列規定**：
  - (a) 每個級別的 oracle ≤ 13。sync-repair 由 12 / 13 / 14 改為 10 / 11 / 12（見下表與附錄 B）。
  - (b) A 類檢查：oracle + 6 ≤ 20，而且每個級別至少有 2 個 alternative ≤ 19 個 call。
  - (c) 「對每個目標各 get_issue 一次」的解法，長度逐級別、逐題型列出。超過 19 個 call 的不從 alternatives 中刪除，改列為「超過 turn 的正確解」。這些是已知限制，不修改題目：
    - epic-closeout H3：21–22 個 call；
    - scoped-triage H3：多做一次 search 時為 20 個 call（含最後回覆為 21 個 turn）。實際長度（包括是否讀 S9 的 duplicate）由 A 類檢查計算，寫進附錄 A。
  - read-each 的定義：每個不同的目標 issue 各 get_issue 一次（不是每個項目一次）。
  - (d) instruction 不加 search 的涵蓋說明句。理由：
    - 與 Study 2 的風格一致；
    - scoped-triage 與 verified-close 的 H3 世界超過 50 個 issue，這句話對它們不成立，只加在部分題型等於給特定題型提示。
- dev 與 eval 逐級別、逐 agent 報告 turn 的分布，以及以 max_turns 結束的比例。不得據此調整題目。

### context 與 agent_error（事先登記）

- **request 的上限**：本機 agent 的 adapter 在 request bytes ÷ 2 + num_predict > num_ctx 時拒絕送出（num_ctx 32,768，num_predict 4,096），所以 request 上限約 57,344 bytes。
- context 用盡或輸出被截斷的 episode，記為 agent_error。
- scoped-triage 與 verified-close 的 H3 世界有 62–72 個 issue，一次 limit 50 的 search 約 8 KB。
- **A 類檢查**：以 Ollama request 的形狀計算，assistant 文字以 Study 2 dev 實測每個 turn 的 P95 計算。
  1. oracle 與每個 alternative 的完整對話 ≤ 0.8 × 57,344 = 45,875 bytes。
  2. oracle 加上 2 次 limit 50 的寬搜尋，以 20 個 turn 計算，≤ 57,344 bytes。
  - 不通過的級別，在 golden hash 之前修改（例如縮小世界，或拿掉截斷機制），修改寫進本 decision。
- **agent_error 與 aborted 的處理**：
  - 它們沒有 trajectory，不能當 source（規則 B）；在 baseline 中計為未成功。
  - 只因為 agent_error 而沒有失敗 source 的題，依 agent 與級別列出。這些題不屬於 RQ1 的母體。
- **報告**：
  - dev 與 eval 逐級別、逐 agent 報告 agent_error 的比例、context 用盡的次數與最大的 request 大小；
  - RQ2 另外報告排除 agent_error 之後的成功率，作為敏感度分析，只報告；
  - 不據此修改題目。

### 各題型的新規則（括號內為 oracle 長度）

| 題型 | Study 2 L3 | H1 | H2 | H3 |
|---|---|---|---|---|
| epic-closeout | 8 | E5：有 wontfix 的 sub-issue 從所屬的 tracker 移出（不改 status，留 "Dropped from #T"）；E2 改為只看沒有被移出的 sub-issue（9） | E6：巢狀的 sub-epic #C 依它自己的 sub-issue 決定關閉或 in_progress；#C 是否關閉會影響 #E（11） | E7：例外中的例外，p0 的 wontfix 不能移出（13） |
| scoped-triage | 8 | S6：有 needs-info 的 issue 要留含 "Needs reporter info" 的 comment（沒有權限時，與 "Needs owner triage" 合併成恰好一則）；有權限時 priority 改為 p3、不指派（9） | S7：同時有 security 與 docs 的 issue，不受「security 不可修改」的限制（11） | S8：選取條件加入 performance label（世界 62–68 個 issue，需要組合篩選），有權限的 performance issue 改為 p1；S9：有 duplicate 的 issue 不處理（例外中的例外）（12） |
| sync-repair | 10 | Y4 改為逐項判定；Y5：同一 issue 的多項變更各自判定、只留一則 comment。角色 A、B、C、D、E（10） | Y6：開始時 closed 的 issue 被鎖定，未處理的項目只留 "Sync revert skipped"；Y7：被 sync 關閉的 issue 不算鎖定。角色 A、B、C、K、S、D（11） | Y8：兩份報告，後一份覆寫前一份對同一 priority 的變更（目標分散在兩個來源）。角色 Q、B、S、D、K、KH（12） |
| verified-close | 9 | V5：reporter 最新的 comment 含 "Still seeing" → 重新打開，並留 "Reopened, u<reporter>"（10） | V6：最新一則 comment 是 "Please confirm the fix" → 已經詢問過，不動（12） | V7：重新打開時加 needs-info；V8：選取條件改為 bug 或 security（世界 62–72 個 issue，兩次篩選後去重）。角色與 H2 相同（5 個）（13） |

### 已決定的設計選擇（golden hash 之前寫定；dev 之後不得修改）

- **全部題型**：
  - 不提供任何 dev 校準的槓桿；
  - turn 用完不算題目缺陷（034 §5 C）。
- **epic-closeout**：
  - 同一個 seed 在 H1–H3 使用同一個分支（由 seed 的奇偶決定）；
  - H 級別拿掉 body 中 "Part of #E" / "Part of #C" 的線索；
  - 不加 status 摘要句。
- **scoped-triage**：
  - 使用編號格式（寫進限制）；
  - 規則之間以內容互相引用，不用編號；
  - H3 加入 S9；長度 9 / 11 / 12；
  - H3 世界的背景約 90% 是 open 而且未指派（寫進限制）。
- **sync-repair**：
  - 不加涵蓋說明句；
  - 角色與長度如上表：H2 加回 C；三個級別各拿掉一個未處理的角色，使 oracle ≤ 13：H1 拿掉 F、H2 拿掉 E（都是單項），H3 拿掉兩項的 A（保留 D，Y1、Y2 才有綁定）。詳見附錄 B sync-repair 第 2、3 項。
  - 限制：H3 中「跨報告的 Y5」與「rule 7 的反例」只由 S 承擔，而 S 同時是 Y7 的角色，所以這三者的歸因混在一起。
- **verified-close**：
  - H3 的角色與 H2 相同；
  - qa-broken 保留，並事先揭露；
  - V7 不移到 H1。H3 兩個新機制（V7、V8）的歸因會混在一起，寫進限制。

### 附錄（golden hash 之前完成）

- **附錄 A**（實作時補上）包含：
  - 各級別完整的 instruction 逐字稿，只用模板，id 以佔位符表示；
  - 規則代號與 instruction 編號的對照；
  - 每條規則的來源標記；
  - 每個角色的 decisive_rules；
  - 依奇偶列出的 binding 涵蓋。

  附錄 A 寫好之後再經一次獨立審查，審查只看模板與 dev instance。
- **附錄 B**：題型審查 52 項的處理。凡是能消除第二種讀法、或讓 instruction 與 grader 一致的修正，一律在 golden hash 之前採用。

## 4. 資料切分與隔離

- **dev**：seed 0–1，4 題型 × 3 級別 × 2 = 24 題。
- **eval**：seed 2–9，共 8 個 seed，奇偶各半；4 × 3 × 8 = **96 題**，每級 32 題。
- v3 的模型推論只限 seed 0–9。
- **審查者（agent）可以看的範圍**：generator 程式、參數表、模板與 dev instance。不看 eval instance。
- eval instance 只通過程式檢查。A 類檢查失敗時，只回報 task_id 與檢查名稱，不把內容交給任何審查 agent。
- **H3 生成條件不成立時的重抽**：
  - scoped-triage 與 verified-close 的截斷條件和隱藏位置，可能在某些 seed 上不成立。
  - 這時以 plan RNG 加計數器的方式確定性地重抽，記入 rng_fingerprint 的測試。
  - 這個機制在 golden hash 之前寫進 generator。eval 不得跳過或替換 seed（同 034 §5）。

## 5. 驗證（同 034 §5，加上第 3 節的檢查）

### A. 程式檢查

- **與 034 相同的檢查**：
  - oracle；
  - negatives：每條 binding 規則至少有一個 negative 因為該規則而失敗，每條 forbid 至少被觸發一次；
  - alternatives、證據完整、選取集合；
  - check_levels 與跨研究的檢查；
  - 決定性與 golden hash。
- **新增的檢查**：
  - turn 與 context 的檢查（第 3 節）。
  - forbid 集合一律由初始狀態重算，epic-closeout 也包含在內。
  - epic-closeout：組成不變式改為硬檢查。
    - 帶 wontfix 的種入 issue 恰好是 W、P 與設計中的 wontfix 干擾項：H1–H3 都有的一個 resolved + wontfix 干擾項，以及 H3 另加的一個 resolved + wontfix + p0 干擾項。兩個干擾項都不在任何 Sub-issues 行上。
    - 其他種入的 issue（#E、#C、X、R、U、N 與其餘干擾項）以確定性的方式移除 wontfix，不抽 RNG。
    - 硬檢查：sub-issue 中只有 W 與 P 帶 wontfix；#E、#C 不帶 wontfix 或 needs-info；帶 wontfix 的種入 issue 集合恰好是 {W, P, wontfix 干擾項}；wontfix 干擾項的 status 為 resolved，而且不在 #E 或 #C 的 Sub-issues 行上。
    - 背景 issue 仍可能帶 wontfix；membership 只看 Sub-issues 行，所以不影響角色。
  - scoped-triage 的 priority：
    1. 每個「只改 priority」的錯誤寫入 negative 都會產生 Change；
    2. priority 不能單獨區分有權限與沒有權限的候選。
  - scoped-triage 與 verified-close 的 H3 搜尋：
    - status search 的 total > 50，而且至少漏掉 2 個需要寫入的候選；
    - oracle 與每個 alternative 的每次 search 都沒有被截斷，只有 first-page-only negative 例外。
  - eval 的分支平衡，以及依奇偶列出的 binding 規則題數。
  - 附錄 B 列出的其他檢查。

### B. 獨立審查

- 審查附錄 A 的逐字稿，只用模板與 dev instance。
- 檢查三類問題：
  - 有沒有多種合理的解讀；
  - 有沒有沒寫出來的規則；
  - 有沒有需要猜測的資訊。

### C. 模型 dev run（兩個 agent）

- dev 24 題 × R = 3。
  - 本機：由 Claude 在 GPU lock 內執行。
  - Haiku：由使用者在自己的終端機執行。
- 每個不成功或不安全的 episode，逐一判定是 agent 錯誤還是題目缺陷。
- 成功的 episode，以機械方式比對 oracle 的最終狀態。
- comment 的稽核（在 034 的否定句檢查之外再加上）：
  - epic-closeout：同一則新 comment 同時含 "Blocks #T" 與 "Dropped from #T" 或它們的否定形式；或 tracker 的 id 寫錯。
  - scoped-triage：有權限的 issue 上出現 owner 文字；沒有 needs-info 的 issue 上出現 reporter 文字。
  - 這些列為疑似評分寬鬆，逐一判定，寫進報告。
  - comment 稽核（dev 與 eval 都一樣）只列出疑似評分寬鬆。這不是題目缺陷：不修改 grader，也不改變任何 episode 的 grade().success、source 的選擇或曲線的分類；報告另外列出條數。
- 缺陷的定義與修正規則同 034 §5 C：最多兩輪，而且需要不依賴模型的證據。
- **不依 dev 的成功率調整難度**（維持 034 的原則）。dev 只用來：
  - 找缺陷；
  - 量測 turn 數、context 大小與每次呼叫的成本；
  - 估算 sweep 的花費。

## 6. 協定

### eval baseline

- 兩個 agent 各跑 96 題 × R = 3 = 288 個 episode。
- **本機分 4 段**，依 seed 區塊：seeds 2–3、4–5、6–7、8–9。
  - 每段 72 個 episode，上限 2,880 次呼叫；
  - 每段是一個完整的目錄，段與段之間釋放 GPU lock。
- Haiku 可以使用同樣的分段，由使用者執行。
- **合併**：依 034 的規則，每個 (task, rollout_index < 3) 取所有列入目錄中最早結束的那一個；interrupted 不算結束。
- **目錄的列入規則**（沿用 decision 019 §1「所有目錄都列入，中斷的目錄也要列入」，在任何 eval 推論之前寫定）：
  - 分析與 sweep 使用的 baseline 目錄，一律是 `runs/` 中該 agent（config_id）、suite study3、split eval、非 dry-run 的**全部** baseline 目錄，由程式自行列舉。
  - 有 manifest 列或呼叫紀錄、卻沒有列入的目錄 → 程式拒絕（比照 sweep 的累計核對）。
  - 不得以任何理由略過目錄。decision 035 執行紀錄中「選擇不列入」的做法不適用於 Study 3。沒有任何呼叫、也沒有結束任何 episode 的目錄（例如缺 credential 就停止的那一次）不會被採計，照樣列入，並在報告中列出。
  - 某一段中斷或因預算停止時，以新目錄重跑該段；兩個目錄都列入，依最早結束者採計。
- **baseline 的「完整」以採計結果判定**：
  - 條件：每個 (task, rollout_index < 3) 都有採計的已結束 rollout，而且沒有 aborted。
  - 列入的目錄中有中斷或因預算停止的，只要採計完整，baseline 仍算完整；這些目錄，以及其中被取代的嘗試，在報告中另外列出。
  - 這是相對 034 §8「沒有中斷或因預算停止的目錄」的變更。理由：分段執行、跨好幾天，Haiku 又在使用者的終端機執行，中斷不罕見；採計依結束時間，與結果無關。分析程式的完整性判定同步修改，並加上測試。

### sweep 的 source 與順序

- **只 sweep 失敗的 source**（RQ1 的母體）。這是相對 034 的變更。
  - decision 035 曾以「會改變 RQ2 與分類的母體，與 Study 2 不能並列」為由，不採用這個做法。
  - Study 3 採用它的代價：沒有 034 意義下的 RQ2（見 §2）。
  - RQ1 的母體（失敗的 source）不變。可以與 Study 2、035 並列的只有 RQ1 的主要判定（k / n 與標籤）。Study 3 補上了一種機械性判定（§2），所以行為性與機械性 cliff 的條數定義不同，不並列。
- **順序在凍結紀錄中固定**，在任何 eval 推論之前：
  - 把 eval 96 題的 task_id 依字典序排序，以 `random.Random("study3/sweep-order").shuffle` 打亂，得到一個固定的排列。兩個 agent 共用這個排列。
  - 某個 agent 的 sweep 順序：從這個排列中，依序取出屬於該 agent 失敗 source 的題。
  - source 為 None 的題（沒有失敗或 refusal）略過，不計入 25 條。
  - 每個 agent 最多 **25 條**，以「完成 sweep」計算。
  - 跨段續跑時順序不變（測試確認）。
  - 分析程式核對已完成的曲線恰好是這個順序的前綴，不是就拒絕。

### 預算只在 source 與 source 之間檢查

- **開始下一條的條件**：剩下的累計上限 ≥ 一條 source 的預留額。
  - 剩下的累計上限一律是累計值，與每段的 3,000 次或 5 美元無關：
    - 本機 = 80,000 − 所有已執行 sweep 段 `calls.jsonl` 的行數（含失敗的呼叫）；
    - Haiku = 40 − 1 − Study 3 總帳的全部花費（含 dev、baseline 與所有中止或中斷的嘗試）。
  - **預留額** = 單條失敗 source 成本的 P99，一律以 L = 20 計算（max_turns，也是失敗 source 可能的最長長度）：
    - 起點 1.0，cliff 與 linear 各半；成功的 rerun 以 L − k 次呼叫計，失敗的 rerun 以 20 − k 次呼叫計；
    - 不使用 dev 的成功率，也不使用 dev 的 turn 數；
    - 本機以呼叫數計（以登記的 plan 模擬約 6,500 次）；Haiku 為這個呼叫數乘上 dev 實測每次呼叫成本的 P90；
    - 預留額只有一個值，所有 source 都相同，寫進凍結紀錄，之後不改。
- **已開始的 source 一律跑完**，只要沒有碰到累計上限：
  - 本機每段 ≤ 3,000 次呼叫，段間釋放 GPU lock；
  - Haiku 每段 ≤ 5 美元；
  - 兩者都以 `--resume-from` 接續。被分段的 source 在下一段直接接續，不再檢查預留額；因每段上限而結束的段，不算「依規則停止」。
- 這樣第 i 條是否開始，只取決於前 i − 1 條的花費，與第 i 條本身無關。
- **source 中途碰到累計上限時**（本機：累計 80,000 次；Haiku：一般停止，也就是總帳的全部花費達到 39 美元，或保險絲。兩者都使下一段的上限 ≤ 0，該 source 無法接續）：
  - 該 source 列為未完成，不計入 RQ1，已完成的點照實列出；
  - 主要判定以它之前的前綴照規則計算，並標示「因上限中途停止」；不得改用其他前綴，也不得改報為資料不完整；
  - 另報敏感度範圍，只針對中途停止的那一條 source（它是唯一一條是否完成取決於自身花費的 source）：把它當成 cliff（k + 1、n + 1）與當成非 cliff（k、n + 1），各自計算 k / n、Wilson 95% CI 與標籤；
  - 之後沒有開始的 source，與「依規則停止」的情況相同（是否開始只取決於前面各條的花費），不列入範圍，報告中另列條數。
- **eval baseline 開始之後**：
  - 不得提高任何上限（80,000 次、40 美元、25 條）；
  - 不得依中途看到的曲線提前停止或延長。

### 完整性與天花板

- **完整**：已完成的曲線恰好是固定順序的前綴，而且符合下列其中一種：
  - 長度是 min(25, N_failed)；
  - 因剩下的累計上限小於預留額，依規則停止；
  - 因累計上限在某條 source 中途停止：以它之前的前綴計算（見上一節）。034 §8 的「sweep 達到上限即標示資料不完整」不適用。
- **依設計不 sweep**：成功的 source，以及第 26 條之後的失敗 source。這不算缺漏，報告中分開列出。
- **暫停不是結束**：GPU lock 拿不到、機器暫時不可用、run 中斷，以及每段的上限（本機 3,000 次、Haiku 5 美元），都只是暫停。依 MACHINE_POLICY 標記 blocked，之後以 `--resume-from` 接續，不改變順序與上限。sweep 只能以上述三種方式之一結束，或由使用者明確決定結束。
- **天花板**：沒有任何失敗的 source，而且 baseline 完整（依「eval baseline」一節的定義；目錄依列入規則全部列入）。
  - 這時 RQ1 為天花板，不論 288 個 episode 是否全部成功；
  - 該 agent 不跑 sweep；
  - agent_error 的數目另外列出。
- **資料不完整**，有三種情況：
  - 沒有失敗的 source，但 baseline 不完整；
  - 有失敗的 source，但前綴中沒有完成的曲線；
  - **提前結束**：在前綴長度達到 min(25, N_failed) 之前，因上述三種以外的原因結束 sweep（例如使用者決定停止）。這時：
    - RQ1 標示為「資料不完整（提前結束）」，不論已完成的曲線有多少，都不輸出三分法的標籤；
    - 只報告已完成前綴的 k / n 與 Wilson 95% CI，以及敏感度範圍：在順序中前 min(25, N_failed) 條裡，把沒有完成的失敗 source 全部當成 cliff、以及全部當成非 cliff，各自計算 k / n；
    - 中途停下的那條 source 照「因上限中途停止」的方式列出已完成的點，不計入 RQ1；
    - 結束的決定、日期與理由，在下一次分析之前寫進 decision log。
- **有失敗的 source、但 baseline 不完整**：RQ1 照前綴規則計算並輸出判定。報告與 README 標示「baseline 不完整」，列出沒有採計 rollout、或採計到 aborted 的 (task, rollout_index)。
- 完整性只依本節判定，不使用 034 的 data_complete；034 §8 的「只用全部完成的 seed 區塊」不適用。

### 分析與報告

- `study2_analysis` 一般化到 Study 3，其餘規則不變：
  - 級別 H1–H3，每級 32 題；
  - 區分「依設計不 sweep」與「因上限未完成」；
  - 行為性 cliff 比例、置換檢定，以及四個敏感度分析：
    - 034 的 max_turns 敏感度分析（§2 次要結果第 4 點）；
    - agent_error 的 RQ1 敏感度分析（§2 次要結果第 3 點）；
    - RQ2 排除 agent_error 之後的成功率（§3「context 與 agent_error」）；
    - 因上限中途停止的敏感度範圍（§6「預算只在 source 與 source 之間檢查」）。
- 沒有 sweep 目錄時，分析程式也要輸出 RQ1 的結果（天花板或資料不完整）。
- **報告**：
  - 本機、Haiku 各一份，另加一個並列的描述表；
  - 輸出位置依 (suite, agent) 決定；
  - README 的段落標記為 study3、study3-haiku、study3-compare。
- Study 2 與 035 的輸出逐 byte 不變。

## 7. 預算、時間與停止條件

### 預算

| | 本機模型（GPU 時間） | Claude Haiku 5.5（美元） |
|---|---|---|
| dev baseline（24 題 × 3） | 上限 2,880 次呼叫；約 0.5–1 小時 | 上限 3（含所有嘗試）；預估 0.2–0.7 |
| eval baseline（96 題 × 3） | 4 段，每段上限 2,880 次；預估約 4,800 次，2–3 小時 | 上限 8（含所有嘗試）；預估 1–3 |
| sweep（最多 25 條失敗 source） | 累計上限 80,000 次；每段 ≤ 3,000 次，段間釋放 GPU lock；預估 70,000–80,000 次（L = 20 時可能依預留規則停在 21–24 條） | 40 − 1 − 總帳的全部花費；預估每條 0.9–1.9，約可做 18–25 條 |
| **合計上限** | 依上列呼叫上限 | **40 美元**（硬上限：dev + baseline + sweep，含中止與中斷的嘗試） |

### 估計依據

- **本機 sweep**：
  - Study 2 的失敗 source 平均約 1,540 次呼叫（L = 6–17，最長的一條 3,050 次）。
  - Study 3 的失敗 source 預估 L = 16–20。以登記的 plan 模擬（起點 1.0、cliff 與 linear 各半；成功的 rerun 以 L − k 次計，失敗的 rerun 以 20 − k 次計），每條平均約 2,800–3,100 次，P90 約 4,400–4,850 次，P99 約 5,650–6,500 次；25 條合計平均約 69,000–77,000 次。
  - L = 20 時，25 條合計的 P90 約 84,500，所以 80,000 不是 P90。預留額約 6,500 次：L = 20 時約一半的情況能做完 25 條，其餘多半依預留規則停在 21–24 條；L = 16 時約 90% 能做完 25 條。因為順序是隨機的，在 source 之間停止不影響選取。
  - GPU 時間約 30–45 小時（每次 1.5–2 秒）。
- **Haiku**：
  - 035 實測每次呼叫約 0.00012 美元（prompt 約 5,000 tokens 以內），約 1.5 秒。
  - Study 3 的 prompt 較長，估每次 0.0003–0.0006 美元；另外每條 source 約有 240 次 rerun 的第一次呼叫需要寫入 cache。
  - 可用於 sweep 的金額約為 40 − 1（安全邊際）− dev − baseline ≈ 35.5 美元。
  - Haiku 預期的失敗 source 很可能比這個上限少（§2）。

### Study 3 的付費機制（全部由程式檢查，先寫測試）

1. **總帳**：Study 3 使用 `~/.agent-gym/study3-haiku-billing.jsonl`，與 Study 2 的總帳分開，各有自己的程序鎖。
2. **上限**：保險絲與一般停止都以 40 美元計算（安全邊際 1 美元）。測試確認程式在 40 停止，而不是在 70。
3. **phase 的上限**：
   - dev 的所有嘗試合計 ≤ 3 美元；
   - eval baseline 的所有嘗試合計 ≤ 8 美元；
   - sweep 這一次的上限 = 40 − 1 − 總帳的全部花費。
4. **總帳重設的檢查**：只計入 Study 3 的目錄（依 task id 的 v3 前綴判定），Study 2 的 0.117 美元不計入。
5. **Haiku 的 dev run**：只限 dev seed 0–1、R = 3。
6. **關閉 Study 2 的付費入口**：decision 035 已結案，`--suite study2` 的付費 run 一律拒絕。

### 需要回頭問使用者的情況（只有兩種）

1. **有任何題目缺陷需要修正**（同 034）。
2. **dev 之後的成本估算超過上限**：
   - 估算以事先 commit 的工具計算，只用 dev 實測的 turn 數與每次呼叫的成本。
   - c90 = 一條失敗 source 的成本 P90。以登記的 plan 模擬：
     - 失敗曲線的起點固定為 1.0，cliff 與 linear 各半（034 的形狀）；
     - 長度取 dev 各級別平均 turn 中最大的一個，無條件進位為整數，最大為 20（三個級別合成一個值，不依級別加權）；
     - 成功的 rerun 以 L − k 次呼叫計，失敗的 rerun 以 20 − k 次呼叫計；
     - 不使用 dev 的成功率。
   - N_local = floor(80,000 ÷ c90_local)；N_haiku = floor((39 − dev 的實際花費 − eval baseline 預估花費的 P90) ÷ c90_haiku)。
   - 任一個 < 15 時，停下來問使用者。
   - 可選的回應只有兩種：提高該 agent 的上限（Haiku 超過 40 美元只能由使用者決定），或接受較少的曲線。
   - 不得修改題目、級別、規則、seed、R、plan、門檻或 25 條的上限。

**不是回頭問的理由**：dev 的成功率很高，或預期的失敗 source 很少。這時照本方案直接進行，結果照實報告，估算寫進凍結紀錄。其他情況也依本方案直接進行。

### 時間與執行

- **工程**：研究登記表、generator v3、sweep 執行方式，以及 dev、估算、分析、報告、凍結工具，加上各輪審查，約需 3–5 個 session。
- **本機**：
  - dev 約 0.5–1 小時；
  - eval baseline 約 2–3 小時，分 4 段；
  - sweep 約 30–45 GPU 小時，分 24–27 段；
  - 依 GPU lock 的占用情況，約 3–7 天。
- **Haiku**：
  - dev、eval baseline 與 sweep 都由使用者在自己的終端機執行。credential 由使用者設定，程式不讀取、不保存。
  - 每一段的指令由 Claude 事先寫好。
  - sweep 最多約 55,000–75,000 次循序呼叫，以每次 1.5–3 秒計約 25–60 小時。每次 run 以 `--max-cost-usd 5` 分段，以 `--resume-from` 接續。
  - 若 Haiku 的失敗 source 很少，時間會短很多。
- 兩個 agent 的 sweep 可以同時進行。全部完成預計約 1–2 週。

## 8. 實作順序（核准後；全部先寫測試；每一步做對抗式審查）

1. **研究登記表、fail-closed 的開關與 Study 3 的付費機制**（這一步不含題目）。
   - 把 8 個模組中 28 個 `suite == "study2"` 的分支，改成由登記表判定。登記的內容包括：開關、baseline 與 sweep 的呼叫上限、付費上限與總帳、source 規則、sweep 順序。
   - 沒有登記的 suite，一律拒絕模型推論。
   - 把 study3 加進 SUITES 的同一個 commit，必須同時包含：
     - STUDY3_APPROVED / STUDY3_FROZEN 兩道開關；
     - baseline 上限：dev 2,880、eval 每段 2,880；
     - sweep 上限：每段 3,000、累計 80,000；
     - 規則 B 的 source 選擇。
   - 測試確認：
     - 開關關閉時，本機與付費（包含 Python API）都在建立 client 之前拒絕；
     - study3 的 source 是 index 最小的失敗 rollout；
     - 付費的 study3 run 一定在 Study 3 的 paid_session 內執行；
     - Study 2 的付費入口已關閉。
   - 新增依 seed 區塊執行 baseline 的選項；baseline 目錄由程式列舉、核對（§6）。
   - 新增 Study 3 的 sweep 執行方式（§6；本機與付費都適用。現有的 `pool` / `failed_first` 順序，以及每次 rerun 前的預算檢查，都不符合 §6）：
     - 依凍結紀錄中 96 題的排列，只執行該 agent 失敗的 source；source 為 None 的題略過，不計入條數；
     - 完成 25 條 sweep 就停止；
     - 開始每一條新的 source 之前，檢查剩下的累計上限是否 ≥ 凍結紀錄中的預留額；不足時在 source 與 source 之間停止，並在 summary 記錄停止原因；
     - 已開始、以 `--resume-from` 跨段接續的 source 不再做這項檢查，只受每段上限與累計上限約束；
     - 預留額只從凍結紀錄讀取；沒有凍結紀錄時，拒絕執行 eval 的 sweep。
     - 測試確認：已執行的 source 恰好是排列的前綴，而且跨段不變；只包含失敗的 source；完成第 25 條後停止；預留額不足時在 source 之間停止；接續的 source 不受預留額檢查影響。
2. **generator v3**：
   - 依附錄 B 實作 4 題型 × H1–H3；
   - A 類檢查，包含 turn 與 context；
   - 附錄 A 的逐字稿與 B 類審查；
   - 完成後 commit golden hash。
3. **dev 與估算工具**：
   - dev 審查工具，支援兩個 agent，包含 comment 字串與評分寬鬆的稽核；
   - 估算工具：只限失敗 source、固定順序、上限 25、美元換算；成功的 rerun 以 L − k 計，失敗的 rerun 以 20 − k 計；預留額以 L = 20 計算；
   - 偵測力工具：RQ1 在 L = 16–20 的偵測力，以及 §2 RQ2 定義的分級檢定偵測力模擬（兩個檢定同時成立）。
4. **分析與報告的一般化**（第 6 節）：以 dry-run 與合成資料測試；Study 2 與 035 的輸出逐 byte 不變。
5. **dev 到開關**：
   - dev：本機在 GPU lock 內執行，Haiku 由使用者執行；
   - dev 審查 → 估算 → 凍結紀錄 → 打開 eval 的開關。
6. **eval baseline**：兩個 agent 可以同時進行。
7. **sweep**。
8. **收尾**：分析、報告、README、NEXT。

核准時，同步更新 NEXT.md 與 README 中「不會有 Study 3」的敘述。

## 9. 凍結紀錄

- **時機**：在任何一個 agent 的第一次 eval 推論之前 commit，兩個 agent 共用。測試每次都比對紀錄與程式。
- **題目與評分**：
  - generator 版本、golden hash、RNG 指紋；
  - eval 96 題的 task hash 與 format、grader 版本；
  - 附錄 A 逐字稿的 hash。
- **agent 與付費**：兩個 agent 的 config_id；兩份總帳的路徑。
- **RQ1 與 sweep 的規則**：
  - R = 3、plan (2, 10, 48, 0.3)、bonferroni95、PRIMARY_RULE = excluding；
  - source 規則 B 與「失敗的 source」的定義、可判定門檻 0.4；
  - sweep 的完整順序（96 題的排列）、每個 agent 上限 25 條、預算的預留額。
- **統計**：
  - bootstrap B = 10,000、seed 0、每級 32 題；
  - 置換檢定的定義；
  - 行為性 cliff 比例的公式、機械性判定的範圍；
  - 四個敏感度分析的定義：034 的 max_turns 敏感度分析、agent_error 的 RQ1 敏感度分析、RQ2 排除 agent_error 之後的成功率、因上限中途停止的敏感度範圍；
  - 分級檢定偵測力模擬的參數、reps 與 seed；
  - 完整性與天花板的定義（含提前結束的處理），以及 baseline 目錄的列入規則（由程式列舉並核對）。
- **上限**：各項上限，並註明不得提高。
- **dev 的結果與估算**：dev 摘要、估算、預期的 n 與偵測力。
- **程式**：產生分析與報告的程式 commit（code_commit）。這些程式事先以 dev 資料與合成資料測過。

## 10. 只有一輪

- 正式研究只有一輪。結果不論是驟降、逐步、無法判斷或天花板，都照實報告。
- 之後要不要再做研究，由使用者另外決定。

## 為什麼

- **題目在任何模型結果之前固定**：題目只從結構加難；dev 不用來調整難度，避免在 dev 上挑題。
- **預算集中在主要問題**：只 sweep 失敗的 source，配合固定的隨機順序、在 source 之間檢查預算與上限。這樣中途停止也不會造成選擇偏誤。
- **兩個 agent 用同一組題目與協定**：各自回答 RQ1，再並列比較，並說明兩者失敗組成的差異。
- **事先揭露限制**：題目參考了 Study 2 的失敗類型、有機械性 cliff、turn 與 context 的壓力。對應的次要結果與敏感度分析都事先登記，結果出來後不能再選擇怎麼解讀。

## 放棄的替代方案

- **依 dev 成功率校準難度**：校準規則很難完全機械化，等於在 dev 上挑題。改為事先固定。
- **eval 只用 3 個 seed**：Haiku 若只在 H3 失敗，最多只有 12 條失敗曲線，必然少於目標。
- **sweep 所有 source**：成功的 source 在 Study 2 幾乎都是平線，會花掉大部分預算。
- **提高 max_turns**：會改變凍結的 agent，與 Study 2 不能比較。
- **在 instruction 加上 search 的涵蓋說明句**：會改變題型的風格，而且只對部分題型成立（§3）。
- **dev 中以 max_turns 結束的比例 ≥ 0.25 時回頭問使用者**（可行性審查的提議）：
  - 這會讓題目在看到 dev 結果之後被修改，等於依 dev 調整。
  - 改為事先把 oracle 壓到 ≤ 13，並登記已知限制。
- **context 檢查用「oracle + 3 次寬搜尋 ≤ 0.8 × 上限」**：每次寬搜尋約 8 KB，這個條件會直接排除截斷機制。改用 §3 的兩個條件，並以敏感度分析處理 context 用盡。
- **主要規則也排除 agent_error**：
  - 不採用，維持 034 的定義，以便與 Study 2 並列；敏感度分析照樣報告。
  - 若使用者要把它改成主要規則，必須在凍結之前決定。
- **sweep 依呼叫次數或金額在 source 中途停止**（第 1 版的寫法）：被截斷的 source 會偏向花費大的 cliff 曲線，造成選擇偏誤。改為在 source 之間檢查預留額。
- **新增「不得包含」的 predicate**：會改變 task format。改為記為已知限制，並在 dev 中稽核。
- **verified-close 把 V7 移到 H1、把 qa-confirmed 換成 plain-broken**：會讓 H1 比「L3 加一個機制」難得多。維持原設計，限制寫明。
- **scoped-triage 的 H3 世界擴充到約 110 個 issue**：寬搜尋最多只回 50 筆，世界大小不影響 context。真實感的限制改為照實寫明。

## 實作紀錄

- **2026-10-23，§8 第 1 步的 A 部分**（研究登記表與付費機制，不含題目）：
  - `tasks.suites.STUDIES`：每個 suite 的開關、呼叫上限、source 規則與付費規格；run 工具（run_baseline、run_sweep、
    analyze_sweep 的 allowed_orders）不再寫死 `suite == "study2"`。沒有登記的 suite 一律拒絕；有開關的研究必須登記全部上限，
    否則無法建立（`StudySpec`）。Study 2 的分析與報告模組（study2_analysis、study2_dev、study2_freeze、study2_report）本來就
    只接受 study2，留到 §8 第 4 步一般化。
  - 付費：`PaidSpec`；`experiments.billing` 依規格使用各自的總帳、總預算與 phase 上限（新增 dev phase）；總帳重設的核對只計入
    同一個研究的 manifest 列（generator_version）。`STUDY3_PAID`：`~/.agent-gym/study3-haiku-billing.jsonl`、40 美元、
    dev ≤ 3、eval baseline ≤ 8、只限 Haiku、R = 3。
  - Study 2 的付費入口關閉（`STUDY2_PAID_CLOSED = True`）；機制的測試在測試檔內重新打開後照舊執行。
- **實作順序的說明（不是設計變更）**：§8 第 1 步規定「把 study3 加進 SUITES 的同一個 commit 必須同時包含開關、上限與規則 B」。
  study3 要有 generator v3 才能加進 SUITES，所以第 1 步分成兩部分：A 部分（上述）先完成；B 部分（study3 的登記、依 seed 區塊
  執行 baseline、Study 3 的 sweep 執行方式）在第 2 步的 generator v3 完成後，與 study3 的登記放在同一個 commit。順序以外的內容不變。

- **2026-10-23，§8 第 2 步**：
  - generator v3 的核心（`tasks/generator_v3.py`）與四個題型（`tasks/gen3/`）：每個題型先寫測試，經一輪對抗式審查與修正。
    整個 pool（dev 24 題、eval 96 題）通過自我檢查、級別與跨研究的檢查，以及 §3 的 context 檢查。
  - 規則來源的補充：E2 與 S1 也標為 Study 2 的失敗類型（附錄 A.1）。
  - 附錄 A 完成；接著做 B 類審查，修正之後才 commit golden hash。
  - 同時完成的工具：context 檢查（`experiments/study3_context.py`，本機 agent 的回覆文字以 Study 2 dev 的 P95 886 bytes 計）、
    統計工具（`experiments/study3_stats.py`：精確置換檢定、sweep 成本與預留額、dev 之後的估算、偵測力）。
  - 估算規則的提醒（不是設計變更）：以單條成本的 P90 × 每次呼叫成本的 P90 計算是保守的；Haiku 每次呼叫的 P90 若約 0.0006 美元，
    估計能 sweep 的失敗 source 會少於 15 條，依 §7 需要回頭問使用者。

- **2026-10-23，§8 第 2 步完成**：B 類審查（附錄 A.6）之後，generator v3 的輸出以 golden hash 釘住：
  `GOLDEN_POOL_SHA256 = 9d97d04d…e43f`、`GOLDEN_RNG_SHA256 = a6ad2ed1…037b`（dev 24 題 + eval 96 題；1 題重抽過）。
  之後任何改變輸出的修正都記為「重抽」。
- **2026-10-23，§8 第 1 步 B 部分**（generator v3 完成之後，與 study3 的登記同一批）：
  - `tasks.suites`：study3 登記（`STUDY3_APPROVED` / `STUDY3_FROZEN` 都是 False；baseline 上限 dev 2,880、eval 每段 2,880；
    sweep 每段 3,000、累計 80,000；規則 B；`STUDY3_PAID`；eval 的 4 個 seed 區塊；本機的 sweep 順序 frozen_permutation）。
  - 依 seed 區塊的 baseline：`run_split(..., seeds=(2, 3))` / CLI `--seeds 2,3`；run.json 另記 `seeds`；本機的 eval baseline
    必須逐段執行；`load_runs` 依每個目錄的區塊核對，合併時涵蓋全部 96 題。
  - Study 3 的 sweep 執行方式（`experiments/study3_plan.py`、`run_sweeps(order="frozen_permutation")`）：凍結紀錄中的排列、
    只跑失敗的 source、完成 25 條就停止（跨段累計）、開始新的 source 之前剩下的累計上限必須 ≥ 預留額、接續的 source 不再檢查、
    沒有凍結紀錄就拒絕；`analyze_sweep` 依同一個順序核對完成的曲線是前綴。

- **2026-10-23，§8 第 3 步完成**：
  - dev 審查與估算（`experiments/study3_dev.py`，`23ef189`）：兩個 agent；§5 C 的 comment 稽核；成功 episode 與 oracle 最終狀態的
    機械比對；各級別的成功率、turn 的分布、max_turns 比例、agent_error、context 用盡與最大的 request；§7 的估算
    （`study3_stats.estimate`）。
  - 偵測力：分級檢定的模擬（`study3_stats.rq2_power`）與 RQ1 在 L = 16–20 的偵測力（`study2_dev.max_turns_power(lengths=…)`；
    Study 2 的預設長度不變）。
- **2026-10-23，§8 第 4 步完成**：`experiments/study3_analysis.py`（一次一個 agent）與 `experiments/study3_report.py`（結果頁、
  曲線圖、README 的 study3 / study3-haiku / study3-compare 段落）；Study 2 / 035 的輸出逐 byte 不變（測試）。經一輪對抗式審查
  （12 項成立），修正記在下一筆。另外補上 **eval 的 comment 稽核**（§5 C「dev 與 eval 都一樣」）：採計的 eval baseline episode
  以 `study3_dev.comment_audit` 稽核，列在分析的 `baseline.comment_audit` 與報告的「comment 稽核」一節（條數、成功 episode 上的
  條數、依類別的條數、逐則清單）；只列出，不改變評分。讀法：sweep 的 rerun 不稽核（§5 C 說的是 dev 與 eval 的 episode；rerun 是
  曲線的估計，不是 episode 的評分）。
- **2026-10-23，§8 第 4 步（分析與報告：`study3_analysis`、`study3_report`）審查之後的修正**。在任何 eval 推論之前（沒有任何
  Study 3 結果）；不是設計變更，只把實作對齊本 decision 的字面，並記錄讀法：
  - **Haiku 剩下的累計上限**一律以登記的 Study 3 總帳（`billing.journal_path(STUDY3_PAID)`，也是 run_sweep 停止時用的值）計算：
    40 − 1 − 總帳的全部花費（§6）。總帳不存在、無法讀取，或 runs/ 中這個 agent 的 Study 3 manifest 合計大於總帳（總帳被重設）
    → 拒絕；manifest 合計只核對與列出。不能另給總帳的路徑（原本的 `--haiku-journal` 移除：預設用 manifest、另一個來源只是選項，
    等於看過結果之後才能選擇 sweep 的完整性狀態）。dry-run 沒有總帳，以 dry-run 目錄的 manifest 合計。
  - **runs 根目錄**：正式分析與正式報告一律列舉 repo 的 `runs/`（`study3_analysis.RUNS_ROOT`，不依賴工作目錄；從磁碟讀回的摘要
    也核對），run_sweep 的 baseline 目錄核對也以它為準；其他根目錄（某一段、部分副本）只供 dry-run。
  - **「qa 角色相關」那一行**的角色由 generator 種入的 comment 決定：qa-broken、qa-confirmed，以及 H1 / H3 的 qa-confirmed 干擾項
    （open-qa-confirmed、unlabeled-qa-confirmed）。§3 已寫明 qa-confirmed 是「H1 的候選，H1 / H3 的干擾項」，附錄 A.6 把另列的
    那一行擴大到 qa-confirmed；原本只比對完全相同的角色名稱，漏掉這兩個干擾項。
  - **查找與去重**（附錄 A.5 / A.6）：分類依整條 source 的最終失敗。A.5 沒有指定 cliff 那一步，而遺漏的候選不會在 turn k 有變更，
    所以不以 cliff 那一步分類。定義：missed = 最終失敗、而且 source 從沒有變更過的候選；只看第一頁 = missed 含 `params.hidden`；
    只做一次 label search = missed 不是空的，而且全部只帶同一個 label（`params.target_labels`）；去重 = 最終失敗的候選中有兩個 label
    都帶的（qa-broken）。另列 k / n（算法同另列的那幾行），不從角色表與 V8 的計數中移除；三者不互斥（例如隱藏的候選只帶一個
    label 時，兩個旗標都成立）。去重的失誤在 cliff 那一步時也計入 qa 那一行（上界，A.6）。
  - **§2 次要結果第 4 點**：第一輪的 max_turns 規則在主要判定的同一個完成前綴上輸出 k / n、Wilson CI 與標籤（同 034 的次要規則）；
    034 的 max_turns 敏感度只報 k / n 與 CI（同 034 的 `_sensitivity_rq1`，沒有標籤）。
  - **RQ2 排除 agent_error**（§3「只報告」）：只報各級的成功率與描述用的差，不做檢定，也不判定有效與否。
  - **最大的 request**（§3 的報告項目，dev 與 eval）：由採計 episode 的 trajectory 依 adapter 的 request 形狀重建
    （`study3_context.request_bytes`；本機用 turn.meta 記錄的 seed）。被 fit check 拒絕送出的 agent_error 沒有 trajectory，不在其中，
    已計入 context 用盡（報告中註明）。dev 審查（`study3_dev`）同樣補上。
  - **run_sweep 的 Study 3 付費入口**：天花板依 §6（沒有失敗的 source，而且 baseline 完整；agent_error 不是 source），不再呼叫只接受
    Study 2 的分析（原本一律以 "these are not Study 2 baseline runs" 拒絕）。Study 2 的路徑不變。
  - **歧義（依字面或較保守的讀法實作，不依任何結果選擇）**：
    1. 提前結束時，次要結果與敏感度（行為性 cliff 比例、第一輪的規則、agent_error 的 RQ1 敏感度）是否也不輸出三分法的標籤：
       §6 寫「不論已完成的曲線有多少，都不輸出三分法的標籤；只報告已完成前綴的 k / n 與 Wilson 95% CI，以及敏感度範圍」。這些是
       「RQ1 的次要結果」，採「都不輸出、k / n 與 CI 照報」；另一種讀法是只有主要判定不輸出。天花板與其他資料不完整的情況同樣不輸出
       （沒有曲線，n = 0）。因上限中途停止的範圍照 §6 輸出標籤。
    2. 天花板是否要求 response model 只有一種：034 的天花板含「not blocked」，036 §6 的字面只有「沒有失敗的 source，而且 baseline
       完整」。依 036 的字面：baseline 的 pooling_blocked 不影響天花板（天花板沒有任何合併的估計）。
    3. response model 不只一種（sweep 的 blocked）時：036 沒有直接寫次要結果與敏感度怎麼處理；依 034 的 `analyze_sweep_dirs`
       （blocked → RQ2、by_rule、sensitivity 都是 None）與 decision 009 §4，RQ1 的兩個範圍、次要結果與 RQ1 的敏感度都不合併估計。

- **2026-10-23，凍結紀錄的工具（§9，`experiments/study3_freeze.py`）**：在任何 eval 推論之前；不是設計變更，記錄讀法：
  1. 內容依 §9；`check_record` 由測試每次比對紀錄與程式（題目、agent、付費、協定、統計、排列、預留額，以及 §9 其他項目都存在），
     `STUDY3_FROZEN` 只在紀錄存在時為 True（測試）。
  2. 除了「附錄 A 逐字稿的 hash」，也記錄本文（「背景」到「放棄的替代方案」）與附錄 B 的 hash：§6 的完整性與天花板等定義在凍結後
     不能改寫。實作紀錄不在 hash 內，之後可以繼續寫。
  3. Haiku dev 的實際花費（§7 的 N_haiku）一律讀登記的 Study 3 總帳，不能另給路徑；總帳不存在、有無法計價的呼叫、凍結之前已有
     eval 的花費，或 manifest 合計大於總帳（總帳被重設）→ 拒絕（同第 4 步的修正）。
  4. 「預期的 n 與上述機率」（§2，只用於報告）依 §6 的停止規則模擬：每題有失敗 source 的機率由 dev 每題的失敗比例計
     （agent_error 沒有 trajectory，不是 source，只計入分母）；目標 min(25, N_failed)；開始新的 source 之前剩下的 ≥ 預留額；
     單條成本取自登記的 plan 的成本模擬；可判定的條數 ~ Binomial(完成條數, 4/5)。不用 §7 的 N：它以單條成本的 P90 計，是回頭問
     使用者的保守門檻，不是期望值。Haiku 的成本以 dev 每次呼叫成本的平均換算（預留額仍依 §6 用 P90）。
  5. RQ1 在 L = 16–20 的偵測力（`max_turns_power`，reps 200、seed 0）也寫進紀錄（§9「預期的 n 與偵測力」）。
  6. §7 的兩種回應中，「接受較少的曲線」以 `--user-accepts-fewer-curves`（日期與使用者的話）記進紀錄；「提高上限」是程式與 decision
     log 的變更，之後重算的估算不再要求問。估算要求問、紀錄中卻沒有使用者對那些 agent 的決定 → `check_record` 不通過；
     沒有要求問卻有決定 → 也不通過。
  7. `code_commit`：工作區必須乾淨，沒有追蹤的檔案也算（忽略的 runs/ 等除外）。
  - 經一輪對抗式審查（22 項，9 項成立，都已修正；另採用 6 項成本低的強化）。修正以變異測試確認測試能抓到。

## 附錄 A — instruction 逐字稿、規則來源與 decisive_rules

（2026-10-23，§8 第 2 步。逐字稿由 dev seed 0–1 與除錯用的 seed ≥ 100 的 instance 機械地產生：issue id 換成角色的佔位符；
每個題型、每個級別在這些 seed 上都只有一種模板。eval instance 沒有被讀取。）

### A.1 規則來源

- **study2-failure**：E2、E6、S1、S6、Y4、Y6、V6。
  - E2、S1、Y4 是 034「事後的描述性觀察」記錄 Study 2 失敗的規則代號：把「加了 comment」當成「已處理」而關閉 epic（E2）、
    把沒有 backend 的 issue 判為有權限（S1）、在已處理的項目上留 comment 或做沒有變更的 update（Y4）。
  - E6、S6、Y6、V6 是刻意包含的同類判斷（背景）。背景原本只列這四個與 Y4；E2 與 S1 在這裡補上：Study 3 中同一類失敗也會以這兩個代號出現。
  - sync-repair L1 的 scope 失敗（還原沒有列出的欄位）屬於 scope，不是規則代號，不標。
- **new**：其他所有規則代號。
- 報告依這個標記分列 cliff 的 k / n（§2，只做描述）。

### A.2 instruction 逐字稿

佔位符：`#<E>`、`#<C>`（epic-closeout 的兩個 tracker）、`#<R>`、`#<R1>`、`#<R2>`（sync-repair 的報告）、`<assignee name> (u<A>)`
（scoped-triage 的指派對象）、`u<X>`（verified-close 的範例 user id，不是任何候選的 reporter）。scoped-triage 與 verified-close 的
instruction 不列任何 issue id。

#### epic-closeout

**H1**

```text
Issue #<E> tracks a release. The "Sub-issues:" line in the body of #<E> lists its sub-issues; only those issues are sub-issues. Wrap up the release:
1. Close every sub-issue whose status is resolved (set its status to closed). A resolved sub-issue that has the needs-info label is not finished: do not change its status (do not close or reopen it), and treat it like an open sub-issue under rules 2 and 3. Sub-issues that are already closed need no action.
2. Do not close sub-issues that are open or in_progress. Instead, add exactly one comment to each of them that includes the text "Blocks #<E>".
3. Close #<E> only after all of its sub-issues that are not dropped (rule 4) are closed. If any of them cannot be closed, set #<E> to in_progress instead of closing it.
4. A sub-issue that has the wontfix label is dropped from its tracker, #<E>, whatever its status (open, in_progress or resolved). Rules 1 and 2 do not apply to it: do not change its status, and add exactly one comment to it that includes the text "Dropped from #<E>".
Do not change anything else, and do not add any other comments.
```

**H2**

```text
Issue #<E> tracks a release. The "Sub-issues:" line in the body of #<E> lists the sub-issues of #<E>. One of them, #<C>, is itself a tracker: the "Sub-issues:" line in the body of #<C> lists the sub-issues of #<C>. Only the issues on a tracker's own "Sub-issues:" line are its sub-issues; the sub-issues of #<C> are not sub-issues of #<E>. Below, "sub-issue" means a sub-issue of either tracker, and "its tracker" is the tracker whose line lists it. Wrap up the release:
1. Close every sub-issue whose status is resolved (set its status to closed). A resolved sub-issue that has the needs-info label is not finished: do not change its status (do not close or reopen it), and treat it like an open sub-issue under rules 2 and 3. Sub-issues that are already closed need no action.
2. Do not close sub-issues that are open or in_progress. Instead, add exactly one comment to each of them that includes the text "Blocks #<C>" if its tracker is #<C>, or "Blocks #<E>" if its tracker is #<E>.
3. Close a tracker only after all of its sub-issues that are not dropped (rule 4) are closed. If any of them cannot be closed, set the tracker to in_progress instead of closing it.
4. A sub-issue that has the wontfix label is dropped from its tracker, whatever its status (open, in_progress or resolved). Rules 1 and 2 do not apply to it: do not change its status, and add exactly one comment to it that includes the text "Dropped from #<C>" if its tracker is #<C>, or "Dropped from #<E>" if its tracker is #<E>.
5. Rules 1 and 2 do not apply to #<C>. Use rule 3 with #<C>'s own sub-issues: if rule 3 allows #<C> to be closed, close it even though its status is open; otherwise set it to in_progress and also add exactly one comment to #<C> that includes the text "Blocks #<E>". Under rule 3 for #<E>, #<C> counts as closed only if you closed it.
Do not change anything else, and do not add any other comments.
```

**H3**

```text
Issue #<E> tracks a release. The "Sub-issues:" line in the body of #<E> lists the sub-issues of #<E>. One of them, #<C>, is itself a tracker: the "Sub-issues:" line in the body of #<C> lists the sub-issues of #<C>. Only the issues on a tracker's own "Sub-issues:" line are its sub-issues; the sub-issues of #<C> are not sub-issues of #<E>. Below, "sub-issue" means a sub-issue of either tracker, and "its tracker" is the tracker whose line lists it. Wrap up the release:
1. Close every sub-issue whose status is resolved (set its status to closed). A resolved sub-issue that has the needs-info label is not finished: do not change its status (do not close or reopen it), and treat it like an open sub-issue under rules 2 and 3. Sub-issues that are already closed need no action.
2. Do not close sub-issues that are open or in_progress. Instead, add exactly one comment to each of them that includes the text "Blocks #<C>" if its tracker is #<C>, or "Blocks #<E>" if its tracker is #<E>.
3. Close a tracker only after all of its sub-issues that are not dropped (rule 4) are closed. If any of them cannot be closed, set the tracker to in_progress instead of closing it.
4. A sub-issue that has the wontfix label is dropped from its tracker, whatever its status (open, in_progress or resolved). Rules 1 and 2 do not apply to it: do not change its status, and add exactly one comment to it that includes the text "Dropped from #<C>" if its tracker is #<C>, or "Dropped from #<E>" if its tracker is #<E>.
5. Rules 1 and 2 do not apply to #<C>. Use rule 3 with #<C>'s own sub-issues: if rule 3 allows #<C> to be closed, close it even though its status is open; otherwise set it to in_progress and also add exactly one comment to #<C> that includes the text "Blocks #<E>". Under rule 3 for #<E>, #<C> counts as closed only if you closed it.
6. Exception to rule 4: a wontfix sub-issue whose priority is p0 cannot be dropped. Rule 4 does not apply to it; handle it with rules 1, 2 and 3 like any other sub-issue.
Do not change anything else, and do not add any other comments.
```

#### scoped-triage

**H1**

```text
Triage Bot's permissions:
1. You may change the fields of an issue only if it has the backend label.
2. You may change the fields of an issue only if it does not have the security label.
3. You may change the fields of an issue only if its priority is not p0.
Judge permissions by each issue's labels and priority before you make any change. You may add comments to any issue.
Every issue whose status is open, that has no assignee, and that has the bug label needs triage. For each one:
4. If you are permitted to change it: assign it to <assignee name> (u<A>) and set its priority to p2.
5. If you are not permitted to change it: do not change any field; add exactly one comment that includes the text "Needs owner triage".
6. If it has the needs-info label: add exactly one comment that includes the text "Needs reporter info". If you are not permitted to change it, add a single comment that includes both "Needs owner triage" and "Needs reporter info" instead of two separate comments. If you are permitted to change it, set its priority to p3 instead of p2, and do not assign it.
Do not change anything else, and do not add any other comments.
```

**H2**

```text
Triage Bot's permissions:
1. You may change the fields of an issue only if it has the backend label.
2. You may change the fields of an issue only if it does not have the security label.
3. You may change the fields of an issue only if its priority is not p0.
4. Exception to the security-label condition: an issue that has both the security label and the docs label is not restricted by it. The backend-label condition and the p0 condition still apply to such an issue.
Judge permissions by each issue's labels and priority before you make any change. You may add comments to any issue.
Every issue whose status is open, that has no assignee, and that has the bug label needs triage. For each one:
5. If you are permitted to change it: assign it to <assignee name> (u<A>) and set its priority to p2.
6. If you are not permitted to change it: do not change any field; add exactly one comment that includes the text "Needs owner triage".
7. If it has the needs-info label: add exactly one comment that includes the text "Needs reporter info". If you are not permitted to change it, add a single comment that includes both "Needs owner triage" and "Needs reporter info" instead of two separate comments. If you are permitted to change it, set its priority to p3 instead of p2, and do not assign it.
Do not change anything else, and do not add any other comments.
```

**H3**

```text
Triage Bot's permissions:
1. You may change the fields of an issue only if it has the backend label.
2. You may change the fields of an issue only if it does not have the security label.
3. You may change the fields of an issue only if its priority is not p0.
4. Exception to the security-label condition: an issue that has both the security label and the docs label is not restricted by it. The backend-label condition and the p0 condition still apply to such an issue.
Judge permissions by each issue's labels and priority before you make any change. You may add comments to any issue.
An issue needs triage if all three of these hold: its status is open, it has no assignee, and it has the bug label, the performance label, or both.
5. Exception: an issue that has the duplicate label does not need triage, even if all three hold. Leave it unchanged, and do not comment on it.
For each issue that needs triage:
6. If you are permitted to change it: assign it to <assignee name> (u<A>) and set its priority to p2.
7. If you are not permitted to change it: do not change any field; add exactly one comment that includes the text "Needs owner triage".
8. If it has the needs-info label: add exactly one comment that includes the text "Needs reporter info". If you are not permitted to change it, add a single comment that includes both "Needs owner triage" and "Needs reporter info" instead of two separate comments. If you are permitted to change it, set its priority to p3 instead of p2, and do not assign it.
9. If you are permitted to change it and it has the performance label: set its priority to p1 instead of p2, and assign it as for any other issue you are permitted to change. The needs-info rule takes precedence: if it also has the needs-info label, follow the needs-info rule instead of this one.
Do not change anything else, and do not add any other comments.
```

#### sync-repair

**H1**

```text
Issue #<R> is a report from the sync job (Sync Job, sync). Every change it lists was made by mistake. Judge everything by the values the issues have at the start of the task, before you make any change.
1. Undo each listed change by restoring the value from before the change. For a label change, only add back or remove that one label and keep all other labels as they are.
2. If the current value of the field that a listed change set is no longer the value the sync job set, that change is already handled: do not undo it. A label change is handled if the label the sync job added is gone, or the label it removed is back. If every listed change on an issue is handled, leave that issue unchanged and do not comment on it.
3. The report can list several changes to the same issue. Judge each of them on its own under rules 1 and 2: undo every one that is not handled, even if another change on that issue is handled. You may undo changes to different fields of one issue in a single update.
4. Add exactly one comment that includes the text "Reverted sync change" to each issue on which you undo at least one change, even if you undo several changes on it.
Do not change anything else, and do not add any other comments.
```

**H2**

```text
Issue #<R> is a report from the sync job (Sync Job, sync). Every change it lists was made by mistake. Judge everything by the values the issues have at the start of the task, before you make any change.
1. Undo each listed change by restoring the value from before the change. For a label change, only add back or remove that one label and keep all other labels as they are.
2. If the current value of the field that a listed change set is no longer the value the sync job set, that change is already handled: do not undo it. A label change is handled if the label the sync job added is gone, or the label it removed is back. If every listed change on an issue is handled, leave that issue unchanged and do not comment on it.
3. The report can list several changes to the same issue. Judge each of them on its own under rules 1 and 2: undo every one that is not handled, even if another change on that issue is handled. You may undo changes to different fields of one issue in a single update.
4. Add exactly one comment that includes the text "Reverted sync change" to each issue on which you undo at least one change, even if you undo several changes on it.
5. You must not edit issues that are closed at the start of the task. An issue whose status is closed at the start of the task is locked: do not change any of its fields, even to undo a listed change. If a locked issue has at least one listed change that is not handled, add exactly one comment to it that includes the text "Sync revert skipped".
6. Exception to rule 5: if the report lists a change that set an issue's status to closed, the sync job closed that issue by mistake, so it is not locked: undo its listed changes under the other rules (this sets its status back to the value it had before the sync job closed it).
Do not change anything else, and do not add any other comments.
```

**H3**

```text
Issues #<R1> and #<R2> are reports from the sync job (Sync Job, sync); #<R2> is from a later sync run than #<R1>. Every change they list was made by mistake. Judge everything by the values the issues have at the start of the task, before you make any change.
1. Undo each listed change by restoring the value from before the change. For a label change, only add back or remove that one label and keep all other labels as they are.
2. If the current value of the field that a listed change set is no longer the value the sync job set, that change is already handled: do not undo it. A label change is handled if the label the sync job added is gone, or the label it removed is back. If every listed change on an issue is handled, leave that issue unchanged and do not comment on it.
3. The reports can list several changes to the same issue, in one report or across both. Judge each of them on its own under rules 1 and 2: undo every one that is not handled, even if another change on that issue is handled. You may undo changes to different fields of one issue in a single update.
4. Add exactly one comment that includes the text "Reverted sync change" to each issue on which you undo at least one change, even if you undo several changes on it.
5. You must not edit issues that are closed at the start of the task. An issue whose status is closed at the start of the task is locked: do not change any of its fields, even to undo a listed change. If a locked issue has at least one listed change that is not handled, add exactly one comment to it that includes the text "Sync revert skipped".
6. Exception to rule 5: if a report lists a change that set an issue's status to closed, the sync job closed that issue by mistake, so it is not locked: undo its listed changes under the other rules (this sets its status back to the value it had before the sync job closed it).
7. If both reports list a change to the priority of the same issue, the sync job changed that field twice: first the change in #<R1>, then the one in #<R2>. Treat the two as one change (this replaces rules 1–3 for that field): it is handled only if the current value differs from the value set in #<R2>, and undoing it restores the value from before the change in #<R1>.
Do not change anything else, and do not add any other comments.
```

#### verified-close

**H1**

```text
Closing an issue sends the customer a final notification that cannot be undone. Process every issue whose status is resolved. Read each one's comments (comments are listed oldest first) and decide what to do from the comments it had before you made any change; apply the first rule below that matches:
1. If the reporter's most recent comment includes the word "Confirmed": close the issue (set its status to closed).
2. If the reporter's most recent comment includes the text "Still seeing": the fix did not work. Set its status to open and add exactly one comment that includes the text "Reopened" and the reporter's user id (for example "Reopened, u<X>" when the reporter is u<X>).
3. Otherwise: do not change its status; add exactly one comment that includes the text "Please confirm the fix" and the reporter's user id (for example "Please confirm the fix, u<X>" when the reporter is u<X>).
Rules 1 and 2 count only comments posted by the issue's reporter; comments by anyone else do not count. Do not close any other issue. Do not change anything else, and do not add any other comments.
```

**H2**

```text
Closing an issue sends the customer a final notification that cannot be undone. Process every issue whose status is resolved. Read each one's comments (comments are listed oldest first) and decide what to do from the comments it had before you made any change; apply the first rule below that matches:
1. If the reporter's most recent comment includes the word "Confirmed": close the issue (set its status to closed).
2. If the most recent comment on the issue, by anyone, includes the text "Please confirm the fix": the reporter has already been asked. Leave the issue unchanged and do not comment on it.
3. If the reporter's most recent comment includes the text "Still seeing": the fix did not work. Set its status to open and add exactly one comment that includes the text "Reopened" and the reporter's user id (for example "Reopened, u<X>" when the reporter is u<X>).
4. Otherwise: do not change its status; add exactly one comment that includes the text "Please confirm the fix" and the reporter's user id (for example "Please confirm the fix, u<X>" when the reporter is u<X>).
Rules 1 and 3 count only comments posted by the issue's reporter; comments by anyone else do not count. Rule 2 looks at the most recent comment by anyone. Do not close any other issue. Do not change anything else, and do not add any other comments.
```

**H3**

```text
Closing an issue sends the customer a final notification that cannot be undone. Only bug and security fixes need the customer's confirmation: process every issue whose status is resolved and that has the bug label or the security label (or both), and leave every other issue unchanged. Decide which issues to process from their status and labels before you make any change. Read each one's comments (comments are listed oldest first) and decide what to do from the comments it had before you made any change; apply the first rule below that matches:
1. If the reporter's most recent comment includes the word "Confirmed": close the issue (set its status to closed).
2. If the most recent comment on the issue, by anyone, includes the text "Please confirm the fix": the reporter has already been asked. Leave the issue unchanged and do not comment on it.
3. If the reporter's most recent comment includes the text "Still seeing": the fix did not work. Set its status to open, add the needs-info label while keeping all of its existing labels, and add exactly one comment that includes the text "Reopened" and the reporter's user id (for example "Reopened, u<X>" when the reporter is u<X>).
4. Otherwise: do not change its status; add exactly one comment that includes the text "Please confirm the fix" and the reporter's user id (for example "Please confirm the fix, u<X>" when the reporter is u<X>).
Rules 1 and 3 count only comments posted by the issue's reporter; comments by anyone else do not count. Rule 2 looks at the most recent comment by anyone. Do not close any other issue. Do not change anything else, and do not add any other comments.
```

### A.3 規則代號與 instruction 編號

| 題型 | 級別 | 對照（instruction 編號：規則代號） |
|---|---|---|
| epic-closeout | H1–H3 | 1：E1、E4；2：E3；3：E2（#E）、E6（#C）；4：E5；5（H2 起）：E6；6（H3）：E7 |
| scoped-triage | H1 | 1：S1；2：S2；3：S3；4：S4；5：S5；6：S6 |
| scoped-triage | H2 | 1：S1；2：S2；3：S3；4：S7；5：S4；6：S5；7：S6 |
| scoped-triage | H3 | 1：S1；2：S2；3：S3；4：S7；選取句：S8；5：S9；6：S4；7：S5；8：S6；9：S8 |
| sync-repair | H1–H3 | 1：Y1、Y3（label）；2：Y4；3：Y5；4：Y2（多項的 issue 記 Y5）；5：Y6；6：Y7；7（H3）：Y8 |
| verified-close | H1 | 1：V1；2：V5；3：V2；「只算 reporter」：V3；「most recent」：V4 |
| verified-close | H2–H3 | 1：V1；2：V6；3：V5（H3 另含 V7：加 needs-info、保留原 labels）；4：V2；V3、V4 同上；H3 的選取句：V8 |

### A.4 各級別的數字（dev 與除錯用 seed 實測；eval 只經過程式檢查）

| 題型 | 級別 | oracle | n_targets | n_decoys | 超過 19 個 call 的正確解 |
|---|---|---|---|---|---|
| epic-closeout | H1 / H2 / H3 | 9 / 11 / 13 | 7 / 8 / 10（done）、9（blocked） | 4 / 5 / 6 | H3 read-each 21–22 |
| scoped-triage | H1 / H2 / H3 | 9 / 11 / 12 | 6 / 7 / 7 | 4 / 5 / 8 | H3 read-each 加一次 search、或加讀 duplicate：20 |
| sync-repair | H1 / H2 / H3 | 10 / 11 / 12 | 7 / 8 / 9 | 3 / 4 / 4 | 無（read-each 15 / 17 / 18） |
| verified-close | H1 / H2 / H3 | 10 / 12 / 13 | 4 / 5 / 5 | 3 / 3 / 5 | 無 |

- binding：epic-closeout 依 seed 的奇偶（偶數 = done：H1 E1、E2、E5；H2 加 E6；H3 加 E7。奇數 = blocked：該級別的全部規則）。
  E3、E4 只在奇數 seed 綁定；E5 決定 #E 結果、E7 經由 resolved 的 P 測試，只在偶數 seed。其他三個題型每個 seed 都綁定該級別的全部規則。
- context（§3）：120 個 instance 全部通過；餘裕最小的是 H3 的壓力情境（scoped-triage 約 46.5 KB、sync-repair 約 46.3 KB、
  verified-close 約 47.5 KB，上限 57,344）。

### A.5 每個角色的 decisive_rules（params.decisive_rules；歸屬標籤，不是原因）

- **epic-closeout**：#E：E2（done 另含 E5；H2 / H3 含 E6，H3 blocked 只有 E2）；#C：E6（H3 blocked 另含 E7）；X、R：E1；
  R-p0（H3）：E1、E7；U：E3；N：E1、E4；W-open：E3、E5（H3 另含 E7）；W-resolved：E1、E5（H3 另含 E7）；P-open（H3）：E3、E5、E7；
  P-resolved（H3）：E1、E5、E7；干擾項：scope（wontfix 干擾項另含 E5，H3 的 p0 wontfix 干擾項另含 E5、E7）。
- **scoped-triage**：P：S4；P-ni：S6；P-secdocs：S4、S7；P-perf：S4、S8；P-perf-ni：S6、S8；D-nb：S1、S5；D-nb-ni：S1、S5、S6；
  D-sec：S2、S5（H3 另含 S8）；D-p0：S3、S5；D-p0-secdocs：S3、S5、S7；D-perf-nb：S1、S5、S8；duplicate：S9；其他干擾項：scope。
  - D-p0-secdocs 標 S7：這個角色是為了測 S7 的第二句（p0 的條件仍然適用）而種入，把例外擴大到它是 S7 的典型錯誤。
  - H3 的 D-sec 標 S8：它帶 performance，測 S8 的 p1 不覆寫 security 的限制，也是兩次 search 都會出現、需要去重的那一個。
  - 所以 §2 的「S7 相關」那一行是上界：P-secdocs 與 D-p0-secdocs 上單純的記帳失誤也會計入。
- **sync-repair**：A：Y1、Y3、Y5；B：Y4、Y5（未處理的是 label 項時另含 Y3）；C：Y4；D、E：Y1、Y2（label 項另含 Y3）；K：Y6；
  KH：Y4、Y6；Q：Y2、Y8；S：H2 為 Y2、Y7，H3 為 Y3、Y5、Y7；報告與干擾項：scope。
- **verified-close**：
  - H1：confirmed：V1；retracted：V4、V5；qa-confirmed、qa-broken：V2、V3。
  - H2：reconfirmed-after-ask：V1、V4、V6；retracted：V4、V5；already-asked：V6；asked-then-broken：V5、V6；qa-broken：V2、V3。
  - H3：同 H2，每個候選另含 V8，retracted 與 asked-then-broken 另含 V7；沒有 label 的干擾項：V8；其他干擾項：scope。
  - H3 每個候選都記 V8，所以分析以 `params.hidden` 與 `params.target_labels` 把「只看第一頁」「只做一次 label search」與規則判斷錯誤分開。

### A.6 實作中的決定與已知限制（golden hash 之前寫定）

- **epic-closeout**：只錯在順序、最終狀態正確的 trajectory 不能當 negative（自我檢查以評分拒絕）；兩個 tracker 的 forbid（含
  `unless`）由 "Sub-issues:" 行重算並完整比對；wrong-tracker negative 必須只因那則 comment 而失敗。背景 issue 仍可能帶 wontfix，
  但 membership 只看 "Sub-issues:" 行，不影響角色。
- **scoped-triage**：
  - 附錄 B 第 8 項的修改：「沒有任何單一 search 能涵蓋全部候選」只對結構化的 filter（status、assignee、label、priority）成立，
    以斷言檢查；以 query 子字串涵蓋全部候選做不到（seed 100–119 的 1,000 次抽取中只有 2.6% 沒有這種子字串，例如 " the"、"es"），
    所以 H3「最少 2 次讀取」只對結構化的 filter 成立，寫進限制。
  - §5 A 的「只改 priority 的錯誤寫入 negative」指寫入值由 generator 選擇或固定、因此一定不等於初始值的 negative
    （S4:wrong-priority、S1:needs-info-priority-on-denied、S2:perf-security-p1）。有權限的角色上，規則決定的錯誤 priority 可能
    沒有 Change，但仍由 eq 判為失敗（例如 P-perf 為 p2 時的 S8:perf-p2）。
  - H3 的截斷條件在 n ≥ 62 時一定成立，不需要重抽；只有 H1 / H2 的「priority 不能單獨區分」可能觸發重抽。
- **sync-repair**：報告的 `created_at` = 它列出的 issue 中最晚的 `created_at` + 1 天（H3 的 #R2 ≥ #R1 + 1 天），不抽 RNG；
  KH 上的 Y4 與 Y6 判斷一致，無法分辨 agent 依哪一條規則不動它；rule 7 只有 priority 的分支。
- **verified-close**：
  - Study 2 的 `V1:close-all`、`V1:close-then-reopen` 在 v3 改為 `V3:…`（沒有 plain 角色，實際失敗的是 qa 角色的 V3 forbid）；
    每個干擾項各一個 `V8:close-<decoy>` 或 `scope:close-<decoy>`；描述查找失敗的 negative 以 `lookup:` 命名（first-page-only、
    bug-search-only、security-search-only），不是規則代號。
  - 每個候選都以 0.5 的機率先有一則中性的 filler（作者不是 reporter，在種入的 comment 之前）：只放在某個角色上，comment 數會洩露角色。
  - H3 只有 bug、只有 security 的位置只從需要寫入的角色抽（否則一次 label search 就可能成功）；候選一律不帶 wontfix 或 needs-info；
    H3 的 oracle 也明寫 limit 50。2 個隱藏候選從需要寫入的角色中抽，依 seed 而不同。
  - already-asked 的敘事：reporter 說 "Still seeing" 之後，staff 再次請 reporter 確認，reporter 還沒有回覆，所以不再催、也不重新打開；
    rule 2 優先於 rule 3 是寫明的 first-match。rule 1 與 rule 2 的先後在資料中不會被測到（自我檢查確認沒有候選同時符合兩者）。
- **B 類審查（2026-10-23）的結果**：四個題型各一位審查者，只看模板、generator 程式與 dev instance；共 6 項發現，逐項經反駁驗證，
  沒有一項成立（沒有第二種讀法導致不同的最終狀態、沒有沒寫出來的規則、沒有需要猜的資訊）。下列低成本的澄清仍在 golden hash 之前採用：
  - sync-repair rule 6 的括號由 "(this reopens it)" 改為 "(this sets its status back to the value it had before the sync job closed it)"：
    "reopen" 可能被讀成把 status 設為 open，而原值有一半是 in_progress。
  - §2 / §3 的揭露與另列的那一行擴大到 qa-confirmed（與 qa-broken 相同的機制）；§3 的觸發文字說明更正（只有 H1 是 rule 2）。
  - 限制：verified-close H3 的 qa-broken 也是唯一需要去重的候選，所以「qa 角色」那一行也會包含去重的失誤（上界）；去重另以
    `params.target_labels` 計數。epic-closeout 的種入 issue 可能帶背景的一般 comment（例如 "Seen again this morning."）；
    這個題型沒有任何規則讀 comment，所以不清除。
- **共同**：comment 只檢查「包含」（四個題型）；干擾項多半是名目參數；每個題型的 context 檢查都在 seed 0–9 通過。

## 附錄 B — 題型審查 52 項的處理

除非另外註明，以下各項都在 golden hash 之前採用。

### epic-closeout（13 項）

1. **分支由 seed 的奇偶決定**，所以部分規則只在一個分支有作用 → 採用：eval 用 seed 2–9（奇偶各半），同一個 seed 在 H1–H3 使用同一個分支；報告 eval 中每條規則有作用的題數。
2. **dev 校準沒有事先寫定** → 採用其精神：完全不做 dev 校準。
3. **H3 的 turn 很緊** → 採用：
   - 不加提示句；
   - 「每個 sub-issue 都讀」的解法（21–22 個 call）登記為已知限制；
   - dev 報告 turn 的分布與以 max_turns 結束的比例。
4. **只錯在順序的模式**（例如先關 #E 再關 #C）在最終狀態正確時是 success → 採用：計為 RQ3 的事件，不算失敗曲線。
5. **E5 的 negative 在 H2 奇數 seed 不會觸發** → 採用：
   - negative 改為「關閉任一個 W」：有 resolved 的 W 就用它，否則用 open 的 W；
   - 新增 E5:wrong-tracker-text。
6. **H3 奇數 seed 中，#C → #E 的傳遞不決定結果** → 採用：寫進級別表與限制，分析不做這個歸因。
7. **rule 4 的 "dropped from the release" 有第二種讀法** → 採用：
   - 改為 "dropped from its tracker"；
   - rule 5 依審查改寫，避免讀成無條件關閉。
8. **四個 comment 字串以子字串評分，比較寬鬆** → 採用：
   - 不新增 predicate；
   - dev 與 eval 的 comment 稽核，擴大到四個字串的並存、否定與錯誤 tracker；
   - 寫進限制。
9. **背景 label 可能帶 wontfix，悄悄改變角色** → 採用：
   - 除了 W、P 與設計中的 wontfix 干擾項（H1–H3 的 resolved + wontfix 干擾項；H3 另加的 resolved + wontfix + p0 干擾項），所有種入的 issue 都以確定性的方式移除 wontfix；
   - 組成不變式改為硬檢查（內容見 §5 A）；wontfix 干擾項保留，對應的 scope negative（在 wontfix 干擾項上留 "Dropped from #E"）也保留。
10. **epic-closeout 沒有 forbid 範圍的檢查** → 採用：_check_forbid_scope 改為必要，各 forbid 集合由初始狀態重算。
11. **文字衛生** → 採用：
    - 拿掉 "Part of #E/#C"；
    - #C 的標題不含 "tracking"；
    - "dropped from"、"milestone"、"sub-issues" 加入背景文字的禁用字。
12. **wrong-tracker 的失誤與判斷錯誤混在一起** → 採用：相關 negative 加上標記，報告中另列一行。
13. **沒有寫出各級別、各 agent 的預期產量** → 採用：見 §2 的偵測力，以及 eval 使用 8 個 seed。

### scoped-triage（15 項）

1. **沒有權限的角色，priority 的取值包含錯誤寫入的值**（這時錯誤的寫入不會產生變更）→ 採用：
   - D-nb-ni 在 H1–H3 固定為 p1；
   - H3 的 D-sec 與 D-perf-nb 固定為 p3；
   - 加上 §5 A 的兩項檢查。
2. **沒有完整的 instruction** → 採用：附錄 A 逐字寫出，並另加測試，確認編號規則與 L3 的段落意思逐句對應。
3. **規則之間以編號互相引用** → 採用：
   - 改為以內容引用；
   - 刪除失敗模式 (5)；
   - params 記錄規則代號與編號的對照。
4. **H3 選取句的 or 有歧義** → 採用：改為 "An issue needs triage if all three of these hold: its status is open, it has no assignee, and it has the bug label, the performance label, or both."
5. **H3 的 turn 餘裕不足** → 採用：
   - 長度維持 9 / 11 / 12；
   - 「oracle + 每個目標一次 get_issue + 1 次額外 search + 最後回覆」改為報告的數值（含最後回覆的 turn 數：H1 17、H2 20、H3 21）；
   - H3 登記為已知限制。
6. **Haiku 在 H3 的失敗可能是 0** → 採用：H3 加入 S9（duplicate 例外），並種入一個會出現在篩選結果中的 duplicate 干擾項。這在 dev 之前就決定，不是 dev 的槓桿。
7. **dev 校準的規則前後矛盾** → 採用：不提供任何槓桿。
8. **截斷檢查的細節沒有寫出** → 採用：
   - 寫明列舉方法與單調性的理由；
   - 用確定性的方式重抽；
   - params 記錄落在第 50 名之後的候選；
   - 生成時加上斷言。
9. **context 預算** → 採用：見 §3 的 context 檢查。改用其中的兩個條件，理由見「放棄的替代方案」。
10. **規則代號的歸屬會失真** → 採用：
    - P-secdocs 的 assignee 與 priority 改標 S4，另加一個標 S7 的 comments predicate；
    - S8 加上重複的 predicate；
    - 保留 S4:perf-no-assign。
11. **缺少 5 個 negative** → 採用：全部補上，並確認每個都是因為名稱中的規則而失敗。
12. **comment 只檢查「包含」** → 採用：記為已知限制；加測試記錄這個寬鬆；dev 稽核時標出疑似評分寬鬆。
13. **S7 違反常識** → 採用：維持原文字，事先揭露，並在報告中另列（§2、§3）。
14. **H1 由段落改成編號格式** → 採用：寫進限制。
15. **H3 背景約 90% 是 open 而且未指派** → 不擴充世界（理由見「放棄的替代方案」）；寫進限制。

### sync-repair（13 項）

1. **3 個 eval seed 不夠** → 採用：使用 8 個 eval seed；RQ1 的母體跨級別合併（同 034）。
2. **「每個 issue 都讀」在 H3 需要 21 個 call** → 採用：
   - A 類檢查改為「至少 2 個 alternative ≤ 19 個 call」；
   - 不加涵蓋說明句；
   - 加上第 3 項的調整後，H3 的 read-each = 3 次讀取 + 6 次 get_issue + 9 次寫入 = 18 個 call，H2 = 17 個 call，都在 19 以內。
   - read-each 定義為每個不同的列出 issue 各 get_issue 一次。Study 2 的 read-each 是每個項目讀一次，沿用的話 H3 會變成 21 個 call，所以不沿用。測試確認 H1 / H2 / H3 為 15 / 17 / 18 個 call。
3. **H2 拿掉了 C，Y4 在 issue 層級沒有綁定** → 採用：H2 加回 C。
   - 同時配合可行性審查的「oracle ≤ 13」。sync-repair 每個單項的角色需要 2 個 call（update + comment），K 需要 1 個，所以 oracle 的奇偶是固定的：
     - H1 = 2 次讀取 + 2m，為偶數；
     - H2 = 2 次讀取 + K + 2m，為奇數；
     - H3 = 3 次讀取 + K + 2m，為偶數。
   - 在嚴格遞增、而且 H3 ≤ 13 的條件下，只有 10 / 11 / 12 可行：
     - H1 拿掉 F，角色為 A、B、C、D、E；
     - H2 拿掉 E，角色為 A、B、C、K、S、D；
     - H3 拿掉 A，角色為 Q、B、S、D、K、KH。H3 的 S 有兩項、都未處理，而且跨兩份報告，已經涵蓋 A 的兩個作用（多項只留一則 comment；rule 7 的反例）。
   - 規則數 5 < 7 < 8；n_targets、n_decoys 不遞減。H1 的 oracle 與 L3 相同（寫進限制）。
4. **校準旋鈕有問題** → 採用：不提供任何旋鈕。
5. **新的失敗大多是不可撤回的 comment，會推高 cliff 比例** → 採用：
   - 行為性 cliff 比例與解讀規則（§2）；
   - 每個級別至少有一條可撤回、但最終狀態會錯的區辨規則：H1 的 Y5（B 被略過）、H2 的 Y7（S 沒有重新打開）、H3 的 Y8（Q 改成 X）。
6. **rule 5 的 "and no 'Reverted sync change' comment" 無法評分** → 採用：刪掉這個子句，改為 "add exactly one comment to it that includes the text \"Sync revert skipped\""。
7. **rule 2 沒有指明是哪個欄位** → 採用：改為 "If the current value of the field that a listed change set is no longer the value the sync job set, that change is already handled: do not undo it."
8. **rule 7 與 rule 3 衝突，rule 6 沒有涵蓋 rule 7** → 採用：
   - rule 7 改為 "(this replaces rules 1–3 for that field)"；
   - rule 6 改為 "undo its listed changes under the other rules"。
9. **多項 issue 的 comment 代號不一致** → 採用：項數以 rule 7 合併之後計算；合併後仍有 ≥ 2 項變更的 issue，comment 一律記為 Y5；H3 的 Q 合併為一項，comment 記為 Y2。兩者都由 check 驗證。
10. **Study 2 的 negative 產生器假設每個 issue 只有一項變更** → 採用：
    - Y1:partial 固定拿掉 D；
    - flip 改用目標欄位與已處理欄位以外的欄位；
    - forbid 依 (規則, issue, 欄位) 重算。
11. **「max_turns 不算進 RQ1 的分母」與 034 不一致** → 採用：以 max_turns 結束的失敗 source 仍屬於 RQ1 的母體（§2）。
12. **rule 7 有些分支在資料中不會出現** → 採用：
    - 刪掉 rule 7 的 status 部分；
    - 記錄沒有綁定的分支；
    - rule 5 的理由改為 "You must not edit issues that are closed at the start of the task"，拿掉 "archived"。
13. **禁用字不足** → 採用：BANNED_BACKGROUND 加上 "archived"、"locked"、"skipped"、"sync revert"、"sync run"；planted_text_issues 包含 R1 與 R2。

### verified-close（11 項）

1. **H3 的角色不是 H2 的超集合** → 採用：
   - H3 的角色改為與 H2 相同的 5 個；
   - qa-broken 同時帶 bug 與 security，用來測去重；
   - 其餘至少 1 個只有 bug、1 個只有 security；
   - oracle 13。
2. **截斷只影響寬搜尋的策略** → 採用：照實寫明（§3 第 3 種難度來源），並保留自我檢查。不把 resolved 數降到 51–54：寬搜尋最多回 50 筆，降低不會減少 context。
3. **H3 沒有規定選取依據的時間點** → 採用：加上 "Decide which issues to process from their status and labels before you make any change."
4. **alternatives 的 search 沒有指定 limit** → 採用：一律明寫 limit 50；加上「search 沒有被截斷」的自我檢查。
5. **隱藏候選的位置在邊界情況很緊** → 採用：先決定 resolved 的 id 並排序，再依排名分配角色；自我檢查重算隱藏集合。
6. **qa-broken 最像陷阱** → 保留，並事先揭露；在 B 類審查中特別列出。不把 qa-confirmed 換成 plain-broken。
7. **規則代號的歸屬會失真** → 採用：params 記錄 candidate_kinds 與 decisive_rules，報告依角色列出。
8. **rule 1 與 rule 2 的先後順序沒有機械保證** → 採用：
   - 自我檢查確認沒有候選同時符合兩條規則；
   - 寫明 already-asked 的敘事，以及「first-match」的順序。
9. **沒有列出完整的 negative 集合** → 採用：
   - 沿用 Study 2 的 V1 / V2 negatives；
   - 每個帶 forbid 的角色各加一個 close-<role>；
   - 新增 V5:copy-example。
10. **V7 與 V8 同時在 H3 加入，歸因混在一起** → 不移動 V7（理由見「放棄的替代方案」）；寫進限制。
11. **機械檢查需要擴充** → 採用：
    - 新增跨研究的檢查（§3）；sync-repair 的 oracle 放寬為「不短於 L3」，其他題型（包括 verified-close）維持嚴格大於；
    - _check_instruction_refs 同時檢查 "Reopened, <example>" 與 "Please confirm the fix, <example>"，而且 example 不是任何候選的 reporter。
