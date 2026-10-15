# 030 — 凍結本機模型的正式設定（提案，待使用者確認）

日期：2026-10-15　狀態：**提案**。使用者確認前不填入凍結值、不跑任何正式 run（eval baseline、sweep）。

decision 026 §5 要求：正式模型、量化版本、prompt、tool schema、推論參數與預算必須先凍結。這份提案以 2026-10-15 的開發題測試為依據
（decision 029 的 server，GPU lock 內）。

## 依據：開發題測試（`runs/local-dev-smoke-20261015`）

設定是 decision 026 的草案（config `sha256:021ae8cf…d65b`），dev split 5 題 × R = 3：

| 項目 | 結果 |
|---|---|
| 結果 | 15 / 15 成功，0 次中止 |
| 模型呼叫 | 75 次（每個 episode 2–10 次，平均 5 次） |
| 每次呼叫的秒數 | 中位數 0.73、p90 1.30、最大 23.7（第一次呼叫，含載入模型） |
| prompt tokens | 中位數 1,390、最大 2,487 |
| 輸出 tokens | 中位數 66、最大 194（num_predict 4,096 從未接近） |
| tool call | 60 個：未知工具 0、一則訊息多個 call 0、`invalid_arguments` 5 個 |
| request bytes / prompt token | 3.59–3.88（中位數 3.76） |
| 最大的 request | 9.5 KB |
| GPU 記憶體（num_ctx 32,768） | 6.0 GiB（模型 4.7 + context 1.1 + compute 0.24），剩 8.9 GiB |

- 5 個 `invalid_arguments` 都由模型在下一個 turn 自行修正，例如 `limit` 100 超過上限 50。
- bytes / token 的實測值表示送出前的 fit check（bytes ÷ 2）約保守 1.9 倍。
- 截斷自我測試（decision 029）：`truncate: false` 時，超長 prompt 被拒絕而不是靜默截斷。server 的最小 context 是 2,048。
- server 的實際執行方式：llama-server，`-c 32768 -np 1 -b 1024 -ub 1024 --flash-attn auto`，chatml template（prompt 由 Ollama 組成）。

## 提案的凍結值

| 項目 | 值 | 說明 |
|---|---|---|
| model | `qwen3.5:9b-q4_K_M` | 使用者指定 |
| model_digest | `sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | 每次 run 前由 `verify_server` 比對 |
| 量化 | Q4_K_M | |
| ollama_version | `0.35.0` | incident-investigator 部署的執行檔（共用、不修改）；每次 run 前比對 |
| server 環境 | decision 029 §2 的 `SERVER_ENV` | 已在 config_id 中 |
| prompt | `baseline/v1` | 與 Anthropic baseline 相同的 system / developer prompt |
| tool schema | `issue_tracker/v1`（`TOOL_SPECS`） | 不變 |
| think | false | 開發題全部成功，沒有開啟的理由 |
| num_ctx / num_predict | 32,768 / 4,096 | 最大 prompt 的 13 倍；GPU 記憶體足夠 |
| temperature / top_p / top_k / min_p / presence_penalty | 0.7 / 0.8 / 20 / 0 / 1.5 | 必須 > 0（rerun 需要變異） |
| seed | 每次呼叫隨機，記在 meta | |
| client timeout | 600 秒 | |
| max_turns | 20 | 不變 |

提案**完全沿用 026 的草案**：開發題測試沒有顯示任何需要改的地方。看過結果後才調整參數，就等於在 dev 上調參；沒有必要。

## 預算（呼叫次數上限；沒有金錢成本，成本是 GPU 時間）

- **正式 eval baseline**（R = 3，45 個 episode）：`--max-model-calls 1800`。
  - 1800 = 45 × max_turns 20 × MAX_EPISODE_ATTEMPTS 2，也就是理論上限，所以上限不會截斷正常的 run，只防止失控。
  - 預期約 250 次呼叫、5–10 分鐘。
- **正式 sweep**（選項 A：`bonferroni95` + (2, 10, 48, 0.3)）：
  - 呼叫數取決於 eval baseline 的 source 長度。以開發題的長度粗估，上限約 n_fine × Σ L(L+1)/2 ≈ 13,700 次，約 3 小時。
  - **分段執行**：每段 `--max-model-calls 3000`（約 40–60 分鐘），段與段之間釋放 GPU lock，讓其他 Project 可以使用；以
    `--resume-from` 接續到完成。decision 018–024 保證預算停止與續跑不改變結果：65 條隨機續跑鏈的模擬全部與沒有中斷的排程完全相同。
  - 段數不設上限，直到所有 source 完成。
- 拿不到 lock 就不跑，改做其他工作（MACHINE_POLICY）。

## 確認後的步驟

1. 把 `model_digest`、`ollama_version` 填入 `agents.ollama.LOCAL_MODEL_CONFIG`。config_id 會因此改變；開發題測試屬於草案設定，
   不與正式結果合併。
2. 正式 eval baseline：

   ```bash
   tools/with-gpu-lock -- uv run python -m experiments.run_baseline --split eval --rollouts 3 --confirm-local-run --max-model-calls 1800
   ```

3. 正式 sweep 分段執行，直到完成；第二段起加 `--resume-from` 列出所有先前的 sweep 目錄：

   ```bash
   tools/with-gpu-lock -- uv run python -m experiments.run_sweep --baseline-run runs/<eval> --plan 2,10,48,0.3 --criterion bonferroni95 --confirm-local-run --max-model-calls 3000
   ```

4. `analyze_baseline` / `analyze_sweep`，Gate 依 decision 025 §4 機械化判定。

## 觀察（不改變任何設計）

開發題全部成功。如果 eval 的成功率也接近 100%，大部分 source 的曲線可能一直維持在高點，Gate 的結果就會是 `gradual` 或
`no_repeated_pattern`。這是已登記協定下的合法結果（decision 012 §3：成功的 source 也做 sweep，用來區分「一直都可恢復」與「中途出現不可恢復點」）。
不因此調整題目、參數或判定規則。
