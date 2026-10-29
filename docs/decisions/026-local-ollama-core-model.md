# 026 — Core 改用自行部署的本機模型（遠端 gpu-server 的 Ollama）；付費 API 改為 Optional

日期：2026-10-11　狀態：採用（使用者 2026-10-11 指示）。本文件決定架構與流程；正式設定的凍結見 §5
（凍結值待遠端可用、開發題測試之後另立 decision，並經使用者確認）。
（2026-10-19 補註：凍結值已由 decision 030 經使用者確認，於 2026-10-16 填入 `agents.ollama.LOCAL_MODEL_CONFIG`。）

## 背景

- 使用者 2026-10-11 指示：Core 改用免費、自行部署的 LLM，暫時不使用任何付費雲端 API。之後如果取得 Claude Max 方案的免費 API 額度，
  再考慮使用。
- 正式模型預計使用 `Qwen3.5-9B Q4_K_M`，透過遠端 `gpu-server` 的 Ollama 執行。incident-investigator 正在同一台機器部署相同的模型。
- 指示的限制：
  1. 不得自行重新安裝 Ollama，也不得重複下載模型。
  2. 先完成 adapter、tool calling、JSON 格式驗證，以及其他不需要 GPU 的 Core 工作。adapter 要依目前架構選擇，優先支援遠端 Ollama，
     並保留未來更換模型的彈性。
  3. 遠端服務可用後，先用少量 Issue Tracker 開發題測試，確認模型能正確呼叫工具。
  4. 正式模型、量化版本、prompt、tool schema、推論參數與預算必須先凍結。
  5. 正式 GPU 實驗使用 `/tmp/ai-portfolio-gpu.lock`；被其他專案占用時改做其他工作，不搶 GPU。
  6. 原本的付費 API 保留為 Optional，不再是 Core 的 blocker。
- 2026-10-11 以唯讀方式檢查 gpu-server（WSL Ubuntu-22.04）：Ollama 尚未安裝，`/api/version` 沒有回應；GPU（RTX 5070 Ti 16 GB）閒置；
  lock 檔案存在。所以需要 GPU 的步驟目前 blocked。
- 參考（唯讀，不共用程式碼；SPEC「不要抽共用 framework」）：incident-investigator 的 decision 029 記錄了 Ollama 的幾個陷阱：
  - prompt 超過 `num_ctx` 時預設會靜默截斷（`/api/chat` 有 `truncate` / `shift` 欄位）；
  - Ollama 的 cloud 模型會轉送到遠端付費服務；
  - qwen3.5 自帶的預設取樣參數是 temperature 1、presence_penalty 1.5；
  - 計數欄位帶 omitempty，值為 0 時欄位不出現。

## 決策

### 1. adapter：Ollama 原生 `/api/chat`（`agents/ollama.py`，只用標準函式庫）

- **為什麼選原生 API，而不是 OpenAI 相容的 `/v1/chat/completions`**：原生 API 能明確設定 `num_ctx`、`truncate: false`、`shift: false`、
  `think`、取樣參數與 seed，並回傳 token 計數與耗時。OpenAI 相容端點會用 server 的預設 context，可能靜默截斷。
  更換模型只需要改 model_config 中的 model tag 與選項；只有日後真的換成不是 Ollama 的推論服務時，才另寫一個 adapter（不預先建立通用框架）。
- **介面**：`OllamaPolicy(client, agent_config)` 實作既有的 `agents.policy.Policy`。runner、recorder、checkpoint、sweep、分析都不需要知道
  provider。`experiments` 依 `agent_config.policy` 選擇 adapter（`make_policy`）。
- **request**（決定性、JSON-native）：
  - `{"model", "messages", "tools", "stream": false, "think", "options", "truncate": false, "shift": false}`。
  - system 與 developer prompt 合成一則 system 訊息（Ollama 沒有 developer role）。
  - tools 轉成 `{"type": "function", "function": {"name", "description", "parameters"}}`。
  - assistant 訊息以 `provider_content`（模型回傳的 assistant 訊息原樣）重送，所以 restore 後的 request 與原 episode 逐 byte 相同（009 的不變式）。
  - tool 結果送成 `{"role": "tool", "content", "tool_name"}`。
  - `options` = model_config 的取樣參數加上這次呼叫的 seed。
- **tool calling 與 JSON 格式驗證**：
  - 回覆的結構逐欄檢查：`done` 為 true、`message.role` 為 assistant、content 與 thinking 是字串、`tool_calls` 是 list、
    每個 call 的 `function.name` 是非空字串、計數是 int。不符 → 協定錯誤（harness，停止）。
  - 參數的處理：
    - 物件照用；
    - JSON 字串先解析；
    - `null` 視為空物件（Ollama 會把空的 map 寫成 null）；
    - 其他值（解析不了的字串、陣列、數字）原樣交給 env，由 env 記為 `invalid_arguments`，對模型而言就是一次工具錯誤，episode 照常進行。
  - tool call 的 id 由 adapter 依位置產生（`call_<回合>_<序號>`），與 Ollama 自己的 id 無關，所以可重現。
- **錯誤分類**（decision 009 的重試規則）：
  - 連線失敗、逾時、HTTP 5xx → infrastructure（重試一次）；
  - `done_reason` 為 `length`（輸出達到 `num_predict`）→ agent_output（不重試，等同 Anthropic 的 max_tokens）；
  - 結構不符、模型不符、context 防護失敗、HTTP 4xx → harness（停止實驗）。
- **截斷防護**：
  - 每個 request 都送 `truncate: false`、`shift: false`；
  - 送出前檢查 request 的 bytes ÷ 2（保守的 token 上限估計），加上 `num_predict` 不得超過 `num_ctx`，否則不送出；
  - 每個回覆都檢查：`prompt_eval_count` 必須存在且 >= 1；`prompt_eval_count + eval_count < num_ctx`；
  - 回覆帶 `remote_model` / `remote_host`（cloud 模型）→ harness。
- **連線限制**：
  - client 只連 loopback（`127.0.0.1`、`::1`、`localhost`）；從 Mac 連 gpu-server 時使用 SSH 通道，方式待遠端可用時驗證後寫進 MACHINE_POLICY；
  - 只會呼叫 `/api/chat`、`/api/version`、`/api/tags`、`/api/show`，**沒有** pull / create / delete / copy / push，程式上不可能下載或改動模型；
  - model tag 以 `cloud` 結尾 → 拒絕。
- **用量**：`input_tokens = prompt_eval_count`、`output_tokens = eval_count`（欄位不在時為 0）、`cost_usd = 0.0`（明確是本機，不是未知）。
  meta 記錄 `response_model`、`done_reason`、`seed`、`tool_use_blocks` 與各段耗時。

### 2. 預算：本機改以「模型呼叫次數」為上限

- `budget_reached` 與 `run_split` / `run_sweeps` 新增 `max_model_calls`：已記帳的 call 數 >= 上限時，在下一次嘗試之前停止（同 012 §1 的機制）。
- 本機 run 的 CLI 必須給 `--max-model-calls`。`--max-cost-usd` 只用於 Anthropic（Optional），因為本機成本永遠是 0，對本機無效。

### 3. CLI

- `run_baseline` 與 `run_sweep` 新增 `--provider {ollama, anthropic}`，預設 `ollama`，以及三種模式：
  - `--dry-run`：依 provider 使用對應形狀的 oracle 假 client，不連網；
  - `--confirm-local-run`：只限 ollama；
  - `--confirm-paid-run`：只限 anthropic，屬於 Optional。
- 本機 run 必須在 GPU lock 內執行：CLI 要求環境變數 `AI_PORTFOLIO_GPU_LOCK=held`，只有包住指令的 lock wrapper 會設定它；
  沒有就在任何 call 之前拒絕。wrapper（取得 `flock -n /tmp/ai-portfolio-gpu.lock`，拿不到就退出、不搶）待遠端可用、確認連線方式後撰寫並測試。
- **正式 run**（eval baseline、eval sweep）必須使用凍結的設定（§5）：model_config 的 `model_digest` 與 `ollama_version` 已填入，
  執行前確認 server 的版本與模型 digest 相同，不符就拒絕。dev 的開發題測試可以用草案設定。
- decision 025 的 R = 3、SweepPlan 選項 A（`bonferroni95` + (2, 10, 48, 0.3)）與 Gate 規則不變，同樣適用於本機的正式 run。

### 4. Anthropic 路徑改為 Optional

- `AnthropicPolicy`、`make_anthropic_client`、成本估算與 decision 009 / 012 / 014 / 025 的付費規則全部保留，但不再是 Core 的 blocker。
- 日後若取得 Claude Max 方案的免費 API 額度，另立 decision 決定是否使用；換模型等於換 agent（不同的 config_id），
  兩邊的結果不得合併（009 §4）。

### 5. 凍結（formal run 之前）

正式 run 之前必須凍結以下項目，寫成一份新的 decision 並經使用者確認：model tag、量化版本與 digest、Ollama 版本、prompt
（`baseline/v1`）、tool schema（env 的 5 個 tools）、推論參數，以及各階段的模型呼叫上限。

草案（`agents.ollama.LOCAL_MODEL_CONFIG`，`model_digest` / `ollama_version` 為 null，表示尚未凍結；2026-10-16 已由 decision 030 填入並凍結）：

| 項目 | 草案值 | 理由 |
|---|---|---|
| model | `qwen3.5:9b-q4_K_M` | 使用者指定；與 incident-investigator 相同，避免重複下載 |
| think | false | 與 Anthropic baseline 的設定無關；先以較快、較少 token 的模式測 tool calling，開發題測試後確認 |
| num_ctx / num_predict | 32768 / 4096 | 本題的對話很短，32k 留足餘裕；KV cache 加權重應能放進 16 GB（待實測） |
| temperature / top_p / top_k / min_p / presence_penalty | 0.7 / 0.8 / 20 / 0 / 1.5 | **必須 > 0**：failure localization 要從同一個 checkpoint 重複抽樣，greedy（temperature 0）會讓每次 rerun 都相同，P(success \| k) 失去意義；數值取 Qwen3 系列的 non-thinking 建議值，並明確覆寫模型自帶的預設 |
| seed | 每次呼叫隨機（記在 meta） | 同一個 seed 會讓 rerun 完全相同；記錄 seed 讓每次呼叫可以事後重現 |
| timeout | 600 秒 | 與 Anthropic baseline 相同 |

### 6. GPU lock wrapper（2026-10-11 補充，已實作並實測）

- `tools/with-gpu-lock [--lock-file PATH] [--local-port N] [--remote-port N] -- <command>`：
  - 經 `ssh gpu-server` 在 WSL 內以 `flock -n /tmp/ai-portfolio-gpu.lock` 取得 lock，由遠端一個讀 stdin 的 process 持有；
  - 同一條連線以 `-L 127.0.0.1:11435:127.0.0.1:11434` 建立通道（Windows 的 localhost 會轉到 WSL 的 loopback）；
  - 取得後才執行指令，並設定 `AI_PORTFOLIO_GPU_LOCK=held` 與 `AGENT_GYM_OLLAMA_URL`（CLI 的 `--ollama-url` 預設取它）；
  - 指令結束、中斷或 ssh 斷線時釋放 lock；
  - exit code：lock 被占用 → 75（不執行、不等待、不搶）；遠端無法連線或逾時 → 69；用法錯誤 → 64。
- 2026-10-11 以**測試用的 lock 檔**（不是 `/tmp/ai-portfolio-gpu.lock`）與 WSL 內 60 秒的臨時 HTTP 服務實測（不使用 GPU）：
  - 通道回傳 HTTP 200；
  - 指令執行期間 lock 被持有，結束後釋放；
  - 關閉 stdin 或終止本機的 ssh 都會釋放 lock；
  - lock 被其他 session 持有時 exit 75。

### 7. gpu-server 上 Ollama 的實際部署方式（2026-10-11 發現；2026-10-15 由本 Project 的 decision 029 解決）

- 12:25 時 Ollama 0.35.0 有回應、模型已載入，GPU lock 被持有。13:08 lock 釋放後，Ollama 就不在了：WSL 與 Windows 都連不到，也沒有 process。
- 以唯讀方式讀 incident-investigator 的 `tools/with-remote-ollama`：
  - 它在 WSL 使用者的 home 下放了釘選的 Ollama 0.35.0 執行檔；
  - 模型在它自己的 models 目錄，註明「只給本 project 用」；
  - 每次執行時在 GPU lock 內啟動 `ollama serve`，結束就停止。

  所以 gpu-server 上**沒有常駐、共用的 Ollama 服務**。
- 本 Project 在使用者決定之前不啟動那個 server、不使用它的 models 目錄、也不複製模型。這是跨 project 的資源決策，列在 NEXT 的
  「Blocked / 需要使用者」。2026-10-11 13:08 的開發題測試因此在 server 檢查時就停止：沒有任何模型呼叫，也沒有建立 run 目錄。
- **後續**：使用者選擇共用模型檔案、各自管理 server。本 Project 以自己的 models 目錄（manifest 複本 + 指向共用 blob 的 symlink）在
  GPU lock 內啟動自己的 server；§6 的 wrapper 隨之改寫（本 Project 的 decision 029，不是 incident-investigator 的 029）。§6 的通道終點從 11434 改為本 Project 的 server（11436）。

## 為什麼

- 使用者明確改變了 Core 的模型來源；SPEC 沒有指定 provider，Policy 介面本來就與 provider 無關，所以只需要新增 adapter，
  其他基礎設施與已登記的分析規則都沿用。
- 在 adapter 裡防範截斷與 cloud 模型，可以確保結果真的來自釘選的本機模型，而且完整看到對話。

## 放棄的替代方案

- OpenAI 相容端點：無法可靠設定 context 與防止截斷。
- 加入 `ollama` Python 套件：多一個依賴，而標準函式庫的 HTTP 就足夠；也比較容易確保只呼叫允許的端點。
- temperature 0：rerun 沒有變異，sweep 的統計設計會失效。
- 與 incident-investigator 共用 adapter：SPEC 明確要求兩邊 Core 都完成之前不抽共用 framework。

## 對實驗的影響

- 正式結果的模型從 `claude-opus-5-5` 改為本機的 qwen3.5 9B（凍結後）；config_id 不同，與任何 Anthropic 結果不得合併。
- R、SweepPlan、驟降判定、Gate 規則、分析方法都不變。
- 目前還沒有任何真實 LLM 結果。
