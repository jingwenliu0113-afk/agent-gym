# 029 — agent-gym 自己的 Ollama server：共用模型檔案、各自管理 server

日期：2026-10-15（使用者決定：2026-10-11）　狀態：採用（解決 decision 026 §7；不改模型、prompt、tool schema、推論參數、SweepPlan、
判定規則或主要 metric）。審查後的修正見 decision 031：目錄的安全檢查、以 launcher 啟動、時鐘無關的身分 token、stop 模式只看
本 session 的程序、CLI 綁定 wrapper 確認的 URL 與版本。

## 背景

- decision 026 §7：gpu-server 上沒有常駐、共用的 Ollama 服務。incident-investigator 在自己的 GPU lock 內，以釘選的 Ollama 0.35.0 和
  它專用的 models 目錄（`~/ii-ollama-models`）啟動 `ollama serve`，結束就停止。
- 使用者決定（2026-10-11，NEXT A0 的選項 1）：
  - 兩個專案共用已下載的 Qwen3.5-9B Q4_K_M 模型檔案，但各自管理 Ollama server。
  - 不重複下載模型，也不修改或刪除 incident-investigator 的模型檔案。
  - 正式 GPU 工作必須取得 `/tmp/ai-portfolio-gpu.lock`，整個推論期間都要持有，結束後關閉自己啟動的 server。
  - 驗證 adapter、tool calling 與模型推論正常。
- 唯讀檢查的發現（2026-10-11）：共用目錄裡有 `metadata/sha256-<model blob>.json`，內容是 `{"ollama_version":"0.35.0", ...}`。
  它的建立時間與 `~/.ollama/id_ed25519` 同一分鐘，也就是 Ollama 第一次使用模型時寫的快取。所以 **Ollama 執行時會寫入
  `OLLAMA_MODELS`**；直接把共用目錄當成本 Project server 的 `OLLAMA_MODELS`，就可能改動對方的目錄。

## 決策

### 1. 本 Project 自己的 models 目錄；共用目錄只讀

- `~/agent-gym-ollama/models`：
  - manifest 從共用目錄**複製**（709 bytes，內容相同，所以 `/api/tags` 的 digest 也相同）；
  - 每個 blob 是指向共用目錄檔案的 **symlink**：6.6 GB 的模型只有一份，不複製、不下載。
- Ollama 執行時寫的東西都留在本 Project 的目錄：models 目錄的快取（`metadata/`），以及 server 的 HOME（`~/agent-gym-ollama/home`，
  Ollama 在這裡產生它自己的 identity key）。
- 共用目錄**只讀**：helper 只 lstat 和讀取。啟動前與停止後各取一次快照並比對。快照記錄每個項目（含目錄本身）的：
  - 類型、權限、大小、mtime、ctime、inode、link 數、owner；
  - symlink 的目標，以及 ≤ 1 MiB 檔案的 sha256；
  - 不記 atime，讀取不算改動。

  兩次快照不同 → `MODELS_CHANGED`，wrapper 結束時 exit 70，並提示回報使用者。
- 兩個目錄不可相同或互相包含（違反就不啟動）。共用目錄的 manifest 必須存在；每個 blob 必須是大小相符的一般檔案。

### 2. 每次 session 在 lock 內啟動自己的 server

遠端的工作由 `agents/ollama_serve.py` 完成：只用標準函式庫，wrapper 經 ssh 的 stdin 傳送，以 WSL 的 python3 3.10 執行。

- **lock**：以 flock(2) 非阻塞取得 `/tmp/ai-portfolio-gpu.lock`。這與其他 Project 的 `flock -n` 是同一把鎖。被占用 → exit 75，
  什麼都不啟動。
- **事前檢查**：下列任何一項成立 → exit 75，不啟動也不停止任何程序。
  - 已有 Ollama 程序（同一個執行檔，或程序名稱是 `ollama`）；
  - WSL 內的 port 11436 已有程式在聽；
  - `nvidia-smi` 回報已用 ≥ 4096 MiB（其他 Project 可能已載入模型）。

  `nvidia-smi` 無法執行 → exit 69。
- **固定的 server 環境**（`SERVER_ENV`）：

  | 變數 | 值 |
  |---|---|
  | `OLLAMA_KEEP_ALIVE` | `-1` |
  | `OLLAMA_MAX_LOADED_MODELS` | `1` |
  | `OLLAMA_NOPRUNE` | `1` |
  | `OLLAMA_NO_CLOUD` | `1` |
  | `OLLAMA_NUM_PARALLEL` | `1` |

  - 部署細節另外設定：`OLLAMA_HOST=127.0.0.1:11436`、`OLLAMA_MODELS`、`HOME`。incident-investigator 用 11434。
  - 呼叫端環境裡的其他 `OLLAMA_*` 一律不傳。
  - 其他 server 預設值（flash attention、KV cache 型別等）由釘選的 Ollama 版本決定。2026-10-15 的 log 顯示
    `OLLAMA_FLASH_ATTENTION:false`、`OLLAMA_KV_CACHE_TYPE` 為預設。
- **程序**：
  - server 在自己的 session / process group 執行；
  - 它繼承 lock 的 fd：helper 意外消失時，只要 server 還在，lock 就仍被持有，不會有兩個 Project 同時用 GPU；
  - Linux 上 helper 消失時 server 會收到 SIGTERM（PR_SET_PDEATHSIG）。
- **就緒**：`/api/version` 有回應且程序仍在 → `READY`，內容是 JSON（pid、port、版本、GPU 記憶體、server 環境、models 目錄）。

### 3. 確認通道接到的是這次啟動的 server

- 只有在 `READY` 之後，wrapper 才經通道送出 `GET /api/version?agent_gym_session=<128 位元隨機值>`。在那之前，通道另一端只可能是別人的程式。
- helper 確認這個請求出現在**這次 server 的請求 log**。Ollama 的 log 含 query string，2026-10-15 已實測。
- 沒有出現 → exit 69，指令不執行。這種情況代表 Windows 那一側有其他程式在聽同一個 port，WSL 內的 port 檢查看不到它；那個程式既不使用也不停止。

### 4. 整個推論期間持有 lock，結束時停止自己的 server

- 指令在 lock 內、server 就緒並確認之後才執行。指令結束（或 wrapper 收到 INT / TERM / HUP，這時先以 TERM 結束指令）時，
  wrapper 關閉 helper 的 stdin。helper 接著：
  1. 停止 server 的整個 process group（TERM，30 秒後 KILL）；
  2. 確認沒有留下這個執行檔的程序、port 不再回應 → `STOPPED`；
  3. 比對快照 → `MODELS_UNCHANGED`；
  4. 結束，lock 隨之釋放。
- wrapper 最多等 120 秒。沒有收到 `STOPPED` 時，以**第二條 ssh 連線**執行 helper 的 stop 模式。它只停止 `STARTING` 回報的 helper 與
  server，並且要求身分完全相符：pid、程序開始時間的雜湊、process group leader、命令列 `<binary> serve`。身分不符就什麼都不停止。
- 最後仍無法確認 `STOPPED` 與 `MODELS_UNCHANGED`：
  - 印出警告與檢查方式；
  - 指令成功時 exit 70，指令失敗時保留它的 exit code。
- 不刪除 lock，不停止其他程序。不以命令列比對（pkill）停止：incident-investigator 的 server 用同一個執行檔、同一個命令列。

### 5. 設定與 CLI

- `model_config["server_env"] = SERVER_ENV`，所以 config_id 涵蓋它，`check_local_config` 也會檢查（只能是 `OLLAMA_*` 字串，不可含
  HOST / MODELS）。
- wrapper 設定 `AGENT_GYM_OLLAMA_SERVER=<READY 的 JSON>`。`run_baseline`、`run_sweep`、`local_probe`（截斷測試與 `--verify-server`）都要求它：
  存在、是 JSON 物件、`env` 等於設定的 `server_env`。這個檢查在任何模型呼叫和建立 run 目錄之前。結果是本機 run **只會**對本 Project
  啟動並確認過的 server 送請求，不會誤用其他 Project 的 server。
- `local_probe --verify-server`：只檢查版本、模型 digest、量化與 capabilities，不載入模型。

## 實測（2026-10-15，gpu-server，lock 內）

1. `--verify-server`：
   - Ollama 0.35.0；digest `sha256:6488c96f…3ea7`（與 026 對 incident-investigator server 的唯讀檢查相同）；Q4_K_M；
     capabilities 含 tools。
   - server 的 log 確認環境已生效：NOPRUNE、NO_CLOUD、NUM_PARALLEL 1、MAX_LOADED_MODELS 1、KEEP_ALIVE 無限、`OLLAMA_MODELS` 是本 Project
     的目錄。GPU 偵測為 CUDA、RTX 5070 Ti（compute 12.0），可用 14.7 GiB。
   - 隨機值的請求出現在 log。server 已停止，共用目錄沒有改變，lock 已釋放。
2. `--truncation-check`：4113 tokens 的 prompt 被拒絕（HTTP 400 `exceed_context_size_error`），`truncate: false` 時不會靜默截斷。
   另外發現：要求 num_ctx 1024，server 實際用 2048（最小值）。實際 context 大於要求值只會讓 adapter 的防護更保守。
3. 開發題測試：見 NEXT（`runs/local-dev-smoke-20261015`）。

## 為什麼

- 使用者要求不改動對方的模型檔案。Ollama 會寫入 `OLLAMA_MODELS`，所以只有讓 server 指向本 Project 自己的目錄，才能**在結構上**
  保證不寫入共用目錄；快照比對則是事後的證據。
- 「整個推論期間持有 lock」由兩點保證：指令只在 helper 持有 lock 並確認 server 之後執行；server 繼承 lock 的 fd。
- 隨機值的確認讓「本機 run 用的是這次啟動的、設定已知的 server」成為可驗證的事實，而不是假設。

## 放棄的替代方案

- **直接以共用目錄當 `OLLAMA_MODELS`**（NEXT A0 原本的描述）：Ollama 的 metadata 快取會寫進對方的目錄。
- **hard link**：建立 hard link 會改變共用檔案的 link 數與 ctime，等於修改它的 metadata；對方刪除模型後空間也不會釋放。
- **複製 6.6 GB**：使用者選擇共用。
- **用 bash 寫遠端腳本**（incident-investigator 的做法）：cmd.exe 的引號限制多，而且無法在 Mac 上以假的遠端完整測試。Python helper
  經 stdin 傳送，不受 cmd.exe 8191 字元的限制，也能用 `tests/remote_fakes.py` 端對端測試。
- **以 pkill 依命令列停止**：會連帶比對到 incident-investigator 的 server。

## 對實驗的影響

- 本機 agent 的 config_id 改變（新增 `server_env`）。目前沒有任何正式的本機結果；草案設定的開發題測試使用新的 config。
- 不影響任何估計方法、判定規則、主要 metric 或資料切分。
