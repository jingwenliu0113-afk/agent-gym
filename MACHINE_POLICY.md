# MACHINE_POLICY

## 核心原則

**本 Project 以 Mac 為主要開發環境。**

2026-10-11 起，Core 的 LLM 改用台中 `gpu-server`（RTX 5070 Ti）上以 Ollama 執行的本機模型（decision 026）。
所以只有「模型推論」需要遠端；其他所有工作都留在 Mac。遠端不可用時，只把需要模型的步驟標記 blocked，其他工作繼續。

## Mac 上負責

- Issue Tracker environment
- reset / snapshot / restore
- task generator
- grader
- replay
- 實驗 harness（runner、manifest、sweep 排程）與 Ollama adapter
- dry-run（oracle 假 client，不需要 GPU）
- statistics 與分析

## 遠端 gpu-server（Windows 11 + WSL2 `Ubuntu-22.04`）

用途只有一個：以 Ollama 執行 Core 的本機模型推論（`qwen3.5:9b-q4_K_M`，凍結值見 decision 026 §5 之後的凍結 decision）。
遠端的路徑一律以 `~`（WSL 使用者的 home）表示；WSL 的使用者名稱不得出現在 repo 中。

遠端命令的基本形式：

```bash
ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "<command>"'
```

### Ollama 與模型（decision 026、029）

- 模型檔案與 incident-investigator **共用**、各自管理 server（使用者決定，2026-10-11）：
  - 執行檔：incident-investigator 部署的 Ollama 0.35.0（`~/opt/ollama-0.35.0/bin/ollama`），只執行、不修改。
  - 共用的 models 目錄（`~/ii-ollama-models`）**只讀**：不寫入、不刪除、不在裡面建立任何東西。
  - 本 Project 自己的目錄 `~/agent-gym-ollama/`，Ollama 執行時寫的快取都在這裡（每一層都必須是真正的目錄，decision 031）：
    - `models/`：manifest 的複本與指向共用 blob 的 symlink，不複製模型；
    - `home/`：server 的 HOME；
    - `bin/agent-gym-ollama`：指向 Ollama 執行檔的 launcher。server 以它啟動，命令列因此不同於 incident-investigator 的 server。
- 本 Project **不得**安裝、升級、重新安裝 Ollama，**不得**下載（pull）、建立、刪除或複製任何模型檔案。
- server 只由 `tools/with-gpu-lock` 在 GPU lock 內啟動（127.0.0.1:11436，固定環境見 decision 029 §2），指令結束就停止。
  不使用、不停止其他 Project 的 server。
- 每次結束都比對共用目錄的快照；有任何改變 → wrapper exit 70，回報使用者。
- adapter 只呼叫 `/api/chat`、`/api/version`、`/api/tags`、`/api/show`，只連 loopback（從 Mac 經 SSH 通道）。
- 不使用 Ollama 的 cloud 模型（tag 以 `cloud` 結尾者一律拒絕；server 以 `OLLAMA_NO_CLOUD=1` 啟動）。
- 正式 run 前確認 server 的 Ollama 版本與模型 digest 等於凍結值；不符就不執行，回報使用者。

### GPU lock

- 統一使用 WSL 內的 `/tmp/ai-portfolio-gpu.lock`（flock(2)，與 `flock -n` 是同一把鎖）。
- 本 Project **所有**會把模型載進 GPU 的工作（開發題測試、正式 eval baseline、正式 sweep）都必須在 lock 內執行，並且整個推論期間持有：
  `tools/with-gpu-lock -- <command>`（decision 026 §6、029）。wrapper 會：
  1. 取得 lock；
  2. 啟動本 Project 的 server；
  3. 確認通道接到的正是這次啟動的 server；
  4. 執行指令，並設定 `AI_PORTFOLIO_GPU_LOCK=held`、`AGENT_GYM_OLLAMA_URL`、`AGENT_GYM_OLLAMA_SERVER`；
  5. 結束時停止 server、比對快照，再釋放 lock。

  server 繼承 lock 的 fd，所以 server 還在，lock 就不會被釋放。
- CLI 沒有 wrapper 設定的環境變數就拒絕執行。
- 拿不到 lock（其他 Project 正在使用）→ exit 75：不執行、不等待、不搶，改做其他工作。下列情況同樣 exit 75，而且不使用也不停止那些程序：
  - 已有任何 Ollama 程序；
  - port 被占用；
  - `nvidia-smi` 回報已用 ≥ 4096 MiB（其他 Project 已把模型載進 GPU；16 GB 容不下兩邊同時載入）。
- 無法確認 server 已停止、共用目錄沒有改變，或無法證明什麼都沒啟動 → wrapper 印出警告與檢查方式，保留診斷檔案（exit 70；
  只有指令自己失敗時才保留它的 exit code）。**不得**自行刪除 lock 或終止其他程序；回報使用者。
- 共用目錄的比較在 helper 仍持有 lock 時進行，結論記在 `~/agent-gym-ollama/sessions/`。wrapper 的第二條連線（stop 模式）採用這份結論；
  只有在它剛結束本 session 仍持有 lock 的程序時，才短暫取得 lock 比較；否則回報 `MODELS_NOT_COMPARED`（decision 031 §5）。
- wrapper 內的指令拿不到終端機的 stdin（得到 `/dev/null`），不能互動；Ctrl-Z 被忽略。
- 不得刪除 `/tmp/ai-portfolio-gpu.lock` 或其他 Project 的 lock 檔，也不得終止其他 Project 的 process（wrapper 的 stop 只停止本次
  session 啟動、身分相符的 server；測試用的 lock 檔只用本 Project 自己建立的）。

## Docker

目前 Core 階段，本 Project **不使用 Mac 上的 Docker Desktop**。

不得執行：

- Docker Desktop restart；
- 全域 prune（例如 `docker system prune`、`docker volume prune`、`docker image prune -a`、`docker network prune`）；
- 任何對其他 Project 的 container / volume / network / image 的操作（啟動、停止、刪除、修改）。

若未來真的出現 Docker 的必要性，**先停下來向使用者提出理由**，經同意後才可導入；不得自行導入。

## Credential 安全

遠端只使用：

```bash
ssh gpu-server
```

不得要求、讀取、保存或修改任何 Tailscale / SSH / sudo / token credential。付費雲端 API 的 key 屬於 Optional 路徑，
Core 不需要也不使用。

若遠端不可用，需要模型的步驟標記 blocked，其他工作繼續。
