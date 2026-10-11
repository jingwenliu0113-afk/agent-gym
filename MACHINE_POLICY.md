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

遠端命令的基本形式：

```bash
ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "<command>"'
```

### Ollama 與模型

- Ollama 0.35.0 與模型由 incident-investigator 部署：它的 server 只在它自己的 GPU lock 內啟動、結束就停止，模型放在它專用的
  models 目錄（decision 026 §7）。本 Project **不得**安裝、升級、重新安裝 Ollama，**不得**下載（pull）、建立、刪除或複製任何模型；
  在使用者決定共用方式之前，也不啟動那個 server、不使用它的 models 目錄。
- adapter 程式上只會呼叫 `/api/chat`、`/api/version`、`/api/tags`、`/api/show`，只連 loopback 位址（從 Mac 經 SSH 通道）。
- 不使用 Ollama 的 cloud 模型（tag 以 `cloud` 結尾者一律拒絕）。
- 正式 run 前確認 server 的 Ollama 版本與模型 digest 等於凍結值；不符就不執行，回報使用者。

### GPU lock

- 統一使用 WSL 內的 `/tmp/ai-portfolio-gpu.lock`，以 `flock -n` 取得：

  ```bash
  ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "flock -n /tmp/ai-portfolio-gpu.lock <command>"'
  ```

- 本 Project **所有**會把模型載進 GPU 的工作（開發題測試、正式 eval baseline、正式 sweep）都必須在 lock 內執行。
  CLI 要求環境變數 `AI_PORTFOLIO_GPU_LOCK=held`，這個變數只由 lock wrapper 設定；沒有就拒絕執行。
- 拿不到 lock（其他 Project 正在使用）→ 不執行、不等待、不搶，改做其他工作。
- 執行前另外看 `nvidia-smi`：其他 Project 已經把模型載進 GPU 時不執行（16 GB 容不下兩邊同時載入）。
- 不得刪除 `/tmp/ai-portfolio-gpu.lock` 或其他 Project 的 lock 檔，也不得終止其他 Project 的 process（wrapper 測試用的 lock 檔只用本 Project 自己建立的）。
- lock wrapper：`tools/with-gpu-lock -- <command>`（decision 026 §6）。Mac 端經 `ssh gpu-server` 在 WSL 內取得 `flock -n`，
  同一條連線建立到 Ollama 的通道（`http://127.0.0.1:11435` → 遠端 loopback 的 11434），設定 `AI_PORTFOLIO_GPU_LOCK=held`；
  指令結束或中斷時釋放。lock 被占用 → exit 75，改做其他工作。

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
