# MACHINE_POLICY

## 核心原則

**本 Project 完全以 Mac 為主要開發環境。**

Core 不需要 RTX 5070 Ti。

## Mac 上負責

- Issue Tracker environment
- reset / snapshot / restore
- task generator
- grader
- replay
- API-based evaluation
- statistics

## 遠端桌機

只有未來 Optional 出現：

- small local tool-use model
- local model fine-tuning
- 大量 local inference

才可能使用 RTX。

遠端關機不影響 Core。

## Docker

目前 Core 階段，本 Project **不使用 Mac 上的 Docker Desktop**。

不得執行：

- Docker Desktop restart；
- 全域 prune（例如 `docker system prune`、`docker volume prune`、`docker image prune -a`、`docker network prune`）；
- 任何對其他 Project 的 container / volume / network / image 的操作（啟動、停止、刪除、修改）。

若未來真的出現 Docker 的必要性，**先停下來向使用者提出理由**，經同意後才可導入；不得自行導入。

## Credential 安全

若未來真的需要遠端，只使用：

```bash
ssh gpu-server
```

不得要求、讀取、保存或修改任何 Tailscale / SSH / sudo / token credential。

若遠端不可用，Optional 工作標記 blocked，其他工作繼續。
