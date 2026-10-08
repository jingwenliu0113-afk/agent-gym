# Agent Gym 實驗規範

## Core

- deterministic environment
- 固定 prompt / model config
- 固定 task generator version
- checkpoint 必須完整可重建
- replay 前後 observable state 要一致

## Statistics

- repeated rollout 以 task 為 cluster
- 使用 task-level bootstrap
- 報 confidence interval
- 不把 10 次同 task rollout 當 10 個獨立 task

## Trajectory Safety

至少分開：

- task success
- constraint violations
- forbidden actions
- unnecessary mutations

## Advanced

tool fault injection 只能在 Advanced。

fault rate 必須預先固定。

verification 策略必須在相同 fault condition 下比較。

## 報告

- 不只報 success rate。
- 必須展示 replay 能回答什麼無 replay 難回答的問題。
- 如果沒有 recoverability cliff，也要報 gradual drift。
