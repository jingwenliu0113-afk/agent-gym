# Agent Gym + Replay SPEC

## 一句話問題

能不能用可重置、可重播的 stateful environment，看出 Agent 從哪一步開始變得難以恢復，以及它在成功過程中有沒有做危險的事？

---

# Core

## 1. Domain

只有 Issue Tracker。

Entities：

- issues
- users
- comments
- labels

Tools：

- search_issues
- get_issue
- update_issue
- add_comment
- assign_issue

## 2. Environment

必須支援：

- reset
- snapshot
- restore

Core 保持 deterministic。

不要在 Core 注入 tool faults。

## 3. Task

每個 task 有：

- initial state
- expected outcome
- success constraints
- forbidden constraints

使用 parameterized generator 產生不同 instance。

避免全部手寫固定題。

## 4. Grader

至少評：

- final-state correctness
- constraint violations
- forbidden actions
- unnecessary mutations
- tool errors
- steps
- token usage
- latency
- cost

「最後成功」不代表 trajectory 安全。

## 5. Recorder

每一步保存：

- model message
- tool call
- args
- result
- state hash before
- state hash after

完整 checkpoint 必須可重建：

- environment state
- conversation history
- tool calls / results
- system prompt
- developer prompt
- prompt version
- model config

核心 invariant：

> restore 後，Agent 與 environment 看到的 observable state 必須與原 checkpoint 一致。

## 6. Failure Localization

從 trajectory checkpoint 重新 rollout。

估計：

`P(success | checkpoint)`

先少量掃描，再對疑似變化處加密取樣。

統計以 task 為 cluster：

- task-level bootstrap
- 不把同一 task 的多次 rollout 當獨立樣本

---

# Advanced

只有找到跨 trajectory 重複的 recoverability pattern 才進。

## A. Tool Fault Injection

例如：

- silent failure
- stale read
- timeout

比較：

- ReAct
- ReAct + verify-after-write

目的：

研究 verification vs cost trade-off。

## B. 第二個 App

例如：

- CRM
- Inventory

再做 cross-app task。

## C. Failure Type Taxonomy

例如：

- premature commitment
- wrong tool
- bad arguments
- constraint loss
- unrecoverable mutation

---

# Optional

- human approval
- side-effect safety policy
- memory
- multi-agent
- small tool-use model fine-tuning
- adaptive security attacks

---

# 不做什麼

- 不做 universal Agent runtime。
- 不做 generic benchmark platform。
- 不接真實高風險 side effects。
- 不做 dashboard 當主產物。

---

# 與 Investigator 的關係

可以形成遞進，而不是重複。

Investigator：

- readonly diagnostic trajectory
- 推理與資訊取得路徑的 recoverability

Gym：

- state-changing trajectory
- snapshot / restore
- side effects
- trajectory safety

若兩題都做到 failure localization，可比較：

- readonly failure structure
- stateful failure structure

但不要抽共用 framework，除非兩邊 Core 都已經完成且真的出現可重用需求。

---

# GitHub 最終成果

README 首屏應能展示：

```text
Task
→ Agent actions
→ state changes
→ grader
→ replay from checkpoint
```

並展示：

- recoverability curve
- trajectory safety
- 最終成功但中途有 forbidden action 的案例

---

# Repo 建議結構

```text
agent-gym/
├── CLAUDE.md
├── SPEC.md
├── NEXT.md
├── env/
├── apps/
│   └── issue_tracker/
├── tasks/
├── agents/
├── grader/
├── replay/
├── experiments/
├── tests/
└── docs/
    ├── EXPERIMENT_RULES.md
    └── decisions/
```

---

# Day 1

1. 定 Issue Tracker schema。
2. 定 5 個 tools。
3. 定 task schema。
4. 定 grader schema。
5. 寫 snapshot / restore 最小原型。
6. 建立 3～5 個 hand-written smoke tasks。

---

# Week 1

1. Issue Tracker local environment。
2. 5 個 tools 全部可用。
3. reset / snapshot / restore 測試。
4. 10～20 個 parameterized task instances。
5. baseline Agent 能跑完整 task。
6. recorder 存完整 trajectory。
7. grader 能評 final state 與 forbidden actions。
8. 能從一個 checkpoint restore 並 rerun。

Week 1 不做 tool fault injection。

---

# Gate

## 進 Advanced

Failure localization 找到：

- 至少一種跨多條 trajectory 重複出現的 recoverability 驟降模式。

## 收尾

若大多數 trajectory 都只是平緩漂移，而且每條都不同：

- 把「失敗是逐步累積」寫成正式結論。
- 不加第二個 app。
