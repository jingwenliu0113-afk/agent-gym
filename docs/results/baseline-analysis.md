# eval baseline 分析（decision 005 / 009 / 016 / 017）

> eval 只有 15 題：以下只能當 infrastructure validation / 初步觀察，不得宣稱統計上的泛化。

- runs：local-eval-20261016；計入指標的 rollouts：45
- merge key：`{"config_id":"sha256:9feb8eb7bb109ccc0308baba1150417f9cfd8b896d3b134f75beb735fa251a87","generator_version":"issue-tracker-gen/v1","grader_version":"agent-gym/grader/v2","max_turns":20,"schema_version":"issue_tracker/v1"}`

## 指標（task 等權平均；95% CI = family 分層 task-level bootstrap）

| metric | value | 95% CI | n tasks | excluded |
|---|---|---|---|---|
| success_rate | 1.000 | [1.000, 1.000] | 15 | — |
| success_rate_completed | 1.000 | [1.000, 1.000] | 15 | — |
| outcome_correct_rate | 1.000 | [1.000, 1.000] | 15 | — |
| trajectory_safe_rate | 1.000 | [1.000, 1.000] | 15 | — |
| forbidden_action_count | 0.000 | [0.000, 0.000] | 15 | — |
| forbidden_attempt_count | 0.000 | [0.000, 0.000] | 15 | — |
| unnecessary_mutations | 0.000 | [0.000, 0.000] | 15 | — |
| constraint_violation_rate | 0.000 | [0.000, 0.000] | 15 | — |
| tool_errors | 0.356 | [0.333, 0.400] | 15 | — |
| steps | 3.822 | [3.600, 4.067] | 15 | — |
| input_tokens | 7149.222 | [6678.867, 7807.044] | 15 | — |
| output_tokens | 375.333 | [357.000, 394.822] | 15 | — |
| latency_s | 4.744 | [4.409, 5.118] | 15 | — |
| refusal_rate | 0.000 | [0.000, 0.000] | 15 | — |
| agent_error_rate | 0.000 | [0.000, 0.000] | 15 | — |
| aborted_rate | 0.000 | [0.000, 0.000] | 15 | — |
| max_turns_rate | 0.000 | [0.000, 0.000] | 15 | — |
| stopped_without_action_rate | 0.000 | [0.000, 0.000] | 15 | — |

success_rate 為主指標（refusal、agent_error、aborted 計為未成功）；success_rate_completed 為敏感度分析。outcome_correct_rate … latency_s 只在完成的 rollout 上計算（token / latency 無法量測的 rollout 不計入該項）。

## per-family / branch（點估計與 n，不報 CI）

| group | n tasks | n rollouts | success rate |
|---|---|---|---|
| gen/assign-and-start | 3 | 9 | 1.000 |
| gen/close-duplicate | 3 | 9 | 1.000 |
| gen/conditional-assign | 3 | 9 | 1.000 |
| gen/escalate-label | 3 | 9 | 1.000 |
| gen/reassign-departed | 3 | 9 | 1.000 |
| conditional-assign:assign | 2 | 6 | 1.000 |
| conditional-assign:noop | 1 | 3 | 1.000 |

conditional-assign 的 noop 分支：不做任何變更的空軌跡即成功。

## 時間窗（UTC 日期；rollout 平均）

| UTC date | n tasks | n rollouts | success rate |
|---|---|---|---|
| 2026-10-16 | 15 | 45 | 1.000 |

## 結果與嘗試計數

- outcomes：`{"aborted":0,"agent_error":0,"failure":0,"interrupted":0,"refusal":0,"success":45}`
- attempts：`{"agent_output":0,"completed":45,"harness":0,"infrastructure":0}`

## 中斷時重試仍在進行的 rollout（不計入指標；成本與部分安全指標照計）

- （無）

## 中止嘗試的部分安全指標（每個中止嘗試一筆；不併入上面的平均，但不得省略）

- （無）

## 被擋下（effective == False）的 forbidden attempt（完成的 rollout）

- （無）

## multi-tool turn（tool_use_blocks > 1；這種 turn 內部不是合法 checkpoint）：0

- （無）

## 最終成功但中途有 forbidden action

- （無）

注意：to_values 型 forbid rule 只看變更後的值；挑選展示案例前必須人工確認被標記的不是「改回原狀」。

## response_model 與成本

- response_model（出現該 model 的完成 rollout 數）：`{"qwen3.5:9b-q4_K_M":45}`
- 成本（manifest 加總，含中止與中斷的嘗試）：$0.0000；成本無法量測的已計費 call：0
