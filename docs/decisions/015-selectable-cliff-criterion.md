# 015 — 驟降判定可選擇（實作 decision 011 的選項 A，預設不變）

日期：2026-10-08　狀態：採用（只是讓選項可用；**正式 sweep 使用哪一個判定仍屬使用者決策**）

## 背景

decision 011 依預先登記的規則沒有合格的 SweepPlan，列出 A / B / C 三個選項待使用者決定；選項 A 的 Bonferroni 判定當時只存在於模擬程式
（標為探索）。使用者決定付費 run 延後、keyless 工作繼續。為了讓「不論選哪個選項，剩下的都只有 paid run」，先把選項 A 正式實作成可選項。

## 決策

- `replay.localization.CLIFF_CRITERIA = ("wilson95", "bonferroni95")`，`DEFAULT_CLIFF_CRITERION = "wilson95"`：
  - `wilson95`：decision 010（兩點的 95% Wilson CI 不重疊）。
  - `bonferroni95`：兩點的 Wilson CI 以 z = Φ⁻¹(1 − 0.025 / m) 計算後不重疊，m = 相鄰已估計點的對數（只有兩點時與 wilson95 相同）。
  - Δ*、k* 與門檻比較兩者相同（decision 013）。
- `detect_cliff(..., criterion=...)`、`sweep(..., criterion=...)`；`SweepResult.criterion` 記錄使用的判定。
- `experiments/run_sweep.py`：`criterion` 與 plan 一樣**沒有預設值**，CLI 必須給 `--criterion`。
- `experiments/sweep_power.py` 改用正式的 `detect_cliff(criterion=...)`，移除探索用的複本；以相同 seed 重算的數字與 decision 011 逐位相同
  （例如 (2, 10, 48, 0.3)：0.830 / 0.030 / 273.9）。

## 為什麼

- 讓判定的選擇變成一個明確的參數，而不是在看到 LLM 結果後改程式。
- 預設維持 decision 010，避免在使用者決定前改變既定協定。

## 放棄的替代方案

- 直接把預設改成 bonferroni95：等同替使用者做了 decision 011 的選擇。
- 讓 run_sweep 沿用預設判定：付費 sweep 可能在沒有明確選擇的情況下使用某個判定。

## 對實驗的影響

- 正式 sweep 必須明確指定 `--criterion`；選擇哪一個仍待使用者決定（decision 011 的選項 A / B / C）。
