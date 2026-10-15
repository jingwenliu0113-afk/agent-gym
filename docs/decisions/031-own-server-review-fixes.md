# 031 — 自己的 Ollama server：審查後的修正（decision 029 的對抗式審查）

日期：2026-10-15　狀態：採用（補足 decision 029；不改模型、prompt、tool schema、推論參數、SweepPlan、判定規則或主要 metric）

## 背景

decision 029 commit（`7992c2e`）之後的對抗式審查分五個面向：helper 的生命週期、共用檔案的安全、wrapper（bash）、CLI 與設定、測試的有效性。
共 34 個 agent，每項發現都另外嘗試推翻或重現；只在本機的假遠端與暫存副本上進行，不連任何遠端。

| 結果 | 數量 | 說明 |
|---|---|---|
| 確認 | 24 | 2 項 medium，是同一個問題；其餘 22 項 low |
| 不確定 | 1 | 與 medium 那項相同 |
| 推翻 | 4 | 見下方「審查判定不成立而未處理」 |

另外，我在實機的 log 中發現 Ollama 0.35 的推論程序是 `<安裝目錄>/lib/ollama/llama-server`，不是 `ollama runner`。

## 決策

### 1. helper（`agents/ollama_serve.py`）

1. **本 Project 的目錄不可能寫進共用目錄**（medium）。
   - 原本的問題：只檢查 models 目錄本身的 realpath。若 `models/blobs` 是指向共用 `blobs/` 的 symlink，`prepare_models` 會刪掉共用的 blob，
     並換成指向自己的 symlink。這無法復原，等於要重新下載。現在的程式不會建立這種佈局，但後果是不可逆的。
   - 現在：
     - 本 Project 的每一層目錄（state、models、blobs、manifest 的上層、home、bin）都必須是**真正的目錄**（lstat，不是 symlink）；
       realpath 在共用目錄之外。在**建立之前**檢查，所以不會經過 symlink 在共用目錄裡建立任何東西。
     - `blobs/` 裡只建立或替換 symlink。一般檔案（可能是共用 blob 的 hard link）→ 拒絕，不刪除。
     - manifest 複本以 `mkstemp` 加上 `os.replace` 寫入，不會經過預先放好的 symlink 寫到別處。
2. **server 以本 Project 的 launcher 啟動**。launcher 是 `~/agent-gym-ollama/bin/agent-gym-ollama`，一個指向 Ollama 執行檔的 symlink。
   - incident-investigator 以 `pkill -x -f '<binary> serve'` 停止自己的 server，並且不先確認那還是它的 server。本 Project 的 server 原本也是
     同一個命令列：如果對方的 server 中途結束、lock 釋放，而我們接著開始 session，它收尾時的 pkill 會停掉我們的 server。
   - 改用 launcher 後，命令列是 `<launcher> serve`。Ollama 以 `/proc/self/exe` 找到推論程式庫，所以行為不變（2026-10-15 實機確認）。
3. **身分 token 不受時鐘影響**。
   - 原本用 `ps -o lstart` 的雜湊。Linux 上它等於開機時間加上程序的開始 ticks，而開機時間會隨牆上時鐘調整而改變；WSL2 在主機休眠後
     會調整時鐘。結果是第二條連線的 stop 模式會判定「不是這次的 server」而什麼都不停止。
   - 現在 Linux 上改用 `/proc/<pid>/stat` 第 22 欄（開機後的 ticks）加上 boot_id。其他平台（測試用的 macOS）沿用 `ps -o lstart`。
4. **判斷「本 session 的程序」改以本 Project 目錄為準**。
   - 停止後的檢查：命令列中有本 Project 目錄（`~/agent-gym-ollama/`）之下路徑的程序，也就是 server（launcher）與 llama-server
     （`--model ~/agent-gym-ollama/models/...`）。其他 Project 的程序不會參照這個目錄，所以不會被算進來，也絕不會被停止。
   - 啟動前的檢查（寧可多擋）：命令列中有 Ollama 執行檔、launcher 或 Ollama 安裝目錄之下的路徑，或程序名稱是 `ollama` / `llama-server`。
5. **stdin 在取得 lock 後就開始讀**：wrapper 在 server 啟動前就結束時，helper 不啟動 server，並回報 `NOT_STARTED`。
6. **以 `os._exit` 結束**。原本收到信號而停止時，讀 stdin 的 daemon thread 持有 stdin 的鎖，Python 結束時會 abort
   （`Fatal Python error: _enter_buffered_busy`，exit −6）。
7. **stop 模式**：
   - 每次送信號前都重新確認身分；程序已經結束不是錯誤（原本 helper 剛好結束時，stop 模式會以 traceback 結束）。
   - `STOPPED` 只依身分確認過的程序判定，不再把同一個執行檔的其他程序算成殘留。
   - 共用目錄的快照只在**取得 GPU lock 的那一瞬間**比較：其他 Project 持有 lock 時，它可能正在合法地使用共用目錄。
     這時回報 `MODELS_NOT_COMPARED`，而不是把對方的改動算在本 session 上。
   - 先讓 helper 自己停止 server，之後才檢查 server 的身分；身分不符的程序再等 5 秒，仍存在才回報。完整測試時發現：第一條連線
     斷掉後，helper 已經在停止 server，第二條連線同時檢查身分，遇到正在結束（命令列已改變）的程序，誤判為「不是這次的 server」。

### 2. wrapper（`tools/with-gpu-lock`）

1. 結束時不論 server 是否已啟動，都等 ssh 結束（最多 `STOP_TIMEOUT_S`）。
   - 原本 server 還沒回報 `STARTING` 時只等 5 秒，之後才啟動的 server 就沒有人停止。
   - 沒有 `STARTING`，又沒有「什麼都沒啟動」的證據（`BUSY`、`FAILED`、`NOT_STARTED`、遠端的指令自己結束、連線前就失敗）→ 警告。
2. 指令在**自己的 process group** 執行（`set -m`），不繼承 FIFO 的 fd。結束時（含正常結束）清掉整個 group。
   - 原本 fd 9 會傳給指令的後代，遠端因此收不到 stdin 的結束。
   - 原本 Ctrl-C 只結束直接的子程序，背景的後代會在 server 停止後繼續存在。
3. session 開始時複製一份 helper，整個 session 都用它。原本執行期間修改 `agents/ollama_serve.py` 時，第二條連線會送出長度不符的程式碼。
4. cleanup 一開始就 `set +e`，所有訊息改用寫入失敗也不中斷的 `say`。原本 stderr 已關閉時，第一個 echo 就會中斷 cleanup、改變 exit code、
   遺留暫存目錄。
5. 通道檢查失敗依原因分別說明：`UNVERIFIED`（Windows 那一側的其他程式）、連線中斷、逾時。原本一律說成 Windows 那一側的問題。
6. 警告時**保留診斷檔案**（協定、server log、ssh 的 stderr），印出路徑與 ssh stderr 的最後幾行。

### 3. CLI（`run_baseline` / `run_sweep` / `local_probe`）

- `connect_local_server`：
  - 使用的 URL 必須**等於 wrapper 確認過的通道**（`AGENT_GYM_OLLAMA_URL`）。原本 `--ollama-url` 可以指向其他 loopback server，
    例如 Mac 上的 Ollama.app，而 decision 029 §5 的保證並沒有被強制執行。
  - server 回報的 Ollama 版本必須等於 wrapper 報告中的版本。
- `local_probe` 的 `--verify-server` 與 `--truncation-check` 都經過同一個檢查。server 的錯誤一律轉成 CLI 錯誤（exit 2），不再是 traceback。

### 4. 測試

- 假 ssh 讓遠端在自己的 session 執行，輸出經過假 ssh 轉寫：殺掉假 ssh 等同真正的斷線。原本第二條連線的測試其實靠的是第一條連線的輸出。
- 新增的測試涵蓋：
  - server 消失之前一直持有 lock（`unlock_before_stop` 變異）；
  - 殘留的 session 程序 → `STOP_FAILED` / exit 70；
  - 其他 Project 的程序在兩條停止路徑上都不會被停止；
  - 經 symlink 的各種目錄佈局；
  - 時鐘無關的 token；
  - 啟動前中斷；
  - 連線在通道檢查時中斷；
  - 診斷檔案的保留。
- wrapper 的測試把 TMPDIR 設在測試的暫存目錄：警告時保留的診斷目錄不會留在系統的暫存目錄。
- Session 的每個等待都有真正的期限（helper 回歸時測試會失敗而不是卡住），測試結束時清掉所有假程序。helper 被殺掉的測試在兩個平台都執行；
  Linux 另有 PDEATHSIG 的測試。lock 被占用的訊息斷言不再依賴暫存目錄的名稱。

### 5. 第二輪審查（對 §1–§4 的修正再做一次對抗式審查：18 個 agent，確認 12 項，全部 low）

**helper**
- `models/metadata/`（Ollama 唯一會寫入的目錄）也必須是真正的目錄；裡面若有 symlink 或 link 數 > 1 的檔案 → 拒絕，不刪除。
  `home/.ollama` 同樣必須是真正的目錄。
- 共用目錄的結論：helper 在**仍持有 lock 時**比較快照，並把結論記在 `~/agent-gym-ollama/sessions/<helper_token>.json`。
  stop 模式的判斷：
  1. 優先採用這份結論；
  2. 沒有結論、而這個 session 的 helper 或 server 在 stop 模式開始時仍持有 lock → 剛結束它們之後立刻取得 lock 比較；
  3. 兩者都已結束（lock 在不知道的時間點釋放）→ `MODELS_NOT_COMPARED`。

  原本 stop 模式會在 lock 已釋放之後才比較，可能把其他 Project 之後的合法改動算在本 session 上。
- server 結束時**不立即回收**（`waitid(..., WNOWAIT)`；沒有 waitid 的平台改看 zombie 狀態）：尚未回收的 server 讓 pid / pgid
  保持被占用，停止 group 的期間不會被其他程序重複使用。stop 模式送信號給 group 之前，確認 group 裡仍有參照本 Project 目錄的程序。
- macOS 對只剩 zombie 的 process group 回 EPERM：`_group_alive` 與 `stop_group` 都處理它。
  這是完整測試時發現的：crash 的 server 不再立即被回收，結果 helper 以 traceback 結束。
- helper 遇到未預期的例外也照樣以 `os._exit` 結束，不會 abort。

**wrapper**
- 指令先在 subshell 裡單獨重導，再 `exec`。原因：bash 3.2 執行 `exec cmd <&8 8<&- 9>&-` 時，會把 fd 8 / 9 複製到 ≥ 10 的 fd 並傳給
  cmd，§2.2 宣稱的「不繼承 FIFO」並沒有生效（指令持有 fd 12）。測試改為檢查指令沒有任何 FIFO 的 fd；變異測試確認舊寫法會被抓到。
- stdin 是終端機時，指令改得到 `/dev/null`：指令在自己的 process group，讀終端機會被 SIGTTIN 停住，`wait` 永遠不返回。
  另外，等待時每秒檢查指令是否被停住，被停住 → 結束它的 process group。
- 「什麼都沒啟動」的證據：ssh 自己的訊息改以 `-E` 寫到另一個檔案，只接受只會在連線建立之前出現的訊息（例如 `ssh: connect to host`、
  `Host key verification failed`）；被信號結束的 ssh（exit ≥ 128）不算證據。原本 ERR 裡任何 `Operation timed out` 都算，
  連線在 LOCKED 之後斷掉時會誤判為「什麼都沒啟動」。
- 無法確認時，**指令還沒執行**也以 exit 70 結束（原本保留 69 / 130 / 143）。只有指令自己失敗時才保留它的 exit code。
- `trap '' PIPE` 移到最前面：任何訊息之前，stderr 已關閉也不會被 SIGPIPE 結束。
- Ctrl-\（QUIT）也走 cleanup（exit 131）；Ctrl-Z 被忽略，不會停住 session。wrapper 被 SIGKILL 時無法清理：指令不在 wrapper 的
  process group，不會一起被結束（已知限制）。
- stdin 已關閉也能執行；建立暫存目錄之後立刻設定刪除它的 trap。

**CLI**：`/api/tags` 的 `models` 不是 list → ValueError（CLI 錯誤），不再是 TypeError 的 traceback。

**推翻 2 項**：
- 移除 cleanup 的 `set +e` 沒有測試會抓到：`test_a_closed_stderr_does_not_change_the_exit_code` 會抓到。
- 測試在系統暫存目錄遺留診斷目錄：測試已改用自己的 TMPDIR。

## 審查判定不成立而未處理

- stop 模式等待後的 pid 重複使用：process group 存在時 pgid 不會被重新分配；送給 helper 的信號現在每次都重新確認身分。
- helper 的 `FAILED` 訊息含有絕對路徑：只出現在 wrapper 的 stderr 與 `runs/`（gitignored），不會進 repo。
- 沒有 `ControlPath=none`：本機的 ssh 設定沒有啟用 multiplexing。
- 無法證明 helper 能在 Python 3.10 執行：已在 gpu-server 的 python3 3.10.12 實際執行。

## 實測（2026-10-15）

在 gpu-server 上以 launcher 啟動 server：
- server 的命令列是 `~/agent-gym-ollama/bin/agent-gym-ollama serve`；
- llama-server 照常啟動（`--model ~/agent-gym-ollama/models/...`），模型載入成功；
- 結束時 server 已停止、共用目錄沒有改變、lock 已釋放；
- server log 沒有任何警告或錯誤。

## 對實驗的影響

不影響任何估計方法、判定規則、主要 metric 或資料切分；config_id 不變（server 的環境沒有改變）。
