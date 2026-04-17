繁體中文 ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## 專案簡介
本專案基於 [index-tts](https://github.com/index-tts/index-tts)，使用 vLLM 函式庫重新實現 GPT 模型的推理，大幅加速 index-tts 的推理過程。

推理速度在單卡 RTX 4090 上的提升：
- 單個請求的 RTF (Real-Time Factor)：≈0.3 → ≈0.1
- 單個請求的 GPT 模型 decode 速度：≈90 token/s → ≈280 token/s
- 並發量：gpu_memory_utilization 設置為 0.5（約 12GB 顯存）的情況下，vLLM 顯示 `Maximum concurrency for 608 tokens per request: 237.18x`，兩百多並發！當然考慮 TTFT 以及其他推理成本（BigVGAN 等），實測 16 左右的並發無壓力（測速腳本參考 `simple_test.py`）

## 新特性
- **支援多角色音訊混合**：可以傳入多個參考音訊，TTS 輸出的角色聲線為多個參考音訊的混合版本（輸入多個參考音訊會導致輸出的角色聲線不穩定，可以抽卡抽到滿意的聲線再作為參考音訊）
- **Docker 一鍵部署**：支援全自動化容器部署，自動下載模型和轉換格式
- **OpenAI API 相容**：相容 OpenAI TTS API 格式，方便整合現有應用
- **串流 TTS API**：`/tts_url_stream`、`/tts_stream` 支援邊生成邊回傳音訊，首音延遲可比非串流版本降低約 3 倍
- **固定 Seed**：四個 TTS 端點均支援 `seed` 參數，預設 `seed=2`，相同文字與 seed 可穩定重現相同聲音
- **文字替換規則系統**：基於 PostgreSQL，支援多組規則（set）管理，可在 TTS 生成前自動替換特定詞彙的發音；規則以繁體輸入，系統自動轉簡體存儲；支援 `_global_` 全域規則集，套用至所有請求
- **替換規則管理介面**：內建 Admin Web UI（`/replacementweb`，需帳密驗證），支援規則的新增/編輯/刪除、批次 JSON 上傳匯入（覆蓋或合併）、匯出 JSON、複製規則組、全域規則預覽、快速邊界包裝、TTS 即時試聽
- **廠商規則管理頁面**：`/vendorweb?set=<名稱>`，簡化介面供非技術人員自行管理指定規則組；支援比對邊界選擇、忽略大小寫、TTS 即時試聽；無法存取 `_global_`

## 性能表現
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| 模型                    | 中文  | 英文  |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm          | 1.12  | 1.987 |

基本保持了原專案的性能

## 更新日誌

- **[2026-04-17]** 多項功能更新：
    1. `_global_` 全域規則集：廠商規則先執行（組內長度降冪），全域規則後執行（兜底），廠商永遠優先
    2. `/replacementweb` 加入 HTTP Basic Auth（帳密由環境變數 `ADMIN_USER`/`ADMIN_PASSWORD` 設定）
    3. 新增 `/vendorweb?set=<名稱>` 廠商用簡化管理頁面（無法存取 `_global_`）
    4. 新增 `GET /replacements/{set_name}/export` 匯出規則 JSON
    5. `POST /replacements/{set_name}/bulk` 新增 `mode=overwrite|merge` 參數；支援上傳 JSON 檔案
    6. 新增 `POST /replacements/{new_set}/clone/{source_set}` 複製規則組
    7. Admin 頁面：批次匯入/複製規則組改為彈窗、快速邊界包裝按鈕（含說明）、全域規則預覽區、TTS 即時試聽（Web Audio API 串流播放，第一段音頻到即開始播）
    8. Vendor 頁面：比對邊界選擇（英邊界/英數邊界/中英數邊界）、忽略大小寫勾選、TTS 即時試聽（串流播放）、邊界欄位顯示
    9. 四個 TTS 端點（`/tts`、`/tts_url`、`/tts_stream`、`/tts_url_stream`）均支援 `seed` 參數，預設 `seed=2`；相同文字與 seed 可重現相同聲音

- **[2026-04-15]** 新增文字替換規則系統（PostgreSQL）、串流 API、Web 管理介面：
    1. 新增 `/tts_url_stream`、`/tts_stream` 串流端點，首音延遲大幅降低
    2. 新增 PostgreSQL 替換規則系統，支援多組規則（set）管理，繁體輸入自動轉簡體
    3. 新增 `/replacementweb` 管理介面，支援規則增刪改查與批次 JSON 匯入
    4. 新增 `GET /replacements` 列出所有規則組 API
    5. TTS 端點支援 `replacement` 參數指定規則組

- **[2024-08-07]** 支援 Docker 全自動化一鍵部署 API 服務：`docker compose up`

- **[2024-08-06]** 支援 OpenAI 接口格式調用：
    1. 添加 `/audio/speech` API 路徑，相容 OpenAI 接口
    2. 添加 `/audio/voices` API 路徑，獲取 voice/character 列表
    - 對應：[createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## 使用步驟

### 方法一：Docker Compose 部署（強烈推薦）

使用 Docker Compose 可以一鍵部署，無需手動配置環境：

```bash
# 1. Clone 本專案
git clone https://github.com/CreateIntelligens/index-tts-vllm.git
cd index-tts-vllm

# 2. 確保已安裝 Docker 和 Docker Compose

# 3. 複製環境變數配置檔案
cp .env.example .env

# 4. （可選）編輯 .env 檔案，配置模型相關參數
# MODEL=IndexTeam/IndexTTS-1.5
# MODEL_DIR=assets/checkpoints
# PORT=8001
# GPU_MEMORY_UTILIZATION=0.25
# DOWNLOAD_MODEL=1  # 首次啟動時自動下載模型
# CONVERT_MODEL=1   # 自動轉換模型格式

# 5. 啟動服務
docker compose up
```

**Docker 部署的優勢：**
- ✅ 自動下載模型權重（設置 `DOWNLOAD_MODEL=1`）
- ✅ 自動轉換模型格式（設置 `CONVERT_MODEL=1`）
- ✅ 無需手動配置 Python 環境
- ✅ 支援 GPU 加速
- ✅ 日誌自動保存到 `logs/` 目錄

> **注意：** 首次啟動時，如果啟用了自動下載，需要較長時間下載模型（約 3-4 GB）。可以查看 `logs/` 目錄中的日誌檔案以追蹤進度。

### 方法二：手動安裝部署

如果你需要更細緻的控制或開發環境，可以手動安裝：

#### 1. Clone 本專案
```bash
git clone https://github.com/CreateIntelligens/index-tts-vllm.git
cd index-tts-vllm
```

#### 2. 創建並激活 Conda 環境
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```

#### 3. 安裝 PyTorch

優先建議安裝 PyTorch 2.7.0（對應 vLLM 0.9.0），具體安裝指令請參考：[PyTorch 官網](https://pytorch.org/get-started/locally/)

若顯卡不支援，請安裝 PyTorch 2.5.1（對應 vLLM 0.7.3），並將 [requirements.txt](requirements.txt) 中 `vllm==0.9.0` 修改為 `vllm==0.7.3`

#### 4. 安裝依賴套件
```bash
pip install -r requirements.txt
```

#### 5. 下載模型權重

此為官方權重檔案，下載到本地任意路徑即可，支援 IndexTTS-1.5 的權重：

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |

#### 6. 模型權重轉換

將下載的模型權重放置到 `assets/checkpoints/` 目錄下，然後執行轉換腳本：

```bash
bash convert_hf_format.sh assets/checkpoints
```

此操作會將官方的模型權重轉換為 transformers 函式庫相容的版本，保存在模型權重路徑下的 `vllm` 資料夾中，方便後續 vLLM 函式庫加載模型權重。

> **注意：** 如果使用 Docker 部署，模型轉換會在容器啟動時自動完成，無需手動執行此步驟。

#### 7. 啟動 Web UI

將 [`webui.py`](webui.py) 中的 `model_dir` 修改為模型權重路徑（預設為 `assets/checkpoints/`），然後執行：

```bash
VLLM_USE_V1=0 python webui.py
```

第一次啟動可能會久一些，因為要對 BigVGAN 進行 CUDA 核編譯。

**注意：** 一定要帶上 `VLLM_USE_V1=0`，因為本專案沒有對 vLLM 的 v1 版本做相容。


## API 部署

### 方法一：直接執行 Python 腳本

使用 FastAPI 封裝的 API 接口，啟動範例如下：

```bash
VLLM_USE_V1=0 python api_server.py --model_dir assets/checkpoints --port 8001
```

**注意：** 一定要帶上 `VLLM_USE_V1=0`，因為本專案沒有對 vLLM 的 v1 版本做相容。

#### 啟動參數
- `--model_dir`: 模型權重路徑，預設為 `assets/checkpoints`
- `--host`: 服務 IP 位址，預設為 `0.0.0.0`
- `--port`: 服務埠口，預設為 `8001`
- `--gpu_memory_utilization`: vLLM 顯存佔用率，預設設置為 `0.25`

### 方法二：Docker Compose 部署（推薦）

使用 Docker Compose 可以一鍵部署，無需手動配置環境：

```bash
# 1. 確保已安裝 Docker 和 Docker Compose
# 2. 複製環境變數配置檔案
cp .env.example .env

# 3. 編輯 .env 檔案，配置模型相關參數（可選）
# MODEL=IndexTeam/IndexTTS-1.5
# MODEL_DIR=assets/checkpoints
# PORT=8001
# GPU_MEMORY_UTILIZATION=0.25
# DOWNLOAD_MODEL=1  # 首次啟動時自動下載模型
# CONVERT_MODEL=1   # 自動轉換模型格式
# VLLM_USE_MODELSCOPE=1  # 使用 ModelScope 下載（中國地區推薦）

# 4. 啟動服務
docker compose up

# 背景執行
docker compose up -d
```

**Docker 部署的優勢：**
- ✅ 自動下載模型權重（設置 `DOWNLOAD_MODEL=1`）
- ✅ 自動轉換模型格式（設置 `CONVERT_MODEL=1`）
- ✅ 無需手動配置 Python 環境
- ✅ 支援 GPU 加速
- ✅ 日誌自動保存到 `logs/` 目錄
- ✅ 支援 ModelScope 和 HuggingFace 雙源下載

> **注意：** 首次啟動時，如果啟用了自動下載，需要較長時間下載模型（約 3-4 GB）。可以查看 `logs/` 目錄中的日誌檔案追蹤進度。

### API 端點總覽

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| POST | `/tts_url` | TTS 生成（傳入音訊路徑，回傳完整音訊） |
| POST | `/tts` | TTS 生成（使用預註冊角色，回傳完整音訊） |
| POST | `/tts_url_stream` | TTS 串流生成（傳入音訊路徑，邊生成邊回傳） |
| POST | `/tts_stream` | TTS 串流生成（使用預註冊角色，邊生成邊回傳） |
| POST | `/audio/speech` | OpenAI 相容 TTS 接口 |
| GET | `/audio/voices` | 取得可用角色列表 |
| GET | `/replacementweb` | 替換規則 Admin 管理介面（需 Basic Auth） |
| GET | `/vendorweb?set=<name>` | 廠商規則管理頁面（簡化介面，無需驗證） |
| GET | `/replacements` | 列出所有規則組及規則數量（`?hide_global=true` 隱藏 `_global_`） |
| GET | `/replacements/{set_name}` | 查看某組所有規則 |
| POST | `/replacements/{set_name}` | 新增單條規則 |
| PUT | `/replacements/{set_name}/{id}` | 修改某條規則 |
| DELETE | `/replacements/{set_name}/{id}` | 刪除某條規則 |
| GET | `/replacements/{set_name}/export` | 匯出規則組為 JSON 檔案 |
| POST | `/replacements/{set_name}/bulk` | 批次匯入（`?mode=overwrite` 覆蓋，`?mode=merge` 合併） |
| POST | `/replacements/{new_set}/clone/{source_set}` | 複製規則組（`?mode=overwrite\|merge`） |

### API 請求範例

#### 基本 TTS 請求（音訊路徑）
```python
import requests

url = "http://localhost:8001/tts_url"
data = {
    "text": "還是會想你，還是想登你",
    "audio_paths": [  # 支援多參考音訊
        "audio1.wav",
        "audio2.wav"
    ],
    "seed": 2          # 固定種子，預設 2；相同 seed 可重現相同聲音
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 串流 TTS（邊生成邊播放，首音延遲更低）
```python
import requests

url = "http://localhost:8001/tts_url_stream"
data = {
    "text": "還是會想你，還是想登你",
    "audio_paths": ["audio1.wav"]
}

with requests.post(url, json=data, stream=True) as response:
    with open("output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                f.write(chunk)
```

#### 使用預註冊角色
```python
import requests

url = "http://localhost:8001/tts"
data = {
    "text": "你好，這是測試文本",
    "character": "test",  # 使用 assets/speaker.json 中定義的角色
    "seed": 2             # 固定種子，預設 2
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Seed 說明

四個 TTS 端點皆支援 `seed` 參數：

| 端點 | 預設 seed |
|------|----------|
| `/tts` | `2` |
| `/tts_url` | `2` |
| `/tts_stream` | `2` |
| `/tts_url_stream` | `2` |

相同的 `text`、`character`（或 `audio_paths`）與 `seed` 組合，每次呼叫會產生相同的聲音輸出。若需要每次有不同變化，可傳入 `"seed": null`。

#### 使用 OpenAI 格式 API
```python
import requests

url = "http://localhost:8001/audio/speech"
data = {
    "model": "tts-1",
    "input": "這是使用 OpenAI 格式的測試",
    "voice": "test"  # 使用預註冊的角色
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 獲取可用角色列表
```python
import requests

url = "http://localhost:8001/audio/voices"
response = requests.get(url)
print(response.json())
# 輸出: {"voices": ["test", "abin", "ann", "hayley", ...]}
```

### OpenAI API 相容性

本專案支援 OpenAI TTS API 格式，可以直接整合到現有應用中：

- **`/audio/speech`**：相容 OpenAI 的 TTS 接口
- **`/audio/voices`**：獲取可用的 voice/character 列表

詳見：[OpenAI createSpeech API](https://platform.openai.com/docs/api-reference/audio/createSpeech)

### 自定義角色聲線

您可以在 `assets/speaker.json` 中註冊自己的角色聲線：

```json
{
  "my_character": [
    "assets/voices/my_character/voice1.wav",
    "assets/voices/my_character/voice2.wav"
  ],
  "another_character": [
    "assets/voices/another/sample.wav"
  ]
}
```

然後在 API 請求中使用 `"character": "my_character"` 即可。

## 文字替換規則（Replacement Rules）

TTS 生成前可套用文字替換規則，將特定詞彙轉換為較易發音的形式（例如 `ISO 9001` → `iso 九零零一`）。

> **背景說明：** 本服務底層模型吃簡體中文，外部呼叫端傳入文字通常已完成繁→簡轉換。替換規則請依照**傳入 API 時的實際文字**撰寫（即簡體），或透過管理介面輸入繁體，系統會自動轉換。

### 規則執行順序

每次 TTS 請求套用規則時，依以下邏輯執行：

1. **廠商規則組先執行**（該 `set_name` 的所有規則，組內依 pattern 字元長度**由長至短**排列）
2. **`_global_` 全域規則後執行**（同樣依長度由長至短）

**長度降冪的意義**：較長的 pattern 優先比對，避免短規則先拆解長詞導致後續找不到完整詞彙。
**廠商先、全域後的意義**：廠商對某個詞的設定永遠優先；全域規則只處理廠商沒有明確定義的部分。

| 情境 | 廠商規則 | 全域規則 | 輸入 | 結果 |
|------|----------|----------|------|------|
| 廠商覆蓋全域 | `血`→`寫` | `血漿`→`雪漿` | `血漿` | `寫漿`（廠商先命中 `血`） |
| 廠商精確詞 | `血漿`→`寫江` | `血`→`寫` | `血漿` | `寫江`（廠商長詞先命中） |
| 全域兜底 | （無） | `血`→`寫` | `血漿` | `寫漿`（全域處理） |

> **注意**：若廠商與全域規則互相指向對方（A→B 且 B→A），為迴圈衝突，任何順序皆無法解決，需重新設計規則。

### 管理介面

| 頁面 | 網址 | 說明 |
|------|------|------|
| Admin 管理頁 | `http://localhost:8001/replacementweb` | 需 HTTP Basic Auth，可管理所有規則組（含全域） |
| 廠商頁 | `http://localhost:8001/vendorweb?set=<名稱>` | 無需驗證，鎖定單一規則組，簡化介面 |

**Admin 頁功能：**
- 規則新增/編輯/刪除
- 批次 JSON 上傳匯入（覆蓋或合併）、匯出 JSON（彈窗介面）
- 複製規則組（覆蓋或合併，彈窗介面）
- 查看全域規則（`_global_`）預覽區
- Pattern 快速邊界包裝（中文詞邊界／英邊界／英數邊界／中英數邊界／`\b` 詞邊界），點選即高亮，附效果說明
- **TTS 即時試聽**：使用 Web Audio API 串流播放，第一段音頻到達即開始播放，支援隨時停止；自動套用當前載入的規則組

**Vendor 頁功能：**
- 針對指定規則組新增/編輯/刪除純文字規則
- 比對邊界選擇（不加邊界／英邊界／英數邊界／中英數邊界），附效果說明表格
- 忽略大小寫勾選
- 表格顯示各規則套用的邊界類型
- **TTS 即時試聽**：串流播放，自動套用該規則組，支援隨時停止
- 無法存取 `_global_`；`?set=_global_` 會被攔截

### TTS 請求加上替換規則

在任一 TTS 端點的 request body 加入 `replacement` 欄位：

```python
import requests

data = {
    "text": "我们通过了ISO 9001认证",
    "audio_paths": ["assets/voices/xxx.wav"],
    "replacement": "jti"   # 使用名為 "jti" 的規則組
}
response = requests.post("http://localhost:8001/tts_url", json=data)
```

不帶 `replacement` 欄位則不做任何替換。

### 規則管理 API

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/replacements` | 列出所有規則組及各組規則數量 |
| GET | `/replacements/{set_name}` | 查看某組所有規則（回傳原始繁體輸入） |
| POST | `/replacements/{set_name}` | 新增單條規則 |
| PUT | `/replacements/{set_name}/{id}` | 修改某條規則 |
| DELETE | `/replacements/{set_name}/{id}` | 刪除某條規則 |
| GET | `/replacements/{set_name}/export` | 匯出規則組為 JSON 檔案 |
| POST | `/replacements/{set_name}/bulk` | 批次匯入（`?mode=overwrite` 覆蓋，`?mode=merge` 合併） |
| POST | `/replacements/{new_set}/clone/{source_set}` | 複製規則組（`?mode=overwrite\|merge`） |

### 規則格式

```json
{
  "pattern": "ISO 9001",
  "replacement": "iso 九零零一",
  "flags": ["IGNORECASE"],
  "is_regex": false,
  "order_num": 0
}
```

| 欄位 | 說明 |
|------|------|
| `pattern` | 比對字串（繁體輸入，系統自動存繁/簡兩份） |
| `replacement` | 替換結果 |
| `flags` | Regex flags，常用 `IGNORECASE` |
| `is_regex` | `false`（預設）= 純文字比對；`true` = 完整 Regex |
| `order_num` | 同組內次要排序依據（主要排序為 pattern 長度降冪）；相同長度時數字小的先執行 |

**`is_regex: false` 範例（廠商一般用法）：**
```json
{"pattern": "ISO 9001", "replacement": "iso 九零零一"}
```

**`is_regex: true` 範例（進階用法）：**
```json
{
  "pattern": "(?<![A-Za-z0-9])ISO\\s*9001(?![A-Za-z0-9])",
  "replacement": "iso 九零零一",
  "flags": ["IGNORECASE"],
  "is_regex": true
}
```

### 批次匯入範例

```bash
curl -X POST http://localhost:8001/replacements/jti/bulk \
  -H "Content-Type: application/json" \
  -d '[
    {"pattern": "ISO 9001", "replacement": "iso 九零零一", "flags": ["IGNORECASE"]},
    {"pattern": "ODM", "replacement": "歐低M", "flags": ["IGNORECASE"], "is_regex": false}
  ]'
```

### PostgreSQL 連線資訊

規則儲存於 PostgreSQL，預設設定：

| 項目 | 值 |
|------|----|
| Host（外部） | `localhost:5422` |
| Host（容器內） | `postgres:5432` |
| Database | `indextts` |
| User / Password | `indextts` / `indextts` |

可透過 `.env` 檔案覆寫 `DB_NAME`、`DB_USER`、`DB_PASSWORD`。

## 併發測試

參考 [`simple_test.py`](simple_test.py)，需先啟動 API 服務：

```bash
# 基本併發測試
python simple_test.py --url http://localhost:8001/tts --concurrency 16

# 測試多個端點
python simple_test.py --url http://server1:8001/tts http://server2:8001/tts --concurrency 32
```

## 微調模型部署

如果您使用 `index-tts-lora` 訓練了微調模型，請參考 [deploy_finetuned_model.md](deploy_finetuned_model.md) 了解如何將微調模型部署到本專案。

## 常見問題

**Q: Docker 容器啟動後無法訪問 API？**

A: 請確認：
1. 端口映射是否正確（查看 `docker-compose.yaml` 中的 `ports`）
2. 防火牆是否允許該端口
3. 查看容器日誌：`docker compose logs`

**Q: 模型轉換失敗怎麼辦？**

A: 請確認：
1. `config.yaml` 中的 `gpt_checkpoint` 路徑是否正確
2. 所有必需的檔案（`gpt.pth`、`dvae.pth`、`bigvgan_generator.pth`、`bpe.model`）是否存在
3. 查看 `logs/` 目錄中的詳細錯誤訊息

**Q: GPU 顯存不足怎麼辦？**

A: 調低 `.env` 檔案中的 `GPU_MEMORY_UTILIZATION` 參數，建議值：
- 8GB 顯存：`0.15-0.2`
- 12GB 顯存：`0.25-0.3`
- 24GB 顯存：`0.4-0.5`

**Q: 如何使用自己的聲音作為參考？**

A: 
1. 將你的音訊檔案放到 `assets/voices/` 目錄
2. 在 `assets/speaker.json` 中註冊你的角色
3. 使用 API 時指定 `character` 參數

**Q: 支援哪些語言？**

A: 本專案支援中文和英文，模型會自動檢測輸入文本的語言。

## 責任聲明

請參閱 [DISCLAIMER](DISCLAIMER) 和 [LICENSE](LICENSE) 了解使用條款。

## 致謝

- 原始 [index-tts](https://github.com/index-tts/index-tts) 專案
- [vLLM](https://github.com/vllm-project/vllm) 高效推理框架
