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
# PORT=8011
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
VLLM_USE_V1=0 python api_server.py --model_dir assets/checkpoints --port 8011
```

**注意：** 一定要帶上 `VLLM_USE_V1=0`，因為本專案沒有對 vLLM 的 v1 版本做相容。

#### 啟動參數
- `--model_dir`: 模型權重路徑，預設為 `assets/checkpoints`
- `--host`: 服務 IP 位址，預設為 `0.0.0.0`
- `--port`: 服務埠口，預設為 `8011`
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
# PORT=8011
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

### API 請求範例

#### 基本 TTS 請求
```python
import requests

url = "http://localhost:8011/tts_url"
data = {
    "text": "還是會想你，還是想登你",
    "audio_paths": [  # 支援多參考音訊
        "audio1.wav",
        "audio2.wav"
    ]
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 使用預註冊角色
```python
import requests

url = "http://localhost:8011/tts"
data = {
    "text": "你好，這是測試文本",
    "character": "test"  # 使用 assets/speaker.json 中定義的角色
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 使用 OpenAI 格式 API
```python
import requests

url = "http://localhost:8011/audio/speech"
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

url = "http://localhost:8011/audio/voices"
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

## 併發測試

參考 [`simple_test.py`](simple_test.py)，需先啟動 API 服務：

```bash
# 基本併發測試
python simple_test.py --url http://localhost:8011/tts --concurrency 16

# 測試多個端點
python simple_test.py --url http://server1:8011/tts http://server2:8011/tts --concurrency 32
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
