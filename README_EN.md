<a href="README.md">中文</a> ｜ English

<div align="center">

# IndexTTS-vLLM
</div>

## Introduction
This project reimplements the inference of the GPT model from [index-tts](https://github.com/index-tts/index-tts) using the vllm library, accelerating the inference process of index-tts.

The inference speed improvement on a single RTX 4090 is as follows:
- Real-Time Factor (RTF) for a single request: ≈0.3 -> ≈0.1
- GPT model decode speed for a single request: ≈90 tokens/s -> ≈280 tokens/s
- Concurrency: With `gpu_memory_utilization` set to 0.5 (about 12GB of VRAM), vllm shows `Maximum concurrency for 608 tokens per request: 237.18x`. That's over 200 concurrent requests, man! Of course, considering TTFT and other inference costs (bigvgan, etc.), a concurrency of around 16 was tested without pressure (refer to `simple_test.py` for the speed test script).

## New Features
- **Support for multi-character audio mixing**: You can input multiple reference audios, and the TTS output voice will be a mixed version of the reference audios. (Inputting multiple reference audios may lead to an unstable output voice; you can try multiple times to get a satisfactory voice and then use it as a reference audio).

## Performance
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm          | 1.12  | 1.987 |

The performance is basically on par with the original project.

## Update Log

- **[2024-08-07]** Added support for fully automated one-click deployment of the API service using Docker: `docker compose up`

- **[2024-08-06]** Added support for OpenAI API format calls:
    1. Added `/audio/speech` API path to be compatible with the OpenAI interface.
    2. Added `/audio/voices` API path to get the list of voices/characters.
    - Corresponds to: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Usage Steps

### 1. Clone this project
```bash
git clone https://github.com/CreateIntelligens/index-tts-vllm.git
cd index-tts-vllm
```


### 2. Create and activate a conda environment
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```


### 3. Install PyTorch

It is recommended to install PyTorch 2.7.0 (corresponding to vllm 0.9.0). Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for specific installation instructions.

If your graphics card does not support it, please install PyTorch 2.5.1 (corresponding to vllm 0.7.3) and change `vllm==0.9.0` to `vllm==0.7.3` in [requirements.txt](requirements.txt).


### 4. Install dependencies
```bash
pip install -r requirements.txt
```


### 5. Download model weights

These are the official weight files. Download them to any local path. Weights for IndexTTS-1.5 are supported.

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |

### 6. Convert model weights

Place the downloaded model weights in the `assets/checkpoints/` directory, then run the conversion script:

```bash
bash convert_hf_format.sh assets/checkpoints
```

This operation will convert the official model weights to a format compatible with the transformers library, saving them in the `vllm` folder under the model weight path, which facilitates subsequent loading of model weights by the vllm library.

> **Note:** If using Docker deployment (see Docker Deployment section below), the model conversion will be completed automatically when the container starts, and you do not need to manually execute this step.

### 7. Launch the web UI!
Modify the `model_dir` in [`webui.py`](webui.py) to your model weight path (default is `assets/checkpoints/`), and then run:

```bash
VLLM_USE_V1=0 python webui.py
```
The first launch might take a while because it needs to compile the CUDA kernel for bigvgan.

Note: You must include `VLLM_USE_V1=0`, as this project is not compatible with v1 of vllm.


## API Deployment

### Method 1: Run Python Script Directly

The API is encapsulated using FastAPI. Here is an example of how to start it:

```bash
VLLM_USE_V1=0 python api_server.py --model_dir assets/checkpoints --port 8011
```

Note: You must include `VLLM_USE_V1=0`, as this project is not compatible with v1 of vllm.

### Startup Parameters
- `--model_dir`: Model weight path, default is `assets/checkpoints`
- `--host`: Service IP address, default is `0.0.0.0`
- `--port`: Service port, default is `8011`
- `--gpu_memory_utilization`: vllm GPU memory utilization rate, default is `0.25`

### Method 2: Docker Compose Deployment (Recommended)

Using Docker Compose allows for one-click deployment without manual environment configuration:

```bash
# 1. Ensure Docker and Docker Compose are installed
# 2. Copy the environment variable configuration file
cp .env.example .env

# 3. Edit the .env file to configure model-related parameters (optional)
# MODEL_DIR=assets/checkpoints
# PORT=8011
# GPU_MEMORY_UTILIZATION=0.25
# DOWNLOAD_MODEL=1  # Automatically download model on first start
# CONVERT_MODEL=1   # Automatically convert model format

# 4. Start the service
docker compose up
```

Advantages of Docker deployment:
- ✅ Automatically download model weights (set `DOWNLOAD_MODEL=1`)
- ✅ Automatically convert model format (set `CONVERT_MODEL=1`)
- ✅ No need to manually configure Python environment
- ✅ GPU acceleration support
- ✅ Logs automatically saved to `logs/` directory

> **Note:** On the first startup, if automatic download is enabled, it will take a considerable amount of time to download the model (approximately 3-4 GB).

### Request Examples

#### Basic TTS Request
```python
import requests

url = "http://localhost:8011/tts_url"
data = {
    "text": "Still thinking of you, still want to see you.",
    "audio_paths": [  # Supports multiple reference audios
        "audio1.wav",
        "audio2.wav"
    ]
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Using Pre-registered Characters
```python
import requests

url = "http://localhost:8011/tts"
data = {
    "text": "Hello, this is test text",
    "character": "test"  # Use character defined in assets/speaker.json
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### OpenAI API
- Added `/audio/speech` API path to be compatible with the OpenAI interface.
- Added `/audio/voices` API path to get the list of voices/characters.

For details, see: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## Concurrency Test
Refer to [`simple_test.py`](simple_test.py). You need to start the API service first.
