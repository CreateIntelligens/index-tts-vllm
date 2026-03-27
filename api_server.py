import os
import asyncio
import io
import struct
import traceback
import tempfile
import uuid
import subprocess
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import numpy as np
import soundfile as sf

from indextts.infer_vllm import IndexTTS

# 🚀 提升音頻處理並行數至 20，消除轉檔排隊瓶頸
audio_processing_semaphore = asyncio.Semaphore(20)

async def convert_audio_with_ffmpeg(input_data, target_sample_rate=16000):
    cmd = [
        'ffmpeg', '-y',
        '-i', 'pipe:0',
        '-ar', str(target_sample_rate),
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        '-f', 'wav',
        'pipe:1'
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_data)
        if process.returncode != 0: return input_data 
        return stdout
    except: return input_data

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS(model_dir=args.model_dir, cfg_path=cfg_path, gpu_memory_utilization=args.gpu_memory_utilization)
    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        def load_speakers():
            with open(speaker_path, 'r') as f: return json.load(f)
        speaker_dict = await asyncio.to_thread(load_speakers)
        for speaker, audio_paths in speaker_dict.items():
            audio_paths_ = [os.path.join(cur_dir, p) for p in audio_paths]
            tts.registry_speaker(speaker, audio_paths_)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def make_wav_header(sample_rate=24000, channels=1, bits_per_sample=16):
    """Build a streaming-friendly WAV header with unknown data size (0xFFFFFFFF)."""
    data_size = 0xFFFFFFFF
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', data_size, b'WAVE',
        b'fmt ', 16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b'data', data_size,
    )

def wav_to_bytes(wav_data, sampling_rate):
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, wav_data, sampling_rate, format='WAV')
        return wav_buffer.getvalue()

@app.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "healthy", "timestamp": time.time()})

@app.post("/tts_url")
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        audio_paths = data.get("audio_paths", [])
        seed = data.get("seed", 8)
        
        print(f"\n--- [TTS_URL Request] ---")
        print(f"Text: {text}")
        print(f"Audio Paths: {audio_paths}")
        print(f"Seed: {seed}")
        print(f"-------------------------\n")

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        async with audio_processing_semaphore:
            wav_bytes = await asyncio.to_thread(wav_to_bytes, wav, sr)
            wav_bytes_16k = await convert_audio_with_ffmpeg(wav_bytes)
            
        return Response(content=wav_bytes_16k, media_type="audio/wav")
    except Exception as ex: return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts")
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        character = data.get("character", "")
        
        print(f"\n--- [TTS Request] ---")
        print(f"Text: {text}")
        print(f"Character: {character}")
        print(f"---------------------\n")

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        async with audio_processing_semaphore:
            wav_bytes = await asyncio.to_thread(wav_to_bytes, wav, sr)
            wav_bytes_16k = await convert_audio_with_ffmpeg(wav_bytes)
            
        return Response(content=wav_bytes_16k, media_type="audio/wav")
    except Exception as ex: return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts_url_stream")
async def tts_url_stream(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        audio_paths = data.get("audio_paths", [])
        seed = data.get("seed", None)

        print(f"\n--- [TTS_URL_STREAM Request] ---")
        print(f"Text: {text}")
        print(f"Audio Paths: {audio_paths}")
        print(f"--------------------------------\n")

        global tts

        async def generate():
            yield make_wav_header(sample_rate=16000)
            async for wav_chunk in tts.infer_stream(audio_paths, text, seed=seed):
                yield wav_chunk.tobytes()

        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={"X-Sample-Rate": "16000", "X-Channels": "1"},
        )
    except Exception as ex:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts_stream")
async def tts_stream(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        character = data.get("character", "")

        print(f"\n--- [TTS_STREAM Request] ---")
        print(f"Text: {text}")
        print(f"Character: {character}")
        print(f"----------------------------\n")

        global tts

        async def generate():
            yield make_wav_header(sample_rate=16000)
            async for wav_chunk in tts.infer_with_ref_audio_embed_stream(character, text):
                yield wav_chunk.tobytes()

        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={"X-Sample-Rate": "16000", "X-Channels": "1"},
        )
    except Exception as ex:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.get("/")
async def frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/audio/voices")
async def tts_voices():
    speaker_path = os.path.join(os.path.dirname(__file__), "assets/speaker.json")
    if os.path.exists(speaker_path): return json.load(open(speaker_path, 'r'))
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11996)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.1)
    args = parser.parse_args()
    uvicorn.run(app=app, host=args.host, port=args.port)
