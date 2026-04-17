import os
import re
import base64
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
import asyncpg
import opencc

from indextts.infer_vllm import IndexTTS

_t2s = opencc.OpenCC('t2s')

def to_hans(text: str) -> str:
    return _t2s.convert(text)

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
db_pool = None

async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS replacement_rules (
                id                   SERIAL PRIMARY KEY,
                set_name             VARCHAR(100) NOT NULL,
                pattern_original     TEXT NOT NULL,
                pattern_hans         TEXT NOT NULL,
                replacement_original TEXT NOT NULL,
                replacement_hans     TEXT NOT NULL,
                flags                TEXT[] DEFAULT '{}',
                is_regex             BOOLEAN DEFAULT FALSE,
                order_num            INT DEFAULT 0,
                created_at           TIMESTAMP DEFAULT NOW(),
                updated_at           TIMESTAMP DEFAULT NOW()
            )
        """)
        # 欄位遷移（若已存在舊 schema）
        for col, definition in [
            ("pattern_original",     "TEXT NOT NULL DEFAULT ''"),
            ("pattern_hans",         "TEXT NOT NULL DEFAULT ''"),
            ("replacement_original", "TEXT NOT NULL DEFAULT ''"),
            ("replacement_hans",     "TEXT NOT NULL DEFAULT ''"),
            ("is_regex",             "BOOLEAN DEFAULT FALSE"),
        ]:
            await conn.execute(f"""
                ALTER TABLE replacement_rules
                ADD COLUMN IF NOT EXISTS {col} {definition}
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_replacement_rules_set_name
            ON replacement_rules(set_name, order_num)
        """)

async def apply_replacements(set_name: str, text: str) -> str:
    if not db_pool:
        return text

    async def _apply_rows(rows, t: str) -> str:
        for row in rows:
            flag_val = 0
            for f in (row['flags'] or []):
                flag_val |= getattr(re, f, 0)
            t = re.sub(row['pattern_hans'], row['replacement_hans'], t, flags=flag_val)
        return t

    async with db_pool.acquire() as conn:
        if set_name and set_name != '_global_':
            # 廠商組先跑（組內長度降冪），全域組後跑（組內長度降冪）
            # 廠商先佔位，全域只處理廠商沒動到的部分
            vendor_rows = await conn.fetch(
                "SELECT pattern_hans, replacement_hans, flags FROM replacement_rules "
                "WHERE set_name = $1 "
                "ORDER BY LENGTH(pattern_hans) DESC, order_num, id",
                set_name
            )
            global_rows = await conn.fetch(
                "SELECT pattern_hans, replacement_hans, flags FROM replacement_rules "
                "WHERE set_name = '_global_' "
                "ORDER BY LENGTH(pattern_hans) DESC, order_num, id"
            )
            text = await _apply_rows(vendor_rows, text)
            text = await _apply_rows(global_rows, text)
        else:
            rows = await conn.fetch(
                "SELECT pattern_hans, replacement_hans, flags FROM replacement_rules "
                "WHERE set_name = $1 "
                "ORDER BY LENGTH(pattern_hans) DESC, order_num, id",
                set_name or '_global_'
            )
            text = await _apply_rows(rows, text)
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, db_pool
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
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'indextts')}:{os.getenv('DB_PASSWORD', 'indextts')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME', 'indextts')}"
    )
    db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
    await init_db(db_pool)
    yield
    await db_pool.close()

def _check_admin_auth(request: Request) -> bool:
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin")
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        user, password = decoded.split(":", 1)
        return user == admin_user and password == admin_pass
    except Exception:
        return False

def _require_admin_auth(request: Request):
    if not _check_admin_auth(request):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Admin"'},
            content="Unauthorized",
        )
    return None

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
        seed = data.get("seed", 2)
        replacement_set = data.get("replacement", None)

        print(f"\n--- [TTS_URL Request] ---")
        print(f"Text: {text}")
        print(f"Audio Paths: {audio_paths}")
        print(f"Seed: {seed}")
        print(f"Replacement Set: {replacement_set}")
        print(f"-------------------------\n")

        text = await apply_replacements(replacement_set, text)
        if replacement_set:
            print(f"[After Replacement] Text: {text}\n")

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        async with audio_processing_semaphore:
            wav_bytes = await asyncio.to_thread(wav_to_bytes, wav, sr)
            wav_bytes_16k = await convert_audio_with_ffmpeg(wav_bytes)
            
        return Response(content=wav_bytes_16k, media_type="audio/wav")
    except Exception as ex:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts")
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        character = data.get("character", "")
        seed = data.get("seed", 2)
        replacement_set = data.get("replacement", None)

        print(f"\n--- [TTS Request] ---")
        print(f"Text: {text}")
        print(f"Character: {character}")
        print(f"Seed: {seed}")
        print(f"Replacement Set: {replacement_set}")
        print(f"---------------------\n")

        text = await apply_replacements(replacement_set, text)
        if replacement_set:
            print(f"[After Replacement] Text: {text}\n")

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text, seed=seed)
        
        async with audio_processing_semaphore:
            wav_bytes = await asyncio.to_thread(wav_to_bytes, wav, sr)
            wav_bytes_16k = await convert_audio_with_ffmpeg(wav_bytes)
            
        return Response(content=wav_bytes_16k, media_type="audio/wav")
    except Exception as ex:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts_url_stream")
async def tts_url_stream(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        audio_paths = data.get("audio_paths", [])
        seed = data.get("seed", 2)
        replacement_set = data.get("replacement", None)

        print(f"\n--- [TTS_URL_STREAM Request] ---")
        print(f"Text: {text}")
        print(f"Audio Paths: {audio_paths}")
        print(f"Replacement Set: {replacement_set}")
        print(f"--------------------------------\n")

        text = await apply_replacements(replacement_set, text)
        if replacement_set:
            print(f"[After Replacement] Text: {text}\n")

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
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/tts_stream")
async def tts_stream(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        character = data.get("character", "")
        seed = data.get("seed", 2)
        replacement_set = data.get("replacement", None)

        print(f"\n--- [TTS_STREAM Request] ---")
        print(f"Text: {text}")
        print(f"Character: {character}")
        print(f"Seed: {seed}")
        print(f"Replacement Set: {replacement_set}")
        print(f"----------------------------\n")

        text = await apply_replacements(replacement_set, text)
        if replacement_set:
            print(f"[After Replacement] Text: {text}\n")

        global tts

        async def generate():
            yield make_wav_header(sample_rate=16000)
            async for wav_chunk in tts.infer_with_ref_audio_embed_stream(character, text, seed=seed):
                yield wav_chunk.tobytes()

        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={"X-Sample-Rate": "16000", "X-Channels": "1"},
        )
    except Exception as ex:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.get("/")
async def frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/replacementweb")
async def replacement_web(request: Request):
    denied = _require_admin_auth(request)
    if denied:
        return denied
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replacement_web.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/vendorweb")
async def vendor_web():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_web.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/replacements")
async def list_sets(hide_global: bool = False):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT set_name, COUNT(*) AS rule_count "
            "FROM replacement_rules GROUP BY set_name ORDER BY set_name"
        )
    result = [dict(r) for r in rows]
    if hide_global:
        result = [r for r in result if r["set_name"] != "_global_"]
    return result

def build_pattern_hans(pattern_orig: str, is_regex: bool) -> str:
    """is_regex=False 時用 re.escape 轉成字面比對；True 時直接轉簡體保留 regex 語法。"""
    if is_regex:
        return to_hans(pattern_orig)
    return re.escape(to_hans(pattern_orig))

@app.get("/replacements/{set_name}")
async def list_rules(set_name: str):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, set_name, pattern_original AS pattern, replacement_original AS replacement, "
            "flags, is_regex, order_num "
            "FROM replacement_rules WHERE set_name = $1 ORDER BY order_num, id",
            set_name
        )
    return [dict(r) for r in rows]

@app.post("/replacements/{set_name}")
async def add_rule(set_name: str, request: Request):
    data = await request.json()
    pattern_orig = data["pattern"]
    replacement_orig = data["replacement"]
    is_regex = data.get("is_regex", False)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO replacement_rules "
            "(set_name, pattern_original, pattern_hans, replacement_original, replacement_hans, flags, is_regex, order_num) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
            "RETURNING id, set_name, pattern_original AS pattern, replacement_original AS replacement, flags, is_regex, order_num",
            set_name,
            pattern_orig,
            build_pattern_hans(pattern_orig, is_regex),
            replacement_orig,
            to_hans(replacement_orig),
            data.get("flags", []),
            is_regex,
            data.get("order_num", 0),
        )
    return dict(row)

@app.put("/replacements/{set_name}/{rule_id}")
async def update_rule(set_name: str, rule_id: int, request: Request):
    data = await request.json()
    pattern_orig = data["pattern"]
    replacement_orig = data["replacement"]
    is_regex = data.get("is_regex", False)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE replacement_rules "
            "SET pattern_original=$1, pattern_hans=$2, replacement_original=$3, replacement_hans=$4, "
            "flags=$5, is_regex=$6, order_num=$7, updated_at=NOW() "
            "WHERE id=$8 AND set_name=$9 "
            "RETURNING id, set_name, pattern_original AS pattern, replacement_original AS replacement, flags, is_regex, order_num",
            pattern_orig,
            build_pattern_hans(pattern_orig, is_regex),
            replacement_orig,
            to_hans(replacement_orig),
            data.get("flags", []),
            is_regex,
            data.get("order_num", 0),
            rule_id,
            set_name,
        )
    if not row:
        return JSONResponse(status_code=404, content={"error": "rule not found"})
    return dict(row)

@app.delete("/replacements/{set_name}/{rule_id}")
async def delete_rule(set_name: str, rule_id: int):
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM replacement_rules WHERE id=$1 AND set_name=$2",
            rule_id, set_name
        )
    deleted = int(result.split()[-1])
    if deleted == 0:
        return JSONResponse(status_code=404, content={"error": "rule not found"})
    return {"deleted": rule_id}

@app.get("/replacements/{set_name}/export")
async def export_rules(set_name: str):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT pattern_original AS pattern, replacement_original AS replacement, "
            "flags, is_regex, order_num "
            "FROM replacement_rules WHERE set_name = $1 ORDER BY order_num, id",
            set_name
        )
    data = [dict(r) for r in rows]
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{set_name}_rules.json"'},
    )

@app.post("/replacements/{set_name}/bulk")
async def bulk_import_rules(set_name: str, request: Request, mode: str = "overwrite"):
    rules = await request.json()
    async with db_pool.acquire() as conn:
        if mode == "overwrite":
            await conn.execute("DELETE FROM replacement_rules WHERE set_name=$1", set_name)
            start_order = 0
        else:
            existing = await conn.fetch(
                "SELECT pattern_original FROM replacement_rules WHERE set_name=$1", set_name
            )
            existing_patterns = {r["pattern_original"] for r in existing}
            rules = [r for r in rules if r["pattern"] not in existing_patterns]
            max_order = await conn.fetchval(
                "SELECT COALESCE(MAX(order_num), -1) FROM replacement_rules WHERE set_name=$1", set_name
            )
            start_order = max_order + 1

        await conn.executemany(
            "INSERT INTO replacement_rules "
            "(set_name, pattern_original, pattern_hans, replacement_original, replacement_hans, flags, is_regex, order_num) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            [
                (
                    set_name,
                    r["pattern"],
                    build_pattern_hans(r["pattern"], r.get("is_regex", False)),
                    r["replacement"],
                    to_hans(r["replacement"]),
                    r.get("flags", []),
                    r.get("is_regex", False),
                    start_order + i,
                )
                for i, r in enumerate(rules)
            ]
        )
    return {"imported": len(rules), "set_name": set_name, "mode": mode}

@app.post("/replacements/{new_set}/clone/{source_set}")
async def clone_rules(new_set: str, source_set: str, mode: str = "overwrite"):
    async with db_pool.acquire() as conn:
        source_rows = await conn.fetch(
            "SELECT pattern_original, pattern_hans, replacement_original, replacement_hans, "
            "flags, is_regex, order_num "
            "FROM replacement_rules WHERE set_name = $1 ORDER BY order_num, id",
            source_set
        )
        if not source_rows:
            return JSONResponse(status_code=404, content={"error": f"source set '{source_set}' not found or empty"})

        if mode == "overwrite":
            await conn.execute("DELETE FROM replacement_rules WHERE set_name=$1", new_set)
            rows_to_insert = source_rows
            start_order = 0
        else:
            existing = await conn.fetch(
                "SELECT pattern_original FROM replacement_rules WHERE set_name=$1", new_set
            )
            existing_patterns = {r["pattern_original"] for r in existing}
            rows_to_insert = [r for r in source_rows if r["pattern_original"] not in existing_patterns]
            max_order = await conn.fetchval(
                "SELECT COALESCE(MAX(order_num), -1) FROM replacement_rules WHERE set_name=$1", new_set
            )
            start_order = max_order + 1

        await conn.executemany(
            "INSERT INTO replacement_rules "
            "(set_name, pattern_original, pattern_hans, replacement_original, replacement_hans, flags, is_regex, order_num) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            [
                (
                    new_set,
                    r["pattern_original"],
                    r["pattern_hans"],
                    r["replacement_original"],
                    r["replacement_hans"],
                    list(r["flags"]) if r["flags"] else [],
                    r["is_regex"],
                    start_order + i,
                )
                for i, r in enumerate(rows_to_insert)
            ]
        )
    return {"cloned": len(rows_to_insert), "source": source_set, "target": new_set, "mode": mode}

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
