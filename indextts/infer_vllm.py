import os
import re
import time
import asyncio
from subprocess import CalledProcessError
import traceback
from typing import List

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model_vllm import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer

import matplotlib.pyplot as plt

def trim_and_pad_silence(wav_data, threshold=1000, min_silence=int(24000*0.4)):
    abs_trimmed = np.abs(wav_data).flatten()
    if len(abs_trimmed) == 0: return wav_data
    last_non_silent = len(abs_trimmed) - np.argmax(abs_trimmed[::-1] >= threshold)
    back_silence_length = len(wav_data) - last_non_silent
    if back_silence_length < min_silence:
        pad_length = min_silence - back_silence_length
        padded = np.vstack([wav_data, np.zeros((pad_length, 1))])
    else:
        padded = wav_data
    return padded.astype(np.int16)


class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", gpu_memory_utilization=0.25, is_fp16=True, device=None, use_cuda_kernel=None,
    ):
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.gpt = UnifiedVoice(gpu_memory_utilization, **self.cfg.gpt, model_dir=model_dir)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device).eval()

        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        self.bpe_path = os.path.join(self.model_dir, "bpe.model")
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)

        self.speaker_dict = {}
        self.conditioning_cache = {}
        self.bigvgan_lock = asyncio.Lock()

    async def infer(self, audio_prompt: List[str], text, output_path=None, verbose=False, seed=None):
        print(">> start inference...")
        start_time = time.perf_counter()
        
        abs_prompts = tuple(sorted([os.path.abspath(p) for p in audio_prompt]))
        if abs_prompts in self.conditioning_cache:
            auto_conditioning, speech_conditioning_latent = self.conditioning_cache[abs_prompts]
            print(">> Conditioning Cache Hit!")
        else:
            auto_conditioning = []
            for ap_ in audio_prompt:
                audio, sr = torchaudio.load(ap_)
                audio = torch.mean(audio, dim=0, keepdim=True)
                audio = torchaudio.transforms.Resample(sr, 24000)(audio)
                cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
                auto_conditioning.append(cond_mel)

            speech_conditioning_latent = []
            for cond_mel in auto_conditioning:
                speech_conditioning_latent_ = self.gpt.get_conditioning(
                    cond_mel, torch.tensor([cond_mel.shape[-1]], device=self.device)
                )
                speech_conditioning_latent.append(speech_conditioning_latent_)
            speech_conditioning_latent = torch.stack(speech_conditioning_latent).sum(dim=0) / len(auto_conditioning)
            
            if len(self.conditioning_cache) < 50:
                self.conditioning_cache[abs_prompts] = (auto_conditioning, speech_conditioning_latent)

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        wavs = []
        gpt_gen_time = 0
        bigvgan_time = 0
        
        for sent in sentences:
            text_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), dtype=torch.int32, device=self.device).unsqueeze(0)
            
            m_start = time.perf_counter()
            with torch.no_grad():
                self.gpt.sampling_params.seed = int(seed) if seed is not None else None
                codes, _ = await self.gpt.inference_speech(speech_conditioning_latent, text_tokens)
                
                codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
                latent = self.gpt(speech_conditioning_latent, text_tokens,
                                torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                torch.tensor([codes.shape[-1]], device=codes.device) * self.gpt.mel_length_compression,
                                cond_mel_lengths=torch.tensor([speech_conditioning_latent.shape[-1]], device=text_tokens.device),
                                return_latent=True, clip_inputs=False)
                gpt_gen_time += time.perf_counter() - m_start

                m_start = time.perf_counter()
                async with self.bigvgan_lock:
                    def run_bigvgan():
                        with torch.no_grad():
                            v, _ = self.bigvgan(latent, [ap_.transpose(1, 2) for ap_ in auto_conditioning])
                            # 🚀 修正處：補回縮放係數與裁切
                            v = torch.clamp(32767 * v.squeeze(1), -32767.0, 32767.0)
                            return v.cpu()
                    wav = await asyncio.to_thread(run_bigvgan)
                bigvgan_time += time.perf_counter() - m_start
                wavs.append(wav)

        wav = torch.cat(wavs, dim=1)
        end_time = time.perf_counter()
        
        wav_len = wav.shape[-1] / 24000
        print(f">> gpt_gen_time: {gpt_gen_time:.2f}s | bigvgan_time: {bigvgan_time:.2f}s")
        print(f">> Total: {end_time - start_time:.2f}s | RTF: {(end_time - start_time)/wav_len:.4f}")

        wav_data = wav.type(torch.int16).numpy().T
        wav_data = trim_and_pad_silence(wav_data)
        return (24000, wav_data)

    async def infer_with_ref_audio_embed(self, speaker: str, text):
        start_time = time.perf_counter()
        if speaker not in self.speaker_dict: raise Exception(f"Speaker {speaker} not found")
        auto_conditioning = self.speaker_dict[speaker]["auto_conditioning"]
        speech_conditioning_latent = self.speaker_dict[speaker]["speech_conditioning_latent"]
        
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        wavs = []
        gpt_gen_time = 0
        bigvgan_time = 0

        for sent in sentences:
            text_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), dtype=torch.int32, device=self.device).unsqueeze(0)
            m_start = time.perf_counter()
            with torch.no_grad():
                codes, _ = await self.gpt.inference_speech(speech_conditioning_latent, text_tokens)
                codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
                latent = self.gpt(speech_conditioning_latent, text_tokens,
                                torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                torch.tensor([codes.shape[-1]], device=codes.device) * self.gpt.mel_length_compression,
                                cond_mel_lengths=torch.tensor([speech_conditioning_latent.shape[-1]], device=text_tokens.device),
                                return_latent=True, clip_inputs=False)
                gpt_gen_time += time.perf_counter() - m_start
                
                m_start = time.perf_counter()
                async with self.bigvgan_lock:
                    def run_bigvgan():
                        with torch.no_grad():
                            v, _ = self.bigvgan(latent, [ap_.transpose(1, 2) for ap_ in auto_conditioning])
                            # 🚀 修正處：補回縮放係數與裁切
                            v = torch.clamp(32767 * v.squeeze(1), -32767.0, 32767.0)
                            return v.cpu()
                    wav = await asyncio.to_thread(run_bigvgan)
                bigvgan_time += time.perf_counter() - m_start
                wavs.append(wav)
        
        wav = torch.cat(wavs, dim=1)
        end_time = time.perf_counter()
        wav_len = wav.shape[-1] / 24000
        print(f">> [Speaker] gpt_gen_time: {gpt_gen_time:.2f}s | bigvgan_time: {bigvgan_time:.2f}s")
        print(f">> [Speaker] Total: {end_time - start_time:.2f}s | RTF: {(end_time - start_time)/wav_len:.4f}")

        wav_data = wav.type(torch.int16).numpy().T
        wav_data = trim_and_pad_silence(wav_data)
        return (24000, wav_data)

    @torch.no_grad()
    def registry_speaker(self, speaker: str, audio_paths: List[str]):
        auto_conditioning = []
        for ap_ in audio_paths:
            audio, sr = torchaudio.load(ap_)
            audio = torch.mean(audio, dim=0, keepdim=True)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            auto_conditioning.append(cond_mel)

        speech_conditioning_latent = []
        for cond_mel in auto_conditioning:
            speech_conditioning_latent_ = self.gpt.get_conditioning(
                cond_mel, torch.tensor([cond_mel.shape[-1]], device=self.device)
            )
            speech_conditioning_latent.append(speech_conditioning_latent_)
        speech_conditioning_latent = torch.stack(speech_conditioning_latent).sum(dim=0) / len(auto_conditioning)

        self.speaker_dict[speaker] = {"auto_conditioning": auto_conditioning, "speech_conditioning_latent": speech_conditioning_latent}
        print(f"Speaker: {speaker} registered")
