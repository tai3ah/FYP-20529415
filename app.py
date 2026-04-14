import re
import time
import uuid
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
import librosa
import soundfile as sf
import noisereduce as nr

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from TTS.api import TTS
from resemblyzer import VoiceEncoder, preprocess_wav

# =========================================================
# Config
# =========================================================
BASE_MODEL_ID = "openai/whisper-medium"
LORA_DIR = Path("final_lora")

PIPER_DIR = Path("piper_models")
PIPER_VOICE = "en_US-lessac-medium"
PIPER_MODEL = PIPER_DIR / f"{PIPER_VOICE}.onnx"
PIPER_CFG = PIPER_DIR / f"{PIPER_VOICE}.onnx.json"

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

XTTS_TEMPERATURE = 0.60
XTTS_REPETITION_PENALTY = 5.0
XTTS_TOP_K = 50
XTTS_TOP_P = 0.85
XTTS_SPEED = 1.0
MAX_CHUNK_CHARS = 180

WHISPER_MAX_SECONDS = 60
XTTS_REF_MAX_SECONDS = 30
WHISPER_SR = 16000
XTTS_SR = 22050
XTTS_OUTPUT_SR = 24000

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("outputs/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = "ffmpeg"

SHEET_ID = "1C2ZFxtJ2H4TwnakoV_buVzlNIOkZ2GnBO3o9sIXHR2I"
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScrM4CSuzdfhGflaliiDkBLl4vasHCqA3MQcVI_nS5ZglkeCw/viewform"

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

ST_PROC = None
ST_MODEL = None
XTTS_MODEL = None
TRANSLATION_READY = False
XTTS_READY = False


# =========================================================
# Audio utilities
# =========================================================
def ffmpeg_convert_to_wav(input_path: str, output_path: str, sr: int) -> None:
    cmd = [FFMPEG_BIN, "-y", "-i", input_path, "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio_for_whisper(audio_path: str) -> Tuple[np.ndarray, int]:
    converted = str(TEMP_DIR / f"whisper_{uuid.uuid4().hex}.wav")
    ffmpeg_convert_to_wav(audio_path, converted, sr=WHISPER_SR)
    audio, _ = librosa.load(converted, sr=WHISPER_SR, mono=True)
    max_samples = WHISPER_MAX_SECONDS * WHISPER_SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio, WHISPER_SR


def prepare_xtts_reference(audio_path: str) -> str:
    converted = str(TEMP_DIR / f"xtts_ref_raw_{uuid.uuid4().hex}.wav")
    ffmpeg_convert_to_wav(audio_path, converted, sr=XTTS_SR)
    audio, _ = librosa.load(converted, sr=XTTS_SR, mono=True)
    max_samples = XTTS_REF_MAX_SECONDS * XTTS_SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if len(audio) > XTTS_SR:
        noise_clip = audio[:int(0.5 * XTTS_SR)]
    else:
        noise_clip = audio[:max(1, len(audio) // 4)]
    try:
        audio = nr.reduce_noise(y=audio, sr=XTTS_SR, y_noise=noise_clip, prop_decrease=0.75, stationary=False)
    except Exception:
        pass
    out_path = str(TEMP_DIR / f"xtts_ref_clean_{uuid.uuid4().hex}.wav")
    sf.write(out_path, audio, XTTS_SR)
    return out_path


def persist_audio_file(audio_path: str) -> Optional[str]:
    if audio_path is None or not Path(audio_path).exists():
        return None
    src = Path(audio_path)
    dst = TEMP_DIR / f"ui_audio_{uuid.uuid4().hex}{src.suffix or '.wav'}"
    dst.write_bytes(src.read_bytes())
    return str(dst)


# =========================================================
# Text utilities
# =========================================================
def normalise_text_for_tts(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_text_for_xtts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    text = str(text).strip()
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            words, temp = chunk.split(), ""
            for word in words:
                if not temp:
                    temp = word
                elif len(temp) + 1 + len(word) <= max_chars:
                    temp += " " + word
                else:
                    final_chunks.append(temp.strip())
                    temp = word
            if temp:
                final_chunks.append(temp.strip())
    return final_chunks


# =========================================================
# Model loading
# =========================================================
def validate_piper_assets() -> Optional[str]:
    if not (PIPER_MODEL.exists() and PIPER_CFG.exists()):
        return f"Piper voice files missing at {PIPER_DIR}"
    return None


def load_translation_pipeline() -> str:
    global ST_PROC, ST_MODEL, TRANSLATION_READY
    if TRANSLATION_READY:
        return "Translation model already loaded."
    if not LORA_DIR.exists():
        raise RuntimeError(f"LoRA folder not found at: {LORA_DIR}")
    t0 = time.time()
    ST_PROC = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, low_cpu_mem_usage=True).to(DEVICE)
    base_model.eval()
    ST_MODEL = PeftModel.from_pretrained(base_model, str(LORA_DIR)).to(DEVICE)
    ST_MODEL.eval()
    TRANSLATION_READY = True
    return f"Translation model loaded in {time.time() - t0:.2f}s."


def load_xtts_model() -> str:
    global XTTS_MODEL, XTTS_READY
    if XTTS_READY:
        return "XTTS model already loaded."
    t0 = time.time()
    XTTS_MODEL = TTS(XTTS_MODEL_NAME).to(XTTS_DEVICE)
    XTTS_READY = True
    return f"XTTS loaded in {time.time() - t0:.2f}s."


def initialise_models() -> str:
    messages = []
    try:
        messages.append(load_translation_pipeline())
    except Exception as e:
        messages.append(f"Translation init failed: {e}")
    try:
        err = validate_piper_assets()
        messages.append("Piper TTS assets found." if not err else f"Piper: {err}")
    except Exception as e:
        messages.append(f"Piper check failed: {e}")
    try:
        messages.append(load_xtts_model())
    except Exception as e:
        messages.append(f"XTTS init failed: {e}")
    return "\n".join(messages)


def background_init():
    print("Background model initialisation started...")
    msg = initialise_models()
    print(f"Models ready:\n{msg}")


def _render_model_status() -> str:
    if TRANSLATION_READY and XTTS_READY:
        return """<div class="sb-model-status sb-model-ready">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;">
                <circle cx="7" cy="7" r="6" stroke="#22c55e" stroke-width="1.5"/>
                <path d="M4.5 7l2 2 3-3" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            All models ready
        </div>"""
    return """<div class="sb-model-status sb-model-loading">
        <span class="sb-spin">◌</span> Initialising models in background…
    </div>"""


def poll_model_status():
    ready = TRANSLATION_READY and XTTS_READY
    return _render_model_status(), gr.Button(visible=not ready), gr.Timer(active=not ready)


def do_init_models():
    initialise_models()
    return _render_model_status(), gr.Button(visible=False)


# =========================================================
# Inference
# =========================================================
LANG_MAP = {
    "French (fine-tuned ★)": "french",
    "Other languages":       "other",
}


@torch.inference_mode()
def translate_to_en(audio_16k: np.ndarray, sr: int, source_lang: str = "french") -> str:
    inputs = ST_PROC(audio_16k, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(DEVICE, dtype=TORCH_DTYPE)

    if source_lang == "french":
        # LoRA fine-tuned model — French is the fine-tuned language
        forced_decoder_ids = ST_PROC.get_decoder_prompt_ids(language="french", task="translate")
        gen_ids = ST_MODEL.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=1,
            max_new_tokens=192
        )
    else:
        # Base Whisper-medium with LoRA disabled — auto-detect language, translate to English
        ST_MODEL.disable_adapter_layers()
        try:
            try:
                forced_decoder_ids = ST_PROC.get_decoder_prompt_ids(language=None, task="translate")
            except Exception:
                forced_decoder_ids = None
            gen_ids = ST_MODEL.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=1,
                max_new_tokens=192
            )
        finally:
            ST_MODEL.enable_adapter_layers()

    return ST_PROC.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def piper_tts(text: str, out_wav: Path) -> None:
    subprocess.run(["piper", "--model", str(PIPER_MODEL), "--config", str(PIPER_CFG), "--output_file", str(out_wav)],
                   input=text.encode("utf-8"), check=True)


def xtts_synthesise(text: str, ref_wav_path: str, out_wav: Path) -> int:
    chunks = split_text_for_xtts(text)
    audio_parts = []
    for chunk in chunks:
        wav = XTTS_MODEL.tts(text=chunk, speaker_wav=ref_wav_path, language="en",
                              temperature=XTTS_TEMPERATURE, repetition_penalty=XTTS_REPETITION_PENALTY,
                              top_k=XTTS_TOP_K, top_p=XTTS_TOP_P, speed=XTTS_SPEED)
        audio_parts.append(np.asarray(wav, dtype=np.float32))
    if not audio_parts:
        raise ValueError("XTTS produced no audio output.")
    sf.write(str(out_wav), np.concatenate(audio_parts), XTTS_OUTPUT_SR)
    return len(chunks)


# =========================================================
# Pipeline runners
# =========================================================
def run_standard_pipeline(audio_path: str, source_lang: str = "french", progress=gr.Progress()):
    if audio_path is None or not Path(audio_path).exists():
        return ("No audio provided.", None, "", "Please upload or record a speech file.")
    if not TRANSLATION_READY:
        progress(0.08, desc="Loading translation model")
        load_translation_pipeline()
    err = validate_piper_assets()
    if err:
        return ("Piper TTS unavailable.", None, "", err)
    t0 = time.time()
    progress(0.22, desc="Preparing audio")
    audio_16k, sr = load_audio_for_whisper(audio_path)
    t_load = time.time()
    progress(0.50, desc="Translating to English")
    en_text = translate_to_en(audio_16k, sr, source_lang=source_lang)
    t_trans = time.time()
    en_text_clean = normalise_text_for_tts(en_text)
    if not en_text_clean:
        return ("Translation produced empty text.", None, "", "Try a clearer audio sample.")
    progress(0.84, desc="Generating English speech")
    out_wav = OUT_DIR / f"standard_{uuid.uuid4().hex}.wav"
    piper_tts(en_text_clean, out_wav)
    t_tts = time.time()
    progress(1.0, desc="Done")
    timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
              f"Speech synthesis: {t_tts - t_trans:.2f}s\nTotal: {t_tts - t0:.2f}s")
    lang_label = "French (fine-tuned)" if source_lang == "french" else "Auto-detected"
    return (en_text, str(out_wav), timing, f"Completed. {lang_label} → English translation done.")


def run_voice_clone_pipeline(audio_path: str, source_lang: str = "french", progress=gr.Progress()):
    if audio_path is None or not Path(audio_path).exists():
        return ("No audio provided.", None, "", "Please upload or record a speech file.")
    if not TRANSLATION_READY:
        progress(0.06, desc="Loading translation model")
        load_translation_pipeline()
    if not XTTS_READY:
        progress(0.12, desc="Loading XTTS model")
        load_xtts_model()
    t0 = time.time()
    progress(0.20, desc="Preparing audio")
    audio_16k, sr = load_audio_for_whisper(audio_path)
    t_load = time.time()
    progress(0.42, desc="Translating to English")
    en_text = translate_to_en(audio_16k, sr, source_lang=source_lang)
    t_trans = time.time()
    en_text_clean = normalise_text_for_tts(en_text)
    if not en_text_clean:
        return ("Translation produced empty text.", None, "", "Try a clearer audio sample.")
    progress(0.60, desc="Preparing reference audio")
    ref_wav_path = prepare_xtts_reference(audio_path)
    t_ref = time.time()
    progress(0.78, desc="Cloning voice and generating English speech")
    out_wav = OUT_DIR / f"cloned_{uuid.uuid4().hex}.wav"
    num_chunks = xtts_synthesise(en_text_clean, ref_wav_path, out_wav)
    t_tts = time.time()
    progress(1.0, desc="Done")
    timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
              f"Reference audio prep: {t_ref - t_trans:.2f}s\nVoice cloning ({num_chunks} chunk(s)): {t_tts - t_ref:.2f}s\n"
              f"Total: {t_tts - t0:.2f}s")
    lang_label = "French (fine-tuned)" if source_lang == "french" else "Auto-detected"
    return (en_text, str(out_wav), timing, f"Completed. {lang_label} → English, voice cloned.")


def run_app(mode: str, audio_path: str, input_lang_label: str = "French (fine-tuned ★)", progress=gr.Progress()):
    source_lang = LANG_MAP.get(input_lang_label, "french")
    if mode == "Translate without voice cloning":
        return run_standard_pipeline(audio_path, source_lang=source_lang, progress=progress)
    if mode == "Translate with voice cloning":
        return run_voice_clone_pipeline(audio_path, source_lang=source_lang, progress=progress)
    return ("Invalid mode.", None, "", "Please select a valid mode.")


# =========================================================
# CSS
# =========================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body, .gradio-container {
    background: #09090f !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }

/* Ensure all sections sit above canvas */
#sb-nav, #sb-hero, .sb-stats-strip, section, footer, #sb-feedback,
.gradio-container > * { position: relative; z-index: 1; }

/* NAV */
#sb-nav {
    position: sticky; top: 0; z-index: 100;
    background: rgba(9,9,15,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(59,130,246,0.18);
    padding: 0 48px;
    display: flex; align-items: center; justify-content: space-between;
    height: 68px;
}
.sb-logo {
    font-family: "Outfit", sans-serif;
    font-weight: 800; font-size: 1.45rem; color: #fff;
    letter-spacing: -0.02em;
    display: flex; align-items: center; gap: 10px;
}
.sb-logo-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    flex-shrink: 0;
}
.sb-nav-links { display: flex; gap: 36px; align-items: center; }
.sb-nav-links a { color: #94a3b8; text-decoration: none; font-size: 0.9rem; transition: color 0.2s; }
.sb-nav-links a:hover { color: #e2e8f0; }
.sb-nav-cta {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: #fff !important; padding: 10px 24px;
    font-size: 0.875rem; font-weight: 600;
    text-decoration: none; border-radius: 6px;
    transition: opacity 0.2s;
}
.sb-nav-cta:hover { opacity: 0.88; }

/* HERO */
#sb-hero {
    background: #09090f;
    padding: 100px 48px 80px;
    text-align: center;
    position: relative; overflow: hidden;
    min-height: 90vh;
    display: flex; align-items: center; justify-content: center;
}
.sb-hero-inner { position: relative; z-index: 2; max-width: 900px; margin: 0 auto; }

.sb-hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.28);
    padding: 6px 16px; border-radius: 100px;
    font-size: 0.8rem; color: #93c5fd; font-weight: 500;
    margin-bottom: 32px; letter-spacing: 0.04em;
}
.sb-hero-badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3b82f6;
    animation: pulse-dot 2.5s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

.sb-hero-title {
    font-family: "Outfit", sans-serif; font-weight: 900;
    font-size: clamp(3rem, 7vw, 6rem);
    color: #fff; margin-bottom: 24px; letter-spacing: -0.03em;
}
.sb-hero-title span {
    background: linear-gradient(135deg, #3b82f6, #06b6d4, #93c5fd, #3b82f6);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 5s ease infinite;
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.sb-hero-desc {
    font-size: 1.05rem; line-height: 1.8;
    color: #94a3b8; max-width: 580px;
    margin: 0 auto 40px auto; font-weight: 300;
}
.sb-hero-actions {
    display: flex; gap: 16px; align-items: center;
    justify-content: center; flex-wrap: wrap;
}
.sb-btn-primary {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: #fff; padding: 14px 36px; font-size: 1rem; font-weight: 600;
    text-decoration: none; border-radius: 8px;
    display: inline-block; transition: opacity 0.2s, transform 0.15s;
}
.sb-btn-primary:hover { opacity: 0.88; transform: translateY(-1px); }
.sb-btn-ghost {
    background: transparent; color: #e2e8f0;
    padding: 14px 36px; font-size: 1rem;
    text-decoration: none; border-radius: 8px;
    border: 1px solid rgba(226,232,240,0.18);
    display: inline-flex; align-items: center; gap: 8px;
    transition: all 0.2s;
}
.sb-btn-ghost:hover { border-color: rgba(226,232,240,0.4); }

/* HERO GLOWS */
.sb-hero-glow {
    position: absolute;
    top: 48%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 900px;
    height: 900px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(59,130,246,0.24) 0%,
        rgba(6,182,212,0.12) 24%,
        rgba(147,197,253,0.08) 42%,
        rgba(59,130,246,0.03) 62%,
        transparent 78%
    );
    filter: blur(18px);
    pointer-events: none;
    z-index: 1;
    transition: left 0.65s cubic-bezier(0.23, 1, 0.32, 1), top 0.65s cubic-bezier(0.23, 1, 0.32, 1);
}
.sb-hero-glow-ambient {
    position: absolute; top: 25%; right: 12%;
    width: 420px; height: 420px; border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.10) 0%, transparent 70%);
    pointer-events: none; z-index: 1;
    animation: ambient-drift 9s ease-in-out infinite;
}
.sb-hero-glow-ambient-2 {
    position: absolute; bottom: 15%; left: 12%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 70%);
    pointer-events: none; z-index: 1;
    animation: ambient-drift 11s ease-in-out infinite reverse;
}
@keyframes ambient-drift {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(28px, -22px) scale(1.07); }
    66%       { transform: translate(-18px, 24px) scale(0.94); }
}

/* STATS */
.sb-stats-strip {
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding: 32px 48px;
    display: flex; justify-content: center; gap: 80px; flex-wrap: wrap;
}
.sb-stat-item { text-align: center; }
.sb-stat-num {
    font-family: "Outfit", sans-serif; font-weight: 800; font-size: 2.2rem;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1; margin-bottom: 6px;
}
.sb-stat-label { font-size: 0.78rem; color: #475569; letter-spacing: 0.1em; text-transform: uppercase; }

/* SECTIONS */
.sb-section { padding: 96px 48px; }
.sb-section-dark { background: #09090f; }
.sb-section-alt { background: #0d0d14; }
.sb-section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em;
    text-transform: uppercase; color: #60a5fa; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.sb-section-label::before { content: ''; width: 24px; height: 1px; background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.sb-section-title {
    font-family: "Outfit", sans-serif; font-weight: 800;
    font-size: clamp(2rem, 4vw, 3rem);
    color: #fff; margin-bottom: 16px; letter-spacing: -0.02em;
}
.sb-section-sub { font-size: 1rem; line-height: 1.75; color: #475569; max-width: 560px; font-weight: 300; margin-bottom: 48px; }

/* STEPS */
.sb-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; }
.sb-step {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    padding: 32px 28px; border-radius: 12px;
    position: relative; overflow: hidden; transition: border-color 0.3s, background 0.3s;
}
.sb-step::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.6), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.sb-step:hover { border-color: rgba(59,130,246,0.24); background: rgba(59,130,246,0.04); }
.sb-step:hover::before { opacity: 1; }
.sb-step-num { font-size: 0.72rem; font-weight: 700; color: #60a5fa; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
.sb-step-icon { width: 44px; height: 44px; border-radius: 10px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; font-size: 20px; }
.sb-step-title { font-family: "Outfit", sans-serif; font-weight: 700; font-size: 1.1rem; color: #f1f5f9; margin-bottom: 10px; }
.sb-step-desc { font-size: 0.9rem; line-height: 1.7; color: #475569; font-weight: 300; }

/* VIDEO */
.sb-video-container {
    position: relative; padding-bottom: 56.25%; background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; overflow: hidden;
    margin-top: 40px; max-width: 100%;
    margin-left: auto !important; margin-right: auto !important;
}
.sb-video-container video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
.sb-video-placeholder { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; }
.sb-play-btn { width: 72px; height: 72px; border-radius: 50%; background: linear-gradient(135deg, #2563eb, #06b6d4); display: flex; align-items: center; justify-content: center; transition: transform 0.2s; }
.sb-play-btn:hover { transform: scale(1.06); }
.sb-video-placeholder p { color: #475569; font-size: 0.9rem; }

/* SAMPLES */
.sb-samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin-top: 40px; }
.sb-sample-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 20px 24px; border-radius: 10px; display: flex; flex-direction: column; gap: 12px; transition: border-color 0.2s, background 0.2s; }
.sb-sample-card:hover { border-color: rgba(59,130,246,0.24); background: rgba(59,130,246,0.04); }
.sb-sample-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #60a5fa; }
.sb-sample-name { font-family: "Outfit", sans-serif; font-weight: 700; font-size: 0.95rem; color: #f1f5f9; }
.sb-sample-dl { display: inline-block; margin-top: 4px; font-size: 0.8rem; font-weight: 500; padding: 8px 18px; border-radius: 6px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); color: #60a5fa; text-decoration: none; transition: all 0.2s; }
.sb-sample-dl:hover { background: rgba(59,130,246,0.16); }

/* SYSTEM DEMO */
/* Outer section column — full-width dark background, contains heading + shell */
#sb-system {
    background: #0d0d14 !important;
    padding: 96px 48px 96px !important;
    border: none !important;
    box-shadow: none !important;
}

/* Terminal shell card — sits inside the section, below the heading */
#sb-demo-shell {
    border: 0px solid rgba(59,130,246,0.18) !important;
    background: rgba(255,255,255,0.01) !important;
    border-radius: 0px !important;
    overflow: hidden !important;
    padding: 0 !important;
    box-shadow: none !important;
    margin-top: 0 !important;
}

.sb-system-topbar { background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 14px 24px; display: flex; align-items: center; gap: 10px; }
.sb-topbar-dot { width: 10px; height: 10px; border-radius: 50%; }
.sb-topbar-title { font-size: 0.8rem; color: #334155; margin-left: 8px; letter-spacing: 0.04em; }

/* Inner content area — real DOM parent of all Gradio component blocks */
#sb-demo-inner { padding: 36px !important; background: transparent !important; border: none !important; box-shadow: none !important; }
#sb-demo-inner label, #sb-demo-inner .label-wrap span {
    color: #64748b !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
}
#sb-demo-inner textarea, #sb-demo-inner input[type="text"] {
    background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important;
}
#sb-demo-inner textarea::placeholder { color: #334155 !important; }
#sb-demo-inner .block,
#sb-demo-inner .wrap,
#sb-demo-inner .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#sb-demo-inner button { border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; transition: all 0.2s !important; }
#sb-demo-inner button.primary, #sb-demo-inner button[variant="primary"] {
    background: linear-gradient(135deg, #2563eb, #06b6d4) !important; color: #fff !important; border: none !important;
}
#sb-demo-inner button.primary:hover { opacity: 0.88 !important; }
#sb-demo-inner button.secondary, #sb-demo-inner button[variant="secondary"] {
    background: rgba(59,130,246,0.08) !important; color: #93c5fd !important; border: 1px solid rgba(59,130,246,0.28) !important;
}
#sb-demo-inner button.secondary:hover { background: rgba(59,130,246,0.12) !important; }
#sb-demo-inner [data-testid="audio"], #sb-demo-inner .gr-audio {
    background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important;
}
#sb-demo-inner input[type="radio"] { accent-color: #3b82f6 !important; }
#sb-demo-inner select { background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* Re-box audio + text outputs — uses double-ID specificity to beat #sb-demo-inner .block */
#sb-demo-inner #sb-audio-in,
#sb-demo-inner #sb-audio-out { background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; }
#sb-demo-inner #sb-text-out,
#sb-demo-inner #sb-runtime-box,
#sb-demo-inner #sb-status-box { background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; padding: 12px !important; }

/* Model status indicator */
.sb-model-status {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.82rem; font-weight: 500; padding: 8px 14px;
    border-radius: 6px; margin-bottom: 20px;
}
.sb-model-ready {
    background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.28);
    color: #93c5fd;
}
.sb-model-loading {
    background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.18);
    color: #94a3b8;
}
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.sb-spin { display: inline-block; animation: spin 1.4s linear infinite; }

/* Flat controls — no box for lang row and mode */
.sb-no-box,
.sb-no-box > .block,
.sb-no-box .block,
.sb-no-box .wrap,
.sb-no-box .form,
.sb-no-box fieldset,
.sb-no-box > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.sb-lang-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.sb-lang-arrow { color: #475569; font-size: 1rem; flex-shrink: 0;}

/* Demo divider */
.sb-demo-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 20px 0; }
.sb-demo-section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin-bottom: 8px;
}

/* FEEDBACK */
#sb-feedback { background: #09090f; padding: 96px 48px; position: relative; overflow: hidden; }
#sb-feedback::before {
    content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 500px; height: 280px;
    background: radial-gradient(ellipse, rgba(59,130,246,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.sb-feedback-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 48px; flex-wrap: wrap; gap: 24px; position: relative; z-index: 2; }
.sb-feedback-form-btn { display: inline-flex; align-items: center; gap: 8px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.28); color: #93c5fd; padding: 12px 24px; font-size: 0.875rem; font-weight: 500; text-decoration: none; border-radius: 8px; transition: all 0.2s; }
.sb-feedback-form-btn:hover { background: rgba(59,130,246,0.12); }
.sb-feedback-carousel { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; min-height: 220px; position: relative; z-index: 2; }
.sb-feedback-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 28px; border-radius: 12px; display: flex; flex-direction: column; gap: 14px; transition: border-color 0.3s; }
.sb-feedback-card:hover { border-color: rgba(59,130,246,0.18); }
.sb-feedback-quote { font-size: 0.95rem; line-height: 1.75; color: #64748b; font-style: italic; font-weight: 300; flex: 1; }
.sb-feedback-name { font-size: 0.78rem; font-weight: 600; color: #60a5fa; letter-spacing: 0.08em; text-transform: uppercase; }
.sb-feedback-placeholder { grid-column: 1 / -1; color: #334155; font-size: 0.9rem; font-style: italic; display: flex; align-items: center; justify-content: center; min-height: 120px; text-align: center; border: 1px dashed rgba(255,255,255,0.06); border-radius: 12px; }
.sb-carousel-controls { display: flex; justify-content: center; gap: 8px; margin-top: 32px; position: relative; z-index: 2; }
.sb-carousel-dot { width: 6px; height: 6px; border-radius: 50%; background: rgba(255,255,255,0.1); cursor: pointer; transition: all 0.2s; border: none; }
.sb-carousel-dot.active { background: #60a5fa; }

/* FOOTER */
#sb-footer { background: #070709; border-top: 1px solid rgba(255,255,255,0.04); padding: 56px 48px 40px; }
.sb-footer-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 48px; padding-bottom: 40px; border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 32px; }
.sb-footer-brand { font-family: "Outfit", sans-serif; font-weight: 800; font-size: 1.3rem; color: #fff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.sb-footer-brand-dot { width: 8px; height: 8px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6, #06b6d4); }
.sb-footer-tagline { font-size: 0.85rem; color: #334155; line-height: 1.7; max-width: 260px; }
.sb-footer-col-title { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #475569; margin-bottom: 16px; }
.sb-footer-links { display: flex; flex-direction: column; gap: 10px; }
.sb-footer-links a { color: #334155; text-decoration: none; font-size: 0.875rem; transition: color 0.2s; }
.sb-footer-links a:hover { color: #60a5fa; }
.sb-footer-bottom { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
.sb-footer-copy { font-size: 0.8rem; color: #1e293b; }

/* RESPONSIVE */
@media (max-width: 900px) {
    #sb-nav { padding: 0 24px; }
    .sb-section, #sb-hero, #sb-feedback, #sb-footer, #sb-system { padding-left: 24px; padding-right: 24px; }
    .sb-feedback-carousel { grid-template-columns: 1fr; }
    .sb-footer-grid { grid-template-columns: 1fr; gap: 32px; }
    .sb-nav-links { display: none; }
    .sb-stats-strip { gap: 40px; padding: 32px 24px; }
}
@media (max-width: 640px) {
    .sb-steps { grid-template-columns: 1fr; }
    .sb-samples-grid { grid-template-columns: 1fr; }
    .sb-hero-title { font-size: 2.5rem; }
}
.gradio-container .block { border-radius: 8px !important; }
footer.svelte-1rjryqp { display: none !important; }

#sb-demo-inner {
    padding: 22px 4px 10px 4px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#sb-demo-inner .sb-demo-divider {
    display: none !important;
}

/* top controls */
#sb-control-row {
    gap: 28px !important;
    align-items: start !important;
    margin-top: 18px !important;
    margin-bottom: 18px !important;
}

#sb-lang-panel,
#sb-mode-panel,
#sb-upload-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* labels */
.sb-demo-section-label {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin-bottom: 10px !important;
}

/* language row */
.sb-lang-row {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
}

.sb-lang-arrow {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #64748b !important;
    font-size: 1.4rem !important;
    line-height: 1 !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 30px !important;
}

/* radios inline */
.sb-inline-radio fieldset {
    gap: 18px !important;
}

.sb-inline-radio label {
    margin-right: 18px !important;
}

/* upload area */
#sb-upload-panel {
    margin-top: 6px !important;
    margin-bottom: 24px !important;
}

#sb-demo-inner #sb-audio-in,
#sb-demo-inner #sb-audio-out,
#sb-demo-inner #sb-text-out,
#sb-demo-inner #sb-runtime-box {
    background: #10101a !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
    box-shadow: none !important;
}

#sb-demo-inner #sb-audio-in {
    min-height: 220px !important;
}

#sb-demo-inner #sb-audio-out {
    min-height: 220px !important;
}

#sb-demo-inner #sb-text-out textarea {
    min-height: 220px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#sb-demo-inner #sb-runtime-box textarea {
    min-height: 90px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* helper text */
.sb-helper-text {
    font-size: 0.84rem;
    color: #64748b;
    line-height: 1.7;
    margin-top: 10px;
    margin-bottom: 14px;
}

.sb-helper-text strong {
    color: #60a5fa;
}

/* button */
#sb-demo-inner button.primary,
#sb-demo-inner button[variant="primary"] {
    width: 100% !important;
    border-radius: 999px !important;
    margin-top: 6px !important;
}

/* outputs row */
#sb-output-row {
    gap: 20px !important;
    align-items: stretch !important;
    margin-bottom: 18px !important;
}

#sb-output-row > div {
    display: flex !important;
    flex-direction: column !important;
}

/* hide status box from UI clutter */
#sb-status-box {
    display: none !important;
}

/* make labels cleaner */
#sb-demo-inner label,
#sb-demo-inner .label-wrap span {
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* remove heavy inner borders from textbox wrapper if gradio adds them */
#sb-demo-inner #sb-text-out,
#sb-demo-inner #sb-runtime-box {
    padding: 12px !important;
}

/* mobile */
@media (max-width: 900px) {
    #sb-control-row,
    #sb-output-row {
        flex-direction: column !important;
        gap: 16px !important;
    }
}
"""


# =========================================================
# UI
# =========================================================
with gr.Blocks(title="SpeechBridge") as demo:

    # ── Soundwave canvas (fixed background, behind everything) ──
    gr.HTML("""
   <canvas id="sb-wave-canvas" style="
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    z-index:-1;
    pointer-events:none;
    opacity:0.95;
    filter: blur(0.2px);
"></canvas>
    <script>
    (function() {
        var canvas, ctx, W, H, scrollY = 0;
        var waves = [
            { freq: 0.016, amp: 0.10, speed: 0.55, phase: 0.0,  r:59,  g:130, b:246 },
            { freq: 0.022, amp: 0.07, speed: 0.85, phase: 2.1,  r:6,   g:182, b:212 },
            { freq: 0.011, amp: 0.13, speed: 0.38, phase: 4.4,  r:147, g:197, b:253 },
            { freq: 0.029, amp: 0.05, speed: 1.1,  phase: 1.3,  r:59,  g:130, b:246 }
        ];

        function init() {
            canvas = document.getElementById('sb-wave-canvas');
            if (!canvas) { setTimeout(init, 200); return; }
            ctx = canvas.getContext('2d');
            resize();
            window.addEventListener('resize', resize);
            window.addEventListener('scroll', function() { scrollY = window.scrollY; }, {passive: true});
            requestAnimationFrame(tick);
        }

        function resize() {
            W = canvas.width  = window.innerWidth;
            H = canvas.height = window.innerHeight;
        }

        function drawWave(w, t) {
            var scrollMod = 1 + scrollY * 0.0006;
            var amp = H * w.amp * scrollMod;

            var cy  = H * (0.42 + Math.min(scrollY * 0.00005, 0.12));
            ctx.beginPath();
            for (var x = 0; x <= W + 4; x += 4) {
                var y = cy
                    + Math.sin(x * w.freq + t * w.speed + w.phase) * amp
                    + Math.sin(x * w.freq * 2.7 + t * w.speed * 1.5 + w.phase + 1) * amp * 0.28;
                if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = 'rgba(' + w.r + ',' + w.g + ',' + w.b + ',0.16)';
            ctx.lineWidth = 2.2;
            ctx.stroke();
        }

        function tick(ts) {
            var t = ts * 0.001;
            ctx.clearRect(0, 0, W, H);
            waves.forEach(function(w) { drawWave(w, t); });
            requestAnimationFrame(tick);
        }

        setTimeout(init, 150);
    })();
    </script>
    """)

    # ── Nav ──
    gr.HTML("""
    <nav id="sb-nav">
        <div class="sb-logo">
            <div class="sb-logo-dot"></div>
            SpeechBridge
        </div>
        <div class="sb-nav-links">
            <a href="#sb-how">How it works</a>
            <a href="#sb-demo-video">Demo</a>
            <a href="#sb-samples">Samples</a>
            <a href="#sb-feedback">Feedback</a>
        </div>
        <a class="sb-nav-cta" href="#sb-system">Try it now</a>
    </nav>
    """)

    # ── Hero ──
    gr.HTML("""
    <section id="sb-hero">
        <div class="sb-hero-glow" id="sb-hero-glow"></div>
        <div class="sb-hero-glow-ambient"></div>
        <div class="sb-hero-glow-ambient-2"></div>
        <div class="sb-hero-inner">
            <div class="sb-hero-badge">
                <div class="sb-hero-badge-dot"></div>
                Multilingual to English · Fine-tuned on French
            </div>
            <h1 class="sb-hero-title">
                Speak Any Language.<br><span>Be Heard</span> in English.
            </h1>
            <p class="sb-hero-desc">
                SpeechBridge translates speech into English — preserving not just the words,
                but the voice, tone, and character of the original speaker.
                Powered by a LoRA fine-tuned Whisper model, with primary research focus on French.
            </p>
            <div class="sb-hero-actions">
                <a class="sb-btn-primary" href="#sb-system">Try it yourself</a>
                <a class="sb-btn-ghost" href="#sb-how">
                    See how it works
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </a>
            </div>
        </div>
    </section>
    """)

    # ── Hero mouse tracking ──
    gr.HTML("""
    <script>
    (function() {
        function initGlow() {
            var hero = document.getElementById('sb-hero');
            var glow = document.getElementById('sb-hero-glow');
            if (!hero || !glow) { setTimeout(initGlow, 300); return; }
            hero.addEventListener('mousemove', function(e) {
                var r = hero.getBoundingClientRect();
                glow.style.left = (e.clientX - r.left) + 'px';
                glow.style.top  = (e.clientY - r.top)  + 'px';
            });
            hero.addEventListener('mouseleave', function() {
                glow.style.left = '50%';
                glow.style.top  = '50%';
            });
        }
        setTimeout(initGlow, 600);
    })();
    </script>
    """)

    # ── Stats ──
    gr.HTML("""
    <div class="sb-stats-strip">
        <div class="sb-stat-item">
            <div class="sb-stat-num">2+</div>
            <div class="sb-stat-label">Languages</div>
        </div>
        <div class="sb-stat-item">
            <div class="sb-stat-num">~0.81</div>
            <div class="sb-stat-label">Voice Similarity</div>
        </div>
        <div class="sb-stat-item">
            <div class="sb-stat-num">2</div>
            <div class="sb-stat-label">Output Modes</div>
        </div>
        <div class="sb-stat-item">
            <div class="sb-stat-num">End-to-end</div>
            <div class="sb-stat-label">Pipeline</div>
        </div>
    </div>
    """)

    # ── How it works ──
    gr.HTML("""
    <section id="sb-how" class="sb-section sb-section-alt">
        <div class="sb-section-label">How it works</div>
        <h2 class="sb-section-title">Three steps to translated speech</h2>
        <p class="sb-section-sub">Upload any speech clip and receive translated English audio — optionally in the original speaker's voice. Optimised for French with a LoRA fine-tuned model.</p>
        <div class="sb-steps">
            <div class="sb-step">
                <div class="sb-step-num">01</div>
                <div class="sb-step-title">Upload your audio</div>
                <div class="sb-step-desc">Record or upload a speech clip in any format. Select your source language — French uses the fine-tuned model, other languages use base Whisper.</div>
            </div>
            <div class="sb-step">
                <div class="sb-step-num">02</div>
                <div class="sb-step-title">Choose output mode</div>
                <div class="sb-step-desc">Select standard translation for clean English speech, or voice cloning to preserve the original speaker's voice and identity.</div>
            </div>
            <div class="sb-step">
                <div class="sb-step-num">03</div>
                <div class="sb-step-title">Receive translated speech</div>
                <div class="sb-step-desc">Get translated English text and synthesised audio. With voice cloning, it sounds like the same person speaking English.</div>
            </div>
        </div>
    </section>
    """)

    # ── Demo video ──
    gr.HTML("""
    <section id="sb-demo-video" class="sb-section sb-section-dark">
        <div class="sb-section-label">See it in action</div>
        <h2 class="sb-section-title">Watch a full translation run</h2>
        <div class="sb-video-container">
            <video id="sb-video" controls style="display:none;">
                <source src="demo.mp4" type="video/mp4">
            </video>
            <div class="sb-video-placeholder" id="sb-video-placeholder">
                <div class="sb-play-btn">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="#0D0D0D"><path d="M8 5v14l11-7L8 5z"/></svg>
                </div>
                <p>Demo video coming soon</p>
            </div>
        </div>
    </section>
    """)

    # ── Samples ──
    gr.HTML("""
    <section id="sb-samples" class="sb-section sb-section-alt">
        <div class="sb-section-label">Sample audio</div>
        <h2 class="sb-section-title">Try it with one of our clips</h2>
        <p class="sb-section-sub">Download any French speech sample below and upload it into the system to see it in action.</p>
        <div class="sb-samples-grid">
            <div class="sb-sample-card">
                <div class="sb-sample-label">Sample 01</div>
                <div class="sb-sample-name">Child's voice</div>
                <a class="sb-sample-dl" href="audio_samples/sample_1.wav" download>Download clip →</a>
            </div>
            <div class="sb-sample-card">
                <div class="sb-sample-label">Sample 02</div>
                <div class="sb-sample-name">Adult female voice</div>
                <a class="sb-sample-dl" href="audio_samples/sample_2.wav" download>Download clip →</a>
            </div>
            <div class="sb-sample-card">
                <div class="sb-sample-label">Sample 03</div>
                <div class="sb-sample-name">Adult male voice</div>
                <a class="sb-sample-dl" href="audio_samples/sample_3.wav" download>Download clip →</a>
            </div>
            <div class="sb-sample-card">
                <div class="sb-sample-label">Sample 04</div>
                <div class="sb-sample-name">Young female voice</div>
                <a class="sb-sample-dl" href="audio_samples/sample_4.wav" download>Download clip →</a>
            </div>
            <div class="sb-sample-card">
                <div class="sb-sample-label">Sample 05</div>
                <div class="sb-sample-name">Elderly voice</div>
                <a class="sb-sample-dl" href="audio_samples/sample_5.wav" download>Download clip →</a>
            </div>
        </div>
    </section>
    """)

    # ── System demo ──
    with gr.Column(elem_id="sb-system"):
        gr.HTML("""
        <div class="sb-section-label">Live demo</div>
        <h2 class="sb-section-title">Try SpeechBridge</h2>
        <p class="sb-section-sub">Upload or record speech and receive an English translation — with or without voice cloning. French uses the LoRA fine-tuned model.</p>
        """)

        with gr.Column(elem_id="sb-demo-inner"):
            model_status_display = gr.HTML(value=_render_model_status())
            init_btn = gr.Button(
                "Initialise models",
                variant="secondary",
                visible=not (TRANSLATION_READY and XTTS_READY)
            )
            status_timer = gr.Timer(value=3, active=True)

            with gr.Row(elem_id="sb-control-row"):
                with gr.Column(scale=1, elem_id="sb-lang-panel"):
                    gr.HTML('<div class="sb-demo-section-label">Language pair</div>')
                    with gr.Row(elem_classes=["sb-lang-row"]):
                        input_lang = gr.Dropdown(
                            choices=["French (★)", "Other languages"],
                            value="French (★)",
                            show_label=False,
                            interactive=True,
                            elem_classes=["sb-no-box"]
                        )
                        gr.HTML('<div class="sb-lang-arrow">→</div>')
                        output_lang = gr.Dropdown(
                            choices=["English"],
                            value="English",
                            show_label=False,
                            interactive=False,
                            elem_classes=["sb-no-box"]
                        )

                with gr.Column(scale=1, elem_id="sb-mode-panel"):
                    gr.HTML('<div class="sb-demo-section-label">Translation mode</div>')
                    mode = gr.Radio(
                        choices=["Translate without voice cloning", "Translate with voice cloning"],
                        value="Translate without voice cloning",
                        show_label=False,
                        elem_classes=["sb-no-box", "sb-inline-radio"]
                    )

            with gr.Column(elem_id="sb-upload-panel"):
                gr.HTML('<div class="sb-demo-section-label">Upload audio</div>')
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=None,
                    elem_id="sb-audio-in",
                )

                gr.Markdown(
                    "<div class='sb-helper-text'>"
                    "Upload audio in any format or record directly. For voice cloning, the clip also serves as the speaker reference. "
                    "<strong>French uses the LoRA fine-tuned model.</strong> "
                    "Other languages use base Whisper with auto language detection."
                    "</div>"
                )

                translate_btn = gr.Button("Run translation", variant="primary")

            with gr.Row(elem_id="sb-output-row"):
                with gr.Column(scale=1):
                    output_audio = gr.Audio(
                        type="filepath",
                        label="Generated English speech",
                        elem_id="sb-audio-out"
                    )

                with gr.Column(scale=1):
                    translated_text = gr.Textbox(
                        label="English translation",
                        lines=10,
                        placeholder="Translated English text will appear here.",
                        elem_id="sb-text-out",
                    )

            runtime_info = gr.Textbox(
                label="Runtime breakdown",
                lines=4,
                interactive=False,
                elem_id="sb-runtime-box"
            )

            pipeline_status = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False,
                elem_id="sb-status-box",
                visible=False
            )

    # ── Feedback ──
    gr.HTML(f"""
    <section id="sb-feedback">
        <div class="sb-feedback-header">
            <div>
                <div class="sb-section-label">User feedback</div>
                <h2 class="sb-section-title" style="margin-bottom:0;">What people are saying</h2>
            </div>
            <a class="sb-feedback-form-btn" href="{GOOGLE_FORM_URL}" target="_blank">
                Share your feedback
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </a>
        </div>
        <div class="sb-feedback-carousel" id="sb-feedback-carousel">
            <div class="sb-feedback-placeholder">
                <span>Feedback will appear here once responses are submitted.</span>
            </div>
        </div>
        <div class="sb-carousel-controls" id="sb-carousel-controls" style="display:none;"></div>
    </section>
    """)

    demo.load(
        fn=None,
        js=f"""
        function() {{
            var SHEET_URL = 'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:json';
            var PER_PAGE = 3;
            var allFeedback = [];
            var currentPage = 0;
            var autoTimer = null;

            function renderStars(n) {{
                var s = '';
                var filled = Math.round(n || 0);
                for (var i = 1; i <= 5; i++) {{
                    s += '<span style="color:#60a5fa;font-size:13px;">' + (i <= filled ? '★' : '☆') + '</span>';
                }}
                return s;
            }}

            function renderPage(page) {{
                var carousel = document.getElementById('sb-feedback-carousel');
                var controls = document.getElementById('sb-carousel-controls');
                if (!carousel) return;
                var start = page * PER_PAGE;
                var slice = allFeedback.slice(start, start + PER_PAGE);
                carousel.innerHTML = '';
                slice.forEach(function(fb) {{
                    var avg = ((fb.r1 || 0) + (fb.r3 || 0)) / 2;
                    var card = document.createElement('div');
                    card.className = 'sb-feedback-card';
                    card.innerHTML =
                        '<div style="display:flex;gap:3px;margin-bottom:4px;">' + renderStars(avg) + '</div>' +
                        '<div class="sb-feedback-quote">' + (fb.comment || 'No comment provided.') + '</div>' +
                        '<div class="sb-feedback-name">' + (fb.name || 'Anonymous') + '</div>';
                    carousel.appendChild(card);
                }});
                if (controls) {{
                    var dots = controls.querySelectorAll('.sb-carousel-dot');
                    dots.forEach(function(d, i) {{ d.classList.toggle('active', i === page); }});
                }}
                currentPage = page;
            }}

            function buildControls(totalPages) {{
                var controls = document.getElementById('sb-carousel-controls');
                if (!controls) return;
                controls.innerHTML = '';
                if (totalPages <= 1) return;
                controls.style.display = 'flex';
                for (var i = 0; i < totalPages; i++) {{
                    (function(idx) {{
                        var dot = document.createElement('button');
                        dot.className = 'sb-carousel-dot' + (idx === 0 ? ' active' : '');
                        dot.addEventListener('click', function() {{ clearInterval(autoTimer); renderPage(idx); }});
                        controls.appendChild(dot);
                    }})(i);
                }}
            }}

            function loadFeedback() {{
                fetch(SHEET_URL)
                    .then(function(r) {{ return r.text(); }})
                    .then(function(text) {{
                        var start = text.indexOf('(') + 1;
                        var end = text.lastIndexOf(')');
                        var json = JSON.parse(text.substring(start, end));
                        var rows = json.table && json.table.rows ? json.table.rows : [];
                        allFeedback = rows.map(function(row) {{
                            var c = row.c || [];
                            var nameVal = c[5] && c[5].v ? String(c[5].v).trim() : 'Anonymous';
                            return {{
                                r1: c[1] && c[1].v ? Number(c[1].v) : 0,
                                r2: c[2] && c[2].v ? Number(c[2].v) : 0,
                                r3: c[3] && c[3].v ? Number(c[3].v) : 0,
                                comment: c[4] && c[4].v ? String(c[4].v) : '',
                                name: nameVal.toLowerCase() === 'anonymous' ? 'Anonymous' : nameVal
                            }};
                        }}).filter(function(fb) {{ return fb.comment || fb.r1; }});
                        if (allFeedback.length === 0) return;
                        var totalPages = Math.ceil(allFeedback.length / PER_PAGE);
                        buildControls(totalPages);
                        renderPage(0);
                        if (totalPages > 1) {{
                            autoTimer = setInterval(function() {{ renderPage((currentPage + 1) % totalPages); }}, 5000);
                        }}
                    }})
                    .catch(function(e) {{ console.log('Feedback error:', e); }});
            }}

            setTimeout(loadFeedback, 1500);
        }}
        """
    )

    # ── Footer ──
    gr.HTML("""
    <footer id="sb-footer">
        <div class="sb-footer-grid">
            <div>
                <div class="sb-footer-brand">
                    <div class="sb-footer-brand-dot"></div>
                    SpeechBridge
                </div>
                <p class="sb-footer-tagline">A multilingual speech-to-English translation system combining a LoRA fine-tuned Whisper model with speaker-preserving voice cloning.</p>
            </div>
            <div>
                <div class="sb-footer-col-title">Navigation</div>
                <div class="sb-footer-links">
                    <a href="#sb-how">How it works</a>
                    <a href="#sb-demo-video">Demo video</a>
                    <a href="#sb-samples">Audio samples</a>
                    <a href="#sb-system">Live demo</a>
                    <a href="#sb-feedback">Feedback</a>
                </div>
            </div>
            <div>
                <div class="sb-footer-col-title">Contact</div>
                <div class="sb-footer-links">
                    <a href="#sb-feedback">Leave feedback</a>
                </div>
            </div>
        </div>
        <div class="sb-footer-bottom">
            <span class="sb-footer-copy">SpeechBridge — University Final Year Project</span>
            <span class="sb-footer-copy">Multilingual to English Speech Translation · Fine-tuned on French</span>
        </div>
    </footer>
    """)

    # ── Event handlers ──
    status_timer.tick(fn=poll_model_status, outputs=[model_status_display, init_btn, status_timer])
    init_btn.click(fn=do_init_models, outputs=[model_status_display, init_btn])
    translate_btn.click(
        fn=run_app,
        inputs=[mode, audio_input, input_lang],
        outputs=[translated_text, output_audio, runtime_info, pipeline_status]
    )


if __name__ == "__main__":
    init_thread = threading.Thread(target=background_init, daemon=True)
    init_thread.start()
    demo.launch(css=CUSTOM_CSS)