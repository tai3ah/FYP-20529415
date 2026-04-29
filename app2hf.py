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

from huggingface_hub import snapshot_download

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
TORCH_DTYPE = torch.float32

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

    if not LORA_DIR.exists() or not any(LORA_DIR.iterdir()):
        print("Downloading LoRA weights from HuggingFace...")
        snapshot_download(
            repo_id="tai3ah/whisper-medium-french-lora",
            local_dir=str(LORA_DIR),
            local_dir_use_symlinks=False
        )

    t0 = time.time()
    ST_PROC = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    base_model.eval()
    ST_MODEL = PeftModel.from_pretrained(base_model, str(LORA_DIR)).to(DEVICE)
    ST_MODEL = ST_MODEL.float()
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
                <circle cx="7" cy="7" r="6" stroke="#2f6f73" stroke-width="1.5"/>
                <path d="M4.5 7l2 2 3-3" stroke="#2f6f73" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
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
    "French (★)": "french",
    "Other languages": "other",
}

_VOICE_ENCODER = None

def _get_voice_encoder():
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        _VOICE_ENCODER = VoiceEncoder()
    return _VOICE_ENCODER

def compute_speaker_similarity(input_path: str, output_path: str) -> str:
    try:
        enc = _get_voice_encoder()
        wav_in  = preprocess_wav(input_path)
        wav_out = preprocess_wav(output_path)
        e_in  = enc.embed_utterance(wav_in)
        e_out = enc.embed_utterance(wav_out)
        score = float(np.dot(e_in, e_out) / (np.linalg.norm(e_in) * np.linalg.norm(e_out)))
        return f"{score:.3f}"
    except Exception:
        return "unavailable"


@torch.inference_mode()
def translate_to_en(audio_16k: np.ndarray, sr: int, source_lang: str = "french") -> str:
    inputs = ST_PROC(audio_16k, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(DEVICE).float()

    if source_lang == "french":
        forced_decoder_ids = ST_PROC.get_decoder_prompt_ids(language="french", task="translate")
        gen_ids = ST_MODEL.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=1,
            max_new_tokens=192
        )
    else:
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
    sim = compute_speaker_similarity(audio_path, str(out_wav))
    progress(1.0, desc="Done")
    timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
              f"Speech synthesis: {t_tts - t_trans:.2f}s\nTotal: {t_tts - t0:.2f}s\n"
              f"Speaker similarity: {sim}")
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
    sim = compute_speaker_similarity(audio_path, str(out_wav))
    progress(1.0, desc="Done")
    timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
              f"Reference audio prep: {t_ref - t_trans:.2f}s\nVoice cloning ({num_chunks} chunk(s)): {t_tts - t_ref:.2f}s\n"
              f"Total: {t_tts - t0:.2f}s\n"
              f"Speaker similarity: {sim}")
    lang_label = "French (fine-tuned)" if source_lang == "french" else "Auto-detected"
    return (en_text, str(out_wav), timing, f"Completed. {lang_label} → English, voice cloned.")


def run_app(mode: str, audio_path: str, input_lang_label: str = "French (★)", progress=gr.Progress()):
    source_lang = LANG_MAP.get(input_lang_label, "french")
    if mode == "Translate without voice cloning":
        return run_standard_pipeline(audio_path, source_lang=source_lang, progress=progress)
    if mode == "Translate with voice cloning":
        return run_voice_clone_pipeline(audio_path, source_lang=source_lang, progress=progress)
    return ("Invalid mode.", None, "", "Please select a valid mode.")


# =========================================================
# CSS
# =========================================================
# =========================================================
# CSS
# =========================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #f6f2eb;
    --bg-soft: #efe8dd;
    --paper: rgba(255,255,255,0.58);
    --paper-strong: rgba(255,255,255,0.78);
    --line: rgba(39,72,74,0.12);
    --line-strong: rgba(39,72,74,0.20);
    --text: #1f2d2d;
    --muted: #667575;
    --brand: #375f61;
    --brand-deep: #27484a;
    --accent: #dce8e6;
    --shadow: 0 20px 60px rgba(31,45,45,0.08);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Ensure all sections sit above canvas */
#sb-nav, #sb-hero, .sb-stats-strip, section, footer, #sb-feedback,
.gradio-container > * { position: relative; z-index: 1; }



h1, h2, h3, .sb-logo, .sb-section-title, .sb-hero-title, .sb-step-title, .sb-footer-brand {
    font-family: 'Cormorant Garamond', serif !important;
}

#sb-nav {
    position: sticky; top: 0; z-index: 100;
    background: rgba(246,242,235,0.82);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--line);
    padding: 0 52px;
    display: flex; align-items: center; justify-content: space-between;
    height: 74px;
}
.sb-logo {
    font-weight: 700; font-size: 1.85rem; color: var(--brand-deep);
    display: flex; align-items: center; gap: 10px;
    text-decoration: none;
}
.sb-nav-links { display: flex; gap: 28px; align-items: center; }
.sb-nav-links a {
    color: var(--muted); text-decoration: none;
    font-size: 0.80rem; font-weight: 600;
    letter-spacing: 0.10em; text-transform: uppercase;
    transition: color 0.2s; cursor: pointer;
}
.sb-nav-links a:hover { color: var(--brand-deep); }
.sb-nav-cta {
    background: linear-gradient(135deg, var(--brand), var(--brand-deep));
    color: #f7f3ee !important;
    padding: 10px 22px; font-size: 0.875rem; font-weight: 600;
    text-decoration: none; border-radius: 999px; border: none;
    box-shadow: 0 10px 28px rgba(39,72,74,0.16);
    transition: box-shadow 0.2s ease, transform 0.15s ease;
    display: inline-block;
}
.sb-nav-cta:hover { transform: translateY(-1px); }

#sb-hero {
    position: relative;
    min-height: 100vh;
    padding: 120px 52px 92px;
    display: flex; align-items: center; justify-content: center;
    text-align: center;
    overflow: hidden;
    background:
      radial-gradient(circle at 14% 18%, rgba(55,95,97,0.12), transparent 26%),
      radial-gradient(circle at 82% 72%, rgba(220,232,230,0.95), transparent 24%),
      linear-gradient(180deg, #f7f3ee 0%, #f6f2eb 100%);
}
.sb-hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(circle at 50% 40%, rgba(255,255,255,0.42), transparent 28%),
      linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0));
    pointer-events: none;
}
.sb-hero-inner {
    position: relative; z-index: 3;
    max-width: 940px; margin: 0 auto;
}
.sb-hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.48);
    border: 1px solid var(--line);
    padding: 7px 16px; border-radius: 999px;
    font-size: 0.78rem; color: var(--brand);
    margin-bottom: 24px; backdrop-filter: blur(8px);
    text-transform: uppercase; letter-spacing: 0.10em; font-weight: 700;
}
.sb-hero-badge-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--brand);
}
.sb-hero-title {
    font-weight: 700;
    font-size: clamp(3.4rem, 7vw, 5.8rem);
    color: var(--brand-deep); margin-bottom: 20px; line-height: 1.02;
}
.sb-wavy { color: var(--brand); }
.sb-hero-desc {
    font-size: 1.02rem; line-height: 1.95; color: var(--muted);
    max-width: 640px; margin: 0 auto 38px; font-weight: 400;
}
.sb-hero-actions {
    display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;
}
.sb-btn-primary, .sb-btn-ghost {
    padding: 14px 30px; font-size: 0.92rem; font-weight: 600;
    text-decoration: none; border-radius: 999px; display: inline-flex; align-items: center; gap: 8px;
}
.sb-btn-primary {
    background: linear-gradient(135deg, var(--brand), var(--brand-deep)); color: #f7f3ee;
    box-shadow: 0 14px 30px rgba(39,72,74,0.16);
}
.sb-btn-ghost {
    background: rgba(255,255,255,0.44); color: var(--brand-deep);
    border: 1px solid var(--line);
}
.sb-hero-panel {
    margin: 46px auto 0;
    width: min(840px, 100%);
    padding: 18px;
    border-radius: 28px;
    background: rgba(255,255,255,0.52);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
}
.sb-hero-wave {
    height: 116px;
    border-radius: 18px;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.54)),
      radial-gradient(circle at 22% 50%, rgba(55,95,97,0.12), transparent 26%),
      radial-gradient(circle at 78% 44%, rgba(220,232,230,0.92), transparent 22%);
    position: relative; overflow: hidden;
    border: 1px solid rgba(39,72,74,0.08);
}
.sb-hero-wave::before, .sb-hero-wave::after {
    content: '';
    position: absolute; left: -10%; width: 120%; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(55,95,97,0.38), transparent);
    animation: sbWave 6s linear infinite;
}
.sb-hero-wave::before { top: 42%; box-shadow: 0 18px 0 rgba(55,95,97,0.18), 0 -18px 0 rgba(55,95,97,0.12); }
.sb-hero-wave::after { top: 60%; animation-duration: 8s; animation-direction: reverse; box-shadow: 0 16px 0 rgba(55,95,97,0.14), 0 -16px 0 rgba(55,95,97,0.10); }
@keyframes sbWave { from { transform: translateX(-6%); } to { transform: translateX(6%); } }

.sb-stats-strip {
    background: rgba(255,255,255,0.38);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 30px 52px;
    display: flex; justify-content: center; gap: 80px; flex-wrap: wrap;
    backdrop-filter: blur(8px);
}
.sb-stat-item { text-align: center; }
.sb-stat-num {
    font-family: 'Cormorant Garamond', serif !important; font-weight: 700; font-size: 2.4rem;
    color: var(--brand-deep); line-height: 1; margin-bottom: 5px;
}
.sb-stat-label {
    font-size: 0.74rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700;
}

.sb-section { padding: 100px 52px; }
.sb-section-dark { background: var(--bg); }
.sb-section-alt  { background: var(--bg-soft); }
.sb-section-label {
    font-size: 0.82rem; font-weight: 700; color: var(--brand);
    margin-bottom: 10px; display: inline-flex; align-items: center; gap: 10px;
    text-transform: uppercase; letter-spacing: 0.12em;
}
.sb-section-label::before {
    content: ''; width: 26px; height: 1px;
    background: linear-gradient(90deg, var(--brand), rgba(55,95,97,0.22));
    border-radius: 2px; flex-shrink: 0;
}
.sb-section-title {
    font-weight: 700; font-size: clamp(2.3rem, 4vw, 3.4rem);
    color: var(--brand-deep); margin-bottom: 14px; line-height: 1.08;
}
.sb-section-sub {
    font-size: 1rem; line-height: 1.95; color: var(--muted);
    max-width: 660px; font-weight: 400; margin-bottom: 44px;
}

.sb-note {
    display: inline-block;
    background: rgba(255,255,255,0.58);
    border: 1px solid var(--line);
    padding: 8px 14px; border-radius: 999px;
    font-size: 0.82rem; color: var(--brand); font-weight: 600;
    box-shadow: 0 8px 20px rgba(39,72,74,0.05);
}
.sb-note-float { position: absolute; z-index: 3; }

.sb-steps-row { display: flex; align-items: flex-start; gap: 10px; }
.sb-step {
    flex: 1;
    background: rgba(255,255,255,0.56);
    border: 1px solid var(--line); border-radius: 24px;
    padding: 30px 24px;
    box-shadow: 0 12px 30px rgba(39,72,74,0.05);
    backdrop-filter: blur(10px); position: relative;
}
.sb-step-num {
    font-size: 0.82rem; font-weight: 700;
    color: var(--brand); margin-bottom: 14px;
    background: rgba(55,95,97,0.10);
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 999px; letter-spacing: 0.06em;
}
.sb-step-title {
    font-weight: 700; font-size: 1.35rem; color: var(--brand-deep); margin-bottom: 10px;
}
.sb-step-desc { font-size: 0.92rem; line-height: 1.85; color: var(--muted); }
.sb-step-arrow {
    flex: 0 0 52px;
    display: flex; align-items: center; justify-content: center; padding-top: 52px;
}
.sb-arrow-svg { width: 52px; height: 28px; overflow: visible; }
.sb-arrow-svg .a-path {
    stroke-dasharray: 200; stroke-dashoffset: 200;
    transition: stroke-dashoffset 0.85s cubic-bezier(0.4, 0, 0.2, 1);
}
.sb-arrow-svg .a-head {
    stroke-dasharray: 35; stroke-dashoffset: 35;
    transition: stroke-dashoffset 0.4s ease 0.75s;
}
.sb-arrow-svg.drawn .a-path { stroke-dashoffset: 0; }
.sb-arrow-svg.drawn .a-head { stroke-dashoffset: 0; }

.sb-video-container {
    position: relative; padding-bottom: 56.25%;
    background: rgba(255,255,255,0.50);
    border: 1px solid var(--line);
    border-radius: 24px; overflow: hidden; margin-top: 36px;
    box-shadow: var(--shadow);
}
.sb-video-container video {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;
}
.sb-video-placeholder {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.25));
}
.sb-play-btn {
    width: 76px; height: 76px; border-radius: 50%; background: linear-gradient(135deg, var(--brand), var(--brand-deep));
    box-shadow: 0 12px 26px rgba(39,72,74,0.18);
    display: flex; align-items: center; justify-content: center;
}
.sb-video-placeholder p { color: var(--muted); font-size: 1rem; }

.sb-samples-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px; margin-top: 40px;
}
.sb-sample-card {
    background: rgba(255,255,255,0.56);
    border: 1px solid var(--line);
    border-radius: 20px; padding: 22px 24px;
    display: flex; flex-direction: column; gap: 10px;
    box-shadow: 0 10px 24px rgba(39,72,74,0.05);
}
.sb-sample-label {
    font-size: 0.74rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.10em;
}
.sb-sample-name {
    font-weight: 700; font-size: 1.12rem; color: var(--brand-deep);
}
.sb-sample-dl {
    display: inline-block; margin-top: 4px;
    font-size: 0.84rem; font-weight: 600;
    padding: 9px 16px; border-radius: 999px;
    background: rgba(220,232,230,0.72);
    border: 1px solid var(--line);
    color: var(--brand-deep); text-decoration: none;
}

/* Demo section */
#sb-system {
    position: relative;
    padding: 108px 52px !important;
    background: linear-gradient(180deg, var(--bg-soft), var(--bg));
}
#sb-demo-inner,
#sb-demo-inner > .block,
#sb-demo-inner > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.sb-demo-shell {
    position: relative;
    padding: 28px;
    border-radius: var(--radius-xl);
    background: rgba(255,255,255,0.52);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
}


.sb-model-status {
    display: inline-flex;
    align-items: center;
    gap: 10px;

    margin-bottom: 22px;
    padding: 10px 16px;
    border-radius: 999px;
    border: 1px solid var(--line);

    background: rgba(255,255,255,0.58);
    color: var(--brand);
    font-size: 0.82rem;
    font-weight: 600;

    box-shadow: 0 8px 20px rgba(39,72,74,0.05);
}
.sb-model-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--brand);
}
.sb-spin {
    display: inline-block;
    animation: sbSpin 1.3s linear infinite;
}
@keyframes sbSpin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.sb-demo-grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 22px;
    align-items: start;
}
.sb-panel {
    padding: 22px;
    border-radius: 24px;
    background: rgba(255,255,255,0.56);
    border: 1px solid var(--line);
}
.sb-panel-title {
    margin: 0 0 18px 0;
    color: var(--brand-deep);
    font-size: 1.1rem;
    font-weight: 700;
}
.sb-panel-subtitle {
    margin: 0 0 18px 0;
    color: var(--muted);
    font-size: 0.90rem;
    line-height: 1.8;
}

#sb-demo-inner .block,
#sb-demo-inner .wrap,
#sb-demo-inner .form,
#sb-demo-inner fieldset,
#sb-demo-inner .gr-group,
#sb-demo-inner .gr-box,
#sb-demo-inner .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#sb-demo-inner .label-wrap,
#sb-demo-inner [data-testid='block-label'] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0 8px 0 !important;
}
#sb-demo-inner .label-wrap *,
#sb-demo-inner [data-testid='block-label'] * {
    color: var(--text) !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    background: transparent !important;
}

#input-lang label,
#output-lang label,
#input-lang [data-testid='block-label'] *,
#output-lang [data-testid='block-label'] * {
    color: var(--text) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    font-weight: 700 !important;
}

#sb-demo-inner textarea,
#sb-demo-inner input,
#sb-demo-inner .gradio-textbox,
#sb-demo-inner [data-testid='textbox'],
#sb-demo-inner [data-testid='dropdown'],
#sb-demo-inner .gr-dropdown,
#sb-demo-inner select {
    color: var(--text) !important;
}

#sb-demo-inner textarea {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    line-height: 1.75 !important;
    font-size: 0.94rem !important;
}
#sb-demo-inner textarea::placeholder {
    color: #839191 !important;
}

#sb-demo-inner [data-testid='dropdown'],
#sb-demo-inner .gr-dropdown,
#sb-demo-inner select,
#sb-demo-inner .wrap.svelte-1hnfib6 {
    background: rgba(255,255,255,0.76) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
}
#sb-demo-inner [data-testid='dropdown'] input,
#sb-demo-inner input[type='text'],
#sb-demo-inner input[type='search'] {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
}

#sb-demo-inner fieldset > div,
#sb-demo-inner [role='radiogroup'] {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
}
#sb-demo-inner fieldset label,
#sb-demo-inner [role='radiogroup'] label {
    margin: 0 !important;
    padding: 12px 14px !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.66) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
}
#sb-demo-inner fieldset label:has(input:checked),
#sb-demo-inner [role='radiogroup'] label:has(input:checked) {
    background: rgba(220,232,230,0.80) !important;
    border-color: rgba(39,72,74,0.24) !important;
}
#sb-demo-inner input[type='radio'] {
    accent-color: var(--brand) !important;
}

#sb-demo-inner button,
#sb-demo-inner .btn {
    border-radius: var(--radius-pill) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
#sb-demo-inner button.primary,
#sb-demo-inner button[variant='primary'] {
    background: linear-gradient(135deg, var(--brand), var(--brand-deep)) !important;
    color: #f7f3ee !important;
    border: none !important;
    box-shadow: 0 12px 28px rgba(39,72,74,0.16) !important;
}
#sb-demo-inner button.secondary,
#sb-demo-inner button[variant='secondary'] {
    background: rgba(255,255,255,0.64) !important;
    color: var(--brand-deep) !important;
    border: 1px solid var(--line) !important;
}

/* HF/Gradio scrolling and layout stability fix */
html, body {
    overflow-x: hidden !important;
}

.gradio-container,
.gradio-container > div {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

#sb-demo-inner,
#sb-demo-inner * {
    max-width: 100% !important;
}

.sb-demo-grid {
    width: 100% !important;
    max-width: 100% !important;
    overflow: visible !important;
}

.sb-demo-grid > div {
    min-width: 0 !important;
}

#sb-system {
    overflow-x: hidden !important;
}

#sb-hero {
    min-height: 600px !important;
}



#sb-audio-in,
#sb-audio-out,
#sb-text-out,
#sb-runtime-box {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    box-shadow: none !important;
}
#sb-audio-in,
#sb-audio-out {
    min-height: 180px !important;
}
#sb-text-out textarea {
    min-height: 180px !important;
    background: transparent !important;
    border: none !important;
}
#sb-runtime-box textarea {
    min-height: 110px !important;
    background: transparent !important;
    border: none !important;
}
#sb-status-box {
    display: none !important;
}

.sb-helper {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.8;
    margin-top: 12px;
}
.sb-helper strong { color: var(--brand); }

#sb-feedback {
    background: var(--bg);
    padding: 100px 52px;
    position: relative; overflow: hidden;
}
.sb-feedback-header {
    display: flex; justify-content: space-between; align-items: flex-end;
    margin-bottom: 48px; flex-wrap: wrap; gap: 24px;
}
.sb-feedback-form-btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.70);
    border: 1px solid var(--line);
    color: var(--brand-deep); padding: 10px 22px;
    font-size: 0.875rem; font-weight: 600;
    text-decoration: none; border-radius: 999px;
    box-shadow: 0 8px 20px rgba(39,72,74,0.05);
}
.sb-feedback-carousel {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 20px; min-height: 220px;
}
.sb-feedback-card {
    background: rgba(255,255,255,0.56);
    border: 1px solid var(--line);
    border-radius: 20px; padding: 28px;
    display: flex; flex-direction: column; gap: 14px;
    box-shadow: 0 10px 24px rgba(39,72,74,0.05);
}
.sb-feedback-quote {
    font-size: 0.95rem; line-height: 1.8;
    color: var(--muted); font-style: italic; flex: 1;
}
.sb-feedback-name {
    font-size: 1rem; font-weight: 700; color: var(--brand-deep);
}
.sb-feedback-placeholder {
    grid-column: 1 / -1; color: var(--muted);
    font-size: 1rem;
    display: flex; align-items: center; justify-content: center;
    min-height: 120px; text-align: center;
    border: 1px dashed var(--line); border-radius: 20px;
    background: rgba(255,255,255,0.38);
}
.sb-carousel-controls {
    display: flex; justify-content: center; gap: 8px; margin-top: 28px;
}
.sb-carousel-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: rgba(55,95,97,0.2); cursor: pointer; border: none;
    transition: background 0.2s ease, transform 0.2s ease;
}
.sb-carousel-dot.active { background: var(--brand); transform: scale(1.3); }

#sb-footer { background: var(--brand-deep); padding: 64px 52px 44px; }
.sb-footer-grid {
    display: grid; grid-template-columns: 1.5fr 1fr 1fr;
    gap: 48px; padding-bottom: 40px;
    border-bottom: 1px solid rgba(220,239,240,0.10);
    margin-bottom: 32px;
}
.sb-footer-brand {
    font-weight: 700; font-size: 1.75rem; color: #dceff0; margin-bottom: 10px;
}
.sb-footer-tagline {
    font-size: 0.9rem; color: rgba(220,239,240,0.66);
    line-height: 1.8; max-width: 320px;
}
.sb-footer-col-title {
    font-size: 0.82rem;
    font-weight: 700; color: rgba(220,239,240,0.82);
    margin-bottom: 16px; letter-spacing: 0.12em; text-transform: uppercase;
}
.sb-footer-links { display: flex; flex-direction: column; gap: 10px; }
.sb-footer-links a {
    color: rgba(220,239,240,0.66); text-decoration: none;
    font-size: 0.9rem; transition: color 0.2s;
}
.sb-footer-links a:hover { color: #dceff0; }
.sb-footer-bottom {
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 12px;
}
.sb-footer-copy { font-size: 0.8rem; color: rgba(220,239,240,0.42); }

.sb-reveal {
    opacity: 0; transform: translateY(30px);
    transition: opacity 0.7s ease, transform 0.7s ease;
}
.sb-reveal.sb-revealed { opacity: 1; transform: translateY(0); }
.sb-d1 { transition-delay: 0.10s; }
.sb-d2 { transition-delay: 0.20s; }
.sb-d3 { transition-delay: 0.30s; }
.sb-d4 { transition-delay: 0.40s; }
.sb-d5 { transition-delay: 0.50s; }

@media (max-width: 900px) {
    #sb-nav { padding: 0 24px; }
    .sb-section, #sb-hero, #sb-feedback, #sb-footer, #sb-system {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
    .sb-feedback-carousel { grid-template-columns: 1fr; }
    .sb-footer-grid { grid-template-columns: 1fr; gap: 32px; }
    .sb-nav-links { display: none; }
    .sb-stats-strip { gap: 40px; padding: 28px 24px; }
    .sb-steps-row { flex-direction: column; gap: 16px; }
    .sb-step-arrow { display: none; }
}
@media (max-width: 640px) {
    .sb-samples-grid { grid-template-columns: 1fr; }
    .sb-hero-title { font-size: 3rem; }
    #sb-control-row, #sb-output-row {
        flex-direction: column !important;
        gap: 16px !important;
    }
}

.gradio-container .block { border-radius: 12px !important; }
"""

# =========================================================
# UI
# =========================================================
with gr.Blocks(title="SpeechBridge", css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <nav id="sb-nav">
        <div class="sb-logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" style="flex-shrink:0;">
                <circle cx="14" cy="14" r="12" stroke="#375f61" stroke-width="1.8" fill="rgba(220,239,240,0.3)"/>
                <path d="M6 14 Q10 8 14 14 Q18 20 22 14" stroke="#375f61" stroke-width="2.2" fill="none" stroke-linecap="round"/>
            </svg>
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

    gr.HTML("""
    <section id="sb-hero">
        <div class="sb-hero-inner">
            <div class="sb-hero-badge">
                <div class="sb-hero-badge-dot"></div>
                Multilingual to English, fine-tuned on French
            </div>
            <h1 class="sb-hero-title">
                Speak any language.<br><span class="sb-wavy">Be heard in English.</span>
            </h1>
            <p class="sb-hero-desc">
                SpeechBridge translates speech into English while preserving the natural feel of the original speaker.
                Choose standard speech output or optional voice cloning.
            </p>
            <div class="sb-hero-actions">
                <a class="sb-btn-primary" href="#sb-system">Try it yourself</a>
                <a class="sb-btn-ghost" href="#sb-how">See how it works</a>
            </div>
            <div class="sb-hero-panel">
                <div class="sb-hero-wave"></div>
            </div>
        </div>
    </section>
    """)

    gr.HTML("""
    <div class="sb-stats-strip">
        <div class="sb-stat-item sb-reveal">
            <div class="sb-stat-num">99</div>
            <div class="sb-stat-label">Languages</div>
        </div>
        <div class="sb-stat-item sb-reveal sb-d1">
            <div class="sb-stat-num">0.81</div>
            <div class="sb-stat-label">Voice Similarity</div>
        </div>
        <div class="sb-stat-item sb-reveal sb-d2">
            <div class="sb-stat-num">2</div>
            <div class="sb-stat-label">Output Modes</div>
        </div>
    </div>
    """)

    gr.HTML("""
    <section id="sb-how" class="sb-section sb-section-alt" style="overflow:hidden;">
        <div class="sb-section-label">How it works</div>
        <h2 class="sb-section-title">Three steps to translated speech</h2>
        <p class="sb-section-sub">
            Upload any speech clip and receive translated English audio, optionally in the original speaker's voice.
            Optimised for French with a LoRA fine-tuned Whisper model.
        </p>
        <div class="sb-steps-row">
            <div class="sb-step sb-reveal">
                <div class="sb-step-num">01</div>
                <div class="sb-step-title">Upload your audio</div>
                <div class="sb-step-desc">Record or upload a clip in any format. Select your source language. French uses the fine-tuned model, other languages fall back to base Whisper.</div>
            </div>
            <div class="sb-step-arrow sb-reveal sb-d1">
                <svg class="sb-arrow-svg" viewBox="0 0 52 28" fill="none">
                    <path class="a-path" d="M2 14 C14 6 32 6 46 14" stroke="#375f61" stroke-width="2" stroke-linecap="round"/>
                    <path class="a-head" d="M40 8 L47 14 L40 20" stroke="#375f61" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="sb-step sb-reveal sb-d2" style="position:relative;">
                <div class="sb-note sb-note-float" style="top:-14px;right:-8px;">~30s per clip</div>
                <div class="sb-step-num">02</div>
                <div class="sb-step-title">Choose output mode</div>
                <div class="sb-step-desc">Select standard translation for clean English speech, or voice cloning to preserve the original speaker's voice and identity.</div>
            </div>
            <div class="sb-step-arrow sb-reveal sb-d3">
                <svg class="sb-arrow-svg" viewBox="0 0 52 28" fill="none">
                    <path class="a-path" d="M2 14 C14 22 32 22 46 14" stroke="#375f61" stroke-width="2" stroke-linecap="round"/>
                    <path class="a-head" d="M40 8 L47 14 L40 20" stroke="#375f61" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="sb-step sb-reveal sb-d4">
                <div class="sb-step-num">03</div>
                <div class="sb-step-title">Receive translated speech</div>
                <div class="sb-step-desc">Get translated English text and synthesised audio. With voice cloning, it sounds like the same person speaking English.</div>
            </div>
        </div>
    </section>
    """)

    gr.HTML("""
    <section id="sb-demo-video" class="sb-section sb-section-dark">
        <div class="sb-section-label">See it in action</div>
        <h2 class="sb-section-title sb-reveal">Watch a full translation run</h2>
    </section>
    """)

    with gr.Column(elem_id="sb-demo-video-box", elem_classes=["sb-section", "sb-section-dark"]):
        gr.HTML("""
        <div style="position:relative;padding-bottom:56.25%;border-radius:12px;overflow:hidden;">
            <iframe
                style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;border-radius:12px;"
                src="https://www.youtube.com/embed/Q9OkTI7ZvgE"
                allowfullscreen>
            </iframe>
        </div>
        """)

    gr.HTML("""
    <section id="sb-samples" class="sb-section sb-section-alt">
        <div class="sb-section-label">Sample audio</div>
        <h2 class="sb-section-title sb-reveal">Try it with one of our clips</h2>
        <p class="sb-section-sub sb-reveal sb-d1">Download any French speech sample below and upload it into the system to see it in action.</p>
        <div class="sb-samples-grid">
            <div class="sb-sample-card sb-reveal">
                <div class="sb-sample-label">Sample 1</div>
                <div class="sb-sample-name">(Male)</div>
                <a class="sb-sample-dl" href="/audio/audio6.m4a" download>Download clip →</a>
            </div>
            <div class="sb-sample-card sb-reveal sb-d1">
                <div class="sb-sample-label">Sample 2</div>
                <div class="sb-sample-name">(Young female)</div>
                <a class="sb-sample-dl" href="/audio/audio2.m4a" download>Download clip →</a>
            </div>
            <div class="sb-sample-card sb-reveal sb-d2">
                <div class="sb-sample-label">Sample 3</div>
                <div class="sb-sample-name">(Young female)</div>
                <a class="sb-sample-dl" href="/audio/audio3.m4a" download>Download clip →</a>
            </div>
            <div class="sb-sample-card sb-reveal sb-d3">
                <div class="sb-sample-label">Sample 4</div>
                <div class="sb-sample-name">(Young female)</div>
                <a class="sb-sample-dl" href="/audio/audio7.m4a" download>Download clip →</a>
            </div>
            <div class="sb-sample-card sb-reveal sb-d4">
                <div class="sb-sample-label">Sample 5</div>
                <div class="sb-sample-name">(Adult female)</div>
                <a class="sb-sample-dl" href="/audio/audio9.m4a" download>Download clip →</a>
            </div>
        </div>
    </section>
    """)

    with gr.Column(elem_id="sb-system"):
        gr.HTML(
            """
            <div class="sb-section-label">Live demo</div>
            <h2 class="sb-section-title">Try SpeechBridge</h2>
            <p class="sb-section-sub">
                Upload or record speech and receive an English translation with or without voice cloning.
            </p>
            <div class="sb-note">Best results start with French input</div>
            """
        )

        with gr.Column(elem_id="sb-demo-inner"):
            with gr.Group(elem_classes=["sb-demo-shell"]):
                model_status_display = gr.HTML(value=_render_model_status())
                init_btn = gr.Button(
                    "Initialise models",
                    variant="secondary",
                    visible=not (TRANSLATION_READY and XTTS_READY),
                )
                status_timer = gr.Timer(value=8, active=True)

                with gr.Row(elem_classes=["sb-demo-grid"]):
                    with gr.Column(elem_classes=["sb-panel"]):
                        gr.HTML('<h3 class="sb-panel-title">Controls</h3>')
                        gr.HTML(
                            '<p class="sb-panel-subtitle">Choose the source language path, select the output mode, then upload or record your speech sample.</p>'
                        )

                        input_lang = gr.Dropdown(
                            choices=["French (★)", "Other languages"],
                            value="French (★)",
                            label="Input language",
                            interactive=True,
                            elem_id="input-lang"
                        )
                        output_lang = gr.Dropdown(
                            choices=["English"],
                            value="English",
                            label="Output language",
                            interactive=True,
                            elem_id="output-lang"
                        )
                        mode = gr.Radio(
                            choices=["Translate without voice cloning", "Translate with voice cloning"],
                            value="Translate without voice cloning",
                            label="Translation mode",
                        )
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Upload or record audio",
                            elem_id="sb-audio-in",
                        )
                        gr.HTML(
                            "<div class='sb-helper'>"
                            "Use <strong>voice cloning</strong> when you want the English output to mimic the original speaker. "
                            "The same clip is used as the XTTS reference audio."
                            "</div>"
                        )
                        run_btn = gr.Button("Generate output", variant="primary")
                        status_text = gr.Textbox(label="Status", interactive=False, elem_id="sb-status-box")

                    with gr.Column(elem_classes=["sb-panel"]):
                        gr.HTML('<h3 class="sb-panel-title">Results</h3>')
                        gr.HTML(
                            '<p class="sb-panel-subtitle">The translated text, generated English audio, and runtime breakdown will appear here.</p>'
                        )
                        translated_text = gr.Textbox(
                            label="Translated English text",
                            interactive=False,
                            elem_id="sb-text-out",
                            placeholder="Your translated English text will appear here.",
                        )
                        audio_output = gr.Audio(
                            label="Generated English speech",
                            type="filepath",
                            interactive=False,
                            elem_id="sb-audio-out",
                        )
                        runtime_box = gr.Textbox(
                            label="Runtime details",
                            interactive=False,
                            elem_id="sb-runtime-box",
                            placeholder="Processing time and stage timings will appear here.",
                        )


    gr.HTML(f"""
    <section id="sb-feedback">
        <div class="sb-feedback-header">
            <div>
                <div class="sb-section-label">Feedback</div>
                <h2 class="sb-section-title">What users thought</h2>
            </div>
            <a class="sb-feedback-form-btn" href="{GOOGLE_FORM_URL}" target="_blank">Share your thoughts!</a>
        </div>
        <div class="sb-feedback-carousel" id="sb-feedback-carousel">
            <div class="sb-feedback-placeholder">Feedback cards will appear here once responses are loaded.</div>
        </div>
        <div class="sb-carousel-controls" id="sb-carousel-controls"></div>
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

    gr.HTML("""
    <footer id="sb-footer">
        <div class="sb-footer-grid">
            <div>
                <div class="sb-footer-brand">SpeechBridge</div>
                <div class="sb-footer-tagline">A final year project interface for multilingual to English speech translation, with optional voice cloning.</div>
            </div>
            <div>
                <div class="sb-footer-col-title">Explore</div>
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
            <span class="sb-footer-copy">SpeechBridge, University Final Year Project</span>
            <span class="sb-footer-copy">Multilingual to English Speech Translation, Fine-tuned on French</span>
        </div>
    </footer>
    """)

    # ── JS: nav smooth scroll + scroll reveal + arrow draw-in + feedback ──
    demo.load(
        fn=None,
        js=f"""
        function() {{

            /* ── Nav smooth scroll (fixes HF iframe anchor issue) ── */
            function initNavScroll() {{
                document.querySelectorAll('a[href^="#"]').forEach(function(a) {{
                    a.addEventListener('click', function(e) {{
                        var id = this.getAttribute('href').slice(1);
                        var target = document.getElementById(id);
                        if (!target) return;
                        e.preventDefault();
                        target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }});
                }});
            }}
            setTimeout(initNavScroll, 600);

            /* ── Scroll reveal ── */
            function initReveal() {{
                var els = document.querySelectorAll('.sb-reveal');
                if (!els.length) {{ setTimeout(initReveal, 400); return; }}
                var obs = new IntersectionObserver(function(entries) {{
                    entries.forEach(function(entry) {{
                        if (!entry.isIntersecting) return;
                        entry.target.classList.add('sb-revealed');
                        obs.unobserve(entry.target);
                    }});
                }}, {{ threshold: 0.12, rootMargin: '0px 0px -40px 0px' }});
                els.forEach(function(el) {{ obs.observe(el); }});
            }}
            setTimeout(initReveal, 500);

            /* ── Arrow draw-in ── */
            function initArrows() {{
                var how = document.getElementById('sb-how');
                if (!how) {{ setTimeout(initArrows, 400); return; }}
                var arrowObs = new IntersectionObserver(function(entries) {{
                    if (!entries[0].isIntersecting) return;
                    setTimeout(function() {{
                        document.querySelectorAll('.sb-arrow-svg').forEach(function(svg) {{
                            svg.classList.add('drawn');
                        }});
                    }}, 350);
                    arrowObs.disconnect();
                }}, {{ threshold: 0.25 }});
                arrowObs.observe(how);
            }}
            setTimeout(initArrows, 500);

            /* ── Feedback carousel ── */
            var SHEET_URL = 'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:json';
            var PER_PAGE = 3;
            var allFeedback = [];
            var currentPage = 0;
            var autoTimer = null;

            function renderStars(n) {{
                var s = '';
                var filled = Math.round(n || 0);
                for (var i = 1; i <= 5; i++) {{
                    s += '<span style="color:#2f6f73;font-size:13px;">' + (i <= filled ? '★' : '☆') + '</span>';
                }}
                return s;
            }}
            function renderPage(page) {{
                var carousel = document.getElementById('sb-feedback-carousel');
                var controls = document.getElementById('sb-carousel-controls');
                if (!carousel) return;
                var slice = allFeedback.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE);
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
                    controls.querySelectorAll('.sb-carousel-dot').forEach(function(d, i) {{
                        d.classList.toggle('active', i === page);
                    }});
                }}
                currentPage = page;
            }}
            function buildControls(total) {{
                var controls = document.getElementById('sb-carousel-controls');
                if (!controls || total <= 1) return;
                controls.innerHTML = '';
                controls.style.display = 'flex';
                for (var i = 0; i < total; i++) {{
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
                        var json = JSON.parse(text.substring(text.indexOf('(') + 1, text.lastIndexOf(')')));
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
                        if (!allFeedback.length) return;
                        var total = Math.ceil(allFeedback.length / PER_PAGE);
                        buildControls(total);
                        renderPage(0);
                        if (total > 1) {{
                            autoTimer = setInterval(function() {{
                                renderPage((currentPage + 1) % total);
                            }}, 5000);
                        }}
                    }})
                    .catch(function(e) {{ console.log('Feedback error:', e); }});
            }}
            setTimeout(loadFeedback, 1500);
        }}
        """
    )

    # ── Event handlers ──
    status_timer.tick(fn=poll_model_status, outputs=[model_status_display, init_btn, status_timer])
    init_btn.click(fn=do_init_models, outputs=[model_status_display, init_btn])
    run_btn.click(
        fn=run_app,
        inputs=[mode, audio_input, input_lang],
        outputs=[translated_text, audio_output, runtime_box, status_text]
    )


if __name__ == "__main__":
    from fastapi.staticfiles import StaticFiles

    def delayed_init():
        time.sleep(10)
        background_init()

    init_thread = threading.Thread(target=delayed_init, daemon=True)
    init_thread.start()

    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    try:
        demo.server_app.mount("/audio", StaticFiles(directory="audio_samples"), name="audio")
    except Exception as e:
        print(f"Audio mount failed: {e}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        demo.close()
        print("\nStopped.")
        print("\nStopped.")