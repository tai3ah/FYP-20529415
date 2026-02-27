import re
import time
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# -----------------------------
# Config (CPU-only)
# -----------------------------
BASE_MODEL_ID = "openai/whisper-medium"
LORA_DIR = Path("final_lora")

PIPER_DIR = Path("piper_models")
PIPER_VOICE = "en_US-lessac-medium"
PIPER_MODEL = PIPER_DIR / f"{PIPER_VOICE}.onnx"
PIPER_CFG = PIPER_DIR / f"{PIPER_VOICE}.onnx.json"

DEVICE = "cpu"
TORCH_DTYPE = torch.float32

MAX_AUDIO_SECONDS = 20  # keep short for CPU practicality


def normalise_text_for_tts(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def piper_tts(text: str, out_wav: Path):
    cmd = [
        "piper",
        "--model", str(PIPER_MODEL),
        "--config", str(PIPER_CFG),
        "--output_file", str(out_wav),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def load_audio_mono_16k(path: str, max_seconds: int):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) > max_seconds * 16000:
        y = y[: max_seconds * 16000]
    return y, 16000


@torch.inference_mode()
def translate_fr_to_en(audio_16k: np.ndarray, sr: int) -> str:
    inputs = ST_PROC(audio_16k, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(DEVICE, dtype=TORCH_DTYPE)

    gen_ids = ST_MODEL.generate(
        input_features=input_features,
        task="translate",
        language="en",
        num_beams=1,
        max_new_tokens=192,
    )
    text = ST_PROC.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.strip()


# -----------------------------
# Startup checks + model load
# -----------------------------
if not (PIPER_MODEL.exists() and PIPER_CFG.exists()):
    raise RuntimeError(
        f"Piper voice not found. Expected:\n{PIPER_MODEL}\n{PIPER_CFG}\n"
        f"Run: python -m piper.download_voices {PIPER_VOICE} --download-dir {PIPER_DIR}"
    )

if not LORA_DIR.exists():
    raise RuntimeError(f"LoRA folder not found at {LORA_DIR}. Copy final_lora into the project folder.")

ST_PROC = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

_base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(DEVICE)
_base.eval()

ST_MODEL = PeftModel.from_pretrained(_base, str(LORA_DIR)).to(DEVICE)
ST_MODEL.eval()


def run_pipeline(audio_path: str):
    if audio_path is None or not Path(audio_path).exists():
        return "No audio provided.", None

    t0 = time.time()
    audio_16k, sr = load_audio_mono_16k(audio_path, MAX_AUDIO_SECONDS)
    t_load = time.time()

    en_text = translate_fr_to_en(audio_16k, sr)
    t_st = time.time()

    en_text_tts = normalise_text_for_tts(en_text)
    if len(en_text_tts) == 0:
        return "Translation produced empty text.", None

    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "tts.wav"
        piper_tts(en_text_tts, out_wav)
        t_tts = time.time()

        info = (
            f"{en_text}\n\n"
            f"Timings (CPU): load {t_load - t0:.2f}s, "
            f"ST {t_st - t_load:.2f}s, "
            f"TTS {t_tts - t_st:.2f}s, "
            f"total {t_tts - t0:.2f}s"
        )

        return info, str(out_wav)


demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Audio(type="filepath", label="Upload French speech audio (.wav/.mp3)"),
    outputs=[
        gr.Textbox(label="English translation and timings"),
        gr.Audio(type="filepath", label="English speech output (Piper, no voice cloning)"),
    ],
    title="FR → EN Speech-to-Speech Translation (CPU demo, no voice cloning)",
    description="Whisper-medium + LoRA for FR→EN speech translation, then Piper TTS for English speech.",
)

if __name__ == "__main__":
    demo.launch()