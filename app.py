# import re
# import time
# import uuid
# import subprocess
# import threading
# from pathlib import Path
# from typing import Optional, Tuple
#
# import gradio as gr
# import numpy as np
# import torch
# import librosa
# import soundfile as sf
# import noisereduce as nr
#
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from peft import PeftModel
# from TTS.api import TTS
# from resemblyzer import VoiceEncoder, preprocess_wav
#
# # =========================================================
# # Config
# # =========================================================
# BASE_MODEL_ID = "openai/whisper-medium"
# LORA_DIR = Path("final_lora")
#
# PIPER_DIR = Path("piper_models")
# PIPER_VOICE = "en_US-lessac-medium"
# PIPER_MODEL = PIPER_DIR / f"{PIPER_VOICE}.onnx"
# PIPER_CFG = PIPER_DIR / f"{PIPER_VOICE}.onnx.json"
#
# XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# XTTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
#
# XTTS_TEMPERATURE = 0.60
# XTTS_REPETITION_PENALTY = 5.0
# XTTS_TOP_K = 50
# XTTS_TOP_P = 0.85
# XTTS_SPEED = 1.0
# MAX_CHUNK_CHARS = 180
#
# WHISPER_MAX_SECONDS = 60
# XTTS_REF_MAX_SECONDS = 30
# WHISPER_SR = 16000
# XTTS_SR = 22050
# XTTS_OUTPUT_SR = 24000
#
# OUT_DIR = Path("outputs")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# TEMP_DIR = Path("outputs/temp")
# TEMP_DIR.mkdir(parents=True, exist_ok=True)
#
# FFMPEG_BIN = "ffmpeg"
#
# SHEET_ID = "1C2ZFxtJ2H4TwnakoV_buVzlNIOkZ2GnBO3o9sIXHR2I"
# GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScrM4CSuzdfhGflaliiDkBLl4vasHCqA3MQcVI_nS5ZglkeCw/viewform"
#
# torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
#
# ST_PROC = None
# ST_MODEL = None
# XTTS_MODEL = None
# TRANSLATION_READY = False
# XTTS_READY = False
#
#
# # =========================================================
# # Audio utilities
# # =========================================================
# def ffmpeg_convert_to_wav(input_path: str, output_path: str, sr: int) -> None:
#     cmd = [FFMPEG_BIN, "-y", "-i", input_path, "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", output_path]
#     subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
#
# def load_audio_for_whisper(audio_path: str) -> Tuple[np.ndarray, int]:
#     converted = str(TEMP_DIR / f"whisper_{uuid.uuid4().hex}.wav")
#     ffmpeg_convert_to_wav(audio_path, converted, sr=WHISPER_SR)
#     audio, _ = librosa.load(converted, sr=WHISPER_SR, mono=True)
#     max_samples = WHISPER_MAX_SECONDS * WHISPER_SR
#     if len(audio) > max_samples:
#         audio = audio[:max_samples]
#     return audio, WHISPER_SR
#
#
# def prepare_xtts_reference(audio_path: str) -> str:
#     converted = str(TEMP_DIR / f"xtts_ref_raw_{uuid.uuid4().hex}.wav")
#     ffmpeg_convert_to_wav(audio_path, converted, sr=XTTS_SR)
#     audio, _ = librosa.load(converted, sr=XTTS_SR, mono=True)
#     max_samples = XTTS_REF_MAX_SECONDS * XTTS_SR
#     if len(audio) > max_samples:
#         audio = audio[:max_samples]
#     if len(audio) > XTTS_SR:
#         noise_clip = audio[:int(0.5 * XTTS_SR)]
#     else:
#         noise_clip = audio[:max(1, len(audio) // 4)]
#     try:
#         audio = nr.reduce_noise(y=audio, sr=XTTS_SR, y_noise=noise_clip, prop_decrease=0.75, stationary=False)
#     except Exception:
#         pass
#     out_path = str(TEMP_DIR / f"xtts_ref_clean_{uuid.uuid4().hex}.wav")
#     sf.write(out_path, audio, XTTS_SR)
#     return out_path
#
# def persist_audio_file(audio_path: str) -> Optional[str]:
#     if audio_path is None or not Path(audio_path).exists():
#         return None
#     src = Path(audio_path)
#     dst = TEMP_DIR / f"ui_audio_{uuid.uuid4().hex}{src.suffix or '.wav'}"
#     dst.write_bytes(src.read_bytes())
#     return str(dst)
# # =========================================================
# # Text utilities
# # =========================================================
# def normalise_text_for_tts(text: str) -> str:
#     return re.sub(r"\s+", " ", text.strip())
#
#
# def split_text_for_xtts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
#     text = str(text).strip()
#     if len(text) <= max_chars:
#         return [text]
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     sentences = [s.strip() for s in sentences if s.strip()]
#     chunks, current = [], ""
#     for sentence in sentences:
#         if not current:
#             current = sentence
#         elif len(current) + 1 + len(sentence) <= max_chars:
#             current += " " + sentence
#         else:
#             chunks.append(current.strip())
#             current = sentence
#     if current:
#         chunks.append(current.strip())
#     final_chunks = []
#     for chunk in chunks:
#         if len(chunk) <= max_chars:
#             final_chunks.append(chunk)
#         else:
#             words, temp = chunk.split(), ""
#             for word in words:
#                 if not temp:
#                     temp = word
#                 elif len(temp) + 1 + len(word) <= max_chars:
#                     temp += " " + word
#                 else:
#                     final_chunks.append(temp.strip())
#                     temp = word
#             if temp:
#                 final_chunks.append(temp.strip())
#     return final_chunks
#
#
# # =========================================================
# # Model loading
# # =========================================================
# def validate_piper_assets() -> Optional[str]:
#     if not (PIPER_MODEL.exists() and PIPER_CFG.exists()):
#         return f"Piper voice files missing at {PIPER_DIR}"
#     return None
#
#
# def load_translation_pipeline() -> str:
#     global ST_PROC, ST_MODEL, TRANSLATION_READY
#     if TRANSLATION_READY:
#         return "Translation model already loaded."
#     if not LORA_DIR.exists():
#         raise RuntimeError(f"LoRA folder not found at: {LORA_DIR}")
#     t0 = time.time()
#     ST_PROC = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
#     base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, low_cpu_mem_usage=True).to(DEVICE)
#     base_model.eval()
#     ST_MODEL = PeftModel.from_pretrained(base_model, str(LORA_DIR)).to(DEVICE)
#     ST_MODEL.eval()
#     TRANSLATION_READY = True
#     return f"Translation model loaded in {time.time() - t0:.2f}s."
#
#
# def load_xtts_model() -> str:
#     global XTTS_MODEL, XTTS_READY
#     if XTTS_READY:
#         return "XTTS model already loaded."
#     t0 = time.time()
#     XTTS_MODEL = TTS(XTTS_MODEL_NAME).to(XTTS_DEVICE)
#     XTTS_READY = True
#     return f"XTTS loaded in {time.time() - t0:.2f}s."
#
#
# def initialise_models() -> str:
#     messages = []
#     try:
#         messages.append(load_translation_pipeline())
#     except Exception as e:
#         messages.append(f"Translation init failed: {e}")
#     try:
#         err = validate_piper_assets()
#         messages.append("Piper TTS assets found." if not err else f"Piper: {err}")
#     except Exception as e:
#         messages.append(f"Piper check failed: {e}")
#     try:
#         messages.append(load_xtts_model())
#     except Exception as e:
#         messages.append(f"XTTS init failed: {e}")
#     return "\n".join(messages)
#
#
# def background_init():
#     print("Background model initialisation started...")
#     msg = initialise_models()
#     print(f"Models ready:\n{msg}")
#
#
# # =========================================================
# # Inference
# # =========================================================
# # @torch.inference_mode()
# # def translate_fr_to_en(audio_16k: np.ndarray, sr: int) -> str:
# #     inputs = ST_PROC(audio_16k, sampling_rate=sr, return_tensors="pt")
# #     input_features = inputs["input_features"].to(DEVICE, dtype=TORCH_DTYPE)
# #     gen_ids = ST_MODEL.generate(input_features=input_features, task="translate", language="en", num_beams=1, max_new_tokens=192)
# #     return ST_PROC.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
#
# LANG_MAP = {
#     "French (fine-tuned)": "french",
#     "Arabic":              "arabic",
#     "Urdu":                "urdu",
#     "Malay":               "malay",
# }
#
# @torch.inference_mode()
# def translate_to_en(audio_16k: np.ndarray, sr: int, source_lang: str = "french") -> str:
#     inputs = ST_PROC(audio_16k, sampling_rate=sr, return_tensors="pt")
#     input_features = inputs["input_features"].to(DEVICE, dtype=TORCH_DTYPE)
#     forced_decoder_ids = ST_PROC.get_decoder_prompt_ids(language=source_lang, task="translate")
#     if source_lang == "french":
#
#         print(">>> Using LoRA fine-tuned model")
#         # LoRA fine-tuned model
#         gen_ids = ST_MODEL.generate(
#             input_features=input_features,
#             forced_decoder_ids=forced_decoder_ids,
#             num_beams=1,
#             max_new_tokens=192
#         )
#     else:
#         print(f">>> Using BASE Whisper for {source_lang}")
#         # Base Whisper-medium (LoRA disabled)
#         ST_MODEL.disable_adapter_layers()
#         try:
#             gen_ids = ST_MODEL.generate(
#                 input_features=input_features,
#                 forced_decoder_ids=forced_decoder_ids,
#                 num_beams=1,
#                 max_new_tokens=192
#             )
#         finally:
#             ST_MODEL.enable_adapter_layers()
#     return ST_PROC.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
#
#
# def piper_tts(text: str, out_wav: Path) -> None:
#     subprocess.run(["piper", "--model", str(PIPER_MODEL), "--config", str(PIPER_CFG), "--output_file", str(out_wav)],
#                    input=text.encode("utf-8"), check=True)
#
#
# def xtts_synthesise(text: str, ref_wav_path: str, out_wav: Path) -> int:
#     chunks = split_text_for_xtts(text)
#     audio_parts = []
#     for chunk in chunks:
#         wav = XTTS_MODEL.tts(text=chunk, speaker_wav=ref_wav_path, language="en",
#                               temperature=XTTS_TEMPERATURE, repetition_penalty=XTTS_REPETITION_PENALTY,
#                               top_k=XTTS_TOP_K, top_p=XTTS_TOP_P, speed=XTTS_SPEED)
#         audio_parts.append(np.asarray(wav, dtype=np.float32))
#     if not audio_parts:
#         raise ValueError("XTTS produced no audio output.")
#     sf.write(str(out_wav), np.concatenate(audio_parts), XTTS_OUTPUT_SR)
#     return len(chunks)
#
#
# # =========================================================
# # Pipeline runners
# # =========================================================
# # def run_standard_pipeline(audio_path: str, progress=gr.Progress()):
# def run_standard_pipeline(audio_path: str, source_lang: str = "french", progress=gr.Progress()):
#     if audio_path is None or not Path(audio_path).exists():
#         return ("No audio provided.", None, "", "Please upload or record a French speech file.")
#     if not TRANSLATION_READY:
#         progress(0.08, desc="Loading translation model")
#         load_translation_pipeline()
#     err = validate_piper_assets()
#     if err:
#         return ("Piper TTS unavailable.", None, "", err)
#     t0 = time.time()
#     progress(0.22, desc="Preparing audio")
#     audio_16k, sr = load_audio_for_whisper(audio_path)
#     t_load = time.time()
#     # progress(0.50, desc="Translating French to English")
#     # en_text = translate_fr_to_en(audio_16k, sr)
#     progress(0.50, desc="Translating to English")
#     en_text = translate_to_en(audio_16k, sr, source_lang=source_lang)
#
#     t_trans = time.time()
#     en_text_clean = normalise_text_for_tts(en_text)
#     if not en_text_clean:
#         return ("Translation produced empty text.", None, "", "Try a clearer audio sample.")
#     progress(0.84, desc="Generating English speech")
#     out_wav = OUT_DIR / f"standard_{uuid.uuid4().hex}.wav"
#     piper_tts(en_text_clean, out_wav)
#     t_tts = time.time()
#     progress(1.0, desc="Done")
#     timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
#               f"Speech synthesis: {t_tts - t_trans:.2f}s\nTotal: {t_tts - t0:.2f}s")
#     # return (en_text, str(out_wav), timing, "Completed. Standard French-to-English translation done.")
#     return (en_text, str(out_wav), timing, f"Completed. {source_lang.title()} speech translated to English.")
#
#
# # def run_voice_clone_pipeline(audio_path: str, progress=gr.Progress()):
# def run_voice_clone_pipeline(audio_path: str, source_lang: str = "french", progress=gr.Progress()):
#     if audio_path is None or not Path(audio_path).exists():
#         return ("No audio provided.", None, "", "Please upload or record a French speech file.")
#     if not TRANSLATION_READY:
#         progress(0.06, desc="Loading translation model")
#         load_translation_pipeline()
#     if not XTTS_READY:
#         progress(0.12, desc="Loading XTTS model")
#         load_xtts_model()
#     t0 = time.time()
#     progress(0.20, desc="Preparing audio")
#     audio_16k, sr = load_audio_for_whisper(audio_path)
#     t_load = time.time()
#     # progress(0.42, desc="Translating French to English")
#     # en_text = translate_fr_to_en(audio_16k, sr)
#     progress(0.42, desc="Translating to English")
#     en_text = translate_to_en(audio_16k, sr, source_lang=source_lang)
#
#     t_trans = time.time()
#     en_text_clean = normalise_text_for_tts(en_text)
#     if not en_text_clean:
#         return ("Translation produced empty text.", None, "", "Try a clearer audio sample.")
#     progress(0.60, desc="Preparing reference audio")
#     ref_wav_path = prepare_xtts_reference(audio_path)
#     t_ref = time.time()
#     progress(0.78, desc="Cloning voice and generating English speech")
#     out_wav = OUT_DIR / f"cloned_{uuid.uuid4().hex}.wav"
#     num_chunks = xtts_synthesise(en_text_clean, ref_wav_path, out_wav)
#     t_tts = time.time()
#     progress(1.0, desc="Done")
#     timing = (f"Audio preparation: {t_load - t0:.2f}s\nSpeech translation: {t_trans - t_load:.2f}s\n"
#               f"Reference audio prep: {t_ref - t_trans:.2f}s\nVoice cloning ({num_chunks} chunk(s)): {t_tts - t_ref:.2f}s\n"
#               f"Total: {t_tts - t0:.2f}s")
#     # return (en_text, str(out_wav), timing, "Completed. French speech translated and synthesised with voice cloning.")
#     return (en_text, str(out_wav), timing, f"Completed. {source_lang.title()} speech translated and synthesised with voice cloning.")
#
#
# # def run_app(mode: str, audio_path: str, progress=gr.Progress()):
# #     if mode == "Translate without voice cloning":
# #         return run_standard_pipeline(audio_path, progress=progress)
# #     if mode == "Translate with voice cloning":
# #         return run_voice_clone_pipeline(audio_path, progress=progress)
# #     return ("Invalid mode.", None, "", "Please select a valid mode.")
#
# def run_app(mode: str, audio_path: str, input_lang_label: str = "French (fine-tuned)", progress=gr.Progress()):
#     source_lang = LANG_MAP.get(input_lang_label, "french")
#     if mode == "Translate without voice cloning":
#         return run_standard_pipeline(audio_path, source_lang=source_lang, progress=progress)
#     if mode == "Translate with voice cloning":
#         return run_voice_clone_pipeline(audio_path, source_lang=source_lang, progress=progress)
#     return ("Invalid mode.", None, "", "Please select a valid mode.")
#
#
# # =========================================================
# # CSS
# # =========================================================
# CUSTOM_CSS = """
# @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');
#
# *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
# html { scroll-behavior: smooth; }
#
# body, .gradio-container {
#     background: #09090f !important;
#     font-family: 'Inter', sans-serif !important;
#     color: #e2e8f0 !important;
# }
# .gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
#
# /* NAV */
# #sb-nav {
#     position: sticky; top: 0; z-index: 100;
#     background: rgba(9,9,15,0.85);
#     backdrop-filter: blur(20px);
#     border-bottom: 1px solid rgba(249,115,22,0.15);
#     padding: 0 48px;
#     display: flex; align-items: center; justify-content: space-between;
#     height: 68px;
# }
# .sb-logo {
#     font-family: "Outfit", sans-serif;
#     font-weight: 800; font-size: 1.45rem; color: #fff;
#     letter-spacing: -0.02em;
#     display: flex; align-items: center; gap: 10px;
# }
# .sb-logo-dot {
#     width: 10px; height: 10px; border-radius: 50%;
#     background: linear-gradient(135deg, #f97316, #f43f5e);
#     flex-shrink: 0;
# }
# .sb-nav-links { display: flex; gap: 36px; align-items: center; }
# .sb-nav-links a { color: #94a3b8; text-decoration: none; font-size: 0.9rem; transition: color 0.2s; }
# .sb-nav-links a:hover { color: #e2e8f0; }
# .sb-nav-cta {
#     background: linear-gradient(135deg, #ea580c, #f97316);
#     color: #fff !important; padding: 10px 24px;
#     font-size: 0.875rem; font-weight: 600;
#     text-decoration: none; border-radius: 6px;
#     transition: opacity 0.2s;
# }
# .sb-nav-cta:hover { opacity: 0.88; }
#
# /* HERO */
# #sb-hero {
#     background: #09090f;
#     padding: 100px 48px 80px;
#     text-align: center;
#     position: relative; overflow: hidden;
#     min-height: 90vh;
#     display: flex; align-items: center; justify-content: center;
# }
# .sb-hero-inner { position: relative; z-index: 2; max-width: 900px; margin: 0 auto; }
#
# .sb-hero-badge {
#     display: inline-flex; align-items: center; gap: 8px;
#     background: rgba(249,115,22,0.08);
#     border: 1px solid rgba(249,115,22,0.25);
#     padding: 6px 16px; border-radius: 100px;
#     font-size: 0.8rem; color: #fb923c; font-weight: 500;
#     margin-bottom: 32px; letter-spacing: 0.04em;
# }
# .sb-hero-badge-dot {
#     width: 6px; height: 6px; border-radius: 50%;
#     background: #f97316;
#     animation: pulse-dot 2.5s infinite;
# }
# @keyframes pulse-dot {
#     0%, 100% { opacity: 1; }
#     50% { opacity: 0.35; }
# }
#
# .sb-hero-title {
#     font-family: "Outfit", sans-serif; font-weight: 900;
#     font-size: clamp(3rem, 7vw, 6rem);
#     color: #fff; margin-bottom: 24px; letter-spacing: -0.03em;
# }
# .sb-hero-title span {
#     background: linear-gradient(135deg, #f97316, #f43f5e, #fb923c, #f97316);
#     background-size: 300% 300%;
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     animation: gradient-shift 5s ease infinite;
# }
# @keyframes gradient-shift {
#     0%   { background-position: 0% 50%; }
#     50%  { background-position: 100% 50%; }
#     100% { background-position: 0% 50%; }
# }
#
# .sb-hero-desc {
#     font-size: 1.05rem; line-height: 1.8;
#     color: #94a3b8; max-width: 560px;
#     margin: 0 auto 40px auto; font-weight: 300;
# }
# .sb-hero-actions {
#     display: flex; gap: 16px; align-items: center;
#     justify-content: center; flex-wrap: wrap;
# }
# .sb-btn-primary {
#     background: linear-gradient(135deg, #ea580c, #f97316);
#     color: #fff; padding: 14px 36px; font-size: 1rem; font-weight: 600;
#     text-decoration: none; border-radius: 8px;
#     display: inline-block; transition: opacity 0.2s, transform 0.15s;
# }
# .sb-btn-primary:hover { opacity: 0.88; transform: translateY(-1px); }
# .sb-btn-ghost {
#     background: transparent; color: #e2e8f0;
#     padding: 14px 36px; font-size: 1rem;
#     text-decoration: none; border-radius: 8px;
#     border: 1px solid rgba(226,232,240,0.18);
#     display: inline-flex; align-items: center; gap: 8px;
#     transition: all 0.2s;
# }
# .sb-btn-ghost:hover { border-color: rgba(226,232,240,0.4); }
#
# /* HERO GLOWS */
# .sb-hero-glow {
#     position: absolute; top: 50%; left: 50%;
#     transform: translate(-50%, -50%);
#     width: 650px; height: 650px; border-radius: 50%;
#     background: radial-gradient(circle, rgba(249,115,22,0.13) 0%, rgba(249,115,22,0.04) 40%, transparent 70%);
#     pointer-events: none; z-index: 1;
#     transition: left 0.65s cubic-bezier(0.23, 1, 0.32, 1), top 0.65s cubic-bezier(0.23, 1, 0.32, 1);
# }
# .sb-hero-glow-ambient {
#     position: absolute; top: 25%; right: 12%;
#     width: 420px; height: 420px; border-radius: 50%;
#     background: radial-gradient(circle, rgba(244,63,94,0.09) 0%, transparent 70%);
#     pointer-events: none; z-index: 1;
#     animation: ambient-drift 9s ease-in-out infinite;
# }
# .sb-hero-glow-ambient-2 {
#     position: absolute; bottom: 15%; left: 12%;
#     width: 320px; height: 320px; border-radius: 50%;
#     background: radial-gradient(circle, rgba(251,146,60,0.08) 0%, transparent 70%);
#     pointer-events: none; z-index: 1;
#     animation: ambient-drift 11s ease-in-out infinite reverse;
# }
# @keyframes ambient-drift {
#     0%, 100% { transform: translate(0, 0) scale(1); }
#     33%       { transform: translate(28px, -22px) scale(1.07); }
#     66%       { transform: translate(-18px, 24px) scale(0.94); }
# }
#
# /* STATS */
# .sb-stats-strip {
#     background: rgba(255,255,255,0.02);
#     border-top: 1px solid rgba(255,255,255,0.05);
#     border-bottom: 1px solid rgba(255,255,255,0.05);
#     padding: 32px 48px;
#     display: flex; justify-content: center; gap: 80px; flex-wrap: wrap;
# }
# .sb-stat-item { text-align: center; }
# .sb-stat-num {
#     font-family: "Outfit", sans-serif; font-weight: 800; font-size: 2.2rem;
#     background: linear-gradient(135deg, #f97316, #f43f5e);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
#     line-height: 1; margin-bottom: 6px;
# }
# .sb-stat-label { font-size: 0.78rem; color: #475569; letter-spacing: 0.1em; text-transform: uppercase; }
#
# /* SECTIONS */
# .sb-section { padding: 96px 48px; }
# .sb-section-dark { background: #09090f; }
# .sb-section-alt { background: #0d0d14; }
# .sb-section-label {
#     font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em;
#     text-transform: uppercase; color: #f97316; margin-bottom: 16px;
#     display: flex; align-items: center; gap: 8px;
# }
# .sb-section-label::before { content: ''; width: 24px; height: 1px; background: linear-gradient(90deg, #f97316, #f43f5e); }
# .sb-section-title {
#     font-family: "Outfit", sans-serif; font-weight: 800;
#     font-size: clamp(2rem, 4vw, 3rem);
#     color: #fff; margin-bottom: 16px; letter-spacing: -0.02em;
# }
# .sb-section-sub { font-size: 1rem; line-height: 1.75; color: #475569; max-width: 560px; font-weight: 300; margin-bottom: 48px; }
#
# /* STEPS */
# .sb-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; }
# .sb-step {
#     background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
#     padding: 32px 28px; border-radius: 12px;
#     position: relative; overflow: hidden; transition: border-color 0.3s, background 0.3s;
# }
# .sb-step::before {
#     content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
#     background: linear-gradient(90deg, transparent, rgba(249,115,22,0.6), transparent);
#     opacity: 0; transition: opacity 0.3s;
# }
# .sb-step:hover { border-color: rgba(249,115,22,0.25); background: rgba(249,115,22,0.03); }
# .sb-step:hover::before { opacity: 1; }
# .sb-step-num { font-size: 0.72rem; font-weight: 700; color: #f97316; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
# .sb-step-icon { width: 44px; height: 44px; border-radius: 10px; background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; font-size: 20px; }
# .sb-step-title { font-family: "Outfit", sans-serif; font-weight: 700; font-size: 1.1rem; color: #f1f5f9; margin-bottom: 10px; }
# .sb-step-desc { font-size: 0.9rem; line-height: 1.7; color: #475569; font-weight: 300; }
#
# /* VIDEO */
# .sb-video-container {
#     position: relative; padding-bottom: 56.25%; background: rgba(255,255,255,0.02);
#     border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; overflow: hidden; margin-top: 40px; max-width: 900px;
# }
# .sb-video-container video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
# .sb-video-placeholder { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; }
# .sb-play-btn { width: 72px; height: 72px; border-radius: 50%; background: linear-gradient(135deg, #ea580c, #f97316); display: flex; align-items: center; justify-content: center; transition: transform 0.2s; }
# .sb-play-btn:hover { transform: scale(1.06); }
# .sb-video-placeholder p { color: #475569; font-size: 0.9rem; }
#
# /* SAMPLES */
# .sb-samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin-top: 40px; }
# .sb-sample-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 20px 24px; border-radius: 10px; display: flex; flex-direction: column; gap: 12px; transition: border-color 0.2s, background 0.2s; }
# .sb-sample-card:hover { border-color: rgba(249,115,22,0.25); background: rgba(249,115,22,0.03); }
# .sb-sample-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #f97316; }
# .sb-sample-name { font-family: "Outfit", sans-serif; font-weight: 700; font-size: 0.95rem; color: #f1f5f9; }
# .sb-sample-dl { display: inline-block; margin-top: 4px; font-size: 0.8rem; font-weight: 500; padding: 8px 18px; border-radius: 6px; background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.2); color: #f97316; text-decoration: none; transition: all 0.2s; }
# .sb-sample-dl:hover { background: rgba(249,115,22,0.16); }
#
# /* SYSTEM DEMO */
# #sb-system { background: #0d0d14; padding: 0 48px 96px; }
# .sb-system-shell { border: 1px solid rgba(249,115,22,0.18); background: rgba(255,255,255,0.01); border-radius: 16px; overflow: hidden; }
# .sb-system-topbar { background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 14px 24px; display: flex; align-items: center; gap: 10px; }
# .sb-topbar-dot { width: 10px; height: 10px; border-radius: 50%; }
# .sb-topbar-title { font-size: 0.8rem; color: #334155; margin-left: 8px; letter-spacing: 0.04em; }
# .sb-system-inner { padding: 36px; }
# .sb-system-inner label, .sb-system-inner .label-wrap span {
#     color: #64748b !important; font-size: 0.78rem !important;
#     font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
# }
# .sb-system-inner textarea, .sb-system-inner input[type="text"] {
#     background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important;
#     border-radius: 8px !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important;
# }
# .sb-system-inner textarea::placeholder { color: #334155 !important; }
# .sb-system-inner .block, .sb-system-inner .wrap { background: transparent !important; border-color: rgba(255,255,255,0.06) !important; }
# .sb-system-inner button { border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; transition: all 0.2s !important; }
# .sb-system-inner button.primary, .sb-system-inner button[variant="primary"] {
#     background: linear-gradient(135deg, #ea580c, #f97316) !important; color: #fff !important; border: none !important;
# }
# .sb-system-inner button.primary:hover { opacity: 0.88 !important; }
# .sb-system-inner button.secondary, .sb-system-inner button[variant="secondary"] {
#     background: rgba(249,115,22,0.06) !important; color: #f97316 !important; border: 1px solid rgba(249,115,22,0.25) !important;
# }
# .sb-system-inner button.secondary:hover { background: rgba(249,115,22,0.12) !important; }
# .sb-system-inner [data-testid="audio"], .sb-system-inner .gr-audio {
#     background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important;
# }
# .sb-system-inner input[type="radio"] { accent-color: #f97316 !important; }
# .sb-system-inner select { background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; color: #e2e8f0 !important; border-radius: 8px !important; }
#
# /* FEEDBACK */
# #sb-feedback { background: #09090f; padding: 96px 48px; position: relative; overflow: hidden; }
# #sb-feedback::before {
#     content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
#     width: 500px; height: 280px;
#     background: radial-gradient(ellipse, rgba(249,115,22,0.05) 0%, transparent 70%);
#     pointer-events: none;
# }
# .sb-feedback-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 48px; flex-wrap: wrap; gap: 24px; position: relative; z-index: 2; }
# .sb-feedback-form-btn { display: inline-flex; align-items: center; gap: 8px; background: rgba(249,115,22,0.06); border: 1px solid rgba(249,115,22,0.25); color: #f97316; padding: 12px 24px; font-size: 0.875rem; font-weight: 500; text-decoration: none; border-radius: 8px; transition: all 0.2s; }
# .sb-feedback-form-btn:hover { background: rgba(249,115,22,0.12); }
# .sb-feedback-carousel { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; min-height: 220px; position: relative; z-index: 2; }
# .sb-feedback-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 28px; border-radius: 12px; display: flex; flex-direction: column; gap: 14px; transition: border-color 0.3s; }
# .sb-feedback-card:hover { border-color: rgba(249,115,22,0.18); }
# .sb-feedback-quote { font-size: 0.95rem; line-height: 1.75; color: #64748b; font-style: italic; font-weight: 300; flex: 1; }
# .sb-feedback-name { font-size: 0.78rem; font-weight: 600; color: #f97316; letter-spacing: 0.08em; text-transform: uppercase; }
# .sb-feedback-placeholder { grid-column: 1 / -1; color: #334155; font-size: 0.9rem; font-style: italic; display: flex; align-items: center; justify-content: center; min-height: 120px; text-align: center; border: 1px dashed rgba(255,255,255,0.06); border-radius: 12px; }
# .sb-carousel-controls { display: flex; justify-content: center; gap: 8px; margin-top: 32px; position: relative; z-index: 2; }
# .sb-carousel-dot { width: 6px; height: 6px; border-radius: 50%; background: rgba(255,255,255,0.1); cursor: pointer; transition: all 0.2s; border: none; }
# .sb-carousel-dot.active { background: #f97316; }
#
# /* FOOTER */
# #sb-footer { background: #070709; border-top: 1px solid rgba(255,255,255,0.04); padding: 56px 48px 40px; }
# .sb-footer-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 48px; padding-bottom: 40px; border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 32px; }
# .sb-footer-brand { font-family: "Outfit", sans-serif; font-weight: 800; font-size: 1.3rem; color: #fff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
# .sb-footer-tagline { font-size: 0.85rem; color: #334155; line-height: 1.7; max-width: 260px; }
# .sb-footer-col-title { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #475569; margin-bottom: 16px; }
# .sb-footer-links { display: flex; flex-direction: column; gap: 10px; }
# .sb-footer-links a { color: #334155; text-decoration: none; font-size: 0.875rem; transition: color 0.2s; }
# .sb-footer-links a:hover { color: #f97316; }
# .sb-footer-bottom { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
# .sb-footer-copy { font-size: 0.8rem; color: #1e293b; }
#
# /* RESPONSIVE */
# @media (max-width: 900px) {
#     #sb-nav { padding: 0 24px; }
#     .sb-section, #sb-hero, #sb-feedback, #sb-footer, #sb-system { padding-left: 24px; padding-right: 24px; }
#     .sb-feedback-carousel { grid-template-columns: 1fr; }
#     .sb-footer-grid { grid-template-columns: 1fr; gap: 32px; }
#     .sb-nav-links { display: none; }
#     .sb-stats-strip { gap: 40px; padding: 32px 24px; }
# }
# @media (max-width: 640px) {
#     .sb-steps { grid-template-columns: 1fr; }
#     .sb-samples-grid { grid-template-columns: 1fr; }
#     .sb-hero-title { font-size: 2.5rem; }
# }
# .gradio-container .block { border-radius: 8px !important; }
# footer.svelte-1rjryqp { display: none !important; }
#
# /* Video — centred and full width */
# .sb-video-container {
#     margin-left: auto !important;
#     margin-right: auto !important;
#     max-width: 100% !important;
# }
#
# /* Demo area helpers */
# .sb-demo-divider {
#     height: 1px;
#     background: rgba(255,255,255,0.06);
#     margin: 18px 0 16px;
# }
# .sb-demo-section-label {
#     font-size: 0.68rem;
#     font-weight: 700;
#     letter-spacing: 0.12em;
#     text-transform: uppercase;
#     color: #475569;
#     margin-bottom: 8px;
# }
#
# """
#
#
# # =========================================================
# # UI
# # =========================================================
# # with gr.Blocks(css=CUSTOM_CSS, title="SpeechBridge") as demo:
# with gr.Blocks(title="SpeechBridge") as demo:
#     gr.HTML("""
#     <nav id="sb-nav">
#         <div class="sb-logo">
#             <div class="sb-logo-mark"></div>
#             SpeechBridge
#         </div>
#         <div class="sb-nav-links">
#             <a href="#sb-how">How it works</a>
#             <a href="#sb-samples">Samples</a>
#             <a href="#sb-system">Demo</a>
#             <a href="#sb-feedback">Feedback</a>
#         </div>
#         <a class="sb-nav-cta" href="#sb-system">Try it now</a>
#     </nav>
#     """)
#
#     gr.HTML("""
#     <section id="sb-hero">
#         <div class="sb-hero-glow" id="sb-hero-glow"></div>
#         <div class="sb-hero-glow-ambient"></div>
#         <div class="sb-hero-glow-ambient-2"></div>
#         <div class="sb-hero-inner">
#             <div class="sb-hero-badge">
#                 <div class="sb-hero-badge-dot"></div>
#                 French to English · Real-time Speech Translation
#             </div>
#             <h1 class="sb-hero-title">
#                 Speak French.<br><span>Be Heard</span> in English.
#             </h1>
#             <p class="sb-hero-desc">
#                 SpeechBridge translates spoken French into English — preserving
#                 not just the words, but the voice, tone, and character of the original speaker.
#             </p>
#             <div class="sb-hero-actions">
#                 <a class="sb-btn-primary" href="#sb-system">Try it yourself</a>
#                 <a class="sb-btn-ghost" href="#sb-how">
#                     See how it works
#                     <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
#                         <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
#                     </svg>
#                 </a>
#             </div>
#         </div>
#     </section>
#     """)
#
#     gr.HTML("""
#     <script>
#     (function() {
#         function initGlow() {
#             var hero = document.getElementById('sb-hero');
#             var glow = document.getElementById('sb-hero-glow');
#             if (!hero || !glow) { setTimeout(initGlow, 300); return; }
#             hero.addEventListener('mousemove', function(e) {
#                 var r = hero.getBoundingClientRect();
#                 glow.style.left = (e.clientX - r.left) + 'px';
#                 glow.style.top  = (e.clientY - r.top)  + 'px';
#             });
#             hero.addEventListener('mouseleave', function() {
#                 glow.style.left = '50%';
#                 glow.style.top  = '50%';
#             });
#         }
#         setTimeout(initGlow, 600);
#     })();
#     </script>
#     """)
#
#
#     gr.HTML("""
#     <div class="sb-stats-strip">
#         <div class="sb-stat-item">
#             <div class="sb-stat-num">2</div>
#             <div class="sb-stat-label">Languages</div>
#         </div>
#         <div class="sb-stat-item">
#             <div class="sb-stat-num">~0.81</div>
#             <div class="sb-stat-label">Voice Similarity</div>
#         </div>
#         <div class="sb-stat-item">
#             <div class="sb-stat-num">2</div>
#             <div class="sb-stat-label">Output Modes</div>
#         </div>
#         <div class="sb-stat-item">
#             <div class="sb-stat-num">End-to-end</div>
#             <div class="sb-stat-label">Pipeline</div>
#         </div>
#     </div>
#     """)
#
#     gr.HTML("""
#     <section id="sb-how" class="sb-section sb-section-alt">
#         <div class="sb-section-label">How it works</div>
#         <h2 class="sb-section-title">Three steps to translated speech</h2>
#         <p class="sb-section-sub">Upload any French audio clip and receive translated English speech, optionally in the original speaker's voice.</p>
#         <div class="sb-steps">
#             <div class="sb-step">
#                 <div class="sb-step-num">01</div>
#                 <div class="sb-step-title">Upload your audio</div>
#                 <div class="sb-step-desc">Record or upload a French speech clip in any format. The system handles all conversion automatically.</div>
#             </div>
#             <div class="sb-step">
#                 <div class="sb-step-num">02</div>
#                 <div class="sb-step-title">Choose output mode</div>
#                 <div class="sb-step-desc">Select standard translation for clean English speech, or voice cloning to preserve the original speaker's voice.</div>
#             </div>
#             <div class="sb-step">
#                 <div class="sb-step-num">03</div>
#                 <div class="sb-step-title">Receive translated speech</div>
#                 <div class="sb-step-desc">Get translated English text and synthesised audio — with cloning, it sounds like the same person speaking English.</div>
#             </div>
#         </div>
#     </section>
#     """)
#
#
#     gr.HTML("""
#     <section class="sb-section sb-section-dark">
#         <div class="sb-section-label">See it in action</div>
#         <h2 class="sb-section-title">Watch a full translation run</h2>
#         <div class="sb-video-container">
#             <video id="sb-video" controls style="display:none;">
#                 <source src="demo.mp4" type="video/mp4">
#             </video>
#             <div class="sb-video-placeholder" id="sb-video-placeholder">
#                 <div class="sb-play-btn">
#                     <svg width="22" height="22" viewBox="0 0 24 24" fill="#0D0D0D"><path d="M8 5v14l11-7L8 5z"/></svg>
#                 </div>
#                 <p>Demo video coming soon</p>
#             </div>
#         </div>
#     </section>
#     """)
#
#
#     gr.HTML("""
#     <section id="sb-samples" class="sb-section sb-section-alt">
#         <div class="sb-section-label">Sample audio</div>
#         <h2 class="sb-section-title">Try it with one of our clips</h2>
#         <p class="sb-section-sub">Download any French speech sample below and upload it into the system to see it in action.</p>
#         <div class="sb-samples-grid">
#             <div class="sb-sample-card">
#                 <div class="sb-sample-label">Sample 01</div>
#                 <div class="sb-sample-name">Child's voice</div>
#                 <a class="sb-sample-dl" href="audio_samples/sample_1.wav" download>Download clip →</a>
#             </div>
#             <div class="sb-sample-card">
#                 <div class="sb-sample-label">Sample 02</div>
#                 <div class="sb-sample-name">Adult female voice</div>
#                 <a class="sb-sample-dl" href="audio_samples/sample_2.wav" download>Download clip →</a>
#             </div>
#             <div class="sb-sample-card">
#                 <div class="sb-sample-label">Sample 03</div>
#                 <div class="sb-sample-name">Adult male voice</div>
#                 <a class="sb-sample-dl" href="audio_samples/sample_3.wav" download>Download clip →</a>
#             </div>
#             <div class="sb-sample-card">
#                 <div class="sb-sample-label">Sample 04</div>
#                 <div class="sb-sample-name">Young female voice</div>
#                 <a class="sb-sample-dl" href="audio_samples/sample_4.wav" download>Download clip →</a>
#             </div>
#             <div class="sb-sample-card">
#                 <div class="sb-sample-label">Sample 05</div>
#                 <div class="sb-sample-name">Elderly voice</div>
#                 <a class="sb-sample-dl" href="audio_samples/sample_5.wav" download>Download clip →</a>
#             </div>
#         </div>
#     </section>
#     """)
#
#     gr.HTML("""
#     <section id="sb-system" class="sb-section sb-section-dark" style="padding-bottom:96px;">
#         <div class="sb-section-label">Live demo</div>
#         <h2 class="sb-section-title">Try SpeechBridge</h2>
#         <p class="sb-section-sub">Upload or record French speech and receive an English translation — with or without voice cloning.</p>
#         <div class="sb-system-shell">
#             <div class="sb-system-topbar">
#                 <div class="sb-topbar-dot" style="background:#ef4444;"></div>
#                 <div class="sb-topbar-dot" style="background:#f59e0b;"></div>
#                 <div class="sb-topbar-dot" style="background:#10b981;"></div>
#                 <span class="sb-topbar-title">speechbridge — live translation interface</span>
#             </div>
#             <div class="sb-system-inner">
#     """)
#
#     # with gr.Row():
#     #     with gr.Column(scale=1):
#     #         model_status = gr.Textbox(
#     #             label="System status",
#     #             value="Models initialising in the background. Ready shortly.",
#     #             interactive=False, lines=3,
#     #         )
#     #         init_btn = gr.Button("Initialise models manually", variant="secondary")
#     #
#     #         gr.HTML('<div style="margin:20px 0 8px;font-size:0.75rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#475569;">Input language</div>')
#     #         input_lang = gr.Dropdown(choices=["French", "More languages coming soon…"], value="French", label="", interactive=False)
#     #
#     #         gr.HTML('<div style="margin:16px 0 8px;font-size:0.75rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#475569;">Output language</div>')
#     #         output_lang = gr.Dropdown(choices=["English", "More languages coming soon…"], value="English", label="", interactive=False)
#     #
#     #         mode = gr.Radio(
#     #             choices=["Translate without voice cloning", "Translate with voice cloning"],
#     #             value="Translate without voice cloning",
#     #             label="Translation mode"
#     #         )
#     #
#     #         audio_input = gr.Audio(
#     #             sources=["upload", "microphone"],
#     #             type="filepath",
#     #             label="Upload or record French speech",
#     #             # format="wav",
#     #         )
#     #
#     #         gr.Markdown("<div style='font-size:0.8rem;color:#334155;line-height:1.6;margin-top:6px;'>Upload any format or record directly. For voice cloning, the uploaded clip also serves as the speaker reference.</div>")
#     #
#     #         translate_btn = gr.Button("Run translation", variant="primary")
#     #
#     #     with gr.Column(scale=1):
#     #         translated_text = gr.Textbox(label="English translation", lines=6, placeholder="Translated English text will appear here.")
#     #         output_audio = gr.Audio(type="filepath", label="Generated English speech")
#     #         runtime_info = gr.Textbox(label="Runtime breakdown", lines=6, interactive=False)
#     #         pipeline_status = gr.Textbox(label="Status", lines=3, interactive=False)
#     #
#     # gr.HTML("</div></div></section>")
#
#
#     with gr.Row():
#         with gr.Column(scale=1):
#             model_status = gr.Textbox(
#                 label="System status",
#                 value="Models initialising in the background. Ready shortly.",
#                 interactive=False, lines=2,
#             )
#             init_btn = gr.Button("Initialise models manually", variant="secondary")
#
#             gr.HTML('<div class="sb-demo-divider"></div>')
#             gr.HTML('<div class="sb-demo-section-label">Language pair</div>')
#
#             with gr.Row():
#                 input_lang = gr.Dropdown(
#                     choices=["French (fine-tuned)", "Arabic", "Urdu", "Malay"],
#                     value="French (fine-tuned)",
#                     label="From",
#                     interactive=True
#                 )
#
#                 output_lang = gr.Dropdown(
#                     choices=["English", "More languages coming soon…"],
#                     value="English", label="To", interactive=True
#                 )
#
#             mode = gr.Radio(
#                 choices=["Translate without voice cloning", "Translate with voice cloning"],
#                 value="Translate without voice cloning",
#                 label="Translation mode"
#             )
#
#             audio_input = gr.Audio(
#                 sources=["upload", "microphone"],
#                 type="filepath",
#                 label="Upload or record French speech",
#             )
#
#             # audio_state = gr.State()
#             #
#             # audio_input.change(
#             #     fn=persist_audio_file,
#             #     inputs=audio_input,
#             #     outputs=audio_state
#             # )
#
#
#
#             # gr.Markdown("<div style='font-size:0.78rem;color:#475569;line-height:1.6;margin-top:4px;'>Upload any format or record directly. For voice cloning, the clip also serves as the speaker reference.</div>")
#             gr.Markdown(
#                 "<div style='font-size:0.78rem;color:#475569;line-height:1.6;margin-top:4px;'>Upload any format or record directly. For voice cloning, the clip also serves as the speaker reference. <strong style='color:#f97316;'>Fine-tuned for French → English.</strong> Arabic, Urdu and Malay use base Whisper (experimental).</div>")
#
#             translate_btn = gr.Button("Run translation", variant="primary")
#
#         with gr.Column(scale=1):
#             translated_text = gr.Textbox(
#                 label="English translation",
#                 lines=8,
#                 placeholder="Translated English text will appear here."
#             )
#             output_audio = gr.Audio(type="filepath", label="Generated English speech")
#             with gr.Row():
#                 runtime_info = gr.Textbox(label="Runtime breakdown", lines=4, interactive=False)
#                 pipeline_status = gr.Textbox(label="Status", lines=4, interactive=False)
#
#     gr.HTML("</div></div></section>")
#
#
#     gr.HTML(f"""
#     <section id="sb-feedback">
#         <div class="sb-feedback-header">
#             <div>
#                 <div class="sb-section-label">User feedback</div>
#                 <h2 class="sb-section-title" style="margin-bottom:0;">What people are saying</h2>
#             </div>
#             <a class="sb-feedback-form-btn" href="{GOOGLE_FORM_URL}" target="_blank">
#                 Share your feedback
#                 <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
#                     <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
#                 </svg>
#             </a>
#         </div>
#         <div class="sb-feedback-carousel" id="sb-feedback-carousel">
#             <div class="sb-feedback-placeholder">
#                 <span>Feedback will appear here once responses are submitted.</span>
#             </div>
#         </div>
#         <div class="sb-carousel-controls" id="sb-carousel-controls" style="display:none;"></div>
#     </section>
#     """)
#
#     demo.load(
#         fn=None,
#         js=f"""
#         function() {{
#             var SHEET_URL = 'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:json';
#             var PER_PAGE = 3;
#             var allFeedback = [];
#             var currentPage = 0;
#             var autoTimer = null;
#
#             function renderStars(n) {{
#                 var s = '';
#                 var filled = Math.round(n || 0);
#                 for (var i = 1; i <= 5; i++) {{
#                     s += '<span style="color:#f97316;font-size:13px;">' + (i <= filled ? '★' : '☆') + '</span>';
#                 }}
#                 return s;
#             }}
#
#             function renderPage(page) {{
#                 var carousel = document.getElementById('sb-feedback-carousel');
#                 var controls = document.getElementById('sb-carousel-controls');
#                 if (!carousel) return;
#                 var start = page * PER_PAGE;
#                 var slice = allFeedback.slice(start, start + PER_PAGE);
#                 carousel.innerHTML = '';
#                 slice.forEach(function(fb) {{
#                     var avg = ((fb.r1 || 0) + (fb.r3 || 0)) / 2;
#                     var card = document.createElement('div');
#                     card.className = 'sb-feedback-card';
#                     card.innerHTML =
#                         '<div style="display:flex;gap:3px;margin-bottom:4px;">' + renderStars(avg) + '</div>' +
#                         '<div class="sb-feedback-quote">' + (fb.comment || 'No comment provided.') + '</div>' +
#                         '<div class="sb-feedback-name">' + (fb.name || 'Anonymous') + '</div>';
#                     carousel.appendChild(card);
#                 }});
#                 if (controls) {{
#                     var dots = controls.querySelectorAll('.sb-carousel-dot');
#                     dots.forEach(function(d, i) {{ d.classList.toggle('active', i === page); }});
#                 }}
#                 currentPage = page;
#             }}
#
#             function buildControls(totalPages) {{
#                 var controls = document.getElementById('sb-carousel-controls');
#                 if (!controls) return;
#                 controls.innerHTML = '';
#                 if (totalPages <= 1) return;
#                 controls.style.display = 'flex';
#                 for (var i = 0; i < totalPages; i++) {{
#                     (function(idx) {{
#                         var dot = document.createElement('button');
#                         dot.className = 'sb-carousel-dot' + (idx === 0 ? ' active' : '');
#                         dot.addEventListener('click', function() {{ clearInterval(autoTimer); renderPage(idx); }});
#                         controls.appendChild(dot);
#                     }})(i);
#                 }}
#             }}
#
#             function loadFeedback() {{
#                 fetch(SHEET_URL)
#                     .then(function(r) {{ return r.text(); }})
#                     .then(function(text) {{
#                         var start = text.indexOf('(') + 1;
#                         var end = text.lastIndexOf(')');
#                         var json = JSON.parse(text.substring(start, end));
#                         var rows = json.table && json.table.rows ? json.table.rows : [];
#                         allFeedback = rows.map(function(row) {{
#                             var c = row.c || [];
#                             var nameVal = c[5] && c[5].v ? String(c[5].v).trim() : 'Anonymous';
#                             return {{
#                                 r1: c[1] && c[1].v ? Number(c[1].v) : 0,
#                                 r2: c[2] && c[2].v ? Number(c[2].v) : 0,
#                                 r3: c[3] && c[3].v ? Number(c[3].v) : 0,
#                                 comment: c[4] && c[4].v ? String(c[4].v) : '',
#                                 name: nameVal.toLowerCase() === 'anonymous' ? 'Anonymous' : nameVal
#                             }};
#                         }}).filter(function(fb) {{ return fb.comment || fb.r1; }});
#                         if (allFeedback.length === 0) return;
#                         var totalPages = Math.ceil(allFeedback.length / PER_PAGE);
#                         buildControls(totalPages);
#                         renderPage(0);
#                         if (totalPages > 1) {{
#                             autoTimer = setInterval(function() {{ renderPage((currentPage + 1) % totalPages); }}, 5000);
#                         }}
#                     }})
#                     .catch(function(e) {{ console.log('Feedback error:', e); }});
#             }}
#
#             setTimeout(loadFeedback, 1500);
#
#         }}
#         """
#     )
#
#     gr.HTML("""
#     <footer id="sb-footer">
#         <div class="sb-footer-grid">
#             <div>
#                 <div class="sb-footer-brand">
#                     <div class="sb-footer-brand-dot"></div>
#                     SpeechBridge
#                 </div>
#                 <p class="sb-footer-tagline">A speech-to-speech translation system combining neural translation with speaker-preserving voice cloning.</p>
#             </div>
#             <div>
#                 <div class="sb-footer-col-title">Navigation</div>
#                 <div class="sb-footer-links">
#                     <a href="#sb-how">How it works</a>
#                     <a href="#sb-samples">Audio samples</a>
#                     <a href="#sb-system">Live demo</a>
#                     <a href="#sb-feedback">Feedback</a>
#                 </div>
#             </div>
#             <div>
#                 <div class="sb-footer-col-title">Contact</div>
#                 <div class="sb-footer-links">
#                     <a href="#sb-feedback">Leave feedback</a>
#                 </div>
#             </div>
#         </div>
#         <div class="sb-footer-bottom">
#             <span class="sb-footer-copy">SpeechBridge — University Final Year Project</span>
#             <span class="sb-footer-copy">French to English Speech Translation</span>
#         </div>
#     </footer>
#     """)
#
#     init_btn.click(fn=initialise_models, outputs=model_status)
#     translate_btn.click(
#         fn=run_app,
#         inputs=[mode, audio_input, input_lang],
#         outputs=[translated_text, output_audio, runtime_info, pipeline_status]
#     )
#
#
# if __name__ == "__main__":
#     init_thread = threading.Thread(target=background_init, daemon=True)
#     init_thread.start()
#     demo.launch(css=CUSTOM_CSS)

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
    border-bottom: 1px solid rgba(249,115,22,0.15);
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
    background: linear-gradient(135deg, #f97316, #f43f5e);
    flex-shrink: 0;
}
.sb-nav-links { display: flex; gap: 36px; align-items: center; }
.sb-nav-links a { color: #94a3b8; text-decoration: none; font-size: 0.9rem; transition: color 0.2s; }
.sb-nav-links a:hover { color: #e2e8f0; }
.sb-nav-cta {
    background: linear-gradient(135deg, #ea580c, #f97316);
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
    background: rgba(249,115,22,0.08);
    border: 1px solid rgba(249,115,22,0.25);
    padding: 6px 16px; border-radius: 100px;
    font-size: 0.8rem; color: #fb923c; font-weight: 500;
    margin-bottom: 32px; letter-spacing: 0.04em;
}
.sb-hero-badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #f97316;
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
    background: linear-gradient(135deg, #f97316, #f43f5e, #fb923c, #f97316);
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
    background: linear-gradient(135deg, #ea580c, #f97316);
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
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 650px; height: 650px; border-radius: 50%;
    background: radial-gradient(circle, rgba(249,115,22,0.13) 0%, rgba(249,115,22,0.04) 40%, transparent 70%);
    pointer-events: none; z-index: 1;
    transition: left 0.65s cubic-bezier(0.23, 1, 0.32, 1), top 0.65s cubic-bezier(0.23, 1, 0.32, 1);
}
.sb-hero-glow-ambient {
    position: absolute; top: 25%; right: 12%;
    width: 420px; height: 420px; border-radius: 50%;
    background: radial-gradient(circle, rgba(244,63,94,0.09) 0%, transparent 70%);
    pointer-events: none; z-index: 1;
    animation: ambient-drift 9s ease-in-out infinite;
}
.sb-hero-glow-ambient-2 {
    position: absolute; bottom: 15%; left: 12%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(251,146,60,0.08) 0%, transparent 70%);
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
    background: linear-gradient(135deg, #f97316, #f43f5e);
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
    text-transform: uppercase; color: #f97316; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.sb-section-label::before { content: ''; width: 24px; height: 1px; background: linear-gradient(90deg, #f97316, #f43f5e); }
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
    background: linear-gradient(90deg, transparent, rgba(249,115,22,0.6), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.sb-step:hover { border-color: rgba(249,115,22,0.25); background: rgba(249,115,22,0.03); }
.sb-step:hover::before { opacity: 1; }
.sb-step-num { font-size: 0.72rem; font-weight: 700; color: #f97316; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
.sb-step-icon { width: 44px; height: 44px; border-radius: 10px; background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; font-size: 20px; }
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
.sb-play-btn { width: 72px; height: 72px; border-radius: 50%; background: linear-gradient(135deg, #ea580c, #f97316); display: flex; align-items: center; justify-content: center; transition: transform 0.2s; }
.sb-play-btn:hover { transform: scale(1.06); }
.sb-video-placeholder p { color: #475569; font-size: 0.9rem; }

/* SAMPLES */
.sb-samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin-top: 40px; }
.sb-sample-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 20px 24px; border-radius: 10px; display: flex; flex-direction: column; gap: 12px; transition: border-color 0.2s, background 0.2s; }
.sb-sample-card:hover { border-color: rgba(249,115,22,0.25); background: rgba(249,115,22,0.03); }
.sb-sample-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #f97316; }
.sb-sample-name { font-family: "Outfit", sans-serif; font-weight: 700; font-size: 0.95rem; color: #f1f5f9; }
.sb-sample-dl { display: inline-block; margin-top: 4px; font-size: 0.8rem; font-weight: 500; padding: 8px 18px; border-radius: 6px; background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.2); color: #f97316; text-decoration: none; transition: all 0.2s; }
.sb-sample-dl:hover { background: rgba(249,115,22,0.16); }

/* SYSTEM DEMO */
#sb-system { background: #0d0d14; padding: 0 48px 96px; }
.sb-system-shell { border: 1px solid rgba(249,115,22,0.18); background: rgba(255,255,255,0.01); border-radius: 16px; overflow: hidden; }
.sb-system-topbar { background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 14px 24px; display: flex; align-items: center; gap: 10px; }
.sb-topbar-dot { width: 10px; height: 10px; border-radius: 50%; }
.sb-topbar-title { font-size: 0.8rem; color: #334155; margin-left: 8px; letter-spacing: 0.04em; }
.sb-system-inner { padding: 36px; }
.sb-system-inner label, .sb-system-inner .label-wrap span {
    color: #64748b !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
}
.sb-system-inner textarea, .sb-system-inner input[type="text"] {
    background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important;
}
.sb-system-inner textarea::placeholder { color: #334155 !important; }
.sb-system-inner .block, .sb-system-inner .wrap { background: transparent !important; border-color: rgba(255,255,255,0.06) !important; }
.sb-system-inner button { border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; transition: all 0.2s !important; }
.sb-system-inner button.primary, .sb-system-inner button[variant="primary"] {
    background: linear-gradient(135deg, #ea580c, #f97316) !important; color: #fff !important; border: none !important;
}
.sb-system-inner button.primary:hover { opacity: 0.88 !important; }
.sb-system-inner button.secondary, .sb-system-inner button[variant="secondary"] {
    background: rgba(249,115,22,0.06) !important; color: #f97316 !important; border: 1px solid rgba(249,115,22,0.25) !important;
}
.sb-system-inner button.secondary:hover { background: rgba(249,115,22,0.12) !important; }
.sb-system-inner [data-testid="audio"], .sb-system-inner .gr-audio {
    background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important;
}
.sb-system-inner input[type="radio"] { accent-color: #f97316 !important; }
.sb-system-inner select { background: #13131f !important; border: 1px solid rgba(255,255,255,0.08) !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* Model status indicator */
.sb-model-status {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.82rem; font-weight: 500; padding: 8px 14px;
    border-radius: 6px; margin-bottom: 20px;
}
.sb-model-ready {
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);
    color: #22c55e;
}
.sb-model-loading {
    background: rgba(249,115,22,0.06); border: 1px solid rgba(249,115,22,0.15);
    color: #94a3b8;
}
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.sb-spin { display: inline-block; animation: spin 1.4s linear infinite; }

/* Flat controls — no box for lang row and mode */
.sb-no-box > .block,
.sb-no-box .block { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.sb-lang-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.sb-lang-arrow { color: #475569; font-size: 1rem; flex-shrink: 0; padding-top: 20px; }

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
    background: radial-gradient(ellipse, rgba(249,115,22,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.sb-feedback-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 48px; flex-wrap: wrap; gap: 24px; position: relative; z-index: 2; }
.sb-feedback-form-btn { display: inline-flex; align-items: center; gap: 8px; background: rgba(249,115,22,0.06); border: 1px solid rgba(249,115,22,0.25); color: #f97316; padding: 12px 24px; font-size: 0.875rem; font-weight: 500; text-decoration: none; border-radius: 8px; transition: all 0.2s; }
.sb-feedback-form-btn:hover { background: rgba(249,115,22,0.12); }
.sb-feedback-carousel { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; min-height: 220px; position: relative; z-index: 2; }
.sb-feedback-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 28px; border-radius: 12px; display: flex; flex-direction: column; gap: 14px; transition: border-color 0.3s; }
.sb-feedback-card:hover { border-color: rgba(249,115,22,0.18); }
.sb-feedback-quote { font-size: 0.95rem; line-height: 1.75; color: #64748b; font-style: italic; font-weight: 300; flex: 1; }
.sb-feedback-name { font-size: 0.78rem; font-weight: 600; color: #f97316; letter-spacing: 0.08em; text-transform: uppercase; }
.sb-feedback-placeholder { grid-column: 1 / -1; color: #334155; font-size: 0.9rem; font-style: italic; display: flex; align-items: center; justify-content: center; min-height: 120px; text-align: center; border: 1px dashed rgba(255,255,255,0.06); border-radius: 12px; }
.sb-carousel-controls { display: flex; justify-content: center; gap: 8px; margin-top: 32px; position: relative; z-index: 2; }
.sb-carousel-dot { width: 6px; height: 6px; border-radius: 50%; background: rgba(255,255,255,0.1); cursor: pointer; transition: all 0.2s; border: none; }
.sb-carousel-dot.active { background: #f97316; }

/* FOOTER */
#sb-footer { background: #070709; border-top: 1px solid rgba(255,255,255,0.04); padding: 56px 48px 40px; }
.sb-footer-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 48px; padding-bottom: 40px; border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 32px; }
.sb-footer-brand { font-family: "Outfit", sans-serif; font-weight: 800; font-size: 1.3rem; color: #fff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.sb-footer-brand-dot { width: 8px; height: 8px; border-radius: 50%; background: linear-gradient(135deg, #f97316, #f43f5e); }
.sb-footer-tagline { font-size: 0.85rem; color: #334155; line-height: 1.7; max-width: 260px; }
.sb-footer-col-title { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #475569; margin-bottom: 16px; }
.sb-footer-links { display: flex; flex-direction: column; gap: 10px; }
.sb-footer-links a { color: #334155; text-decoration: none; font-size: 0.875rem; transition: color 0.2s; }
.sb-footer-links a:hover { color: #f97316; }
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
"""


# =========================================================
# UI
# =========================================================
with gr.Blocks(title="SpeechBridge") as demo:

    # ── Soundwave canvas (fixed background, behind everything) ──
    gr.HTML("""
    <canvas id="sb-wave-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;pointer-events:none;"></canvas>
    <script>
    (function() {
        var canvas, ctx, W, H, scrollY = 0;
        var waves = [
            { freq: 0.016, amp: 0.10, speed: 0.55, phase: 0.0,  r:249, g:115, b:22  },
            { freq: 0.022, amp: 0.07, speed: 0.85, phase: 2.1,  r:244, g:63,  b:94  },
            { freq: 0.011, amp: 0.13, speed: 0.38, phase: 4.4,  r:251, g:146, b:60  },
            { freq: 0.029, amp: 0.05, speed: 1.1,  phase: 1.3,  r:249, g:115, b:22  }
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
            var scrollMod = 1 + scrollY * 0.00025;
            var amp = H * w.amp * scrollMod;
            var cy  = H * 0.5;
            ctx.beginPath();
            for (var x = 0; x <= W + 4; x += 4) {
                var y = cy
                    + Math.sin(x * w.freq + t * w.speed + w.phase) * amp
                    + Math.sin(x * w.freq * 2.7 + t * w.speed * 1.5 + w.phase + 1) * amp * 0.28;
                if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = 'rgba(' + w.r + ',' + w.g + ',' + w.b + ',0.07)';
            ctx.lineWidth = 1.5;
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
    gr.HTML("""
    <section id="sb-system" class="sb-section sb-section-dark" style="padding-bottom:96px;">
        <div class="sb-section-label">Live demo</div>
        <h2 class="sb-section-title">Try SpeechBridge</h2>
        <p class="sb-section-sub">Upload or record speech and receive an English translation — with or without voice cloning. French uses the LoRA fine-tuned model.</p>
        <div class="sb-system-shell">
            <div class="sb-system-topbar">
                <div class="sb-topbar-dot" style="background:#ef4444;"></div>
                <div class="sb-topbar-dot" style="background:#f59e0b;"></div>
                <div class="sb-topbar-dot" style="background:#10b981;"></div>
                <span class="sb-topbar-title">speechbridge — live translation interface</span>
            </div>
            <div class="sb-system-inner">
    """)

    with gr.Row():
        with gr.Column(scale=1):

            # Smart model status — no textbox, just clean indicator + conditional button
            model_status_display = gr.HTML(value=_render_model_status())
            init_btn = gr.Button(
                "Initialise models",
                variant="secondary",
                visible=not (TRANSLATION_READY and XTTS_READY)
            )
            status_timer = gr.Timer(value=3, active=True)

            gr.HTML('<div class="sb-demo-divider"></div>')

            # Language pair — flat, no box
            gr.HTML('<div class="sb-demo-section-label">Language pair</div>')
            with gr.Row(elem_classes=["sb-lang-row"]):
                input_lang = gr.Dropdown(
                    choices=["French (fine-tuned ★)", "Other languages"],
                    value="French (fine-tuned ★)",
                    label="From",
                    interactive=True,
                    elem_classes=["sb-no-box"]
                )
                gr.HTML('<div class="sb-lang-arrow">→</div>')
                output_lang = gr.Dropdown(
                    choices=["English"],
                    value="English",
                    label="To",
                    interactive=False,
                    elem_classes=["sb-no-box"]
                )

            gr.HTML('<div class="sb-demo-divider"></div>')

            # Mode — flat, no box
            gr.HTML('<div class="sb-demo-section-label">Translation mode</div>')
            mode = gr.Radio(
                choices=["Translate without voice cloning", "Translate with voice cloning"],
                value="Translate without voice cloning",
                label="",
                elem_classes=["sb-no-box"]
            )

            gr.HTML('<div class="sb-demo-divider"></div>')

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload or record speech",
            )

            gr.Markdown(
                "<div style='font-size:0.78rem;color:#475569;line-height:1.6;margin-top:4px;'>"
                "Upload any format or record directly. For voice cloning, the clip also serves as the speaker reference. "
                "<strong style='color:#f97316;'>French uses the LoRA fine-tuned model.</strong> "
                "Other languages use base Whisper with auto language detection."
                "</div>"
            )

            translate_btn = gr.Button("Run translation", variant="primary")

        with gr.Column(scale=1):
            translated_text = gr.Textbox(
                label="English translation",
                lines=8,
                placeholder="Translated English text will appear here."
            )
            output_audio = gr.Audio(type="filepath", label="Generated English speech")
            with gr.Row():
                runtime_info = gr.Textbox(label="Runtime breakdown", lines=4, interactive=False)
                pipeline_status = gr.Textbox(label="Status", lines=4, interactive=False)

    gr.HTML("</div></div></section>")

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
                    s += '<span style="color:#f97316;font-size:13px;">' + (i <= filled ? '★' : '☆') + '</span>';
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
