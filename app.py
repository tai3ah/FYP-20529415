import re
import time
import uuid
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


# =========================================================
# Config
# =========================================================
BASE_MODEL_ID = "openai/whisper-medium"
LORA_DIR = Path("final_lora")

PIPER_DIR = Path("piper_models")
PIPER_VOICE = "en_US-lessac-medium"
PIPER_MODEL = PIPER_DIR / f"{PIPER_VOICE}.onnx"
PIPER_CFG = PIPER_DIR / f"{PIPER_VOICE}.onnx.json"

DEVICE = "cpu"
TORCH_DTYPE = torch.float32
MAX_AUDIO_SECONDS = 20

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

# Lazy-loaded globals
ST_PROC = None
ST_MODEL = None
MODEL_READY = False


# =========================================================
# Backend utility
# =========================================================
def normalise_text_for_tts(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def startup_validation() -> Optional[str]:
    if not (PIPER_MODEL.exists() and PIPER_CFG.exists()):
        return (
            "Piper voice files are missing.\n"
            f"Expected:\n- {PIPER_MODEL}\n- {PIPER_CFG}\n"
            f"Run:\npython -m piper.download_voices {PIPER_VOICE} --download-dir {PIPER_DIR}"
        )

    if not LORA_DIR.exists():
        return f"LoRA folder not found at: {LORA_DIR}"

    return None


def load_audio_mono_16k(path: str, max_seconds: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) > max_seconds * 16000:
        y = y[: max_seconds * 16000]
    return y, 16000


def piper_tts(text: str, out_wav: Path):
    cmd = [
        "piper",
        "--model", str(PIPER_MODEL),
        "--config", str(PIPER_CFG),
        "--output_file", str(out_wav),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def load_translation_pipeline() -> str:
    global ST_PROC, ST_MODEL, MODEL_READY

    if MODEL_READY and ST_PROC is not None and ST_MODEL is not None:
        return "Model already loaded."

    validation_error = startup_validation()
    if validation_error:
        raise RuntimeError(validation_error)

    t0 = time.time()

    ST_PROC = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    base_model.eval()

    ST_MODEL = PeftModel.from_pretrained(base_model, str(LORA_DIR)).to(DEVICE)
    ST_MODEL.eval()

    MODEL_READY = True
    return f"Translation pipeline loaded in {time.time() - t0:.2f}s."


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


def initialise_models():
    try:
        msg = load_translation_pipeline()
        return f"Ready. {msg}"
    except Exception as e:
        return f"Initialisation failed: {str(e)}"


def run_standard_pipeline(audio_path: str, progress=gr.Progress()):
    if audio_path is None or not Path(audio_path).exists():
        return (
            "No audio provided.",
            None,
            "No timing information available.",
            "Please upload or record a French speech file to continue."
        )

    if not MODEL_READY:
        progress(0.1, desc="Loading translation model")
        load_translation_pipeline()

    t0 = time.time()

    progress(0.25, desc="Loading and preparing audio")
    audio_16k, sr = load_audio_mono_16k(audio_path, MAX_AUDIO_SECONDS)
    t_load = time.time()

    progress(0.55, desc="Translating French speech to English")
    en_text = translate_fr_to_en(audio_16k, sr)
    t_st = time.time()

    en_text_tts = normalise_text_for_tts(en_text)
    if not en_text_tts:
        return (
            "Translation produced empty text.",
            None,
            "Audio was processed, but no valid translated text was produced.",
            "Try a clearer or shorter audio sample."
        )

    progress(0.85, desc="Generating English speech output")
    out_wav = OUT_DIR / f"tts_{uuid.uuid4().hex}.wav"
    piper_tts(en_text_tts, out_wav)
    t_tts = time.time()

    progress(1.0, desc="Completed")

    timing_info = (
        f"Audio loading: {t_load - t0:.2f}s\n"
        f"Speech translation: {t_st - t_load:.2f}s\n"
        f"Speech synthesis: {t_tts - t_st:.2f}s\n"
        f"Total runtime: {t_tts - t0:.2f}s"
    )

    status = (
        "Completed successfully. Standard French-to-English speech translation "
        "and English speech generation finished."
    )

    return en_text, str(out_wav), timing_info, status


def run_voice_clone_placeholder(audio_path: str):
    if audio_path is None or not Path(audio_path).exists():
        return (
            "No audio provided.",
            None,
            "No timing information available.",
            "Please upload or record a French speech file to continue."
        )

    return (
        "Voice cloning mode is currently under development.",
        None,
        "Pipeline timing is not available for this mode yet.",
        (
            "This interface pathway has been prepared for future integration of "
            "speaker-preserving output generation."
        )
    )


def run_app(mode: str, audio_path: str, progress=gr.Progress()):
    if mode == "Translate without voice cloning":
        return run_standard_pipeline(audio_path, progress=progress)

    if mode == "Translate with voice cloning":
        progress(0.2, desc="Opening voice cloning mode")
        return run_voice_clone_placeholder(audio_path)

    return (
        "Invalid mode selected.",
        None,
        "No timing information available.",
        "Please choose a valid translation mode."
    )


# =========================================================
# Content text you can edit easily
# =========================================================
HERO_TITLE = "Affordable translation."
HERO_SUBTITLE = "Drop the communication barrier."

ABOUT_TEXT = (
    "SpeechBridge is a modular speech-to-speech translation system focused on "
    "French-to-English spoken translation. It combines a fine-tuned Whisper-based "
    "translation model with English speech synthesis in a clean, demonstration-ready interface."
)

PIPELINE_CARD_1 = (
    "Record or upload French speech audio using the interface below. "
    "Short clear clips are processed for end-to-end translation."
)

PIPELINE_CARD_2 = (
    "Choose between standard translation or a voice cloning pathway. "
    "The current implemented system performs translation without voice cloning."
)

PIPELINE_CARD_3 = (
    "Receive translated English text and generated English speech output, together with "
    "runtime information for transparent system evaluation."
)

FUTURE_WORK_TEXT = (
    "Current scope is limited to French-to-English speech translation. Future development may "
    "include integrated voice cloning, broader multilingual support, faster inference, and "
    "real-time streaming deployment."
)

FEEDBACK_1 = "Trial feedback from students or evaluators can be inserted here."
FEEDBACK_2 = "Use this section to present usability impressions from your university testing."
FEEDBACK_3 = "You can also summarise comments about translation quality, latency, and clarity here."


# =========================================================
# CSS
# =========================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

html {
    scroll-behavior: smooth;
}

body, .gradio-container {
    background: #f3f3f1 !important;
    font-family: 'Inter', sans-serif !important;
    color: #111111 !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

#main-wrap {
    max-width: 1500px;
    margin: 0 auto;
}

#top-nav {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: #dedcc4;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.nav-inner {
    max-width: 1400px;
    margin: 0 auto;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background:
        radial-gradient(circle at center, #111 12%, transparent 13%),
        repeating-radial-gradient(circle at center, #111 0 1.6px, transparent 1.6px 4px);
    opacity: 0.95;
}

.brand-text {
    display: flex;
    flex-direction: column;
    line-height: 1;
}

.brand-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #111;
}

.brand-sub {
    font-size: 0.54rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #555;
    margin-top: 4px;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 34px;
    flex-wrap: wrap;
    justify-content: center;
}

.nav-links a {
    color: #111;
    text-decoration: none;
    font-size: 1rem;
    font-weight: 500;
}

.nav-links a:hover {
    opacity: 0.7;
}

.nav-btn {
    background: #000;
    color: #fff !important;
    padding: 14px 28px;
    text-decoration: none;
    font-weight: 600;
    display: inline-block;
    min-width: 130px;
    text-align: center;
}

.hero-section {
    background: #f3f3f1;
    padding: 88px 40px 70px 40px;
    text-align: center;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(4rem, 7vw, 6.4rem);
    line-height: 0.95;
    font-weight: 500;
    margin-bottom: 22px;
    color: #111;
}

.hero-subtitle {
    font-size: 1.7rem;
    font-weight: 300;
    margin-bottom: 18px;
    color: #333;
}

.hero-desc {
    max-width: 850px;
    margin: 0 auto 26px auto;
    font-size: 1.02rem;
    line-height: 1.8;
    color: #4b4b4b;
}

.hero-cta {
    display: inline-block;
    background: #000;
    color: #fff !important;
    text-decoration: none;
    padding: 15px 34px;
    font-weight: 600;
    margin-top: 6px;
}

.section-wrap {
    padding: 82px 70px;
}

.section-light {
    background: #f3f3f1;
}

.section-dark {
    background: #000;
    color: #fff;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3rem, 5vw, 5rem);
    line-height: 1.02;
    text-align: center;
    margin-bottom: 28px;
    font-weight: 500;
}

.section-subtext {
    max-width: 940px;
    margin: 0 auto 48px auto;
    text-align: center;
    line-height: 1.8;
    font-size: 1rem;
    color: #4f4f4f;
}

.section-dark .section-subtext {
    color: #e9e9e9;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 26px;
    margin-top: 20px;
}

.info-card {
    background: transparent;
    padding: 10px 4px;
}

.info-card h3 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    margin-bottom: 14px;
    font-weight: 600;
}

.info-card p {
    font-size: 1rem;
    line-height: 1.95;
    color: #333;
}

.center-btn-wrap {
    text-align: center;
    margin-top: 34px;
}

.center-btn {
    display: inline-block;
    background: #000;
    color: #fff !important;
    text-decoration: none;
    padding: 16px 34px;
    font-weight: 600;
    min-width: 180px;
    text-align: center;
}

.demo-shell {
    max-width: 1320px;
    margin: 36px auto 0 auto;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(255,255,255,0.56);
    backdrop-filter: blur(4px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.05);
}

.demo-top-bar {
    background: #dedcc4;
    padding: 14px 18px;
    font-size: 0.95rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}

.demo-note {
    max-width: 900px;
    margin: 0 auto 18px auto;
    text-align: center;
    font-size: 1rem;
    line-height: 1.8;
    color: #444;
}

.feedback-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 30px;
    margin-top: 40px;
}

.feedback-card {
    text-align: center;
    padding: 10px 14px;
}

.feedback-avatar {
    width: 88px;
    height: 88px;
    border-radius: 50%;
    background: linear-gradient(135deg, #b9bccf, #8287a7);
    margin: 0 auto 18px auto;
}

.quote-mark {
    font-size: 4rem;
    line-height: 1;
    color: #7280e4;
    margin-bottom: 10px;
    font-family: 'Cormorant Garamond', serif;
}

.feedback-text {
    font-size: 1rem;
    line-height: 1.8;
    color: #444;
    font-style: italic;
    min-height: 118px;
}

.feedback-name {
    margin-top: 16px;
    font-weight: 700;
    font-size: 1.08rem;
}

.footer {
    background: #dedcc4;
    padding: 44px 60px;
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.7fr 1fr 0.9fr;
    gap: 34px;
    align-items: start;
}

.footer-title {
    font-weight: 700;
    margin-bottom: 12px;
}

.footer-links a,
.footer-text {
    display: block;
    color: #111;
    text-decoration: none;
    line-height: 2;
}

.footer-btn {
    display: inline-block;
    background: #000;
    color: #fff !important;
    text-decoration: none;
    padding: 15px 30px;
    font-weight: 600;
    min-width: 170px;
    text-align: center;
}

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    border-radius: 0 !important;
}

.gradio-container .gr-button-primary,
.gradio-container .gr-button {
    background: #000 !important;
    border: 1px solid #000 !important;
    color: #fff !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

.gradio-container .gr-button-secondary {
    background: #f3f3f1 !important;
    color: #111 !important;
    border: 1px solid #111 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap,
.gradio-container .gr-textbox,
.gradio-container .gr-audio,
.gradio-container .gr-radio,
.gradio-container .gr-group {
    border-radius: 0 !important;
}

#demo-area .gr-group,
#demo-area .gr-box,
#demo-area .gr-form,
#demo-area .gr-column {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#demo-area label,
#demo-area .gradio-container label {
    font-weight: 700 !important;
    color: #111 !important;
}

#demo-area .gr-textbox,
#demo-area .gr-audio,
#demo-area .gr-radio {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    padding: 8px !important;
}

.small-muted {
    color: #666;
    font-size: 0.95rem;
    line-height: 1.7;
}

@media (max-width: 1000px) {
    .card-grid,
    .feedback-grid,
    .footer-grid {
        grid-template-columns: 1fr;
    }

    .nav-inner {
        flex-direction: column;
        align-items: stretch;
    }

    .nav-links {
        gap: 18px;
    }

    .hero-section,
    .section-wrap,
    .footer {
        padding-left: 24px;
        padding-right: 24px;
    }
}
"""


# =========================================================
# UI
# =========================================================
with gr.Blocks(css=CUSTOM_CSS, title="SpeechBridge") as demo:
    gr.HTML("""
    <div id="main-wrap">
        <div id="top-nav">
            <div class="nav-inner">
                <div class="brand-wrap">
                    <div class="brand-logo"></div>
                    <div class="brand-text">
                        <div class="brand-title">SpeechBridge</div>
                        <div class="brand-sub">French to English translation</div>
                    </div>
                </div>

                <div class="nav-links">
                    <a href="#home">Home</a>
                    <a href="#about">About</a>
                    <a href="#pipeline">Our Offering</a>
                    <a href="#contact">Contact</a>
                </div>

                <a class="nav-btn" href="#contact">Contact Us</a>
            </div>
        </div>
    """)

    gr.HTML(f"""
        <section id="home" class="hero-section">
            <div class="hero-title">{HERO_TITLE}</div>
            <div class="hero-subtitle">{HERO_SUBTITLE}</div>
            <div class="hero-desc">{ABOUT_TEXT}</div>
            <a class="hero-cta" href="#pipeline">Explore the System</a>
        </section>
    """)

    gr.HTML(f"""
        <section id="about" class="section-wrap section-light">
            <div class="section-title">More about the pipeline</div>
            <div class="section-subtext">
                SpeechBridge currently demonstrates a modular French-to-English spoken translation workflow.
                Users can record or upload audio, select a translation mode, and receive translated English
                text and speech output through a clean evaluation interface.
            </div>

            <div class="card-grid">
                <div class="info-card">
                    <h3>Speech input</h3>
                    <p>{PIPELINE_CARD_1}</p>
                </div>

                <div class="info-card">
                    <h3>Translation mode</h3>
                    <p>{PIPELINE_CARD_2}</p>
                </div>

                <div class="info-card">
                    <h3>Generated output</h3>
                    <p>{PIPELINE_CARD_3}</p>
                </div>
            </div>

            <div class="center-btn-wrap">
                <a class="center-btn" href="#pipeline">Our offering</a>
            </div>
        </section>
    """)

    gr.HTML("""
        <section id="pipeline" class="section-wrap section-light">
            <div class="section-title">Try SpeechBridge</div>
            <div class="demo-note">
                Record or upload a short French speech sample, then choose whether to run
                standard translation or open the voice cloning pathway currently under development.
            </div>

            <div class="demo-shell">
                <div class="demo-top-bar">SpeechBridge demonstration interface</div>
                <div id="demo-area" style="padding: 28px 26px 22px 26px;">
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_status = gr.Textbox(
                label="System status",
                value="App ready. Models are lazy-loaded so the interface opens faster.",
                interactive=False,
                lines=3
            )
            init_btn = gr.Button("Initialise translation model", variant="secondary")

            mode = gr.Radio(
                choices=[
                    "Translate without voice cloning",
                    "Translate with voice cloning"
                ],
                value="Translate without voice cloning",
                label="Translation mode"
            )

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Record or upload French speech audio"
            )

            gr.Markdown(
                "<div class='small-muted'>Recommended input, short and clear speech under 20 seconds for smoother CPU performance during demonstration.</div>"
            )

            translate_btn = gr.Button("Run translation", variant="primary")

        with gr.Column(scale=1):
            translated_text = gr.Textbox(
                label="English translation",
                lines=8,
                placeholder="Translated English text will appear here."
            )

            output_audio = gr.Audio(
                type="filepath",
                label="Generated English speech"
            )

            runtime_info = gr.Textbox(
                label="Runtime information",
                lines=5,
                interactive=False
            )

            pipeline_status = gr.Textbox(
                label="Processing feedback",
                lines=4,
                interactive=False
            )

    gr.HTML("""
                </div>
            </div>
        </section>
    """)

    gr.HTML(f"""
        <section class="section-wrap section-dark">
            <div class="section-title">Future development</div>
            <div class="section-subtext">{FUTURE_WORK_TEXT}</div>
        </section>
    """)

    gr.HTML("""
        <section class="section-wrap section-light">
            <div class="section-title">The spoken word when it matters.</div>
    """)

    gr.HTML(f"""
            <div class="feedback-grid">
                <div class="feedback-card">
                    <div class="feedback-avatar"></div>
                    <div class="quote-mark">”</div>
                    <div class="feedback-text">{FEEDBACK_1}</div>
                    <div class="feedback-name">Trial Feedback 1</div>
                </div>

                <div class="feedback-card">
                    <div class="feedback-avatar" style="background: linear-gradient(135deg, #9da1b7, #53576f);"></div>
                    <div class="quote-mark">”</div>
                    <div class="feedback-text">{FEEDBACK_2}</div>
                    <div class="feedback-name">Trial Feedback 2</div>
                </div>

                <div class="feedback-card">
                    <div class="feedback-avatar" style="background: linear-gradient(135deg, #b9a5a0, #6b5e5a);"></div>
                    <div class="quote-mark">”</div>
                    <div class="feedback-text">{FEEDBACK_3}</div>
                    <div class="feedback-name">Trial Feedback 3</div>
                </div>
            </div>
        </section>
    """)

    gr.HTML("""
        <footer id="contact" class="footer">
            <div class="footer-grid">
                <div>
                    <div class="brand-wrap" style="align-items:flex-start;">
                        <div class="brand-logo" style="width:28px;height:28px;margin-top:2px;"></div>
                        <div class="brand-text">
                            <div class="brand-title" style="font-size:1.35rem;">SpeechBridge</div>
                            <div class="brand-sub">French to English translation</div>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="footer-title">Home</div>
                    <div class="footer-links">
                        <a href="#home">Home</a>
                        <a href="#about">About</a>
                        <a href="#pipeline">Our Offering</a>
                        <a href="#contact">Contact</a>
                    </div>
                </div>

                <div>
                    <div class="footer-title">Project details</div>
                    <div class="footer-text">French to English speech translation demo</div>
                    <div class="footer-text">Whisper-medium + LoRA + Piper TTS</div>
                    <div class="footer-text">University final year project</div>
                    <div class="footer-text">Voice cloning pathway in progress</div>
                </div>

                <div>
                    <a class="footer-btn" href="#contact">Contact Us</a>
                </div>
            </div>
        </footer>
    </div>
    """)

    init_btn.click(
        fn=initialise_models,
        outputs=model_status
    )

    translate_btn.click(
        fn=run_app,
        inputs=[mode, audio_input],
        outputs=[translated_text, output_audio, runtime_info, pipeline_status]
    )


if __name__ == "__main__":
    demo.launch()