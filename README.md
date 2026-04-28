# SpeechBridge
### Modular Speech-to-Speech Translation System

**Taibah Rushdi Khalid (20529415)**

---

## Live Demonstration / Testing

### **The system must be tested via the hosted Hugging Face Space:**

https://huggingface.co/spaces/tai3ah/speechbridge


**Important:** After opening the page, please select the **App** tab to access the live system interface.

This repository is provided for archival, code review, and reproducibility purposes only. It is not intended to be the marking environment. All functional testing should be carried out using the hosted deployment above. No local setup is required.

---

## Project Overview

SpeechBridge is an end-to-end speech-to-speech translation system that converts spoken audio into English speech output, with an optional voice cloning mode that aims to preserve characteristics of the original speaker's voice.

### Core Functionality

- **French input** uses a LoRA fine-tuned Whisper Medium model specialised for French-to-English translation.
- **Other supported languages** use base Whisper Medium with automatic language detection.
- **Voice cloning mode** uses XTTS v2 with the input clip as the speaker reference.
- **Standard mode** produces clean English speech using Piper TTS.

---

## Suggested Test Cases

1. Upload or record French speech and verify English translated speech output.
2. Test a non-French input and verify automatic language detection.
3. Compare **Standard Mode** and **Voice Cloning Mode** outputs.
4. Evaluate intelligibility, latency, and speaker similarity.

---

## Repository Structure

```text
app.py               # Local archival version
app2hf.py            # Hugging Face Spaces deployment version
audio_samples/       # Sample audio clips for testing
final_lora/          # LoRA adapter weights
piper_models/        # Piper TTS voice model files
requirements.txt     # Python dependencies
```

---

## Local Setup (optional, for reference only)

**Prerequisites:** Python 3.10+, `ffmpeg` installed on your system, GPU recommended.

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
python app.py
```

First run will automatically download the base Whisper and XTTS models (~3GB total). App runs at `http://localhost:7860`.
