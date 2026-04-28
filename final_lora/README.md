---
base_model: openai/whisper-medium
library_name: peft
---

# Whisper-Medium French-to-English LoRA Adapter

## Model Details

### Model Description

LoRA adapter fine-tuned from OpenAI Whisper-Medium for French-to-English speech translation. Developed as part of the SpeechBridge final year project, which investigates modular speech-to-speech translation with deployable voice cloning.

- **Developed by:** Taibah Rushdi Khalid
- **Model type:** PEFT LoRA adapter for automatic speech recognition / speech translation
- **Language(s):** French input, English output
- **License:** Inherits base model licence from OpenAI Whisper
- **Finetuned from model:** openai/whisper-medium

## Uses

### Direct Use

Load this adapter on top of `openai/whisper-medium` to perform French-to-English speech translation.

### Downstream Use

Can be integrated into multilingual speech translation pipelines, research prototypes, and end-to-end speech applications such as SpeechBridge.

### Out-of-Scope Use

Not intended for medical, legal, or safety-critical translation. Performance may degrade on noisy audio, heavy accents, domain-specific terminology, or unsupported languages.

## Bias, Risks, and Limitations

Results depend on training data coverage and evaluation conditions. Translation quality may vary across accents, recording quality, speaking styles, and uncommon vocabulary. This adapter was trained under limited compute and data constraints.

## How to Get Started

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model = PeftModel.from_pretrained(base, "tai3ah/whisper-medium-french-lora")
