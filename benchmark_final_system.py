# benchmark_final_system.py
# Runs your final pipeline automatically on benchmark samples and computes:
# latency, WER, CER, BLEU, speaker similarity, success rate, RTF

import os
import re
import csv
import json
import time
import uuid
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
import jiwer
import sacrebleu
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from TTS.api import TTS
from resemblyzer import VoiceEncoder, preprocess_wav


# ======================================================
# Helpers
# ======================================================

def norm_text(text):
    text = str(text).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def audio_duration(path):
    info = sf.info(path)
    return float(info.duration)


def convert_audio(input_path, output_path, sr=16000):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sr),
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def split_chunks(text, max_chars=180):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks = []
    cur = ""

    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks


# ======================================================
# Benchmark
# ======================================================

class BenchmarkSystem:
    def __init__(
        self,
        base_model_id,
        lora_dir,
        piper_model,
        piper_cfg,
        out_dir,
        device="cpu",
        xtts_model="tts_models/multilingual/multi-dataset/xtts_v2",
        eval_asr_model="openai/whisper-small",
    ):
        self.device = device
        self.out_dir = Path(out_dir)
        self.temp_dir = self.out_dir / "temp"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Load translation model
        self.processor = WhisperProcessor.from_pretrained(base_model_id)

        base = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        base.eval()

        self.model = PeftModel.from_pretrained(base, lora_dir).to(device)
        self.model.eval()

        # Eval ASR
        self.eval_proc = WhisperProcessor.from_pretrained(eval_asr_model)
        self.eval_model = WhisperForConditionalGeneration.from_pretrained(
            eval_asr_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        self.eval_model.eval()

        # XTTS
        self.xtts = TTS(xtts_model).to(device)

        # Similarity
        self.encoder = VoiceEncoder()

        # Piper paths
        self.piper_model = piper_model
        self.piper_cfg = piper_cfg

    # --------------------------------------------
    # Translation
    # --------------------------------------------
    @torch.inference_mode()
    def translate(self, audio_path):
        wav16 = self.temp_dir / f"{uuid.uuid4().hex}_16k.wav"
        convert_audio(audio_path, wav16, sr=16000)

        audio, sr = librosa.load(str(wav16), sr=16000, mono=True)

        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs["input_features"].to(self.device)

        forced_ids = self.processor.get_decoder_prompt_ids(
            language="french",
            task="translate"
        )

        gen = self.model.generate(
            input_features=feats,
            forced_decoder_ids=forced_ids,
            num_beams=1,
            max_new_tokens=192
        )

        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()

    # --------------------------------------------
    # Piper
    # --------------------------------------------
    def run_piper(self, text, out_wav):
        subprocess.run(
            [
                "piper",
                "--model", self.piper_model,
                "--config", self.piper_cfg,
                "--output_file", str(out_wav)
            ],
            input=text.encode("utf-8"),
            check=True
        )

    # --------------------------------------------
    # XTTS
    # --------------------------------------------
    def run_xtts(self, text, ref_audio, out_wav):
        chunks = split_chunks(text)
        parts = []

        for c in chunks:
            wav = self.xtts.tts(
                text=c,
                speaker_wav=ref_audio,
                language="en"
            )
            parts.append(np.asarray(wav, dtype=np.float32))

        final = np.concatenate(parts)
        sf.write(str(out_wav), final, 24000)

    # --------------------------------------------
    # ASR of generated speech
    # --------------------------------------------
    @torch.inference_mode()
    def transcribe_generated(self, audio_path):
        wav16 = self.temp_dir / f"{uuid.uuid4().hex}_eval.wav"
        convert_audio(audio_path, wav16, sr=16000)

        audio, sr = librosa.load(str(wav16), sr=16000, mono=True)

        inputs = self.eval_proc(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs["input_features"].to(self.device)

        forced_ids = self.eval_proc.get_decoder_prompt_ids(
            language="english",
            task="transcribe"
        )

        gen = self.eval_model.generate(
            input_features=feats,
            forced_decoder_ids=forced_ids,
            num_beams=1,
            max_new_tokens=192
        )

        text = self.eval_proc.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()

    # --------------------------------------------
    # Similarity
    # --------------------------------------------
    def speaker_similarity(self, ref_audio, gen_audio):
        try:
            ref = preprocess_wav(ref_audio)
            gen = preprocess_wav(gen_audio)

            emb1 = self.encoder.embed_utterance(ref)
            emb2 = self.encoder.embed_utterance(gen)

            sim = float(
                np.dot(emb1, emb2) /
                (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )
            return sim
        except:
            return None

    # --------------------------------------------
    # One sample
    # --------------------------------------------
    def run_one(self, row, condition):
        sample_id = row["sample_id"]
        audio_path = row["audio_path"]
        ref_text = row["reference_text"]

        result = {
            "sample_id": sample_id,
            "condition": condition,
            "success": False,
            "error": "",
            "translated_text": "",
            "asr_text": "",
            "generated_audio_path": "",
            "translation_wer": None,
            "final_output_wer": None,
            "translation_cer": None,
            "final_output_cer": None,
            "speaker_similarity": None,
            "translation_time_sec": None,
            "synthesis_time_sec": None,
            "total_latency_sec": None,
            "output_duration_sec": None,
            "tts_rtf": None,
            "end_to_end_rtf": None,
        }

        try:
            t0 = time.time()

            # Translate
            t1 = time.time()
            translated = self.translate(audio_path)
            t2 = time.time()

            cond_dir = self.out_dir / condition
            cond_dir.mkdir(parents=True, exist_ok=True)

            # Synthesis
            if condition.endswith("standard"):
                out_wav = cond_dir / f"{sample_id}_standard.wav"
                self.run_piper(translated, out_wav)
                t3 = time.time()
                sim = None

            else:
                out_wav = cond_dir / f"{sample_id}_clone.wav"
                self.run_xtts(translated, audio_path, out_wav)
                t3 = time.time()
                sim = self.speaker_similarity(audio_path, str(out_wav))

            # ASR on output
            asr_text = self.transcribe_generated(str(out_wav))

            # Metrics
            ref_n = norm_text(ref_text)
            hyp_n = norm_text(translated)
            asr_n = norm_text(asr_text)

            trans_wer = jiwer.wer(ref_n, hyp_n)
            final_wer = jiwer.wer(ref_n, asr_n)

            trans_cer = jiwer.cer(ref_n, hyp_n)
            final_cer = jiwer.cer(ref_n, asr_n)

            dur = audio_duration(str(out_wav))
            total = t3 - t0
            synth = t3 - t2

            result.update({
                "success": True,
                "translated_text": translated,
                "asr_text": asr_text,
                "generated_audio_path": str(out_wav),
                "translation_wer": trans_wer,
                "final_output_wer": final_wer,
                "translation_cer": trans_cer,
                "final_output_cer": final_cer,
                "speaker_similarity": sim,
                "translation_time_sec": t2 - t1,
                "synthesis_time_sec": synth,
                "total_latency_sec": total,
                "output_duration_sec": dur,
                "tts_rtf": synth / dur if dur > 0 else None,
                "end_to_end_rtf": total / dur if dur > 0 else None,
            })

        except Exception as e:
            result["error"] = str(e)

        return result

    # --------------------------------------------
    # Full benchmark
    # --------------------------------------------
    def run(self, csv_path, conditions):
        df = pd.read_csv(csv_path)
        rows = []

        for cond in conditions:
            for _, row in df.iterrows():
                out = self.run_one(row, cond)
                rows.append(out)
                print(out)

        results = pd.DataFrame(rows)

        # Summary
        summary = []

        for cond in conditions:
            sub = results[(results["condition"] == cond) & (results["success"] == True)].copy()

            if len(sub) == 0:
                summary.append({
                    "condition": cond,
                    "n": 0,
                    "success_rate": 0.0,
                    "mean_latency_sec": None,
                    "mean_final_output_wer": None,
                    "mean_translation_wer": None,
                    "mean_similarity": None,
                    "mean_tts_rtf": None,
                    "mean_end_to_end_rtf": None,
                    "translation_bleu": None,
                })
                continue

            refs = [norm_text(x) for x in df["reference_text"].tolist()[:len(sub)]]
            hyps = [norm_text(x) for x in sub["translated_text"].fillna("").tolist()]

            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

            sim_series = sub["speaker_similarity"].dropna()
            mean_similarity = sim_series.mean() if len(sim_series) > 0 else None

            summary.append({
                "condition": cond,
                "n": len(sub),
                "success_rate": len(sub) / len(df) if len(df) else 0,
                "mean_latency_sec": sub["total_latency_sec"].mean(),
                "mean_final_output_wer": sub["final_output_wer"].mean(),
                "mean_translation_wer": sub["translation_wer"].mean(),
                "mean_similarity": mean_similarity,
                "mean_tts_rtf": sub["tts_rtf"].mean(),
                "mean_end_to_end_rtf": sub["end_to_end_rtf"].mean(),
                "translation_bleu": bleu,
            })

        results.to_csv(self.out_dir / "per_sample_results.csv", index=False)
        pd.DataFrame(summary).to_csv(self.out_dir / "summary_results.csv", index=False)

        print("\n=== SUMMARY ===")
        print(pd.DataFrame(summary))


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--piper_model", required=True)
    parser.add_argument("--piper_cfg", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conditions", default="local_standard,local_clone")
    args = parser.parse_args()

    bench = BenchmarkSystem(
        base_model_id="openai/whisper-medium",
        lora_dir=args.lora_dir,
        piper_model=args.piper_model,
        piper_cfg=args.piper_cfg,
        out_dir=args.out_dir,
        device=args.device
    )

    conds = [x.strip() for x in args.conditions.split(",") if x.strip()]
    bench.run(args.csv_path, conds)


if __name__ == "__main__":
    main()