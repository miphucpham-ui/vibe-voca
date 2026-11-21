"""Intelligibility of a vibe-voca checkpoint: synthesise held-out sentences, let PhoWhisper (CTranslate2)
transcribe them, and score character and word error rates against the text.

    python -m vibe_voca.evaluate --checkpoint runs/v2f/step-00032000.pt --voice vi_female \
        --manifest $VIBE_VOCA_WORK/infore/infore24/test.jsonl --count 80 --out runs/v2f/eval/step32000_female.json

Accepts an inference export (inference.pt) or any training checkpoint. Vietnamese is transcribed by PhoWhisper
(vinai/PhoWhisper-large converted to CTranslate2, $VIBE_VOCA_MODELS/phowhisper-large-ct2; scripts/setup.sh).
English: ``--language en`` transcribes with Whisper small.en ($VIBE_VOCA_MODELS/whisper-small-en-ct2), e.g. on
tiny-voca's held-out sentences with --voice F2.
Text is compared after lower-casing and removing punctuation; the same sentences (first --count of the
manifest, fixed order) are used every time so numbers are comparable across checkpoints.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
import torchaudio.functional as AF

from .infer import load_model, speak
from .model import Polish
from .paths import PHOWHISPER, WHISPER_EN


def load_ct2(model_dir, device="cuda"):
    """A faster-whisper model. Its CTranslate2 wheel is built against CUDA 12 cuBLAS; where the system has another
    CUDA (Colab: 13), the pip wheel nvidia-cublas-cu12 is loaded first when it is installed."""
    if device == "cuda":
        try:
            import ctypes
            import os
            import nvidia.cublas
            lib = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
            for name in ("libcublasLt.so.12", "libcublas.so.12"):
                ctypes.CDLL(os.path.join(lib, name), mode=ctypes.RTLD_GLOBAL)
        except (ImportError, OSError):
            pass
    from faster_whisper import WhisperModel
    return WhisperModel(str(model_dir), device=device, compute_type="float16" if device == "cuda" else "int8")


def words(text):
    return re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", text.lower())).split()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--save-audio", type=Path, help="directory for the first few wavs")
    parser.add_argument("--language", default="vi", choices=("vi", "en"))
    parser.add_argument("--texture", type=float, default=0.0, help="AR(2) pitch texture amount (1 = recordings)")
    parser.add_argument("--whisper", help="CTranslate2 Whisper model directory (default: PhoWhisper for vi, small.en for en)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    step = model.step
    whisper = load_ct2(args.whisper or {"vi": PHOWHISPER, "en": WHISPER_EN}[args.language], args.device)
    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8")][:args.count]
    char_errors = char_total = word_errors = word_total = 0
    results = []
    for n, row in enumerate(rows):
        audio, _, _ = speak(model, row["text"], args.voice, polish=Polish(texture=args.texture))
        wave = AF.resample(audio.float(), 24000, 16000).numpy()
        segments, _ = whisper.transcribe(wave, language=args.language, beam_size=1, vad_filter=False)
        heard = " ".join(s.text.strip() for s in segments)
        ref, hyp = words(row["text"]), words(heard)
        we, ce = edit_distance(ref, hyp), edit_distance(" ".join(ref), " ".join(hyp))
        word_errors, word_total = word_errors + we, word_total + len(ref)
        char_errors, char_total = char_errors + ce, char_total + len(" ".join(ref))
        results.append({"id": row["id"], "text": row["text"], "heard": heard, "wer": round(we / max(len(ref), 1), 3)})
        if args.save_audio and n < 5:
            import soundfile as sf
            args.save_audio.mkdir(parents=True, exist_ok=True)
            sf.write(args.save_audio / f"{row['id']}.wav", audio.numpy(), 24000)
            (args.save_audio / f"{row['id']}.txt").write_text(row["text"] + "\n", encoding="utf-8")
    summary = {"checkpoint": str(args.checkpoint), "step": step, "language": args.language, "voice": args.voice,
               "texture": args.texture,
               "sentences": len(rows),
               "cer": round(char_errors / max(char_total, 1), 4), "wer": round(word_errors / max(word_total, 1), 4)}
    print(json.dumps(summary, ensure_ascii=False))
    for r in sorted(results, key=lambda r: -r["wer"])[:3]:
        print(f"  WER {r['wer']}: {r['text']}\n           heard: {r['heard']}")
    if args.out:
        args.out.write_text(json.dumps({**summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
