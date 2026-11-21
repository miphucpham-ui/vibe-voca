"""vibe-voca web demo (FastAPI + plain HTML/CSS/JS).

    python -m vibe_voca.web --checkpoint runs/v2f/step-00032000.pt [--voice vi_female] [--port 7860]
    python -m vibe_voca.web --checkpoint ... --test-set vi_female=work/infore/infore24   # "random test sentence"

In Colab open it with: from google.colab import output; output.serve_kernel_port_as_window(7860)

API (relative URLs, so it works behind Colab's port proxy):
  GET  api/info                  checkpoint, voices, examples
  POST api/synth   {text, voice, speed, semitones, pitch_range, seed, polish, texture,
                    comma_ms, stop_ms, gap_ms, punctuate}
                   -> {audio (base64 WAV), seconds, elapsed, spoken, unknown, phonemes[], tracks[]}
  GET  api/random?voice=...      a held-out sentence of that voice (+ its original recording when available)
"""

import argparse
import base64
import io
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..infer import load_model
from ..paths import VIBERT_CAPU
from ..synthesis import Settings, syllables, synthesize, tone_spans

STATIC = Path(__file__).resolve().parent / "static"
VOICE_LABELS = {"vi_nguoi_phu_nu": "Nữ (giọng clone)", "vi_female": "Nữ (InfoRe)", "female": "Nữ (InfoRe)",
                "vi_adam": "Nam (Adam)", "F2": "F2 · tiếng Anh gốc", "M2": "M2 · tiếng Anh gốc"}
EXAMPLES = [
    "Xin chào, tôi là trợ lý ảo tiếng Việt.",
    "Bệnh cường giáp xảy ra khi tuyến giáp sản xuất quá mức hormone thyroxine.",
    "Bác sĩ khuyên dùng vitamin D và omega 3 mỗi ngày, uống sau bữa sáng.",
    "hôm nay trời đẹp chúng ta cùng đi dạo công viên nhé",
    "Năm 2024 có 366 ngày, giá vàng tăng 12,5% so với năm trước.",
    "Tôi học Machine Learning ở MIT và nghe nhạc mỗi ngày.",
    "Universities from the United States dominated the top 10 of a global ranking.",
]
LOCK = threading.Lock()
STATE = {"model": None, "checkpoint": None, "voice": None, "test_sets": {}}
app = FastAPI(title="vibe-voca")


class Request(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.15
    semitones: float = 2.0
    pitch_range: float = 1.15
    seed: int = 0
    polish: bool = True
    texture: float = 0.5
    comma_ms: float = 200
    stop_ms: float = 350
    gap_ms: float = 300
    punctuate: bool = False


def wav_base64(wave, rate):
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(wave, -1, 1), rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode()


def voice_order(voice):
    return list(VOICE_LABELS).index(voice) if voice in VOICE_LABELS else len(VOICE_LABELS)


@app.get("/api/info")
def info():
    model = STATE["model"]
    return {"checkpoint": f"{STATE['checkpoint']} · step {model.step}",
            "voices": [{"id": v, "label": VOICE_LABELS.get(v, v)} for v in sorted(model.config.speakers, key=voice_order)],
            "default_voice": STATE["voice"], "examples": EXAMPLES,
            "punctuation_available": (VIBERT_CAPU / "pytorch_model.bin").exists(),
            "test_sets": sorted(STATE["test_sets"])}


@app.post("/api/synth")
def synth(r: Request):
    model = STATE["model"]
    text, voice = r.text.strip(), r.voice or STATE["voice"]
    if not text:
        raise HTTPException(400, "Nhập câu cần đọc.")
    if voice not in model.config.speakers:
        raise HTTPException(400, f"Không có giọng {voice}")
    settings = Settings(speed=r.speed, semitones=r.semitones, pitch_range=r.pitch_range, seed=r.seed, polish=r.polish,
                        texture=r.texture, comma_ms=r.comma_ms, stop_ms=r.stop_ms,
                        gap_ms=r.gap_ms, punctuate=r.punctuate)
    with LOCK:
        try:
            result = synthesize(model, text, voice, settings)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    rate, hop = model.config.sample_rate, model.config.hop
    tracks, phonemes = [], []
    for sentence in result.sentences:
        phones, ends = sentence.phones, np.concatenate([[0], np.cumsum(sentence.durations)])
        words = []
        for word, i, j, tone, unknown in syllables(phones):
            spans = tone_spans(phones, sentence.durations, i, j, tone)
            tones = [s[2] for s in spans if s[2] != "EN"]
            words.append({"word": word, "a": int(ends[i]), "b": int(ends[j + 1]),
                          "spans": [{"a": a, "b": b, "tone": t} for a, b, t in spans]})
            phonemes.append({"word": word, "phones": [p.removeprefix("vi:") for p in phones.symbols[i:j + 1]],
                             "tone": "+".join(tones) if tones else (tone or ""), "unknown": bool(unknown)})
        tracks.append({"start": round(sentence.start, 4), "step": hop / rate,
                       "hz": [None if not np.isfinite(x) else round(float(x), 1) for x in sentence.hz], "words": words})
    return {"audio": wav_base64(result.audio, rate), "seconds": round(result.seconds, 3), "elapsed": round(result.elapsed, 3),
            "sentences": len(result.sentences), "spoken": result.spoken if result.spoken != text else None,
            "unknown": result.unknown, "phonemes": phonemes, "tracks": tracks}


@app.get("/api/random")
def random_sentence(voice: str | None = None):
    sets = STATE["test_sets"]
    folder = sets.get(voice or STATE["voice"]) or next(iter(sets.values()), None)
    if folder is None:
        raise HTTPException(404, "Chưa có bộ test (chạy server với --test-set voice=thư_mục)")
    manifest = folder / "test.jsonl"
    if not manifest.exists():
        raise HTTPException(404, f"Không có {manifest}")
    row = random.choice([json.loads(line) for line in open(manifest, encoding="utf-8")])
    text = row["text"] + ("." if row.get("ending") == "period" and not row["text"].rstrip().endswith((".", "?", "!")) else "")
    out = {"id": row["id"], "text": text, "audio": None}
    path = folder / row["audio"]
    if path.exists():
        wave, rate = sf.read(path, dtype="float32")
        out["audio"] = wav_base64(wave, rate)
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("VIBE_VOCA_CHECKPOINT"),
                        help="inference.pt or a training checkpoint (default: $VIBE_VOCA_CHECKPOINT)")
    parser.add_argument("--voice", help="default voice (default: the model's first)")
    parser.add_argument("--test-set", action="append", default=[], metavar="VOICE=DIR",
                        help="24 kHz set with test.jsonl + audio/ for the 'random test sentence' button (repeatable)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint (or VIBE_VOCA_CHECKPOINT) is required")
    model = load_model(args.checkpoint, args.device)
    voice = args.voice or model.config.speakers[0]
    if voice not in model.config.speakers:
        parser.error(f"unknown voice {voice!r}; this model has {model.config.speakers}")
    test_sets = dict(item.split("=", 1) for item in args.test_set)
    STATE.update(model=model, checkpoint=args.checkpoint.name if args.checkpoint.parent.name in ("", ".")
                 else f"{args.checkpoint.parent.name}/{args.checkpoint.name}",
                 voice=voice, test_sets={k: Path(v) for k, v in test_sets.items()})
    started = time.perf_counter()                 # warm-up: CUDA kernels, polish, the punctuation model if present
    synth(Request(text="Xin chào các bạn.", punctuate=(VIBERT_CAPU / "pytorch_model.bin").exists()))
    print(f"vibe-voca web: {args.checkpoint} on {args.device} (warm-up {time.perf_counter() - started:.1f} s), "
          f"http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
