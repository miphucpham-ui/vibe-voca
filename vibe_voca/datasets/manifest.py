"""Any single-speaker set of (text, audio) -> vibe-voca's 24 kHz layout, with the same checks as vibe_voca.datasets.infore.

Input: either jsonl manifests (``--manifests``; rows with id, text, audio relative to the manifest's folder), or the
archive of Vietnamese-Adam-TTS-50h (``--archive``, the default voice "vi_adam"):

Source of the archive: HF dataset Phitran21/Vietnamese-Adam-TTS-50h, revision b3b0c6772c2ca61db5437991f26275e0e780311c (CC-BY-4.0
as labelled; synthetic single-speaker speech from an "Adam" voice preset of an unnamed generator: check the
generator's terms before releasing a model trained on it). 18,478 clips, 48 kHz WAV, pre-normalised text WITH
punctuation (numbers and abbreviations already spelled out).

    python -m vibe_voca.datasets.manifest --archive finnal_tts.zip --extracted $VIBE_VOCA_WORK/adam/extracted \
        --output $VIBE_VOCA_WORK/adam/adam24
    python -m vibe_voca.datasets.manifest --manifests my_set/manifest_{train,validation,test}.jsonl \
        --voice vi_myvoice --prefix my --source-url URL --source-license LICENSE --output $VIBE_VOCA_WORK/my/my24

* text is kept as written (case and punctuation): the front end reads commas, and English words go to dizione; a
  clip with a word nothing can read is left out;
* speech edges from energy (infore.speech_edges); the clip is cut 60 ms after speech; a clip whose text ends with
  . ? ! gets the clip's own room tone up to 400 ms after the last word (InfoRe's "period" ending), any other clip
  ends open;
* reading-rate filter (robust z-score of syllables per second of speech), length limits, text-hash splits,
  coverage report (vowels, consonants, tones) as for InfoRe.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import infore
from ..g2p import parse

VOICE = "vi_adam"
SOURCE_URL = "https://huggingface.co/datasets/Phitran21/Vietnamese-Adam-TTS-50h"
SOURCE_LICENSE = "CC-BY-4.0 (synthetic voice; check the generator's terms)"
FINAL = re.compile(r"[.?!…]\s*$")


def syllables(text):
    return [w for w in re.findall(r"[^\W\d_]+", text.lower())]


def scan_job(job):
    key, path = job
    wave, rate = sf.read(path, dtype="float64", always_2d=True)
    wave = wave.mean(1)
    found = infore.speech_edges(wave, rate)
    out = {"key": key, "duration": round(len(wave) / rate, 3)}
    if found:
        out.update(speech_start=round(found[0], 3), speech_end=round(found[1], 3),
                   lead_ms=round(found[0] * 1000), final_ms=round((len(wave) / rate - found[1]) * 1000))
    return out


def write_job(job):
    target, path, edges, period, settings = job
    if Path(target).is_file():
        return target, None
    wave, rate = sf.read(path, dtype="float64", always_2d=True)
    wave = infore.resample(infore.shape(wave.mean(1), rate, edges, period, settings), rate)
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak < 1e-4:
        return target, "silent"
    temporary = target + ".tmp"
    sf.write(temporary, (infore.PEAK * wave / peak).astype(np.float32), infore.RATE, format="FLAC", subtype="PCM_16")
    os.replace(temporary, target)
    return target, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extracted", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+", help="instead of the Adam archive: jsonl rows with id, text, audio (relative to the manifest's folder)")
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--prefix", default="adam")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--source-license", default=SOURCE_LICENSE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-ms", type=int, default=400)
    parser.add_argument("--tail-ms", type=int, default=60)
    parser.add_argument("--lead-ms", type=int, default=None)
    parser.add_argument("--rate-z", type=float, default=4.0)
    parser.add_argument("--min-speech-s", type=float, default=0.6)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--validation-permille", type=int, default=30)
    parser.add_argument("--test-permille", type=int, default=30)
    parser.add_argument("--limit", type=int, help="use only the first N clips (by id)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    if not args.manifests and not (args.archive and args.extracted):
        parser.error("give --manifests, or --archive and --extracted")
    from ..text import phonemize

    if args.manifests:
        rows = [{**r, "path": str(m.parent / r["audio"])} for m in args.manifests
                for r in map(json.loads, open(m, encoding="utf-8"))]
        rows.sort(key=lambda r: str(r["id"]))
    else:
        if not (args.extracted / "data.jsonl").exists():
            print(f"extracting {args.archive}", flush=True)
            with zipfile.ZipFile(args.archive) as archive:
                archive.extractall(args.extracted)
        data = next(args.extracted.rglob("data.jsonl"))
        wav_dir = data.parent / "wav"
        rows = [json.loads(line) for line in open(data, encoding="utf-8") if line.strip()]
        rows.sort(key=lambda r: int(r["id"]))
        for r in rows:
            r["path"], r["id"] = str(wav_dir / f"{r['id']}.wav"), int(r["id"])
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    left_out, examples = collections.Counter(), collections.defaultdict(list)

    readable = {}
    for r in rows:
        text = re.sub(r"\s+", " ", str(r["text"])).strip()
        path = Path(r["path"])
        if not path.exists():
            left_out["audio file missing"] += 1
            continue
        try:
            unknown = phonemize(text).unknown
        except ValueError:
            unknown = ["(nothing to say)"]
        if unknown:
            left_out["word nothing can read"] += 1
            examples["word nothing can read"].append(f"{r['id']}: {unknown[:3]} | {text[:80]}")
            continue
        readable[r["id"]] = {"text": text, "path": str(path)}

    scan_path = args.output / "scan.jsonl"
    scans = {}
    if scan_path.exists():
        for line in open(scan_path, encoding="utf-8"):
            s = json.loads(line)
            scans[s["key"]] = s
    todo = [(k, v["path"]) for k, v in readable.items() if k not in scans]
    if todo:
        print(f"scanning {len(todo)} clips for speech edges", flush=True)
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool, open(scan_path, "a", encoding="utf-8") as sink:
            for n, s in enumerate(pool.map(scan_job, todo, chunksize=8), 1):
                scans[s["key"]] = s
                sink.write(json.dumps(s) + "\n")
                if n % 2000 == 0:
                    print(f"  {n} scanned", flush=True)

    candidates = []
    for key, t in readable.items():
        s = scans[key]
        if "speech_end" not in s:
            left_out["no speech found"] += 1
            continue
        speech = s["speech_end"] - s["speech_start"]
        s["speech_s"], s["rate"] = round(speech, 3), round(len(syllables(t["text"])) / max(speech, 1e-3), 3)
        candidates.append((key, t, s))
    rates = np.array([s["rate"] for _, _, s in candidates])
    median, mad = float(np.median(rates)), float(np.median(np.abs(rates - np.median(rates)))) * 1.4826
    kept = {split: [] for split in infore.SPLITS}
    for key, t, s in candidates:
        z = (s["rate"] - median) / mad
        period = bool(FINAL.search(t["text"]))
        longest = (s["speech_start"] if args.lead_ms is None else args.lead_ms / 1000) + s["speech_s"] + args.period_ms / 1000
        reason = ("speech too short" if s["speech_s"] < args.min_speech_s else
                  "clip too long" if longest > args.max_seconds else
                  "too fast for its text (audio short / text extra)" if z > args.rate_z else
                  "too slow for its text (audio long / text missing)" if z < -args.rate_z else None)
        if reason:
            left_out[reason] += 1
            examples[reason].append(f"{key}: {s['rate']} syll/s, {s['speech_s']} s: {t['text'][:80]}")
            continue
        split = infore.split_of(t["text"], args.validation_permille, args.test_permille)
        kept[split].append({"key": key, "text": t["text"], "path": t["path"], "scan": s, "period": period})

    audio_dir = args.output / "audio"
    audio_dir.mkdir(exist_ok=True)
    settings = {"tail_ms": args.tail_ms, "lead_ms": args.lead_ms, "period_ms": args.period_ms}
    by_key = {c["key"]: c for clips in kept.values() for c in clips}
    for c in by_key.values():
        c["id"] = f"{args.prefix}_{c['key']:05d}" if isinstance(c["key"], int) else f"{args.prefix}_{c['key']}"
        c["audio"] = f"audio/{c['id']}.{'period' if c['period'] else 'open'}.flac"
    print(f"writing {len(by_key)} clips", flush=True)
    failed = set()
    jobs = [(str(args.output / c["audio"]), c["path"], c["scan"], c["period"], settings) for c in by_key.values()]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for n, (target, error) in enumerate(pool.map(write_job, jobs, chunksize=8), 1):
            if error:
                failed.add(target)
                left_out[error] += 1
            if n % 2000 == 0:
                print(f"  {n} written", flush=True)

    entries = {split: [] for split in infore.SPLITS}
    for split, clips in kept.items():
        for c in clips:
            if str(args.output / c["audio"]) in failed:
                continue
            entries[split].append({"id": c["id"], "split": split, "voice_id": args.voice, "text": c["text"],
                                   "ending": "period" if c["period"] else "open", "audio": c["audio"],
                                   "speech_s": c["scan"]["speech_s"], "rate": c["scan"]["rate"], "source_id": c["key"],
                                   "source_license": args.source_license, "source_url": args.source_url})
    for split in infore.SPLITS:
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries[split]),
                                                   encoding="utf-8")
    # coverage on the Vietnamese syllables only (punctuation and English words left aside)
    plain = {s: [{"text": " ".join(w for w in syllables(e["text"]) if parse(w))} for e in v] for s, v in entries.items()}
    cover = infore.coverage(plain, args.min_count)
    (args.output / "coverage.json").write_text(json.dumps(cover, ensure_ascii=False, indent=1), encoding="utf-8")
    commas = sum(e["text"].count(",") for e in entries["train"])
    summary = {"kept": {s: len(entries[s]) for s in infore.SPLITS},
               "speech_hours": {s: round(sum(e["speech_s"] for e in entries[s]) / 3600, 2) for s in infore.SPLITS},
               "period": {s: sum(e["ending"] == "period" for e in entries[s]) for s in infore.SPLITS},
               "commas_in_train": commas, "left_out": dict(left_out),
               "rate": {"median_syll_per_s": round(median, 3), "mad": round(mad, 3)},
               "examples_left_out": {k: v[:8] for k, v in examples.items()}, "coverage_warnings": cover["warnings"]}
    (args.output / f"{args.prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("kept", "speech_hours", "period", "commas_in_train", "left_out", "rate",
                                              "coverage_warnings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
