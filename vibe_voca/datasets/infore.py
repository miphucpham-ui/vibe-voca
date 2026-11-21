"""Import InfoRe №1 into the layout ``vibe_voca.prepare`` reads: normalised text, balanced sentence
endings, duration filtering and a phoneme-coverage report. No forced alignment is used.

Inputs: the parquet shards and the normalised transcripts of ``vibe_voca.datasets.infore_text``
($VIBE_VOCA_DATA/infore1_text.jsonl). Output:

    <output>/<split>.jsonl      id, split, voice_id, text, audio, ... per clip
    <output>/audio/<id>.flac    mono 24 kHz, peak-normalised
    <output>/scan.jsonl         per clip: speech edges, rate (reused when the script runs again)
    <output>/infore.json        what was kept, what was left out and why, rate statistics
    <output>/coverage.json      phonemes, tones and rhyme x tone per split

1. Speech edges from signal energy (10 ms frames): speech is a run of at least 30 ms above
   ref - 30 dB, ref being the clip's 95th percentile level. Breath and room noise stay below.
2. Duration filtering: the reading rate (syllables per second of speech, between the edges) is
   compared with the corpus: a clip whose robust z-score is beyond --rate-z is left out (too fast
   = text the reader skipped or audio cut short; too slow = speech without text, long hesitations).
   Clips with less than --min-speech-s of speech or longer than --max-seconds are left out too.
   Clips with a word the G2P cannot read (abbreviations, letters, loanwords) are left out.
3. Endings, half and half within every split:
   - "period": text ends with "." and the audio has exactly --period-ms of silence after the
     speech (cut --tail-ms after it, then the clip's own room tone, taken from its leading silence);
   - "open": no "." and no pause: cut --tail-ms after the end of speech.
   Clips whose reader already paused for --period-ms go to "period" first; the rest is drawn at
   random (fixed seed).
4. Coverage: every Vietnamese phoneme, every tone and every rhyme vowel x tone pair is counted per
   split; the report lists what falls under --min-count in train or is missing from a held-out split.

    python -m vibe_voca.datasets.infore --output $VIBE_VOCA_WORK/infore/infore24
"""

import argparse
import collections
import concurrent.futures
import glob
import hashlib
import io
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from ..g2p import NUCLEI, TONE_NAMES, parse
from ..paths import DATA, INFORE

SPLITS = ("train", "validation", "test")
PEAK = 0.95
RATE = 24000
VOICE = "female"
SOURCE_URL = "https://huggingface.co/datasets/doof-ferb/infore1_25hours"
SOURCE_LICENSE = "CC-BY-4.0"
_POOL = [None]          # last usable room tone, per worker process


def split_of(text, validation_permille, test_permille):
    bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 1000
    if bucket < validation_permille:
        return "validation"
    return "test" if bucket < validation_permille + test_permille else "train"


def decode(data):
    wave, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    return wave.mean(1), rate


def speech_edges(wave, rate, silent_db=30.0, min_run_ms=30):
    """(start, end) of speech in seconds from 10 ms frame energy, or None when nothing is loud."""
    hop = int(0.01 * rate)
    count = len(wave) // hop
    if count < 3:
        return None
    db = 10 * np.log10(np.mean(wave[:count * hop].reshape(count, hop) ** 2, axis=1) + 1e-12)
    loud = db >= np.percentile(db, 95) - silent_db
    padded = np.concatenate([[False], loud, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run_ms // 10]
    if not runs:
        return None
    return runs[0][0] * hop / rate, runs[-1][1] * hop / rate


def fade(length, rising):
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, length))
    return ramp if rising else ramp[::-1]


def room_tone(wave, rate, speech_start, needed):
    """``needed`` samples of the clip's own background: its leading silence (stopping 80 ms before
    speech), looped with cross-fades. Falls back to the last clip that had 150 ms of it."""
    lead = wave[int(0.02 * rate):max(int((speech_start - 0.08) * rate), 0)]
    if len(lead) >= int(0.15 * rate):
        _POOL[0] = lead
    elif _POOL[0] is not None:
        lead = _POOL[0]
    else:
        return np.zeros(needed, dtype=wave.dtype)
    cross = min(int(0.01 * rate), len(lead) // 4)
    out = lead.copy()
    while len(out) < needed:
        out = np.concatenate([out[:-cross], out[-cross:] * fade(cross, False) + lead[:cross] * fade(cross, True),
                              lead[cross:]])
    return out[:needed]


def shape(wave, rate, edges, period, settings):
    """Cut the clip after its speech and give it the ending of its group."""
    tail, cross = int(settings["tail_ms"] / 1000 * rate), int(0.01 * rate)
    end = min(int(edges["speech_end"] * rate) + tail, len(wave))
    lead_ms = settings["lead_ms"]
    start = max(int((edges["speech_start"] - lead_ms / 1000) * rate), 0) if lead_ms is not None else 0
    body = wave[start:end].copy()
    body[-cross:] *= fade(cross, False)                       # no click where the cut falls
    if not period:
        return body
    pause = int(settings["period_ms"] / 1000 * rate) - tail
    tone = room_tone(wave, rate, edges["speech_start"], pause)
    tone[:cross] *= fade(cross, True)
    return np.concatenate([body, tone])


def resample(wave, rate):
    from scipy.signal import resample_poly
    if rate == RATE:
        return wave
    factor = math.gcd(rate, RATE)
    return resample_poly(wave, RATE // factor, rate // factor)


def scan_job(job):
    key, data = job
    wave, rate = decode(data)
    found = speech_edges(wave, rate)
    out = {"key": key, "duration": round(len(wave) / rate, 3)}
    if found:
        out.update(speech_start=round(found[0], 3), speech_end=round(found[1], 3),
                   lead_ms=round(found[0] * 1000), final_ms=round((len(wave) / rate - found[1]) * 1000))
    return out


def write_job(job):
    target, data, edges, period, settings = job
    if Path(target).is_file():
        return target, None
    wave, rate = decode(data)
    wave = resample(shape(wave, rate, edges, period, settings), rate)
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak < 1e-4:
        return target, "silent"
    temporary = target + ".tmp"
    sf.write(temporary, (PEAK * wave / peak).astype(np.float32), RATE, format="FLAC", subtype="PCM_16")
    os.replace(temporary, target)
    return target, None


def rows(data_dir, wanted=None):
    """(key, audio bytes) for every clip, key = (shard, row); only ``wanted`` keys when given."""
    import pyarrow.parquet as pq
    for file in sorted(glob.glob(f"{data_dir}/*.parquet")):
        shard, index = Path(file).name, 0
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32, columns=["audio"]):
            for row in batch.to_pylist():
                key = (shard, index)
                index += 1
                if wanted is None or key in wanted:
                    yield key, row["audio"]["bytes"]


def coverage(entries, min_count):
    """Counts of phonemes, tones and rhyme vowel x tone per split, and what is thin or missing."""
    vowels = sorted(set(NUCLEI.values()))
    report, warnings = {}, []
    for split, items in entries.items():
        phones, tones, pairs = collections.Counter(), collections.Counter(), collections.Counter()
        for e in items:
            for word in e["text"].rstrip(".").split():
                p = parse(word)
                phones.update(p["phones"])
                tones[TONE_NAMES[p["tone"]]] += 1
                pairs[f"{p['nucleus']}|{TONE_NAMES[p['tone']]}"] += 1
        report[split] = {"phonemes": dict(phones.most_common()), "tones": dict(tones.most_common()),
                         "rhyme_tone": dict(pairs.most_common())}
    every_phone = sorted({p for s in report.values() for p in s["phonemes"]})
    every_pair = [f"{v}|{TONE_NAMES[t]}" for v in vowels for t in range(1, 7)]
    train = report.get("train", {})
    for p in every_phone:
        if train.get("phonemes", {}).get(p, 0) < min_count:
            warnings.append(f"train: phoneme {p} only {train.get('phonemes', {}).get(p, 0)} times")
    for t in range(1, 7):
        if train.get("tones", {}).get(TONE_NAMES[t], 0) < min_count:
            warnings.append(f"train: tone {TONE_NAMES[t]} only {train.get('tones', {}).get(TONE_NAMES[t], 0)} times")
    rare_pairs = {pair: train.get("rhyme_tone", {}).get(pair, 0) for pair in every_pair
                  if train.get("rhyme_tone", {}).get(pair, 0) < 20}
    for split in ("validation", "test"):
        missing = [p for p in every_phone if p not in report.get(split, {}).get("phonemes", {})]
        if missing:
            warnings.append(f"{split}: no {missing}")
        missing_tones = [TONE_NAMES[t] for t in range(1, 7) if TONE_NAMES[t] not in report.get(split, {}).get("tones", {})]
        if missing_tones:
            warnings.append(f"{split}: no tone {missing_tones}")
    return {"splits": report, "rare_rhyme_tone_in_train": rare_pairs, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(INFORE), help="InfoRe parquet shards")
    parser.add_argument("--text", default=str(DATA / "infore1_text.jsonl"), help="vibe_voca.datasets.infore_text output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-ms", type=int, default=400, help="silence after the last word of a '.' clip")
    parser.add_argument("--tail-ms", type=int, default=60, help="audio kept after the end of speech (decay)")
    parser.add_argument("--lead-ms", type=int, default=None, help="if set, leading silence is cut to this length")
    parser.add_argument("--period-share", type=float, default=0.5)
    parser.add_argument("--rate-z", type=float, default=4.0, help="robust z-score of the reading rate beyond which a clip is left out")
    parser.add_argument("--min-speech-s", type=float, default=0.6)
    parser.add_argument("--max-seconds", type=float, default=30.0, help="longest clip after shaping (model limit: 32 s)")
    parser.add_argument("--min-count", type=int, default=200, help="coverage: fewest occurrences wanted in train")
    parser.add_argument("--validation-permille", type=int, default=50)
    parser.add_argument("--test-permille", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    left_out = collections.Counter()
    examples = collections.defaultdict(list)

    # Text: normalised, readable by the G2P.
    texts = {}
    for line in open(args.text, encoding="utf-8"):
        t = json.loads(line)
        texts[(t["shard"], t["row"])] = t
    readable = {}
    for key, t in texts.items():
        if t["unreadable"]:
            left_out["word the G2P cannot read"] += 1
            examples["word the G2P cannot read"].append(t["text"])
        else:
            readable[key] = t

    # 1. Speech edges (cached in scan.jsonl).
    scan_path = args.output / "scan.jsonl"
    scans = {}
    if scan_path.exists():
        for line in open(scan_path, encoding="utf-8"):
            s = json.loads(line)
            scans[tuple(s["key"])] = s
    todo = set(readable) - set(scans)
    if todo:
        print(f"scanning {len(todo)} clips for speech edges", flush=True)
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool, open(scan_path, "a", encoding="utf-8") as sink:
            for n, s in enumerate(pool.map(scan_job, rows(args.data, todo), chunksize=8), 1):
                scans[tuple(s["key"])] = s
                sink.write(json.dumps(s, ensure_ascii=False) + "\n")
                if n % 2000 == 0:
                    print(f"  {n} scanned", flush=True)

    # 2. Duration filtering.
    candidates = []
    for key, t in readable.items():
        s = scans[key]
        if "speech_end" not in s:
            left_out["no speech found"] += 1
            continue
        speech = s["speech_end"] - s["speech_start"]
        syllables = len(t["text"].split())
        s["speech_s"], s["rate"] = round(speech, 3), round(syllables / max(speech, 1e-3), 3)
        candidates.append((key, t, s))
    rates = np.array([s["rate"] for _, _, s in candidates])
    median = float(np.median(rates))
    mad = float(np.median(np.abs(rates - median))) * 1.4826
    kept = {split: [] for split in SPLITS}
    for key, t, s in candidates:
        z = (s["rate"] - median) / mad
        longest = (s["speech_start"] if args.lead_ms is None else args.lead_ms / 1000) + s["speech_s"] + args.period_ms / 1000
        reason = ("speech too short" if s["speech_s"] < args.min_speech_s else
                  "clip too long" if longest > args.max_seconds else
                  "too fast for its text (audio short / text extra)" if z > args.rate_z else
                  "too slow for its text (audio long / text missing)" if z < -args.rate_z else None)
        if reason:
            left_out[reason] += 1
            examples[reason].append(f"{key[0]}#{key[1]} {s['rate']} syll/s, {s['speech_s']} s: {t['text']}")
            continue
        split = split_of(t["text"], args.validation_permille, args.test_permille)
        kept[split].append({"key": key, "text": t["text"], "text_raw": t["text_raw"], "scan": s})

    # 3. Half "period", half "open", in every split.
    rng = random.Random(args.seed)
    for split, clips in kept.items():
        want = round(args.period_share * len(clips))
        natural = [c for c in clips if c["scan"]["final_ms"] >= args.period_ms]
        rest = [c for c in clips if c["scan"]["final_ms"] < args.period_ms]
        rng.shuffle(natural)
        rng.shuffle(rest)
        for c in clips:
            c["period"] = False
        for c in (natural + rest)[:want]:
            c["period"] = True

    # 4. Audio.
    audio_dir = args.output / "audio"
    audio_dir.mkdir(exist_ok=True)
    settings = {"tail_ms": args.tail_ms, "lead_ms": args.lead_ms, "period_ms": args.period_ms}
    by_key = {c["key"]: c for clips in kept.values() for c in clips}
    for c in by_key.values():
        ending = "period" if c["period"] else "open"
        c["id"] = f"infore1_{c['key'][0][6:11]}_{c['key'][1]:04d}"
        c["audio"] = f"audio/{c['id']}.{ending}.flac"          # the ending is in the name: never reused stale

    def jobs():
        for key, data in rows(args.data, set(by_key)):
            c = by_key[key]
            yield str(args.output / c["audio"]), data, c["scan"], c["period"], settings
    print(f"writing {len(by_key)} clips", flush=True)
    failed = set()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for n, (target, error) in enumerate(pool.map(write_job, jobs(), chunksize=8), 1):
            if error:
                failed.add(target)
                left_out[error] += 1
            if n % 2000 == 0:
                print(f"  {n} written", flush=True)

    entries = {split: [] for split in SPLITS}
    for split, clips in kept.items():
        for c in clips:
            if str(args.output / c["audio"]) in failed:
                continue
            entries[split].append({
                "id": c["id"], "split": split, "voice_id": VOICE, "text": c["text"] + ("." if c["period"] else ""),
                "text_raw": c["text_raw"], "ending": "period" if c["period"] else "open", "audio": c["audio"],
                "speech_s": c["scan"]["speech_s"], "rate": c["scan"]["rate"], "origin": c["key"][0],
                "source_row": c["key"][1], "source_license": SOURCE_LICENSE, "source_url": SOURCE_URL})
    for split in SPLITS:
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries[split]), encoding="utf-8")

    cover = coverage(entries, args.min_count)
    (args.output / "coverage.json").write_text(json.dumps(cover, ensure_ascii=False, indent=1), encoding="utf-8")
    hours = {s: round(sum(e["speech_s"] for e in entries[s]) / 3600, 2) for s in SPLITS}
    summary = {"kept": {s: len(entries[s]) for s in SPLITS}, "speech_hours": hours,
               "period": {s: sum(e["ending"] == "period" for e in entries[s]) for s in SPLITS},
               "left_out": dict(left_out), "rate": {"median_syll_per_s": round(median, 3), "mad": round(mad, 3),
                                                    "kept_range": [round(median - args.rate_z * mad, 2),
                                                                   round(median + args.rate_z * mad, 2)]},
               "examples_left_out": {k: v[:8] for k, v in examples.items()},
               "coverage_warnings": cover["warnings"], "settings": {k: str(v) for k, v in vars(args).items()}}
    (args.output / "infore.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("kept", "speech_hours", "period", "left_out", "rate", "coverage_warnings")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
