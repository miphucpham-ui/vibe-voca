"""Long-text synthesis shared by the command line (``vibe_voca.infer``) and the web demo (``vibe_voca.web``).

Text of any length is split into sentences at . ? ! … and line breaks (one utterance holds at most 512 tokens;
very long sentences are cut every 60 words). The sentences are spoken in batches (``infer.speak_batch``; sentence
k uses seed + k) and the pieces are joined: every
sentence carries its own <bos> lead-in and final pause + <eos> tail, which would add up to 0.6-0.9 s between
sentences, so a fixed gap (``gap_ms``) of the sentence's own room tone is kept after its last sound and the next
sentence's lead-in is dropped (except its last 2 frames). Optionally, punctuation is restored first
(``vibe_voca.punctuate``, Vietnamese lines only).
"""

from dataclasses import dataclass, field
import math
import re
import time

import numpy as np
import torch

from .infer import speak_batch
from .model import Polish
from .text import FEATURE_NAMES, PHONE_SET, TONE_OFFSET, UNK, _marked

TRAILING = frozenset({"<eos>", "<sp>", ".", "?", "!", "…", ",", ";", ":"})


@dataclass
class Settings:
    speed: float = 1.0
    semitones: float = 0.0          # pitch shift
    pitch_range: float = 1.0        # scale of pitch movement around the utterance mean
    seed: int = 0
    polish: bool = True             # model.Polish (corrections made while speaking)
    texture: float = 0.0            # AR(2) pitch texture (1 = InfoRe recordings)
    comma_ms: float = 0.0           # minimum pause at commas (0 = as predicted)
    stop_ms: float = 0.0            # minimum pause at . ? ! (0 = as predicted)
    gap_ms: float = 300.0           # pause between sentences
    punctuate: bool = False         # restore punctuation first (vibert-capu)

    @property
    def pitch_shift(self):
        return self.semitones * math.log(2) / 12

    def make_polish(self):
        if not self.polish:
            return None
        return Polish(texture=self.texture)


@dataclass
class Sentence:
    text: str
    phones: object                  # vibe_voca.text.Phonemized
    durations: np.ndarray           # frames per token
    hz: np.ndarray                  # predicted F0 per frame (NaN where unvoiced)
    start: float                    # time of frame 0 in the joined audio (seconds; may be < 0 after the lead cut)


@dataclass
class Result:
    audio: np.ndarray
    sample_rate: int
    spoken: str                     # the text actually read (after punctuation restoration)
    sentences: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def seconds(self):
        return len(self.audio) / self.sample_rate


def split_sentences(text, max_words=60):
    out = []
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?…])\s+", line):          # not inside 1.250.000 or 12.5
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if re.search(r"\w", chunk):
                    out.append(chunk)
    return out


_punctuator = []


def punctuate(text):
    """Add . , : ? to Vietnamese lines typed without them. Lines without Vietnamese letters or tone marks (English)
    are left as typed."""
    if not _punctuator:
        from .punctuate import Punctuator
        _punctuator.append(Punctuator())
    lines = text.splitlines()
    vietnamese = [i for i, line in enumerate(lines) if line.strip() and any(_marked(w) for w in line.split())]
    if vietnamese:
        for i, fixed in zip(vietnamese, _punctuator[0]([lines[i] for i in vietnamese])):
            lines[i] = fixed
    return "\n".join(lines)


def syllables(phones):
    """[(label, first token, last token, tone name or None, unknown?)] from the front end's word runs."""
    labels, out, i, symbols = iter(phones.words), [], 0, phones.symbols
    while i < len(symbols):
        if symbols[i] in PHONE_SET or symbols[i] == UNK:
            j = i
            while j + 1 < len(symbols) and symbols[j + 1] in PHONE_SET:
                j += 1
            top = max(phones.weight[i:j + 1])
            label, language = next(labels, ("", "vi"))
            tone = FEATURE_NAMES[top] if top > TONE_OFFSET else ("EN" if language == "en" else None)
            out.append((label or " ".join(symbols[i:j + 1]), i, j, tone, symbols[i] == UNK))
            i = j + 1
        else:
            i += 1
    return out


def tone_spans(phones, durations, first, last, word_tone):
    """[(first frame, end frame, tone)] of the rhymes of tokens first..last (runs of one tone merged)."""
    ends = np.concatenate([[0], np.cumsum(durations)])
    spans = []
    for k in range(first, last + 1):
        w = phones.weight[k]
        name = FEATURE_NAMES[w] if w > TONE_OFFSET else ("EN" if word_tone == "EN" else None)
        if name:
            if spans and spans[-1][1] == int(ends[k]) and spans[-1][2] == name:
                spans[-1] = (spans[-1][0], int(ends[k + 1]), name)
            else:
                spans.append((int(ends[k]), int(ends[k + 1]), name))
    return spans


def synthesize(model, text, voice, settings=None):
    """Any text -> Result (joined float32 audio, peak-limited to 0.99, plus per-sentence phonemes, durations, F0)."""
    settings = settings or Settings()
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    spoken = punctuate(text) if settings.punctuate else text
    rate, hop = model.config.sample_rate, model.config.hop
    polish = settings.make_polish()
    pauses = {"comma": float(settings.comma_ms), "stop": float(settings.stop_ms)}
    sentences = split_sentences(spoken)
    result = Result(np.zeros(1, dtype=np.float32), rate, spoken)
    waves, offset, started = [], 0.0, time.perf_counter()
    with torch.inference_mode():
        spoken_rows = speak_batch(model, sentences, voice, settings.speed, settings.pitch_shift, settings.seed,
                                  settings.pitch_range, polish, pauses=pauses)
        for n, (sentence, (wave, phones, durations, prosody)) in enumerate(zip(sentences, spoken_rows)):
            lead = 0
            if waves:
                lead = max(0, int(durations[0]) - 2) * hop
                wave = wave[lead:]
            if settings.gap_ms > 0 and n < len(sentences) - 1:
                last = len(phones.symbols) - 1
                while last > 0 and phones.symbols[last] in TRAILING:
                    last -= 1
                end = int(durations[: last + 1].sum()) * hop - lead
                wave = wave[: max(end, 0) + int(settings.gap_ms / 1000 * rate)]
            prosody = prosody[: int(durations.sum())]
            hz = np.where(prosody[:, 1] > 0, model.config.pitch_reference_hz * np.exp(prosody[:, 0] + settings.pitch_shift),
                          np.nan)
            result.sentences.append(Sentence(sentence, phones, durations, hz, offset - lead / rate))
            result.unknown += list(phones.unknown)
            waves.append(wave)
            offset += len(wave) / rate
    result.elapsed = time.perf_counter() - started
    if waves:
        audio = np.concatenate(waves)
        peak = float(np.abs(audio).max())
        result.audio = audio * (0.99 / peak) if peak > 0.99 else audio
    result.unknown = list(dict.fromkeys(result.unknown))
    return result
