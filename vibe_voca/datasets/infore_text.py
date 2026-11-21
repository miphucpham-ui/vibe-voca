"""Normalise InfoRe transcripts: joined words, typos and tone-mark placement.

Rules (each one checked on the InfoRe transcripts, see the report this script prints):

* joined words: a find-and-replace error in the source turned "...a em" into "...anh emm"
  ("củanh emm" = "của em", "nữanh emm" = "nữa em", "xưanh emm" = "xưa em"); a lone "emm" is "em";
  "đigiúp" is "đi giúp";
* typos: "môt" -> "một", "qúy" -> "quý";
* tone-mark placement: one convention for oa/oe/uy without a final consonant. InfoRe mostly writes
  the mark on the first vowel ("hòa" 145x, "hoà" 15x), so "hoà, khoẻ, thuỷ, tuỳ, uỷ" become
  "hòa, khỏe, thủy, tùy, ủy". The pronunciation does not change (the G2P reads both), only the text.

    python -m vibe_voca.datasets.infore_text --out $VIBE_VOCA_DATA/infore1_text.jsonl
"""

import argparse
import collections
import glob
import json
import re
import unicodedata
from pathlib import Path

from ..g2p import TONE_MARKS, parse
from ..paths import INFORE

MARK_OF = {tone: mark for mark, tone in TONE_MARKS.items()}
JOINED = [(re.compile(r"\b(\w+?a)nh emm\b"), r"\1 em", "…anh emm -> …a em"),
          (re.compile(r"\bemm\b"), "em", "emm -> em"),
          (re.compile(r"\bđigiúp\b"), "đi giúp", "đigiúp -> đi giúp")]
TYPOS = {"môt": "một", "qúy": "quý"}
GLIDE_RHYME = re.compile(r"^(.*?)([ou])([aăey])$")      # oa, oe, uy (and oă) with nothing after


def split_tone(word):
    tone, letters = 1, []
    for ch in unicodedata.normalize("NFD", word):
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
        else:
            letters.append(ch)
    return unicodedata.normalize("NFC", "".join(letters)), tone


def place_tone(word):
    """Write the tone of an open oa/oe/uy syllable on its first vowel ("hoà" -> "hòa")."""
    base, tone = split_tone(word)
    if tone == 1:
        return word
    found = GLIDE_RHYME.match(base)
    if not found or found.group(1).endswith("q"):          # qu-: the u belongs to the onset (quý, quỳ)
        return word
    onset, first, second = found.groups()
    marked = unicodedata.normalize("NFC", first + MARK_OF[tone])
    return onset + marked + second


def normalize(text, counts=None):
    counts = counts if counts is not None else collections.Counter()
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()
    for pattern, replacement, name in JOINED:
        text, n = pattern.subn(replacement, text)
        counts[name] += n
    words = []
    for word in text.split():
        if word in TYPOS:
            counts[f"{word} -> {TYPOS[word]}"] += 1
            word = TYPOS[word]
        placed = place_tone(word)
        if placed != word:
            counts[f"tone mark: {word} -> {placed}"] += 1
        words.append(placed)
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(INFORE), help="InfoRe parquet shards")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    counts, changed, unreadable, total = collections.Counter(), 0, collections.Counter(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as sink:
        for file in sorted(glob.glob(f"{args.data}/*.parquet")):
            table = pq.read_table(file, columns=["transcription"]).column(0).to_pylist()
            for index, raw in enumerate(table):
                text = normalize(raw, counts)
                total += 1
                changed += text != re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw).lower()).strip()
                bad = [w for w in text.split() if parse(w) is None]
                unreadable.update(bad)
                sink.write(json.dumps({"shard": Path(file).name, "row": index, "text_raw": raw, "text": text,
                                       "unreadable": bad}, ensure_ascii=False) + "\n")
    print(f"{total} transcripts, {changed} changed -> {args.out}")
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n:
            print(f"  {n:4d}  {name}")
    print(f"still unreadable by the G2P: {len(unreadable)} words, {sum(unreadable.values())} times, in "
          f"{sum(1 for line in open(args.out, encoding='utf-8') if json.loads(line)['unreadable'])} clips")
    print("  ", unreadable.most_common(30))


if __name__ == "__main__":
    main()
