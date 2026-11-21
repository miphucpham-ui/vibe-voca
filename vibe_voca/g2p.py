"""Vietnamese syllable -> phonemes + tone, and the bridge from each Vietnamese phoneme to tiny-voca's
English phoneme rows (the hand prior that SEAM refines with spectral statistics).

A syllable is (C1)(w)V(C2) + tone:

* C1 onset; an empty onset is the glottal stop ʔ (tiny-voca has a ʔ row).
* w  the medial glide written o (hoa, hoe) or u (huy, tuần, thuở, khuya, qu-).
* V  one nucleus token, the three diphthongs iê/ia, ươ/ưa, uô/ua included (iə ɨə uə).
* C2 coda: p t k m n ŋ, ɲ (-nh), c (-ch), and the offglides j (-i/-y) and w (-o/-u).
* tone: 1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng (0 = no tone, for onsets and medials).

Dialects:
* "north" (Hà Nội): d, gi, r -> z; tr = ch -> c; s = x -> s.
* "spelling": keeps the contrasts written in the spelling that many speakers still make:
  tr -> ʈ, s -> ʂ, r -> ʐ (d, gi -> z).

Usage:
    python -m vibe_voca.g2p --dialect north --out lexicon      # lexicon of every InfoRe syllable
"""

import argparse
import collections
import glob
import json
import unicodedata
from pathlib import Path

TONE_MARKS = {"̀": 2, "̃": 3, "̉": 4, "́": 5, "̣": 6}
TONE_NAMES = {0: "none", 1: "ngang", 2: "huyền", 3: "ngã", 4: "hỏi", 5: "sắc", 6: "nặng"}
VOWEL_LETTERS = set("aăâeêioôơuưy")

ONSETS = {
    "ngh": "ŋ", "ng": "ŋ", "nh": "ɲ", "ch": "c", "tr": "c", "th": "tʰ", "ph": "f", "kh": "x", "gh": "ɣ",
    "gi": "z", "qu": "k", "b": "ɓ", "c": "k", "k": "k", "d": "z", "đ": "ɗ", "g": "ɣ", "h": "h", "l": "l",
    "m": "m", "n": "n", "p": "p", "r": "z", "s": "s", "t": "t", "v": "v", "x": "s",
}
SPELLING_ONSETS = {"tr": "ʈ", "s": "ʂ", "r": "ʐ"}
NUCLEI = {
    "iê": "iə", "yê": "iə", "ia": "iə", "ya": "iə", "ươ": "ɨə", "ưa": "ɨə", "uô": "uə", "ua": "uə",
    "oo": "ɔ", "ôô": "o",
    "a": "aː", "ă": "a", "â": "ə", "e": "ɛ", "ê": "e", "i": "i", "y": "i", "o": "ɔ", "ô": "o", "ơ": "əː",
    "u": "u", "ư": "ɨ",
}
CODAS = {"ch": "c", "nh": "ɲ", "ng": "ŋ", "c": "k", "m": "m", "n": "n", "p": "p", "t": "t",
         "i": "j", "y": "j", "o": "w", "u": "w"}
OPEN_ONLY = {"ia", "ya", "ua", "ưa"}            # diphthongs written this way take no coda


def split_tone(syllable):
    tone, letters = 1, []
    for ch in unicodedata.normalize("NFD", syllable.lower()):
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
        else:
            letters.append(ch)
    return unicodedata.normalize("NFC", "".join(letters)), tone


def parse(syllable, dialect="north"):
    """{"onset", "medial", "nucleus", "coda", "tone", "phones"} or None when the syllable is not
    Vietnamese (an abbreviation such as "hlv", a foreign word, a typo)."""
    base, tone = split_tone(syllable)
    onsets = dict(ONSETS, **SPELLING_ONSETS) if dialect == "spelling" else ONSETS
    onset, rest, medial = "ʔ", base, None
    for spelling in sorted(onsets, key=len, reverse=True):
        if base.startswith(spelling):
            onset, rest = onsets[spelling], base[len(spelling):]
            if spelling == "gi" and (not rest or rest[0] not in VOWEL_LETTERS or rest[0] == "ê"):
                rest = "i" + rest                  # gì = z i, gìn = z i n, giếng = z iə ŋ
            if spelling == "qu":
                medial = "w"
            break
    if not rest or rest[0] not in VOWEL_LETTERS:
        return None
    if medial is None:
        if rest[0] == "o" and len(rest) > 1 and rest[1] in "aăe":
            medial, rest = "w", rest[1:]
        elif rest[0] == "u" and len(rest) > 1 and rest[1] in "yêâơ":
            medial, rest = "w", rest[1:]
    nucleus = coda = None
    for spelling in sorted(NUCLEI, key=len, reverse=True):
        if rest.startswith(spelling):
            nucleus, tail = spelling, rest[len(spelling):]
            break
    if nucleus is None:
        return None
    if tail:
        if nucleus in OPEN_ONLY or tail not in CODAS:
            return None
        coda = CODAS[tail]
    if coda in ("p", "t", "k", "c") and tone not in (5, 6):
        return None                                # a checked syllable takes only sắc or nặng
    vowel = NUCLEI[nucleus]
    if nucleus == "a" and (tail in ("u", "y") or tail in ("nh", "ch")):
        vowel = "a"                                # au, ay, anh, ach have the short a of ă
    phones = [onset] + (["w"] if medial else []) + [vowel] + ([coda] if coda else [])
    return {"onset": onset, "medial": medial, "nucleus": vowel, "coda": coda, "tone": tone, "phones": phones}


def tones_for(parsed):
    """Tone feature per phone: on the rhyme (nucleus and coda), none on onset and medial."""
    rhyme_start = 1 + (parsed["medial"] is not None)
    return [0] * rhyme_start + [parsed["tone"]] * (len(parsed["phones"]) - rhyme_start)


# Vietnamese phoneme -> tiny-voca English rows with weights (hand prior; SEAM replaces/refines the weights).
# Every key on the right is one of tiny-voca's 76 symbols. Two Vietnamese phonemes never share the
# same mixture, so no two rows start identical.
TO_ENGLISH = {
    # onsets / codas
    "ɓ": {"b": 1.0}, "ɗ": {"d": 1.0}, "p": {"p": 1.0}, "t": {"t": 1.0}, "tʰ": {"t": 0.75, "h": 0.25},
    "k": {"k": 1.0}, "ʔ": {"ʔ": 1.0}, "c": {"tʃ": 0.6, "j": 0.4}, "ʈ": {"tʃ": 0.6, "t": 0.1, "ɹ": 0.3},
    "m": {"m": 1.0}, "n": {"n": 1.0}, "ŋ": {"ŋ": 1.0}, "ɲ": {"n": 0.6, "j": 0.4},
    "f": {"f": 1.0}, "v": {"v": 1.0}, "s": {"s": 1.0}, "z": {"z": 1.0}, "h": {"h": 1.0}, "l": {"l": 1.0},
    "x": {"k": 0.5, "h": 0.5}, "ɣ": {"ɡ": 0.7, "ð": 0.3}, "ʂ": {"ʃ": 0.7, "ɹ": 0.3}, "ʐ": {"ʒ": 0.7, "ɹ": 0.3},
    "j": {"j": 1.0}, "w": {"w": 1.0},
    # vowels
    "i": {"iː": 1.0}, "e": {"eɪ": 0.6, "ɛ": 0.4}, "ɛ": {"ɛ": 0.8, "æ": 0.2},
    "aː": {"ɑː": 0.6, "æ": 0.4}, "a": {"ʌ": 0.6, "æ": 0.4},
    "ə": {"ʌ": 0.5, "ə": 0.5}, "əː": {"ɜː": 0.8, "ə": 0.2},
    "ɨ": {"ᵻ": 0.4, "uː": 0.4, "ɪ": 0.2}, "u": {"uː": 1.0}, "o": {"oː": 1.0}, "ɔ": {"ɔː": 0.7, "ɔ": 0.3},
    "iə": {"iə": 1.0}, "ɨə": {"ᵻ": 0.35, "uː": 0.15, "ə": 0.5}, "uə": {"uː": 0.5, "ə": 0.5},
}
# Tone -> tiny-voca stress rows (0 none, 1 primary, 2 secondary). A starting point only: the three
# tones English has nothing like (hỏi, ngã, nặng) need their own share added before training.
TONE_TO_STRESS = {0: {0: 1.0}, 1: {0: 1.0}, 2: {0: 0.7, 2: 0.3}, 3: {1: 0.6, 2: 0.4},
                  4: {2: 0.7, 0: 0.3}, 5: {1: 1.0}, 6: {2: 0.5, 0: 0.5}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from .paths import INFORE, TINY_VOCA
    parser.add_argument("--data", default=str(INFORE), help="InfoRe parquet shards")
    parser.add_argument("--dialect", choices=("north", "spelling"), default="north")
    parser.add_argument("--out", default="lexicon")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    counts = collections.Counter()
    for file in sorted(glob.glob(f"{args.data}/*.parquet")):
        for text in pq.read_table(file, columns=["transcription"]).column(0).to_pylist():
            counts.update(unicodedata.normalize("NFC", text).split())

    english = json.loads((TINY_VOCA / "config.json").read_text())["vocab"]
    bad = sorted({s for mix in TO_ENGLISH.values() for s in mix} - set(english))
    if bad:
        raise SystemExit(f"not tiny-voca symbols: {bad}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phones, tones, failed = collections.Counter(), collections.Counter(), collections.Counter()
    with open(out / f"vi_lexicon_{args.dialect}.tsv", "w", encoding="utf-8") as lexicon:
        for syllable, n in sorted(counts.items()):
            parsed = parse(syllable, args.dialect)
            if parsed is None:
                failed[syllable] = n
                continue
            lexicon.write(f"{syllable}\t{' '.join(parsed['phones'])}\t{parsed['tone']}\n")
            for phone in parsed["phones"]:
                phones[phone] += n
            tones[TONE_NAMES[parsed["tone"]]] += n
    missing = sorted(set(phones) - set(TO_ENGLISH))
    (out / f"vi_failed_{args.dialect}.tsv").write_text(
        "".join(f"{s}\t{n}\n" for s, n in failed.most_common()), encoding="utf-8")
    (out / f"vi_phone_counts_{args.dialect}.tsv").write_text(
        "".join(f"{p}\t{n}\t{json.dumps(TO_ENGLISH.get(p), ensure_ascii=False)}\n" for p, n in phones.most_common()),
        encoding="utf-8")
    (out / "vi_to_en.json").write_text(json.dumps({"phonemes": TO_ENGLISH, "tone_to_stress": TONE_TO_STRESS,
                                                   "tones": TONE_NAMES}, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(counts.values())
    print(f"syllables: {len(counts)} types, {total} tokens; parsed {len(counts) - len(failed)} types; "
          f"failed {len(failed)} types / {sum(failed.values())} tokens ({sum(failed.values()) / total:.2%})")
    print("failed:", failed.most_common(40))
    print("phonemes:", len(phones), phones.most_common())
    print("tones:", tones.most_common())
    if missing:
        print("NO ENGLISH MAPPING:", missing)


if __name__ == "__main__":
    main()
