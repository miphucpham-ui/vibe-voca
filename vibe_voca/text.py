# Adapted from Alpa (77ethers/alpa, alpa/text.py), following tiny-voca's token layout (Muhammed Nazeem);
# Apache-2.0, see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md. Modified for vibe-voca.

"""Vietnamese front end: text -> phoneme tokens with a per-token tone feature.

The same code runs during data preparation, training and inference, so tokens can never drift.

* Vocabulary = tiny-voca's 76 symbols, in tiny-voca's order (ids 0..75: specials, punctuation,
  English phonemes), then the Vietnamese phonemes, written ``vi:<IPA>`` (ids 76..). Keeping the
  English rows lets every Vietnamese row start as a mixture of English rows (SEAM) and keeps the
  door open for English words later; no English row is shared with a Vietnamese phoneme, so
  Vietnamese training never rewrites what an English symbol means.
* Per-token feature (the model's ``syllable_weight`` table, 9 rows): 0 none, 1 English primary
  stress, 2 English secondary stress (tiny-voca's own rows), 3..8 the Vietnamese tones ngang,
  huyền, ngã, hỏi, sắc, nặng. A tone is carried by every phone of the rhyme (nucleus and coda);
  onset and medial glide carry 0.
* Each syllable is read by ``vibe_voca/g2p.py`` (rule-based, Northern by default). A
  syllable it cannot read (abbreviation, digit, foreign word) becomes ``<unk>`` and is reported
  in ``Phonemized.unknown`` so data preparation can leave such recordings out.
* Utterance edges, word boundaries and punctuation are explicit tokens that may last zero frames.
* English is read by tiny-voca's own phonemizer (dizione, data in ``tiny-voca/phonemizer``), unchanged:
  a sentence without Vietnamese letters or tone marks goes to dizione whole (same tokens and stress as
  tiny-voca); inside a Vietnamese sentence, words that are not Vietnamese syllables (and acronyms) go to
  dizione, digits are read as Vietnamese numbers. Pure Vietnamese text gives the same tokens as before.
"""

from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from .g2p import TONE_NAMES, parse
from .numbers import expand_numbers
from .paths import TINY_VOCA

FRONTEND_VERSION = 1
DIALECT = os.environ.get("VIBE_VOCA_DIALECT", "north")
# Word segmentation (pyvi): syllables of one Vietnamese word are joined without <sp>, as phonemes inside an English
# word are. A model records which mode it was trained with (frontend fingerprint), so inference always follows the
# checkpoint; this setting only decides the mode of data preparation and training. On by default (the released
# models use it); VIBE_VOCA_WORDS=0 gives one <sp> between all syllables (the first models, runs v1-v2c).
WORDS = os.environ.get("VIBE_VOCA_WORDS", "1") == "1"

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
# tiny-voca's symbols 5..75, in its order (n4ze3m/tiny-voca config.json "vocab").
ENGLISH_PUNCTUATION = (",", ".", "?", "!", ";", ":", "\"", "'", "(", ")", "—", "…")
ENGLISH_PHONES = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "ɹ",
                  "j", "w", "tʃ", "dʒ", "ɾ", "ʔ", "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː",
                  "ʊ", "uː", "ɜː", "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː", "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ",
                  "aɪɚ", "aɪə", "əl", "n̩")
ENGLISH_VOWELS = ENGLISH_PHONES[ENGLISH_PHONES.index("ə"):]

# Vietnamese phonemes (vi_g2p's IPA). Append-only: changing the order invalidates checkpoints.
VI_IPA = ("ɓ", "m", "f", "v", "t", "tʰ", "ɗ", "n", "z", "s", "c", "ɲ", "l", "k", "x", "ŋ", "ɣ", "h", "ʔ", "p",
          "ʈ", "ʂ", "ʐ", "j", "w",
          "i", "e", "ɛ", "a", "aː", "ə", "əː", "ɨ", "u", "o", "ɔ", "iə", "ɨə", "uə")
VI_VOWEL_IPA = VI_IPA[VI_IPA.index("i"):]


def vi(ipa):
    return "vi:" + ipa


VI_PHONES = tuple(vi(p) for p in VI_IPA)
VI_VOWELS = tuple(vi(p) for p in VI_VOWEL_IPA)
PHONES = ENGLISH_PHONES + VI_PHONES
VOWELS = ENGLISH_VOWELS + VI_VOWELS
PUNCTUATION = ENGLISH_PUNCTUATION
SYMBOLS = SPECIALS + ENGLISH_PUNCTUATION + ENGLISH_PHONES + VI_PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOWEL_SET = frozenset(VOWELS)
VI_PHONE_SET = frozenset(VI_PHONES)

# Sound groups (used by the aligner's QC, the annotation probe and model.Polish).
VOICELESS = frozenset(vi(p) for p in ("p", "t", "tʰ", "k", "c", "ʈ", "f", "s", "ʂ", "x", "h")) | \
    frozenset(("p", "t", "k", "tʃ", "f", "θ", "s", "ʃ", "h"))
FRICATIVES = frozenset(vi(p) for p in ("f", "s", "ʂ", "x")) | frozenset(("f", "θ", "s", "ʃ"))
# model.Polish: open = pulses up to its open limit; stops = pulses below the consonant limit (voiced
# fricatives too: voicing below, frication above); hiss = noise only; breath = h; mark = none in Vietnamese.
OPEN = VOWELS + tuple(vi(p) for p in ("m", "n", "ŋ", "ɲ", "l", "j", "w")) + ("m", "n", "ŋ", "l", "ɹ", "j", "w")
STOPS = tuple(vi(p) for p in ("ɓ", "ɗ", "p", "t", "tʰ", "k", "c", "ʈ", "ʔ", "v", "z", "ʐ", "ɣ")) + \
    ("p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "ɾ", "ʔ", "v", "ð", "z", "ʒ")
HISS = tuple(vi(p) for p in ("f", "s", "ʂ", "x")) + ("f", "θ", "s", "ʃ")
BREATH = (vi("h"), "h")
MARK = ()

# Per-token feature: 0 none, 1-2 English stress, 3-8 Vietnamese tones (vi_g2p tone 1..6 + 2).
FEATURE_NAMES = ("none", "primary", "secondary") + tuple(TONE_NAMES[t] for t in range(1, 7))
TONE_OFFSET = 2
FEATURE_LEVELS = len(FEATURE_NAMES)

_MARKS_TO_TOKEN = {",": ",", ";": ",", ":": ",", "…": ",", "–": ",", "—": ",", "-": ",",
                   ".": ".", "?": "?", "!": "!"}
_PIECE = re.compile(r"\s+|([" + re.escape("".join(_MARKS_TO_TOKEN)) + r"])")
_QUOTES = dict.fromkeys(map(ord, "'\"`´‘’‚‛“”„‟«»‹›()[]{}"))


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    weight: tuple       # per-token feature (the model's syllable_weight input): see FEATURE_NAMES
    unknown: tuple      # words the G2P cannot read, given <unk>
    words: tuple = ()   # (label, "vi" | "en") for each run of phone tokens, in order (for display)

    def __len__(self):
        return len(self.ids)


def frontend_identity(words=None):
    """What decides the tokens: the rules version, the dialect, the symbol inventory (and word segmentation)."""
    # "vivoca" is the front end's identity inside the checkpoint fingerprint (the project's old name): keep it.
    identity = {"frontend": "vivoca", "frontend_version": FRONTEND_VERSION, "dialect": DIALECT,
                "symbols": list(SYMBOLS), "features": list(FEATURE_NAMES)}
    if WORDS if words is None else words:
        identity["word_segmentation"] = "pyvi 0.1.1"
    return identity


def frontend_fingerprint(words=None):
    return hashlib.sha256(json.dumps(frontend_identity(words), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frontend_words(fingerprint):
    """The word-segmentation mode a checkpoint was trained with, from its frontend fingerprint."""
    for words in (False, True):
        if fingerprint == frontend_fingerprint(words):
            return words
    raise ValueError("checkpoint was trained with a different phoneme inventory/front end")


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).lower().translate(_QUOTES)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)            # a joining hyphen separates syllables
    return re.sub(r"\s+", " ", text).strip()


def word_phones(word):
    """One written syllable -> (vibe-voca symbols, per-token feature), or None."""
    parsed = parse(word, DIALECT)
    if parsed is None:
        return None
    rhyme = 1 + (parsed["medial"] is not None)
    tone = parsed["tone"] + TONE_OFFSET
    return [vi(p) for p in parsed["phones"]], [0] * rhyme + [tone] * (len(parsed["phones"]) - rhyme)


# ---------------------------------------------------------------- English: tiny-voca's own phonemizer, unchanged
_DIZIONE_DATA = Path(os.environ.get("DIZIONE_DATA", TINY_VOCA / "phonemizer"))
_VI_MARKS = frozenset("̛̣̀́̃̉̂̆")     # tones, circumflex, breve, horn
_LETTERS = re.compile(r"[^\W\d_]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\s*\n\s*")
# English function words that are also Vietnamese syllables: English only between English words ("love the way").
_EN_GLUE = frozenset("a an the to in on at of and or is it be me my we he so no do go up by as".split())


@functools.lru_cache(maxsize=1)
def _dizione():
    os.environ.setdefault("DIZIONE_DATA", str(_DIZIONE_DATA))
    try:
        import dizione
    except ImportError as error:
        raise ImportError("English text needs tiny-voca's phonemizer: pip install "
                          "'git+https://github.com/n4ze3m/dizione@8cd889e4de7be6bd1dd2e6891344a0729fa14dd4'") from error
    return dizione


@functools.lru_cache(maxsize=1)
def _english_words():
    with open(_DIZIONE_DATA / "lexicon.tsv", encoding="utf-8") as lines:
        return frozenset(line.split("\t", 1)[0] for line in lines)


def english_identity():
    """The dizione build (its own fingerprint): record it next to anything trained on English tokens."""
    return _dizione().identity()["fingerprint"]


def _marked(word):
    """Written with Vietnamese letters or tone marks (đ, â, ơ, ư, ă, any tone)."""
    return "đ" in word.lower() or any(ch in _VI_MARKS for ch in unicodedata.normalize("NFD", word))


def _sentence_language(sentence):
    """'vi', 'en', or None (nothing but digits and marks). A sentence with any Vietnamese-marked word is
    Vietnamese (English words inside it are read as English); an unmarked one is English unless most of its
    words are Vietnamese syllables that are not English words ("xin chao ban")."""
    words = _LETTERS.findall(sentence)
    if not words:
        return None
    if any(_marked(w) for w in words):
        return "vi"
    english = sum(w.lower() in _english_words() for w in words)
    vietnamese = sum(w.lower() not in _english_words() and parse(w.lower(), DIALECT) is not None for w in words)
    return "vi" if vietnamese > english else "en"


def _groups(text):
    """Runs of consecutive sentences in one language: [(language, text)]."""
    sentences = [s for s in _SENTENCE_BREAK.split(unicodedata.normalize("NFC", text)) if s and s.strip()]
    languages = [_sentence_language(s) for s in sentences]
    known = [lang for lang in languages if lang] or ["vi"]
    previous, groups = known[0], []
    for sentence, language in zip(sentences, languages):
        language = language or previous
        if groups and groups[-1][0] == language:
            groups[-1][1].append(sentence)
        else:
            groups.append((language, [sentence]))
        previous = language
    return [(language, " ".join(parts)) for language, parts in groups]


def _word_language(word):
    if _marked(word):
        return "vi"
    if not word.isascii():
        return "vi"                                    # not Latin letters: the Vietnamese reader marks it <unk>
    if len(word) >= 2 and word.isupper() and word.isalpha():
        return "en"                                    # acronym: MIT, FBI
    return "vi" if parse(word.lower(), DIALECT) is not None else "en"


def _english_tokens(text):
    """dizione's tokens for ``text`` without <bos>/<eos>: [(symbol, stress)], unknown words."""
    try:
        read = _dizione().phonemize(text)
    except ValueError:                                 # nothing dizione can say (a lone "/", "&"): skip it
        return [], []
    inner = list(zip(read.symbols, read.stress))
    if inner and inner[0][0] == BOS:
        inner = inner[1:]
    if inner and inner[-1][0] == EOS:
        inner = inner[:-1]
    return inner, list(read.unknown)


class _Builder:
    def __init__(self):
        self.symbols, self.weight, self.unknown, self.words = [BOS], [0], [], []

    def mark(self, mark):
        if self.symbols[-1] == SP:
            self.symbols.pop()
            self.weight.pop()
        if self.symbols[-1] != mark and self.symbols[-1] != BOS:     # collapse repeats such as "!!"
            self.symbols.append(mark)
            self.weight.append(0)

    def space(self):
        if self.symbols[-1] in PHONE_SET or self.symbols[-1] == UNK:
            self.symbols.append(SP)
            self.weight.append(0)

    def vietnamese(self, word, join=False):
        if not join:
            self.space()
        read = word_phones(word)
        if read is None:
            self.unknown.append(word)
            self.symbols.append(UNK)
            self.weight.append(0)
        else:
            self.symbols.extend(read[0])
            self.weight.extend(read[1])
        if join and self.words:
            self.words[-1] = (self.words[-1][0] + " " + word, "vi")
        else:
            self.words.append((word, "vi"))

    def english(self, text):
        tokens, unknown = _english_tokens(text)
        self.unknown.extend(unknown)
        if tokens and (tokens[0][0] in PHONE_SET or tokens[0][0] == UNK):
            self.space()
        runs, inside = 0, False
        for symbol, stress in tokens:
            if symbol in _MARKS_TO_TOKEN or symbol in PUNCTUATION:
                self.mark(symbol)
            else:
                self.symbols.append(symbol)
                self.weight.append(int(stress))
            phone = symbol in PHONE_SET or symbol == UNK
            runs, inside = runs + (phone and not inside), phone
        labels = [w for w in _LETTERS.findall(text)]
        labels = labels if len(labels) == runs else [""] * runs
        self.words.extend((label, "en") for label in labels)


@functools.lru_cache(maxsize=1)
def _segmenter():
    from pyvi import ViTokenizer
    return ViTokenizer


def _joins(words):
    """For a run of Vietnamese syllables: True where a syllable continues the previous one's word (pyvi)."""
    joins = []
    for token in _segmenter().tokenize(" ".join(words)).split():
        joins += [False] + [True] * token.count("_")
    return joins if len(joins) == len(words) else [False] * len(words)


def _vietnamese_group(builder, text, segment=False):
    pieces = [p for p in _PIECE.split(_clean(expand_numbers(text))) if p]
    items = [("mark", p) if p in _MARKS_TO_TOKEN else ["word", p, _word_language(p)] for p in pieces]
    for _ in range(3):                                 # glue words between English words become English
        for i, item in enumerate(items):
            if item[0] == "word" and item[2] == "vi" and item[1].lower() in _EN_GLUE and not _marked(item[1]):
                near = [items[j] for j in (i - 1, i + 1) if 0 <= j < len(items)]
                words = [n for n in near if n[0] == "word"]
                if words and len(words) == len(near) and all(n[2] == "en" for n in words):
                    item[2] = "en"
    join = {}
    if segment:                                        # runs of Vietnamese syllables between marks / English words
        run = []
        for i, item in enumerate(items + [("end", None)]):
            if item[0] == "word" and item[2] == "vi":
                run.append(i)
                continue
            for index, joined in zip(run, _joins([items[j][1] for j in run]) if len(run) > 1 else []):
                join[index] = joined
            run = []
    english = []
    for index, item in enumerate(items + [("end", None)]):
        if item[0] == "word" and item[2] == "en":
            english.append(item[1])
            continue
        if english:
            builder.english(" ".join(english))
            english = []
        if item[0] == "mark":
            builder.mark(_MARKS_TO_TOKEN[item[1]])
        elif item[0] == "word":
            builder.vietnamese(item[1].lower(), join.get(index, False))


def _clean(text):
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)            # a joining hyphen separates syllables
    return re.sub(r"\s+", " ", text).strip()


def phonemize(text, words=None):
    """Text -> tokens. Vietnamese sentences go through vi_g2p (digits read as Vietnamese numbers; English words
    inside them through dizione); English sentences go through dizione whole, exactly as tiny-voca reads them."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    builder = _Builder()
    for language, chunk in _groups(text):
        if language == "en":
            builder.english(chunk)
        else:
            _vietnamese_group(builder, chunk, WORDS if words is None else words)
    symbols, weight = builder.symbols + [EOS], builder.weight + [0]
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise ValueError("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(weight), tuple(builder.unknown),
                      tuple(builder.words))
