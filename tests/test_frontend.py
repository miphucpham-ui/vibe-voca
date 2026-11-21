"""Front-end regression tests. The fingerprints below are recorded in every trained checkpoint: if one of these
tests fails, existing checkpoints will refuse to load (or, worse, read text differently from how they were trained)."""

import pytest

from vibe_voca import text
from vibe_voca.g2p import parse
from vibe_voca.numbers import expand_numbers
from vibe_voca.paths import TINY_VOCA

FINGERPRINT_SYLLABLES = "34442037417ee13297e51480409fc75e1a70095df723423797a6053c614581fc"
FINGERPRINT_WORDS = "e692918b108d5463e276fcaaff3a863aefcb186af9e47650b4e492946ddcc4a6"
needs_tiny_voca = pytest.mark.skipif(not (TINY_VOCA / "phonemizer" / "lexicon.tsv").exists(),
                                     reason="tiny-voca snapshot not downloaded (scripts/setup.sh)")


def test_fingerprints_unchanged():
    assert text.frontend_fingerprint(False) == FINGERPRINT_SYLLABLES
    assert text.frontend_fingerprint(True) == FINGERPRINT_WORDS
    assert text.frontend_words(FINGERPRINT_WORDS) is True
    assert text.frontend_words(FINGERPRINT_SYLLABLES) is False


def test_vocabulary_layout():
    assert len(text.SYMBOLS) == 115
    assert text.SYMBOLS[:5] == ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>")
    assert text.SYMBOLS[76] == "vi:ɓ"
    assert len(text.FEATURE_NAMES) == 9


def test_g2p():
    assert parse("việt")["phones"] == ["v", "iə", "t"]
    assert parse("việt")["tone"] == 6
    assert parse("quý")["medial"] is not None
    assert parse("xyz") is None


def test_numbers():
    assert expand_numbers("2024") == "hai nghìn không trăm hai mươi tư"
    assert expand_numbers("21") == "hai mươi mốt"
    assert expand_numbers("12,5") == "mười hai phẩy năm"


@needs_tiny_voca
def test_vietnamese_sentence():
    phones = text.phonemize("xin chào, tôi là trợ lý ảo.", words=False)
    assert phones.symbols[:5] == ("<bos>", "vi:s", "vi:i", "vi:n", "<sp>")
    assert "," in phones.symbols and phones.symbols[-2:] == (".", "<eos>")
    assert not phones.unknown
    assert len(phones.ids) == len(phones.weight)


@needs_tiny_voca
def test_mixed_and_english():
    mixed = text.phonemize("Tôi học Machine Learning ở MIT.", words=True)
    assert [lang for _, lang in mixed.words].count("en") >= 2
    english = text.phonemize("The weather is nice today.")
    assert not any(s.startswith("vi:") for s in english.symbols)
