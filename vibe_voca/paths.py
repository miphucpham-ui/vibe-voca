"""Where vibe-voca looks for things outside the repository. Every location can be set with an environment variable;
the defaults are folders inside the repository (all git-ignored).

    VIBE_VOCA_TINY_VOCA   snapshot of n4ze3m/tiny-voca (tiny-voca.pt, phonemizer/)   default: <repo>/tiny-voca
    VIBE_VOCA_MODELS      helper models (PhoWhisper CT2, vibert-capu)                 default: <repo>/models
    VIBE_VOCA_DATA        source datasets and archives                                default: <repo>/data
    VIBE_VOCA_WORK        features, annotations, 24 kHz audio (large, many files)     default: <repo>/work

On Colab keep VIBE_VOCA_WORK on local disk (/content/...): Drive is slow for many small files.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _dir(variable, default):
    return Path(os.environ.get(variable, REPO / default)).expanduser()


TINY_VOCA = _dir("VIBE_VOCA_TINY_VOCA", "tiny-voca")
MODELS = _dir("VIBE_VOCA_MODELS", "models")
DATA = _dir("VIBE_VOCA_DATA", "data")
WORK = _dir("VIBE_VOCA_WORK", "work")

PHOWHISPER = MODELS / "phowhisper-large-ct2"          # vinai/PhoWhisper-large, CTranslate2 float16
WHISPER_EN = MODELS / "whisper-small-en-ct2"          # openai/whisper-small.en, CTranslate2 float16
VIBERT_CAPU = MODELS / "vibert-capu"                  # dragonSwing/vibert-capu (punctuation restoration)
INFORE = DATA / "infore1_25hours" / "data"            # doof-ferb/infore1_25hours parquet shards
