"""Synthesis tests on a real checkpoint. Set VIBE_VOCA_CHECKPOINT to an inference export or training checkpoint
(e.g. the Hub model: ``hf download PhucMinh111/vibe-voca vibe-voca.pt``); skipped otherwise."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vibe_voca.infer import load_model, speak, speak_batch
from vibe_voca.model import Polish
from vibe_voca.synthesis import Settings, split_sentences, synthesize

CHECKPOINT = os.environ.get("VIBE_VOCA_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT or not Path(CHECKPOINT).exists(), reason="VIBE_VOCA_CHECKPOINT not set")
SENTENCES = ["Xin chào, tôi là trợ lý ảo tiếng Việt.", "Năm 2024 có 366 ngày.",
             "Hôm nay trời đẹp, chúng ta cùng nhau đi dạo trong công viên gần nhà."]


@pytest.fixture(scope="module")
def model():
    return load_model(CHECKPOINT, "cuda" if torch.cuda.is_available() else "cpu")


def test_batch_matches_single_utterances(model):
    """Sentence k of a batch sounds as speak(..., seed=seed + k), whatever the batch size and padding."""
    voice, polish, pauses = model.config.speakers[0], Polish(texture=0.5), {"comma": 200, "stop": 350}
    rows = speak_batch(model, SENTENCES, voice, seed=7, polish=polish, pauses=pauses, batch_size=2)
    for k, (wave, phones, durations, _) in enumerate(rows):
        single, _, details = speak(model, SENTENCES[k], voice, seed=7 + k, polish=polish, pauses=pauses)
        assert (durations == details["durations"][0].cpu().numpy()).all()
        assert wave.shape == tuple(single.shape)
        assert np.abs(wave - single.float().numpy()).max() < 1e-2


def test_synthesize_long_text(model):
    text = " ".join(SENTENCES)
    result = synthesize(model, text, model.config.speakers[0], Settings(texture=0.5))
    assert len(result.sentences) == len(split_sentences(text)) == 3
    assert result.audio.dtype == np.float32 and np.isfinite(result.audio).all()
    assert 3 < result.seconds < 20 and np.abs(result.audio).max() <= 0.99 + 1e-6
