# Adapted from tiny-voca (Muhammed Nazeem, commit 28e0722) src/tiny_voca/infer.py, via Alpa (77ethers/alpa); Apache-2.0,
# see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md. Modified for vibe-voca.

"""Text -> speech from a vibe-voca checkpoint.

    python -m vibe_voca.infer --checkpoint runs/v2f/step-00032000.pt --voice vi_female \\
        --text "Xin chào, tôi là trợ lý ảo tiếng Việt." --output out.wav
"""

import argparse
from pathlib import Path

import torch

from .config import CHECKPOINT_FORMAT, Config
from .model import VibeVoca, Polish
from .text import frontend_words, phonemize


def load_model(path, device="cpu"):
    """A model from an inference export (``inference.pt``) or a training checkpoint (``last.pt``, ``step-*.pt``,
    a transplant start). The model remembers the front-end mode it was trained with (``model.frontend_words``)
    and its training step (``model.step``)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if "model" in state:
        weights = state["model"]
    else:
        weights = {k[len("model."):]: v for k, v in state["system"].items() if k.startswith("model.")}
    frontend = state.get("frontend")
    model = VibeVoca(Config.from_dict(state["config"]))
    model.load_state_dict(weights, strict=True)
    model.frontend_words = frontend_words(frontend["fingerprint"] if isinstance(frontend, dict) else frontend)
    model.step = int(state.get("step", 0))
    return model.to(device).eval()


def speak(model, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, polish=Polish(), pauses=None):
    """One utterance: text -> (float waveform tensor, Phonemized, synthesis details). Same front end as training.
    ``polish=None`` gives the sound exactly as the network was trained to make it."""
    device = next(model.parameters()).device
    phones = phonemize(text, getattr(model, "frontend_words", None))
    if len(phones) > model.config.max_tokens:
        raise ValueError("text is too long for one utterance; split it into sentences")
    tokens = torch.tensor([phones.ids], device=device)
    weight = torch.tensor([phones.weight], device=device)
    if voice not in model.config.speakers:
        raise ValueError(f"unknown voice {voice!r}; this model has {model.config.speakers}")
    speaker = torch.tensor([model.config.speakers.index(voice)], device=device)
    audio, lengths, details = model.synthesize(tokens, weight, torch.tensor([len(phones)], device=device), speaker, speed,
                                               pitch_shift, seed, pitch_range, polish=polish, pauses=pauses)
    if not torch.isfinite(audio).all():
        raise RuntimeError("model produced non-finite audio")
    return audio[0, :int(lengths[0])].cpu(), phones, details


def speak_batch(model, texts, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, polish=Polish(), pauses=None,
                batch_size=16):
    """Several utterances, ``batch_size`` per network call -> [(float waveform array, Phonemized, durations [tokens],
    prosody [frames, 4])]. Utterance k is spoken with seed + k (one noise stream per row), so the first one sounds
    exactly as ``speak`` makes it and every one as ``speak(..., seed=seed + k)``, up to float rounding."""
    device = next(model.parameters()).device
    if voice not in model.config.speakers:
        raise ValueError(f"unknown voice {voice!r}; this model has {model.config.speakers}")
    all_phones = []
    for text in texts:
        phones = phonemize(text, getattr(model, "frontend_words", None))
        if len(phones) > model.config.max_tokens:
            raise ValueError(f"{text!r}: text is too long for one utterance; split it into sentences")
        all_phones.append(phones)
    speaker, out = model.config.speakers.index(voice), []
    for start in range(0, len(all_phones), batch_size):
        group = all_phones[start:start + batch_size]
        size = max(len(p) for p in group)
        tokens = torch.tensor([list(p.ids) + [0] * (size - len(p)) for p in group], device=device)
        weight = torch.tensor([list(p.weight) + [0] * (size - len(p)) for p in group], device=device)
        lengths = torch.tensor([len(p) for p in group], device=device)
        speakers = torch.full((len(group),), speaker, device=device)
        audio, samples, details = model.synthesize(tokens, weight, lengths, speakers, speed, pitch_shift, seed + start,
                                                   pitch_range, polish=polish, pauses=pauses)
        if not torch.isfinite(audio).all():
            raise RuntimeError("model produced non-finite audio")
        audio, samples = audio.float().cpu(), samples.cpu()
        durations, prosody = details["durations"].cpu(), details["prosody"].float().cpu()
        for row, phones in enumerate(group):
            out.append((audio[row, :int(samples[row])].numpy(), phones,
                        durations[row, :len(phones)].numpy().astype(int), prosody[row].numpy()))
    return out


def main():
    from .synthesis import Settings, synthesize

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True, help="any length: split into sentences at . ? ! and line breaks")
    parser.add_argument("--voice", help="one of the model's voices (default: its first)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--semitones", type=float, default=0, help="pitch shift")
    parser.add_argument("--pitch-range", type=float, default=1, help="scale pitch movement around the utterance mean (1 = as predicted)")
    parser.add_argument("--plain", action="store_true", help="no corrections while speaking (model.Polish off)")
    parser.add_argument("--texture", type=float, default=0.0, help="AR(2) pitch texture amount (1 = InfoRe recordings)")
    parser.add_argument("--comma-ms", type=float, default=0, help="minimum pause at commas (0 = as predicted)")
    parser.add_argument("--stop-ms", type=float, default=0, help="minimum pause at . ? ! (0 = as predicted)")
    parser.add_argument("--gap-ms", type=float, default=300, help="pause between sentences")
    parser.add_argument("--punctuate", action="store_true", help="restore punctuation first (needs vibert-capu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import soundfile as sf

    model = load_model(args.checkpoint, args.device)
    settings = Settings(speed=args.speed, semitones=args.semitones, pitch_range=args.pitch_range, seed=args.seed,
                        polish=not args.plain, texture=args.texture,
                        comma_ms=args.comma_ms, stop_ms=args.stop_ms, gap_ms=args.gap_ms, punctuate=args.punctuate)
    result = synthesize(model, args.text, args.voice or model.config.speakers[0], settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, result.audio, model.config.sample_rate, subtype="FLOAT")
    for sentence in result.sentences:
        print(" ".join(sentence.phones.symbols))
    if result.unknown:
        print("unreadable (<unk>):", ", ".join(result.unknown))
    print(f"Saved {args.output}: {result.seconds:.2f} s, {len(result.sentences)} sentence(s), "
          f"synthesis {result.elapsed:.2f} s")


if __name__ == "__main__":
    main()
