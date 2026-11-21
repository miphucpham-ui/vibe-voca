"""Start vibe-voca from tiny-voca's published English weights instead of from random ones (SEAM start).

Adapted from alpa/transplant.py (77ethers/alpa, Apache-2.0; see THIRD_PARTY_NOTICES.md). tiny-voca and vibe-voca share one architecture, so every layer is copied as it is. Only the
language tables are built, without any training:

* tokens: tiny-voca's 76 rows are copied unchanged (vibe-voca keeps tiny-voca's symbols as ids 0..75);
  every Vietnamese phoneme row is a weighted mixture of English rows. The weights come from
  ``vibe_voca/g2p.py`` (TO_ENGLISH, the hand prior) or from ``--map`` (a SEAM map, same shape:
  {"phonemes": {vi_ipa: {english_symbol: weight}}}).
* per-token feature: rows 0-2 are tiny-voca's stress rows (none, primary, secondary); the six tone
  rows are mixtures of them (TONE_TO_STRESS) plus a small fixed perturbation each, so no two rows
  start identical.
* voices: tiny-voca's F2 and M2 become vibe-voca's "female" and "male"; with ``--voices`` each configured voice
  names the tiny-voca voice it starts from (``vi_female=F2,F2=F2,M2=M2`` keeps F2 and M2 for English and adds
  a copy of F2 for the Vietnamese recordings).

    python -m vibe_voca.transplant --source tiny-voca/tiny-voca.pt --config configs/vibe_voca.json \
        --voices vi_female=F2,F2=F2,M2=M2,vi_adam=M2,vi_nguoi_phu_nu=F2 --output runs/start.pt

The output is a step-0 training checkpoint; branch from it with ``vibe_voca.train --init-from START --new-data``.
A model started this way derives from tiny-voca's weights and must say so.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .g2p import TO_ENGLISH, TONE_TO_STRESS

from .config import CHECKPOINT_FORMAT, Config
from .text import ENGLISH_PHONES, FEATURE_LEVELS, SYMBOLS, TONE_OFFSET, VI_IPA, frontend_fingerprint, vi

LANGUAGE_SETTINGS = ("speakers", "vocab_size")


def mixture_rows(table, english, mixtures):
    rows = {}
    for ipa in VI_IPA:
        mix = mixtures.get(ipa)
        if not mix:
            raise ValueError(f"no English mixture for Vietnamese phoneme {ipa}")
        total = sum(mix.values())
        rows[vi(ipa)] = sum(table[english[s]] * (w / total) for s, w in mix.items())
    return rows


def transplant(source, config, mixtures=TO_ENGLISH, tone_mixtures=TONE_TO_STRESS, tone_spread=0.2, seed=0, voices=None):
    if config.adapter_rank:
        raise ValueError("start from a configuration without adapters (configs/base.json); vibe_voca.train adds them "
                         "when it branches with --init-from and a configuration that has them")
    theirs = source["config"]
    defaults = Config().to_dict()                          # settings tiny-voca's config predates (adapters)
    different = sorted(key for key, value in config.to_dict().items()
                       if key not in LANGUAGE_SETTINGS and theirs.get(key, defaults[key]) != value)
    if different:
        raise ValueError(f"the source model has another architecture or audio format: {different}")
    voices = voices or dict(zip(config.speakers, theirs["speakers"]))
    if set(voices) != set(config.speakers) or not set(voices.values()) <= set(theirs["speakers"]):
        raise ValueError(f"every voice of {config.speakers} needs a source among {theirs['speakers']} (--voices)")
    voice_rows = [list(theirs["speakers"]).index(voices[name]) for name in config.speakers]
    symbols = source["frontend"]["symbols"]
    english = {symbol: index for index, symbol in enumerate(symbols)}
    if tuple(symbols) != tuple(SYMBOLS[:len(symbols)]):
        raise ValueError("the source vocabulary is not tiny-voca's 76 symbols in tiny-voca's order")
    missing = sorted({s for mix in mixtures.values() for s in mix} - set(ENGLISH_PHONES))
    if missing:
        raise ValueError(f"mixtures use symbols that are not English phonemes: {missing}")
    weights = {}
    generator = torch.Generator().manual_seed(seed)
    for name, tensor in source["model"].items():
        if name == "tokens.weight":
            vietnamese = mixture_rows(tensor, english, mixtures)
            weights[name] = torch.stack([tensor[english[s]] if s in english else vietnamese[s] for s in SYMBOLS])
        elif name == "stress.weight":
            rows = [tensor[0], tensor[1], tensor[2]]
            scale = tensor.norm(dim=1).mean()
            for tone in range(1, FEATURE_LEVELS - TONE_OFFSET):
                mix = tone_mixtures[tone]
                base = sum(tensor[int(level)] * w for level, w in mix.items()) / sum(mix.values())
                direction = torch.randn(tensor.shape[1], generator=generator)
                rows.append(base + tone_spread * scale * direction / direction.norm())
            weights["syllable_weight.weight"] = torch.stack(rows)
        elif name in ("voices.weight", "frame_voices.weight"):
            weights[name] = tensor[voice_rows].clone()
        else:
            weights[name] = tensor.clone()
    return weights


def report(weights):
    """Closest pair among the Vietnamese rows and among the feature rows (cosine)."""
    table = weights["tokens.weight"]
    ids = [SYMBOLS.index(vi(p)) for p in VI_IPA]
    rows = torch.nn.functional.normalize(table[ids], dim=1)
    cosine = rows @ rows.T - 2 * torch.eye(len(ids))
    i, j = divmod(int(cosine.argmax()), len(ids))
    features = torch.nn.functional.normalize(weights["syllable_weight.weight"], dim=1)
    fc = features @ features.T - 2 * torch.eye(len(features))
    a, b = divmod(int(fc.argmax()), len(features))
    return {"closest_vietnamese_pair": [VI_IPA[i], VI_IPA[j], round(float(cosine[i, j]), 4)],
            "closest_feature_pair": [a, b, round(float(fc[a, b]), 4)]}


def starting_checkpoint(source, config, origin, **options):
    """A step-0 checkpoint that ``vibe_voca.train --init-from ... --new-data`` accepts. It
    holds only the model: the trainer starts its own optimizer, mel head and critics."""
    from .model import VibeVoca
    weights = transplant(source, config, **options)
    VibeVoca(config).load_state_dict(weights, strict=True)      # every tensor present, with the right shape
    return {"format_version": CHECKPOINT_FORMAT, "config": config.to_dict(), "frontend": frontend_fingerprint(),
            "annotation": "transplant", "step": 0, "datasets": [], "recipe": {"gan_start": 0},
            "system": {"model." + name: tensor for name, tensor in weights.items()}, "origin": origin}, report(weights)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="tiny-voca.pt from https://huggingface.co/n4ze3m/tiny-voca")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--map", type=Path, help="SEAM map JSON; default: g2p.TO_ENGLISH / TONE_TO_STRESS")
    parser.add_argument("--tone-spread", type=float, default=0.2, help="perturbation of each tone row, share of a row's norm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voices", help="each voice's tiny-voca source, e.g. vi_female=F2,F2=F2,M2=M2 (default: in order)")
    args = parser.parse_args(arguments)
    config = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    mixtures, tones = TO_ENGLISH, TONE_TO_STRESS
    if args.map:
        seam = json.loads(args.map.read_text(encoding="utf-8"))
        mixtures = seam.get("phonemes", mixtures)
        tones = {int(k): {int(a): b for a, b in v.items()} for k, v in seam.get("tone_to_stress", tones).items()}
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    voices = dict(pair.split("=", 1) for pair in args.voices.split(",")) if args.voices else None
    origin = {"model": "tiny-voca", "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "step": int(source.get("step", 0)), "voices": voices or dict(zip(config.speakers, source["config"]["speakers"])),
              "map": str(args.map) if args.map else "vi_g2p.TO_ENGLISH", "tone_spread": args.tone_spread}
    checkpoint, summary = starting_checkpoint(source, config, origin, mixtures=mixtures, tone_mixtures=tones,
                                              tone_spread=args.tone_spread, seed=args.seed, voices=voices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "tensors": len(checkpoint["system"]), **summary, **origin},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
