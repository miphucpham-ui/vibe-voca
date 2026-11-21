"""Add a voice to a trained vibe-voca checkpoint, starting as a copy of an existing voice (its two voice rows).

    python -m vibe_voca.grow_voice --checkpoint runs/old/last.pt --config configs/new.json \
        --copy vi_new=vi_female --output runs/start_new.pt

The new configuration must list the old voices in the same order, followed by the new ones. Every other tensor is
copied unchanged; the result is a step-N training checkpoint for ``vibe_voca.train --init-from ... --new-data``.
"""

import argparse
import json
from pathlib import Path

import torch

from .config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--copy", action="append", required=True, help="new=existing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    old = Config.from_dict(state["config"])
    new = Config.from_dict(json.loads(args.config.read_text()))
    if tuple(new.speakers[:len(old.speakers)]) != tuple(old.speakers):
        raise SystemExit(f"new voices must follow the old ones: {old.speakers} -> {new.speakers}")
    copies = dict(pair.split("=", 1) for pair in args.copy)
    system = dict(state["system"])
    for name in ("model.voices.weight", "model.frame_voices.weight"):
        table = system[name]
        rows = [table[i] for i in range(len(old.speakers))]
        for voice in new.speakers[len(old.speakers):]:
            rows.append(table[old.speakers.index(copies[voice])].clone())
        system[name] = torch.stack(rows)
    out = {"format_version": state["format_version"], "config": new.to_dict(), "frontend": state["frontend"],
           "annotation": state["annotation"], "step": state["step"], "datasets": state["datasets"],
           "recipe": state["recipe"], "system": system,
           "origin": {"grown_from": str(args.checkpoint), "copies": copies}}
    torch.save(out, args.output)
    print(json.dumps({"output": str(args.output), "step": state["step"], "speakers": new.speakers, "copies": copies}))


if __name__ == "__main__":
    main()
