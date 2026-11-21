"""Join annotation runs (one per dataset) into one manifest per split for vibe_voca.train.

    python -m vibe_voca.combine --root $VIBE_VOCA_WORK --config configs/vibe_voca.json --out $VIBE_VOCA_WORK/annotations_all \
        --source $VIBE_VOCA_WORK/infore/annotations:female=vi_female --source $VIBE_VOCA_WORK/adam/annotations

Each --source is an annotation directory (annotations.json + <split>.jsonl), optionally with voice renames
(old=new, comma-separated). Feature paths are rewritten relative to --root, which must contain every source's
acoustic root. Rows must come from the current front end (VIBE_VOCA_WORDS decides which; on by default).
"""

import argparse
import hashlib
import json
from pathlib import Path

from .config import Config
from .text import frontend_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = Config.from_dict(json.loads(args.config.read_text()))
    frontend = frontend_fingerprint()
    splits, sources = {"train": [], "validation": []}, []
    for spec in args.source:
        directory, _, renames = spec.partition(":")
        directory = Path(directory)
        rename = dict(pair.split("=", 1) for pair in renames.split(",") if pair)
        meta = json.loads((directory / "annotations.json").read_text())
        root = Path(meta["acoustic_root"])
        counts = {}
        for split in splits:
            rows = [json.loads(line) for line in open(directory / f"{split}.jsonl", encoding="utf-8") if line.strip()]
            for row in rows:
                if row.get("frontend") != frontend:
                    raise SystemExit(f"{directory}: rows from another front end (set VIBE_VOCA_WORDS as for that run)")
                row = dict(row, speaker=rename.get(row["speaker"], row["speaker"]),
                           path=str((root / row["path"]).relative_to(args.root)))
                if row["speaker"] not in config.speakers:
                    raise SystemExit(f"{directory}: voice {row['speaker']} is not in {config.speakers}")
                splits[split].append(row)
            counts[split] = len(rows)
        sources.append({"annotations": str(directory), "renames": rename, "fingerprint": meta["fingerprint"], "counts": counts})
    body = {s: "\n".join(json.dumps({k: v for k, v in r.items() if k != "annotation"}, ensure_ascii=False, sort_keys=True)
                         for r in rows) for s, rows in splits.items()}
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps({**row, "acoustic": config.acoustic_fingerprint, "frontend": frontend,
                                       "annotation": fingerprint}, ensure_ascii=False) + "\n")
    meta = {"fingerprint": fingerprint, "complete": True, "acoustic_root": str(args.root), "stage": "combined",
            "sources": sources, "counts": {s: len(r) for s, r in splits.items()}}
    (args.out / "annotations.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"fingerprint": fingerprint, "counts": meta["counts"], "sources": sources}))


if __name__ == "__main__":
    main()
