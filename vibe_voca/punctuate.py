"""Vietnamese punctuation restoration with dragonSwing/vibert-capu (viBERT fine-tuned on OSCAR; restores . , : ?
and capitalisation; CC-BY-SA-4.0), for text typed without punctuation before it is spoken.

    python -m vibe_voca.punctuate "hôm nay trời đẹp chúng ta đi dạo nhé"

The model directory ($VIBE_VOCA_MODELS/vibert-capu, see vibe_voca.paths; revision 261c60f2c30b02455dfce21a43c3ef14fc26992c) carries its
own inference code (gec_model.py ...), written for transformers 4.18. transformers 5 builds models on the meta device
inside from_pretrained, where that code's embedding resize fails, so the model is built here on a real device and the
weights are loaded with torch.load(weights_only=True); its output class is made a dataclass (transformers 5 requires
it). The repository's files are not modified; its tagger then does the rest.
"""

import sys
from pathlib import Path

import torch

from .paths import VIBERT_CAPU as MODEL_DIR


class Punctuator:
    def __init__(self, model_dir=MODEL_DIR, device=None):
        model_dir = Path(model_dir)
        if not (model_dir / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"{model_dir}: download dragonSwing/vibert-capu there first")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        import dataclasses
        import modeling_seq2labels
        from configuration_seq2labels import Seq2LabelsConfig
        from gec_model import GecBERTModel
        from modeling_seq2labels import Seq2LabelsModel
        # transformers 5 requires ModelOutput subclasses to be dataclasses; the repository's output class predates that.
        if not dataclasses.is_dataclass(modeling_seq2labels.Seq2LabelsOutput):
            dataclasses.dataclass(modeling_seq2labels.Seq2LabelsOutput)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tagger = GecBERTModel(vocab_path=str(model_dir / "vocabulary"), model_paths=[], device=device,
                                   split_chunk=True)
        config = Seq2LabelsConfig.from_pretrained(str(model_dir))
        model = Seq2LabelsModel(config)
        missing, unexpected = model.load_state_dict(torch.load(model_dir / "pytorch_model.bin", map_location="cpu",
                                                               weights_only=True), strict=False)
        if [k for k in missing + unexpected if not k.endswith("position_ids")]:      # an old buffer, not a weight
            raise RuntimeError(f"vibert-capu weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.tagger.models.append(model.eval().to(self.tagger.device))
        self.tagger.model_weights = [1.0]
        self.tagger.indexers.append(self.tagger._get_indexer(config.pretrained_name_or_path, config.special_tokens_fix))

    def __call__(self, texts):
        """Punctuated, capitalised versions of ``texts`` (a string or a list of strings)."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        with torch.inference_mode():
            out = self.tagger(items)
        return out[0] if single else out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="lines to punctuate (default: stdin)")
    args = parser.parse_args()
    punctuate = Punctuator()
    for line in args.text or [sys.stdin.read()]:
        print(punctuate(line))


if __name__ == "__main__":
    main()
