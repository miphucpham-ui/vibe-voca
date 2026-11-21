#!/usr/bin/env bash
# Install vibe-voca's dependencies and download the models it needs (pinned revisions). Safe to re-run: finished
# downloads are skipped. On Colab run it again after every runtime restart (pip packages and /content are lost).
#
#     bash scripts/setup.sh            # everything
#     bash scripts/setup.sh core       # packages + tiny-voca + aligner only (enough to prepare data and train)
#
# Locations follow vibe_voca/paths.py (VIBE_VOCA_TINY_VOCA, VIBE_VOCA_MODELS; defaults inside the repository).
set -euo pipefail
cd "$(dirname "$0")/.."
TINY_VOCA=${VIBE_VOCA_TINY_VOCA:-$PWD/tiny-voca}
MODELS=${VIBE_VOCA_MODELS:-$PWD/models}
TINY_VOCA_REVISION=1e73262226dbf4ef02e9df277e81aaa1db5ce16c
PHOWHISPER_REVISION=b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb
VIBERT_CAPU_REVISION=261c60f2c30b02455dfce21a43c3ef14fc26992c
WHAT=${1:-all}

pip install -q -r requirements.txt
# faster-whisper's CTranslate2 wheel needs CUDA 12 cuBLAS (Colab ships CUDA 13): vibe_voca.evaluate loads this wheel.
if [ "$WHAT" = all ] && command -v nvidia-smi > /dev/null; then pip install -q nvidia-cublas-cu12; fi

# tiny-voca: starting weights and the English phonemizer data (dizione)
[ -f "$TINY_VOCA/tiny-voca.pt" ] || hf download n4ze3m/tiny-voca --revision $TINY_VOCA_REVISION --local-dir "$TINY_VOCA" > /dev/null
# phoneme recogniser used only to annotate training data (vibe_voca.annotate)
hf download facebook/wav2vec2-lv-60-espeak-cv-ft > /dev/null
[ "$WHAT" = core ] && { echo "setup (core) done"; exit 0; }

mkdir -p "$MODELS"
# PhoWhisper-large as CTranslate2 float16 (vibe_voca.evaluate)
if [ ! -f "$MODELS/phowhisper-large-ct2/model.bin" ]; then
    snapshot=$(hf download vinai/PhoWhisper-large --revision $PHOWHISPER_REVISION)
    ct2-transformers-converter --model "$snapshot" --output_dir "$MODELS/phowhisper-large-ct2" --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json
fi
# vibert-capu: punctuation restoration (vibe_voca.punctuate; CC-BY-SA-4.0)
[ -f "$MODELS/vibert-capu/pytorch_model.bin" ] || hf download dragonSwing/vibert-capu --revision $VIBERT_CAPU_REVISION \
    --local-dir "$MODELS/vibert-capu" > /dev/null
echo "setup done: tiny-voca in $TINY_VOCA, models in $MODELS"
