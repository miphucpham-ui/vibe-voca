#!/usr/bin/env bash
# vibe-voca end to end, one stage per call. Every dataset gets a folder $VIBE_VOCA_WORK/<name>/ with
#   audio24/      24 kHz clips + train/validation/test.jsonl   (import stages)
#   features/     mel, pitch, energy per clip                   (features)
#   annotations/  phoneme tokens + durations                    (annotate)
#
#   bash scripts/pipeline.sh infore                      InfoRe: normalise transcripts, 24 kHz import
#   bash scripts/pipeline.sh adam ARCHIVE.zip            Vietnamese-Adam-TTS-50h archive -> voice vi_adam
#   bash scripts/pipeline.sh import NAME VOICE M.jsonl.. any single-speaker manifests (id, text, audio)
#   bash scripts/pipeline.sh features NAME               acoustic features
#   bash scripts/pipeline.sh annotate NAME [REVISION]    durations from the wav2vec2 phone recogniser
#   bash scripts/pipeline.sh combine SRC[:old=new] ...   join annotations -> $VIBE_VOCA_WORK/combined
#   bash scripts/pipeline.sh start                       step-0 checkpoint from tiny-voca -> runs/start.pt
#   bash scripts/pipeline.sh grow CKPT NEW=OLD ...       add voices to a checkpoint -> runs/start_grown.pt
#   bash scripts/pipeline.sh train | resume              train on $VIBE_VOCA_WORK/combined
#   bash scripts/pipeline.sh eval CKPT VOICE MANIFEST    PhoWhisper CER/WER
#   bash scripts/pipeline.sh speak CKPT VOICE "text"     -> sample.wav
#   bash scripts/pipeline.sh web CKPT [server options]   web demo on port 7860
#   bash scripts/pipeline.sh export CKPT                 ONNX
#
# Settings (environment): CONFIG (configs/vibe_voca.json), START_CONFIG (configs/base.json), VOICES (tiny-voca source
# of each voice), RUN (runs/main), INIT (runs/start.pt), STEPS, LR, TRAIN_ONLY (all | new-rows), NEW_VOICES, WEIGHTS.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
WORK=${VIBE_VOCA_WORK:-$PWD/work}
DATA=${VIBE_VOCA_DATA:-$PWD/data}
TINY_VOCA=${VIBE_VOCA_TINY_VOCA:-$PWD/tiny-voca}
CONFIG=${CONFIG:-configs/vibe_voca.json}
START_CONFIG=${START_CONFIG:-configs/base.json}
VOICES=${VOICES:-vi_female=F2,F2=F2,M2=M2,vi_adam=M2,vi_nguoi_phu_nu=F2}
RUN=${RUN:-runs/main}
INIT=${INIT:-runs/start.pt}
STEPS=${STEPS:-12000}
LR=${LR:-3e-4}
TRAIN_ONLY=${TRAIN_ONLY:-all}
NEW_VOICES=${NEW_VOICES:-}
WEIGHTS=${WEIGHTS:-'{"duration_spread": 1}'}
INFORE_REVISION=55062d7bb59588fe2ef35bf407fa9a975d20d9ce     # doof-ferb/infore1_25hours
ADAM_REVISION=b3b0c6772c2ca61db5437991f26275e0e780311c       # Phitran21/Vietnamese-Adam-TTS-50h
# one BLAS thread per feature worker; numba's cache on local disk (a cache shared between runtimes can segfault pyin)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$WORK/.numba_cache}

RECIPE=(--batch-frames 9000 --max-batch 32 --learning-rate "$LR" --warmup 200 --teacher-hold 300 --teacher-ramp 300
        --gan-start 1000000 --validation-size 32 --save-every 500 --keep-every 2000 --max-seconds 36000
        --weights "$WEIGHTS" --train-only "$TRAIN_ONLY" ${NEW_VOICES:+--new-voices "$NEW_VOICES"})
usage() { sed -n '2,25p' "$0"; exit 2; }
stage=${1:-}; shift || true

case "$stage" in
infore)
    [ -f "$DATA/infore1_text.jsonl" ] || python -m vibe_voca.datasets.infore_text --out "$DATA/infore1_text.jsonl"
    python -m vibe_voca.datasets.infore --text "$DATA/infore1_text.jsonl" --output "$WORK/infore/audio24" ;;
adam)
    [ $# -ge 1 ] || usage
    python -m vibe_voca.datasets.manifest --archive "$1" --extracted "$WORK/adam/extracted" --output "$WORK/adam/audio24" ;;
import)
    [ $# -ge 3 ] || usage
    name=$1 voice=$2; shift 2
    python -m vibe_voca.datasets.manifest --manifests "$@" --voice "$voice" --prefix "$name" \
        --source-url "${SOURCE_URL:-unknown}" --source-license "${SOURCE_LICENSE:-unknown}" --output "$WORK/$name/audio24" ;;
features)
    [ $# -ge 1 ] || usage
    for split in validation test train; do
        python -m vibe_voca.prepare --inputs "$WORK/$1/audio24/$split.jsonl" --split $split --output "$WORK/$1/features" \
            --config "$CONFIG" --resume
    done ;;
annotate)
    [ $# -ge 1 ] || usage
    case "$1" in infore) rev=$INFORE_REVISION ;; adam) rev=$ADAM_REVISION ;; *) rev=${2:-local} ;; esac
    python -m vibe_voca.annotate --features "$WORK/$1/features" --output "$WORK/$1/annotations" --dataset-revision "$rev" \
        --config "$CONFIG" --device cuda --resume ;;
combine)
    [ $# -ge 1 ] || usage
    sources=()                                   # NAME[:old=new,...] -> $WORK/NAME/annotations[:old=new,...]
    for s in "$@"; do name=${s%%:*}; sources+=(--source "$WORK/$name/annotations${s#"$name"}"); done
    python -m vibe_voca.combine --root "$WORK" --config "$CONFIG" --out "$WORK/combined" "${sources[@]}" ;;
start)
    python -m vibe_voca.transplant --source "$TINY_VOCA/tiny-voca.pt" --config "$START_CONFIG" --voices "$VOICES" --output runs/start.pt ;;
grow)
    [ $# -ge 2 ] || usage
    ckpt=$1; shift
    copies=(); for c in "$@"; do copies+=(--copy "$c"); done
    python -m vibe_voca.grow_voice --checkpoint "$ckpt" --config "$CONFIG" "${copies[@]}" --output runs/start_grown.pt ;;
train)
    python -m vibe_voca.train --train "$WORK/combined/train.jsonl" --validation "$WORK/combined/validation.jsonl" \
        --output "$RUN" --config "$CONFIG" --device cuda --init-from "$INIT" --new-data --steps "$STEPS" "${RECIPE[@]}" ;;
resume)
    python -m vibe_voca.train --train "$WORK/combined/train.jsonl" --validation "$WORK/combined/validation.jsonl" \
        --output "$RUN" --config "$CONFIG" --device cuda --resume "$RUN/last.pt" --steps "$STEPS" "${RECIPE[@]}" ;;
eval)
    [ $# -ge 3 ] || usage
    mkdir -p "$(dirname "$1")/eval"
    python -m vibe_voca.evaluate --checkpoint "$1" --voice "$2" --manifest "$3" --count "${COUNT:-80}" \
        --texture 0.5 --out "$(dirname "$1")/eval/$(basename "$1" .pt)_$2.json" ;;
speak)
    [ $# -ge 3 ] || usage
    python -m vibe_voca.infer --checkpoint "$1" --voice "$2" --text "$3" --texture 0.5 \
        --output "${OUT:-sample.wav}" ;;
web)
    [ $# -ge 1 ] || usage
    ckpt=$1; shift
    python -m vibe_voca.web --checkpoint "$ckpt" "$@" ;;
export)
    [ $# -ge 1 ] || usage
    python -m vibe_voca.export --checkpoint "$1" --output "${1%.pt}.onnx" ;;
*)
    usage ;;
esac
