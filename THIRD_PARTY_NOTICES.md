# Third-party notices

## tiny-voca

vibe-voca is derived from **tiny-voca** by Muhammed Nazeem (https://github.com/n4ze3m/tiny-voca, commit `28e0722`;
weights at https://huggingface.co/n4ze3m/tiny-voca), licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`).

- **Weights.** vibe-voca models start from tiny-voca's published weights (`vibe_voca/transplant.py`): every layer is
  copied, the Vietnamese phoneme rows and tone rows are built from tiny-voca's English rows. A model trained with
  this repository derives from tiny-voca's weights.
- **Code.** These files are adapted from tiny-voca's files of the same names (via Alpa, below): `model.py`,
  `config.py`, `infer.py`, `export.py`, `align.py`, `annotate.py`, `data.py`, `prepare.py`, `train.py`,
  `losses.py`, `journal.py`. Each is marked as modified at its top.
- **English front end.** English text is read by tiny-voca's phonemizer **dizione**
  (https://github.com/n4ze3m/dizione, commit `8cd889e4`) with the data shipped in the tiny-voca snapshot.

## Alpa

Parts of the code (transplant, front-end layout, annotation label sets, ``Polish``) are adapted from **Alpa**
(https://huggingface.co/77ethers/alpa), Apache License 2.0. No Alpa weights are used.

## Models used by the tools (downloaded, not included)

- `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0): phoneme recogniser used only to annotate training data.
- `vinai/PhoWhisper-large` (BSD-3-Clause): speech recogniser used only for evaluation.
- `dragonSwing/vibert-capu` (CC-BY-SA-4.0): optional punctuation restoration before synthesis.
- `pyvi` (MIT): Vietnamese word segmentation in the front end.

## Data (not included)

- **InfoRe Technology №1, 25 h** (`doof-ferb/infore1_25hours`, CC-BY-4.0).
- **Vietnamese-Adam-TTS-50h** (`Phitran21/Vietnamese-Adam-TTS-50h`, labelled CC-BY-4.0; synthetic speech from an
  unnamed generator: check the generator's terms before releasing a model trained on it).
- **ViMedCSS** (`tensorxt/ViMedCSS`, CC-BY-4.0): text only, for a synthetic voice that is not distributed here.
