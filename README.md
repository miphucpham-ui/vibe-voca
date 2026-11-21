# vibe-voca

Small Vietnamese + English TTS fine-tuned from [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca). English voices
F2/M2 stay bit-identical to tiny-voca; Vietnamese is added on top through new phoneme/tone rows and voice-gated adapters.

- **Front end** (`vibe_voca/text.py`): G2P tiếng Việt theo luật (`vibe_voca/g2p.py`, giọng Bắc), 6 thanh điệu làm đặc trưng
  từng token, đọc số (`vibe_voca/numbers.py`), tách từ bằng pyvi; câu/từ tiếng Anh đi qua phonemizer gốc của tiny-voca
  (dizione), nên tiếng Anh được đọc y như tiny-voca.
- **Mô hình** (`vibe_voca/model.py`): transformer feed-forward + đầu tổng hợp source-filter theo cao độ; adapter chỉ chạy
  cho các giọng tiếng Việt.
- **Huấn luyện**: thời lượng âm vị lấy từ bộ nhận dạng âm vị wav2vec2 (chỉ dùng khi chuẩn bị dữ liệu), không có aligner học.

## Kết quả (checkpoint `v2f`, bước 32000)

PhoWhisper-large chép lại 80 câu test của mỗi giọng (CER / WER):

| Giọng | Dữ liệu | Mô hình | Bản ghi gốc |
|---|---|---|---|
| `vi_female` | InfoRe 25 h | 1.05% / 2.47% | – |
| `vi_adam` | Adam-TTS 50 h (tổng hợp) | 3.09% / 6.04% | 2.98% / 5.92% |
| `F2`, `M2` | tiny-voca (tiếng Anh) | giống hệt tiny-voca | – |

Checkpoint không nằm trong repo (xem [Giấy phép](#giấy-phép-và-dữ-liệu)).

## Cài đặt

```bash
git clone <repo-url> vibe-voca && cd vibe-voca
bash scripts/setup.sh          # pip install -r requirements.txt + tải tiny-voca, aligner, PhoWhisper, vibert-capu
python -m pytest               # kiểm tra front end
```

Các thư mục ngoài repo được cấu hình bằng biến môi trường (`vibe_voca/paths.py`), mặc định nằm trong repo và đã
git-ignore:

| Biến | Nội dung | Mặc định |
|---|---|---|
| `VIBE_VOCA_TINY_VOCA` | snapshot `n4ze3m/tiny-voca` (trọng số khởi đầu + dữ liệu phonemizer) | `tiny-voca/` |
| `VIBE_VOCA_MODELS` | PhoWhisper (CTranslate2), vibert-capu | `models/` |
| `VIBE_VOCA_DATA` | dữ liệu nguồn (parquet InfoRe, ...) | `data/` |
| `VIBE_VOCA_WORK` | audio 24 kHz, đặc trưng, annotation (rất nhiều file nhỏ) | `work/` |

Trên Google Colab nên đặt `VIBE_VOCA_WORK=/content/work` (đĩa local; Drive rất chậm với nhiều file nhỏ) và chạy lại
`scripts/setup.sh` sau mỗi lần restart runtime.

## Dùng mô hình

```bash
# đọc một đoạn văn (tự tách câu)
python -m vibe_voca.infer --checkpoint runs/v2f/step-00032000.pt --voice vi_female --device cuda \
    --text "Xin chào, tôi là trợ lý ảo tiếng Việt. Năm 2024 có 366 ngày." \
    --texture 0.5 --output out.wav

# web demo (FastAPI + HTML/JS), cổng 7860
python -m vibe_voca.web --checkpoint runs/v2f/step-00032000.pt --voice vi_female \
    --test-set vi_female=$VIBE_VOCA_WORK/infore/audio24
```

Trong Colab mở web demo bằng `from google.colab import output; output.serve_kernel_port_as_window(7860)`.

Từ Python:

```python
from vibe_voca.infer import load_model
from vibe_voca.synthesis import Settings, synthesize

model = load_model("runs/v2f/step-00032000.pt", "cuda")
result = synthesize(model, "Hôm nay trời đẹp, chúng ta đi dạo nhé.", "vi_female",
                    Settings(texture=0.5, comma_ms=200, stop_ms=350))
# result.audio: numpy float32, model.config.sample_rate (24 kHz)
```

Tuỳ chọn khi đọc (không cần train lại): `speed`, `semitones`, `pitch_range`, `texture` (dao động cao độ tự nhiên, AR(2) khớp trên InfoRe), `comma_ms` /
`stop_ms` (độ dài tối thiểu của khoảng nghỉ ở dấu câu), `punctuate=True` (tự thêm dấu câu bằng vibert-capu). Văn bản
dài được tách câu và đọc theo batch (16 câu một lần gọi mạng; câu thứ k dùng seed + k).

Xuất ONNX: `python -m vibe_voca.export --checkpoint CKPT --output vibe_voca.onnx`.

## Huấn luyện

Mọi bước nằm trong `scripts/pipeline.sh` (chạy từng bước; `bash scripts/pipeline.sh` để xem danh sách).

```bash
# 1. dữ liệu: tải InfoRe vào $VIBE_VOCA_DATA/infore1_25hours (hf download doof-ferb/infore1_25hours --repo-type dataset ...)
bash scripts/pipeline.sh infore                  # chuẩn hoá transcript, cắt/lọc, 24 kHz -> work/infore/audio24
bash scripts/pipeline.sh adam finnal_tts.zip     # (tuỳ chọn) Adam-TTS-50h -> giọng vi_adam
bash scripts/pipeline.sh import myvoice vi_myvoice my_set/manifest_*.jsonl   # bộ dữ liệu bất kỳ (id, text, audio)

# 2. đặc trưng âm học + thời lượng âm vị, cho từng bộ
bash scripts/pipeline.sh features infore && bash scripts/pipeline.sh annotate infore
bash scripts/pipeline.sh features adam   && bash scripts/pipeline.sh annotate adam

# 3. gộp (đổi tên giọng nếu cần) và huấn luyện
bash scripts/pipeline.sh combine infore:female=vi_female adam
bash scripts/pipeline.sh start                   # tiny-voca -> runs/start.pt (configs/base.json, VOICES=...)
RUN=runs/main STEPS=12000 bash scripts/pipeline.sh train     # configs/vibe_voca.json: thêm adapter cho giọng Việt
bash scripts/pipeline.sh resume                  # tiếp tục sau khi bị ngắt

# 4. đánh giá
bash scripts/pipeline.sh eval runs/main/inference.pt vi_female work/infore/audio24/test.jsonl
```

- `configs/base.json`: kiến trúc của tiny-voca + 5 giọng (`vi_female`, `F2`, `M2`, `vi_adam`, `vi_nguoi_phu_nu`), dùng
  cho bước `start`. `configs/vibe_voca.json`: như trên + adapter rank 64 cho các giọng tiếng Việt (adapter khởi tạo bằng 0
  khi rẽ nhánh, nên mô hình ban đầu không đổi).
- `TRAIN_ONLY=new-rows NEW_VOICES=vi_female,vi_adam`: chỉ train hàng âm vị/thanh điệu tiếng Việt, hàng giọng mới và
  adapter; mọi tensor khác đóng băng, nên giọng tiếng Anh giữ nguyên từng bit. Thêm giọng vào checkpoint có sẵn:
  `bash scripts/pipeline.sh grow CKPT vi_new=vi_female` rồi `INIT=runs/start_grown.pt ... train`.
- Checkpoint `v2f` được train theo nhiều giai đoạn (cả mạng → adapter → tách từ + `duration_spread` → thêm giọng
  Adam → thêm giọng thứ năm, mỗi giai đoạn `--init-from` giai đoạn trước); pipeline trên chạy cùng các bước đó với
  tham số qua biến môi trường (`CONFIG`, `INIT`, `STEPS`, `LR`, `TRAIN_ONLY`, `NEW_VOICES`, `WEIGHTS`).
- Front end: mặc định tách từ (`VIBE_VOCA_WORDS=1`); checkpoint tự ghi lại chế độ front end nên lúc đọc luôn khớp.

## Cấu trúc

```
vibe_voca/
  text.py g2p.py numbers.py      front end (tiếng Việt + tiếng Anh qua dizione)
  model.py losses.py config.py   mô hình
  datasets/                      nhập dữ liệu: InfoRe, manifest bất kỳ / Adam-TTS
  prepare.py                     đặc trưng âm học (mel, F0, năng lượng)
  align.py annotate.py           thời lượng âm vị (wav2vec2 phone recogniser)
  combine.py transplant.py grow_voice.py data.py train.py
  infer.py synthesis.py punctuate.py export.py evaluate.py
  web/                           web demo (FastAPI + static HTML/CSS/JS)
configs/  scripts/  tests/
```

## Giấy phép và dữ liệu

Code: Apache-2.0 (dẫn xuất từ tiny-voca, xem `THIRD_PARTY_NOTICES.md`). Repo không chứa dữ liệu, audio hay
trọng số. Mô hình train bằng repo này là dẫn xuất từ trọng số tiny-voca (Apache-2.0) và mang điều kiện của dữ liệu đã
dùng: InfoRe (CC-BY-4.0); Adam-TTS-50h là giọng tổng hợp từ một công cụ không rõ tên, cần kiểm tra điều khoản trước khi
phát hành mô hình; giọng `vi_nguoi_phu_nu` là giọng clone được chủ giọng cho phép dùng riêng, không phát hành.
