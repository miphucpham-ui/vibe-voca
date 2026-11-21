# vibe-voca

Small Vietnamese + English TTS (~5M params, 24 kHz) fine-tuned from [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca).
English voices F2/M2 stay bit-identical to tiny-voca; Vietnamese is added on top through new phoneme/tone rows and
voice-gated adapters. Model: [PhucMinh111/vibe-voca](https://huggingface.co/PhucMinh111/vibe-voca).

| Giọng | Dữ liệu | CER / WER (PhoWhisper, 80 câu test) |
|---|---|---|
| `vi_female` | InfoRe 25 h | 1.05% / 2.47% |
| `vi_adam` | Adam-TTS 50 h (tổng hợp) | 3.09% / 6.04% |
| `vi_nguoi_phu_nu` | giọng clone (dùng riêng) | 4.50% / 8.29% |
| `F2`, `M2` | tiny-voca (tiếng Anh) | giống hệt tiny-voca |

## Cài đặt

```bash
pip install "git+https://github.com/miphucpham-ui/vibe-voca"
```

## Dùng

```python
from vibe_voca.infer import pretrained
from vibe_voca.synthesis import Settings, synthesize

model = pretrained("cuda")   # tải từ Hugging Face (repo private: cần HF_TOKEN)
audio = synthesize(model, "Xin chào, tôi là trợ lý ảo tiếng Việt.", "vi_female", Settings(texture=0.5)).audio
```

```bash
python -m vibe_voca.infer --voice vi_female --text "Xin chào." --output out.wav
python -m vibe_voca.web      # web demo, cổng 7860
```

Tuỳ chọn (`Settings`): `speed`, `semitones`, `pitch_range`, `texture`, `comma_ms`, `stop_ms`, `gap_ms`, `punctuate`.

## Huấn luyện

```bash
git clone https://github.com/miphucpham-ui/vibe-voca && cd vibe-voca
bash scripts/setup.sh                    # thư viện + tiny-voca, aligner, PhoWhisper, vibert-capu
bash scripts/pipeline.sh infore          # dữ liệu -> 24 kHz
bash scripts/pipeline.sh features infore && bash scripts/pipeline.sh annotate infore
bash scripts/pipeline.sh combine infore:female=vi_female
bash scripts/pipeline.sh start           # khởi tạo từ tiny-voca
RUN=runs/main STEPS=12000 bash scripts/pipeline.sh train
bash scripts/pipeline.sh eval runs/main/inference.pt vi_female work/infore/audio24/test.jsonl
```

`bash scripts/pipeline.sh` liệt kê mọi bước. Thư mục làm việc đặt qua `VIBE_VOCA_TINY_VOCA`, `VIBE_VOCA_MODELS`,
`VIBE_VOCA_DATA`, `VIBE_VOCA_WORK` (xem `vibe_voca/paths.py`). Test: `python -m pytest`.

## Giấy phép

Code: Apache-2.0 (dẫn xuất từ tiny-voca, xem `THIRD_PARTY_NOTICES.md`). Mô hình dẫn xuất từ trọng số tiny-voca và mang
điều kiện của dữ liệu: InfoRe (CC-BY-4.0); Adam-TTS là giọng tổng hợp, chưa rõ điều khoản; `vi_nguoi_phu_nu` chỉ dùng
riêng, không phát hành.
