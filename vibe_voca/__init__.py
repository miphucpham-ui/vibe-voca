"""vibe-voca: small Vietnamese + English text-to-speech fine-tuned from tiny-voca.

    from vibe_voca.infer import pretrained
    from vibe_voca.synthesis import Settings, synthesize

    model = pretrained("cuda")                      # PhucMinh111/vibe-voca on the Hugging Face Hub
    audio = synthesize(model, "Xin chào.", "vi_female", Settings(texture=0.5)).audio   # float32, 24 kHz
"""

__version__ = "0.3.0"
