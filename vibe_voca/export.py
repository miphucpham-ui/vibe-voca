# Adapted from tiny-voca (Muhammed Nazeem, commit 28e0722) src/tiny_voca/export.py, via Alpa (77ethers/alpa); Apache-2.0,
# see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md. Modified for vibe-voca.

"""ONNX export of free-running synthesis.

    python -m vibe_voca.export --checkpoint inference.pt --output vibe_voca.onnx

The exported graph speaks one utterance: phoneme ids and syllable weights in,
waveform out. Text is turned into ids outside the graph (``vibe_voca.text.phonemize``).

``Exportable`` computes the same audio as ``VibeVoca.synthesize`` for a single
utterance, written with operations ONNX has: the short-time Fourier transforms
of the source-filter head become real convolutions with fixed kernels, the
complex filter responses are kept as separate real and imaginary parts, and
the frame expansion uses a comparison instead of a sorted search.

Inputs
    tokens       int64 [1, N]   phoneme ids, with the edge tokens
    weight       int64 [1, N]   0 none, 1-2 English stress, 3-8 Vietnamese tones (vibe_voca.text.FEATURE_NAMES)
    voice        int64 [1]      index into ``voices`` in the metadata
    speed        float32 [1]    1.0 = as predicted, larger is faster
    pitch_shift  float32 [1]    added to log F0 (0.058 is about a semitone)
    pitch_range  float32 [1]    scales pitch movement around the mean
Outputs
    audio        float32 [1, samples]
    durations    int64 [1, N]   frames per token (``hop`` samples each)
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .infer import load_model
from .model import ALL_PULSES, NO_PULSES, Polish, treble_kernel
from .text import SYMBOLS, frontend_fingerprint

INPUTS = ("tokens", "weight", "voice", "speed", "pitch_shift", "pitch_range")
OUTPUTS = ("audio", "durations")


def _cepstrum_matrices(positions, fft):
    """Real and imaginary parts of rfft for unit coefficients at ``positions``."""
    buffer = torch.zeros(len(positions), fft, dtype=torch.float64)
    buffer[torch.arange(len(positions)), torch.tensor(positions)] = 1
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return spectrum.real.float(), spectrum.imag.float()


def _analysis_kernels(window):
    """Convolution kernels [bins, 1, fft] giving the real and imaginary STFT."""
    fft = len(window)
    spectrum = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
    return spectrum.real.T[:, None].float().contiguous(), spectrum.imag.T[:, None].float().contiguous()


def _synthesis_kernel(window):
    """Transposed-convolution kernel [2 * bins, 1, fft]: windowed inverse rfft
    of the real parts (first half of the channels) and imaginary parts."""
    fft = len(window)
    bins = fft // 2 + 1
    unit = torch.eye(bins, dtype=torch.complex128)
    real = torch.fft.irfft(unit, n=fft, dim=-1)
    imaginary = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
    return (torch.cat((real, imaginary)) * window.double())[:, None].float().contiguous()


class Exportable(nn.Module):
    """Single-utterance synthesis in ONNX-friendly operations. ``polish`` as in ``VibeVoca.synthesize``."""

    def __init__(self, model, external_noise=False, polish=Polish()):
        super().__init__()
        self.model = model
        self.external_noise = external_noise
        self.polish = polish
        c = model.config
        v = model.vocoder
        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        for name, (real, imaginary) in (("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
                                        ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft))):
            self.register_buffer(f"{name}_real", real, persistent=False)
            self.register_buffer(f"{name}_imaginary", imaginary, persistent=False)
        for name, window in (("synth", v.synth_window), ("noise", v.noise_window)):
            real, imaginary = _analysis_kernels(window)
            self.register_buffer(f"{name}_analysis_real", real, persistent=False)
            self.register_buffer(f"{name}_analysis_imaginary", imaginary, persistent=False)
            self.register_buffer(f"{name}_synthesis", _synthesis_kernel(window), persistent=False)
            self.register_buffer(f"{name}_envelope", (window.double() ** 2).float()[None, None].contiguous(), persistent=False)
        self.register_buffer("ramp", torch.arange(c.hop, dtype=torch.float32) / c.hop, persistent=False)
        self.phase_steps = 1 << 24
        if polish is not None:
            # Which cepstral coefficients sharpening stretches, in the order causal then anticausal.
            chosen = torch.zeros(c.cep_causal)
            chosen[polish.sharpen_from:polish.sharpen_to] = 1
            mirror = torch.zeros(c.cep_anti)
            mirror[max(polish.sharpen_from - 1, 0):max(polish.sharpen_to - 1, 0)] = 1
            self.register_buffer("stretched", torch.cat((chosen, mirror)), persistent=False)
            triangle = torch.cat((torch.arange(1, polish.pitch_smooth + 2), torch.arange(polish.pitch_smooth, 0, -1))).float()
            self.register_buffer("pitch_kernel", (triangle / triangle.sum())[None, None], persistent=False)
            self.register_buffer("rumble_cut", torch.sigmoid((v.synth_frequency - 90.0) / 12.0)[None, :, None], persistent=False)
            self.register_buffer("treble", treble_kernel(polish.treble, c.sample_rate)[None, None], persistent=False)

    def _analysis(self, source, name, fft, hop):
        """Real and imaginary STFT [1, bins, frames] of ``source`` [1, samples]."""
        padded = F.pad(source[:, None], (fft // 2, fft))
        return (F.conv1d(padded, getattr(self, f"{name}_analysis_real"), stride=hop),
                F.conv1d(padded, getattr(self, f"{name}_analysis_imaginary"), stride=hop))

    def _synthesis(self, s_real, s_imaginary, real, imaginary, name, fft, hop, samples):
        """Multiply an STFT by a response given as real and imaginary parts
        [1, frames, bins], and resynthesize ``samples`` samples."""
        real, imaginary = real.transpose(1, 2), imaginary.transpose(1, 2)
        product = torch.cat((s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real), dim=1)
        wave = F.conv_transpose1d(product, getattr(self, f"{name}_synthesis"), stride=hop)
        envelope = F.conv_transpose1d(torch.ones_like(s_real[:, :1]), getattr(self, f"{name}_envelope"), stride=hop)
        return (wave / envelope)[:, 0, fft // 2:fft // 2 + samples]

    def _filter(self, source, real, imaginary, name, fft, hop):
        s_real, s_imaginary = self._analysis(source, name, fft, hop)
        return self._synthesis(s_real, s_imaginary, real, imaginary, name, fft, hop, source.shape[1])

    @staticmethod
    def _response(cepstrum, real, imaginary):
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _pulses(self, f0):
        """``pulse_train`` for one utterance, without 64-bit floats (the web
        runtime has none). The phase is summed inside each frame in float32,
        where it stays below a few cycles, and across frames as 24-bit integer
        fractions of a cycle, which wrap exactly however long the utterance is."""
        c = self.model.config
        knots = torch.log(f0.clamp_min(1.0))
        knots = torch.cat((knots, knots[:, -1:]), dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        frequency = torch.exp(low * (1 - self.ramp) + high * self.ramp)
        within = torch.cumsum(frequency / c.sample_rate, dim=2)
        whole_frame = within[:, :, -1]
        steps = torch.round((whole_frame - torch.floor(whole_frame)) * self.phase_steps).long()
        start = torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype) / self.phase_steps
        cycles = within + start[:, :, None]
        theta = (2 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        frequency = frequency.reshape(1, -1)
        harmonics = (c.harmonic_limit * c.sample_rate / frequency).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denominator = 2 * torch.sin(theta / 2)
        tiny = denominator.abs() < 1e-6
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
        wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
        return wave * torch.sqrt(2.0 / harmonics)

    def _vocoder(self, hidden, f0, total, noise, breath=None, limits=None, open_=None, vowel=None):
        c = self.model.config
        v = self.model.vocoder
        p = self.polish
        last = total - 1
        # Harmonic branch.
        h = v.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        steps = v.harmonic_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        here = torch.div(sub, steps, rounding_mode="floor")
        following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
        envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
        envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
        gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
        envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
        if p is not None and p.sharpen != 1.0:
            # Sharper formants in vowels, nasals and semivowels, with the loudness of each broad band put back.
            beta = (1.0 + (p.sharpen - 1.0) * open_.to(envelope.dtype)).index_select(1, here)[..., None]
            sharp = envelope * (1 + self.stretched * (beta - 1))
            flat_level, sharp_level = (envelope @ self.harmonic_real).clamp(-14, 6), (sharp @ self.harmonic_real).clamp(-14, 6)
            power = lambda level: torch.exp(2 * level) @ v.band_mask.T + 1e-20
            magnitude = torch.exp(sharp_level + (0.5 * torch.log(power(flat_level) / power(sharp_level))) @ v.band_weight)
            angle = sharp @ self.harmonic_imaginary
            harmonic_real, harmonic_imaginary = magnitude * torch.cos(angle), magnitude * torch.sin(angle)
        else:
            harmonic_real, harmonic_imaginary = self._response(envelope, self.harmonic_real, self.harmonic_imaginary)
        # Noise branch.
        n = v.noise(hidden)
        steps = v.noise_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        shape = n[..., :c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor")) * v.noise_lifter
        gain = n[..., c.noise_cep:].flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)
        pulses = self._pulses(f0)
        size = torch.arange(c.hop * total, device=hidden.device)
        noise = torch.randn_like(pulses) if noise is None else noise.index_select(1, size)
        hiss = self._filter(noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop)
        if p is None:
            return self._filter(pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop) + hiss
        # The mixed source: pulses below each sub-frame's limit, noise of the same power above it.
        breath = torch.randn_like(pulses) if breath is None else breath.index_select(1, size)
        limit = limits.index_select(1, here)
        width = torch.where(limit > 2000, torch.full_like(limit, 350.0), torch.full_like(limit, 120.0))
        keep = torch.sigmoid((limit[:, None] - v.synth_frequency[None, :, None]) / width[:, None])
        keep = F.avg_pool1d(F.pad(keep, (1, 1), mode="replicate"), 3, 1)
        other = torch.sqrt((1 - keep ** 2).clamp_min(0)) * self.rumble_cut
        p_real, p_imaginary = self._analysis(pulses, "synth", c.synth_fft, c.synth_hop)
        b_real, b_imaginary = self._analysis(breath * math.sqrt(0.5 / c.harmonic_limit), "synth", c.synth_fft, c.synth_hop)
        harmonic = self._synthesis(p_real * keep + b_real * other, p_imaginary * keep + b_imaginary * other, harmonic_real, harmonic_imaginary,
                                   "synth", c.synth_fft, c.synth_hop, pulses.shape[1])
        if p.noise_cap is not None:
            level = lambda wave: wave.reshape(1, -1, c.hop).pow(2).mean(-1)
            quieter = (level(harmonic) * 10 ** (p.noise_cap / 10) / level(hiss).clamp_min(1e-20)).clamp(max=1.0).sqrt()
            quieter = torch.where(vowel, quieter, torch.ones_like(quieter))
            quieter = F.avg_pool1d(F.pad(quieter[:, None], (1, 1), mode="replicate"), 3, 1)
            hiss = hiss * F.interpolate(quieter, scale_factor=float(c.hop), mode="linear", align_corners=False)[:, 0]
        wave = harmonic + hiss
        if p.treble:
            wave = F.conv1d(wave[:, None], self.treble, padding=self.treble.shape[-1] // 2)[:, 0]
        return wave

    def forward(self, tokens, weight, voice, speed, pitch_shift, pitch_range, noise=None, breath=None):
        m = self.model
        c = m.config
        p = self.polish
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        hidden = m.encoder(m.tokens(tokens) + m.syllable_weight(weight) + m.voices(voice)[:, None], token_mask, m.gate(voice))
        # Durations.
        output = m.duration(hidden, token_mask)
        frames = torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (output[..., 1] > 0)
        durations = torch.where(present, frames, torch.zeros_like(frames))
        # Repeat token states by duration.
        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(total, device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(features) + m.frame_voices(voice)[:, None]
        frame_mask = torch.ones_like(frame, dtype=torch.bool)
        # Pitch, voicing, energy.
        prosody_hidden = m.prosody_stack(expanded, frame_mask, m.gate(voice))
        prosody = m.prosody_head(prosody_hidden)
        predicted = prosody[..., 0]
        if p is not None and p.pitch_smooth:
            predicted = F.conv1d(F.pad(predicted[:, None], (p.pitch_smooth, p.pitch_smooth), mode="replicate"), self.pitch_kernel)[:, 0]
        vowel = m.vowel_tokens[tokens].gather(1, index)
        voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
        voiced = (voicing > 0).to(prosody.dtype)
        mean = (predicted * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (predicted - mean) * pitch_range
        condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1)
        acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask, m.gate(voice))
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
        if p is None:
            return self._vocoder(acoustic, f0, total, noise), durations
        # A frequency limit per frame, from the sound it belongs to (see ``Polish``).
        kind = m.sound_kind[tokens].gather(1, index)
        is_voiced = voicing > 0
        fill = lambda value: torch.full_like(pitch, value)
        limits = torch.where(kind == 1, fill(p.open_limit), fill(ALL_PULSES))
        limits = torch.where(kind == 2, fill(p.consonant_limit), limits)
        limits = torch.where(kind == 3, fill(NO_PULSES), limits)
        limits = torch.where(kind == 4, torch.where(is_voiced, fill(p.consonant_limit), fill(NO_PULSES)), limits)
        limits = torch.where((kind == 5) & ~is_voiced, fill(NO_PULSES), limits)
        return self._vocoder(acoustic, f0, total, noise, breath, limits, kind == 1, vowel), durations


def example_inputs(length=24):
    tokens = torch.randint(5, len(SYMBOLS), (1, length))
    one = lambda value: torch.tensor([value], dtype=torch.float32)
    return tokens, torch.zeros_like(tokens), torch.tensor([0]), one(1.0), one(0.0), one(1.0)


def export(model, path, external_noise=False, opset=17, polish=Polish()):
    wrapper = Exportable(model, external_noise, polish).eval()
    inputs = example_inputs()
    names = list(INPUTS)
    axes = {"tokens": {1: "tokens"}, "weight": {1: "tokens"}, "audio": {1: "samples"}, "durations": {1: "tokens"}}
    if external_noise:
        for name in ("noise",) if polish is None else ("noise", "breath"):
            inputs = (*inputs, torch.randn(1, model.config.hop * model.config.max_frames))
            names.append(name)
            axes[name] = {1: f"{name}_samples"}
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=list(OUTPUTS), dynamic_axes=axes,
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def metadata(model, checkpoint, polish=Polish()):
    c = model.config
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {"sample_rate": c.sample_rate, "hop": c.hop, "voices": list(c.speakers), "symbols": list(SYMBOLS),
            "max_tokens": c.max_tokens, "max_frames": c.max_frames, "step": state.get("step"),
            "frontend_fingerprint": frontend_fingerprint(), "inputs": list(INPUTS), "outputs": list(OUTPUTS),
            "polish": None if polish is None else dataclasses.asdict(polish)}


def main():
    parser = argparse.ArgumentParser(description="Export a vibe-voca inference checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain", action="store_true", help="Leave out the corrections made while speaking (the sound of v0.1 as first published)")
    parser.add_argument("--external-noise", action="store_true", help="Take the noise signals as inputs (for exact comparisons)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    model = load_model(args.checkpoint, "cpu")
    polish = None if args.plain else Polish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.output, args.external_noise, args.opset, polish)
    args.output.with_suffix(".json").write_text(json.dumps(metadata(model, args.checkpoint, polish), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) and {args.output.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
