# Adapted from tiny-voca (Muhammed Nazeem, commit 28e0722) src/tiny_voca/model.py, via Alpa (77ethers/alpa); Apache-2.0,
# see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md. Modified for vibe-voca.

"""vibe-voca: feed-forward transformer with a pitch-driven source-filter head.

All weights start random. Tensor conventions: hidden [B,T,C], frame features
[B,T], waveform [B,S]. Acoustic frame i is centred on sample ``i * hop``.

    phoneme + syllable weight + voice -> transformer encoder -> duration predictor
        -> repeat tokens by duration (supervised durations while training)
        -> frame transformer A -> pitch / voicing / energy
        -> frame transformer B -> filter parameters
        -> harmonic pulse train (from pitch) and noise, each shaped by its
           time-varying filter in the STFT domain -> 24 kHz waveform

The waveform head has no neural upsampling network. The oscillator supplies the
periodicity that a tiny decoder cannot render reliably on its own; the network
only has to predict slowly varying spectral envelopes (complex cepstra), which
is what keeps the voice from sounding buzzy or unstable at this size.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config
from .text import BREATH, FEATURE_LEVELS, HISS, MARK, OPEN, STOPS
from .text import PHONE_IDS, SYMBOL_TO_ID, VOWELS


def length_mask(lengths, size=None):
    size = int(lengths.max()) if size is None else size
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def rotary_table(size, width, device):
    inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width))
    angle = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inverse[None]
    return angle.cos(), angle.sin()


def rotate(x, table):
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Self-attention with rotary (relative) positions: no absolute position
    table, so utterances longer than those seen in training stay in range."""

    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, mask, table):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotate(q, table), rotate(k, table), v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.channels)
        self.attention = Attention(c.channels, c.heads, c.dropout)
        self.norm2 = nn.LayerNorm(c.channels)
        self.depthwise = nn.Conv1d(c.channels, c.channels, c.kernel, padding=c.kernel // 2, groups=c.channels)
        self.expand = nn.Linear(c.channels, c.feedforward)
        self.project = nn.Linear(c.feedforward, c.channels)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, mask, table):
        keep = mask[..., None]
        x = (x + self.dropout(self.attention(self.norm1(x), mask, table))) * keep
        # Mask before the convolution so padded positions cannot leak back in.
        y = self.depthwise((self.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
        y = self.project(self.dropout(F.gelu(self.expand(y))))
        return (x + self.dropout(y)) * keep


class Adapter(nn.Module):
    """Residual bottleneck after a block. The output projection starts at zero, so a new adapter changes nothing."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c.channels)
        self.down = nn.Linear(c.channels, c.adapter_rank)
        self.up = nn.Linear(c.adapter_rank, c.channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(self.norm(x))))


class Stack(nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.blocks = nn.ModuleList(Block(c) for _ in range(layers))
        self.norm = nn.LayerNorm(c.channels)
        self.width = c.channels // c.heads
        self.adapters = nn.ModuleList(Adapter(c) for _ in range(layers)) if c.adapter_rank else None

    def forward(self, x, mask, gate=None):
        """``gate`` [B]: 1 where the utterance's voice uses the adapters, 0 elsewhere (those rows are computed
        exactly as without adapters: the adapter output is never added to them)."""
        table = rotary_table(x.shape[1], self.width, x.device)
        use = self.adapters is not None and gate is not None      # no data-dependent shortcut: ONNX export traces this
        for i, block in enumerate(self.blocks):
            x = block(x, mask, table)
            if use:
                x = torch.where(gate[:, None, None] > 0, (x + self.adapters[i](x)) * mask[..., None], x)
        return self.norm(x) * mask[..., None]


class Predictor(nn.Module):
    def __init__(self, c, outputs):
        super().__init__()
        p = c.predictor_channels
        self.conv1 = nn.Conv1d(c.channels, p, 3, padding=1)
        self.conv2 = nn.Conv1d(p, p, 3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(p), nn.LayerNorm(p)
        self.drop = nn.Dropout(c.dropout)
        self.out = nn.Linear(p, outputs)

    def forward(self, x, mask):
        x = self.conv1((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm1(F.gelu(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm2(F.gelu(x)))
        return self.out(x) * mask[..., None]


def pulse_train(f0, sample_rate, hop, limit=0.47, phase=None):
    """Band-limited pulse train [B, T*hop] with unit power for f0 [B, T] in Hz.

    Frame values sit at samples i*hop and are interpolated in the log domain.
    The sum of all harmonics below ``limit * sample_rate`` has the closed form
    sin((K+1/2)x) / (2 sin(x/2)) - 1/2; the top harmonic fades in and out so the
    signal stays continuous while the pitch glides. Float64 keeps the phase exact.
    """
    frames = f0.shape[1]
    knots = torch.log(f0.double().clamp_min(1.0))
    knots = torch.cat((knots, knots[:, -1:]), dim=1)
    position = torch.arange(frames * hop, device=f0.device, dtype=torch.float64) / hop
    low = position.floor().long()
    weight = position - low
    frequency = torch.exp(knots[:, low] * (1 - weight) + knots[:, low + 1] * weight)
    theta = 2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1)
    if phase is not None:
        theta = theta + phase.double()[:, None]
    theta = torch.remainder(theta, 2 * math.pi)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def filtered(source, response, fft, hop, window):
    """Time-varying filtering: response [B, frames, bins] multiplies the STFT."""
    spectrum = torch.stft(source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, window=window, center=True, length=source.shape[-1])


def cepstrum_response(causal, anticausal, fft):
    """Complex frequency response exp(FFT(cepstrum)). Causal coefficients give a
    minimum-phase filter; anticausal ones add a maximum-phase part (glottal pulse)."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., :causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1]:] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return torch.polar(torch.exp(spectrum.real.clamp(-14, 6)), spectrum.imag)


# Sounds by how they are voiced, for ``Polish``: OPEN, STOPS, HISS, BREATH and MARK come from vibe_voca.text.
NO_PULSES, ALL_PULSES = -1e5, 1e6       # frequency limits (Hz) that mean "noise everywhere" and "pulses everywhere"


@dataclass(frozen=True)
class Polish:
    """Corrections made while speaking; the weights are not touched.

    The harmonic branch was trained with a pulse train as its only source, and the network learnt to build hiss,
    breath and bursts out of those pulses as well, which a listener hears as a second, metallic voice inside the
    consonants. Here every frame gets a frequency limit from the sound it belongs to. Below the limit the branch
    is fed the pulses, above it white noise of the same power:

    * sibilants and fricatives (``HISS`` in text.py): noise at every frequency;
    * stops and voiced fricatives (``STOPS``): pulses below ``consonant_limit`` only; h (``BREATH``) the same where
      the frame is voiced, else noise;
    * vowels, nasals and semivowels: pulses up to ``open_limit``, above which a real voice is breath.

    The other settings: the predicted pitch is averaged over ``pitch_smooth`` frames either side (triangular
    weights); in vowels, nasals and semivowels the peaks and valleys of the spectral envelope are stretched by
    ``sharpen`` (cepstral coefficients ``sharpen_from`` up to ``sharpen_to``, with the loudness of each broad
    band put back); in vowels the noise branch is held ``noise_cap`` dB under the harmonic branch; and the top
    of the spectrum is lifted by ``treble`` dB (from 2 kHz, in full from 5 kHz). The values were chosen against
    the recordings and by ear.

    vibe-voca addition (off by default, so the sound above is unchanged):

    * ``texture`` adds the fast pitch fluctuation of a real voice to the oscillator: an AR(2) process
      (``texture_ar`` = a1, a2, sigma), fitted to InfoRe recordings minus their slow Kalman trend; 1.0 is the
      recordings' amount (about 0.46 semitones), 0 is none."""
    pitch_smooth: int = 3
    consonant_limit: float = 1000.0
    open_limit: float = 4500.0
    sharpen: float = 1.6
    sharpen_from: int = 8
    sharpen_to: int = 40
    noise_cap: float = -15.0
    treble: float = 4.0
    texture: float = 0.0
    texture_ar: tuple = (0.978, -0.2984, 0.01537)


def rts_smooth(x, observed, q, r):
    """Kalman filter + Rauch-Tung-Striebel smoother, constant-velocity model, one sequence (numpy). Used only for
    the training pitch target (``train --pitch-kalman``, see data.py); synthesis smooths with ``Polish.pitch_smooth``."""
    import numpy as np
    n = len(x)
    first = np.flatnonzero(observed)
    if not first.size:
        return x.copy()
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    m, P = np.array([x[first[0]], 0.0]), np.diag([r, 1e-2])
    ms, Ps, mp, Pp = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
    for t in range(n):
        if t:
            m, P = F @ m, F @ P @ F.T + Q
        mp[t], Pp[t] = m, P
        if observed[t]:
            k = P[:, 0] / (P[0, 0] + r)
            m = m + k * (x[t] - m[0])
            P = P - np.outer(k, P[0])
        ms[t], Ps[t] = m, P
    out = ms.copy()
    for t in range(n - 2, -1, -1):
        G = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        out[t] = ms[t] + G @ (out[t + 1] - mp[t + 1])
    return out[:, 0]


class Shaping(NamedTuple):
    """What ``SourceFilter`` needs to apply a ``Polish``: per-frame controls and the replacement noise."""
    polish: Polish
    limits: torch.Tensor        # [B, frames] Hz: pulses below, noise above
    open: torch.Tensor          # [B, frames] bool: vowel, nasal or semivowel
    vowel: torch.Tensor         # [B, frames] bool
    breath: torch.Tensor        # [B, samples] white noise, unit variance


def treble_kernel(gain_db, sample_rate, taps=129):
    """Zero-phase filter that lifts the top of the spectrum: nothing below 2 kHz, ``gain_db`` from 5 kHz up."""
    size = 1024
    frequency = torch.fft.rfftfreq(size, 1 / sample_rate).clamp_min(1.0)
    rise = (torch.log2(frequency / 2000.0) / math.log2(2.5)).clamp(0, 1)
    kernel = torch.fft.irfft(10 ** (gain_db * rise / 20), n=size).roll(taps // 2)[:taps]
    return kernel * torch.hann_window(taps, periodic=False)


class SourceFilter(nn.Module):
    """Harmonic-plus-noise synthesis controlled per frame by two linear heads."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.harmonic_steps, self.noise_steps = c.hop // c.synth_hop, c.hop // c.noise_hop
        self.harmonic = nn.Linear(c.channels, c.cep_causal + c.cep_anti + self.harmonic_steps)
        self.noise = nn.Linear(c.channels, c.noise_cep + self.noise_steps)
        for head, level in ((self.harmonic, -1.5), (self.noise, -2.5)):   # start quiet and spectrally flat
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[0] = level

        def lifter(size, first):
            scale = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4)
            if first == 0:
                scale[0] = 2.0      # coefficient 0 is the log gain and needs range
            return scale
        self.register_buffer("causal_lifter", lifter(c.cep_causal, 0), persistent=False)
        self.register_buffer("anti_lifter", lifter(c.cep_anti, 1), persistent=False)
        self.register_buffer("noise_lifter", lifter(c.noise_cep, 0), persistent=False)
        self.register_buffer("synth_window", torch.hann_window(c.synth_fft), persistent=False)
        self.register_buffer("noise_window", torch.hann_window(c.noise_fft), persistent=False)
        # For ``Polish``: four broad bands of the harmonic filter, and weights that turn one gain per band into a
        # smooth gain per frequency (straight lines between the band centres, level outside them).
        frequency = torch.arange(c.synth_fft // 2 + 1) * (c.sample_rate / c.synth_fft)
        edges, centres = (0.0, 1000.0, 2500.0, 5000.0, float("inf")), torch.tensor([500.0, 1750.0, 3750.0, 8500.0])
        self.register_buffer("band_mask", torch.stack([((frequency >= lo) & (frequency < hi)).float() for lo, hi in zip(edges, edges[1:])]), persistent=False)
        inside = frequency.clamp(float(centres[0]), float(centres[-1]))
        upper = torch.bucketize(inside, centres).clamp(1, len(centres) - 1)
        share = (inside - centres[upper - 1]) / (centres[upper] - centres[upper - 1])
        weight = torch.zeros(len(centres), len(frequency))
        weight[upper - 1, torch.arange(len(frequency))] = 1 - share
        weight[upper, torch.arange(len(frequency))] += share
        self.register_buffer("band_weight", weight, persistent=False)
        self.register_buffer("synth_frequency", frequency, persistent=False)

    @staticmethod
    def _take(values, index):
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))

    def _subframes(self, frames, lengths, steps):
        """Sub-frame j uses frame j // steps; indices past an utterance's end
        repeat its last sub-frame so padding never changes real audio. Two extra
        sub-frames cover the half window of zeros appended to the source."""
        sub = torch.arange(steps * frames + 3, device=lengths.device)[None].expand(len(lengths), -1)
        return torch.minimum(sub, (steps * lengths - 1).clamp_min(0)[:, None])

    @staticmethod
    def _shape(source, response, fft, hop, window):
        # Half a window of trailing zeros: the frames around an utterance's end are
        # then the same whether it stands alone or is padded inside a batch.
        return filtered(F.pad(source, (0, fft // 2)), response, fft, hop, window)[:, :source.shape[-1]]

    def forward(self, hidden, f0, lengths, noise=None, phase=None, shaping=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]
            # Harmonic branch: envelope interpolated between frames, one gain per sub-frame.
            h = self.harmonic(hidden)
            split = c.cep_causal + c.cep_anti
            sub = self._subframes(frames, lengths, self.harmonic_steps)
            here, following = sub // self.harmonic_steps, torch.minimum((sub + 1) // self.harmonic_steps, last)
            causal = 0.5 * (self._take(h[..., :c.cep_causal], here) + self._take(h[..., :c.cep_causal], following)) * self.causal_lifter
            anti = 0.5 * (self._take(h[..., c.cep_causal:split], here) + self._take(h[..., c.cep_causal:split], following)) * self.anti_lifter
            causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
            harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
            # Noise branch: one envelope per frame, four gains per frame for sharp consonants.
            n = self.noise(hidden)
            sub = self._subframes(frames, lengths, self.noise_steps)
            shape = self._take(n[..., :c.noise_cep], sub // self.noise_steps) * self.noise_lifter
            shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
            noise_response = cepstrum_response(shape, None, c.noise_fft)
            # Hold the last real pitch value through padding: the pitch glide inside an
            # utterance's final frame must not depend on what the batch was padded with.
            f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase) * audible
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            if shaping is not None:
                return self._polished(shaping, causal, anti, harmonic_response, here, pulses, noise * audible, noise_response, last, audible)
            wave = (self._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window)
                    + self._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window))
            return wave * audible

    def _polished(self, shaping, causal, anti, response, here, pulses, noise, noise_response, last, audible):
        """The same synthesis with a ``Polish`` applied. ``here`` is the frame of each harmonic sub-frame."""
        c, p = self.config, shaping.polish
        frames = shaping.limits.shape[1]
        samples = frames * c.hop
        frequency = self.synth_frequency
        if p.sharpen != 1.0:
            # Stretch the envelope's detail by ``beta`` (in dB) where the frame is a vowel, nasal or semivowel. The lowest
            # coefficients (loudness and overall slope) and the highest (finer than a formant) are left alone.
            beta = (1.0 + (p.sharpen - 1.0) * shaping.open.float()).gather(1, here)[..., None]
            chosen = torch.zeros(c.cep_causal, device=causal.device)
            chosen[p.sharpen_from:p.sharpen_to] = 1
            mirror = torch.zeros(c.cep_anti, device=causal.device)
            mirror[max(p.sharpen_from - 1, 0):max(p.sharpen_to - 1, 0)] = 1
            sharp = cepstrum_response(causal * (1 + chosen * (beta - 1)), anti * (1 + mirror * (beta - 1)), c.synth_fft)
            # Sharper peaks would move loudness between broad bands; give each band back what the network predicted.
            power = lambda r: r.abs().pow(2) @ self.band_mask.T + 1e-20
            response = sharp * torch.exp(0.5 * torch.log(power(response) / power(sharp)) @ self.band_weight)
        # The mixed source, built in the short-time frames the filter works in: pulses below each sub-frame's
        # limit, noise above it. The switch is 120 Hz wide for a consonant's limit, 350 Hz for the higher one of open
        # sounds, and is softened over three sub-frames in time.
        limit = shaping.limits.gather(1, here)
        keep = torch.sigmoid((limit[:, None] - frequency[None, :, None]) / torch.where(limit > 2000, 350.0, 120.0)[:, None])
        keep = F.avg_pool1d(F.pad(keep, (1, 1), mode="replicate"), 3, 1)
        fft, hop, window = c.synth_fft, c.synth_hop, self.synth_window
        spectrum = lambda source: torch.stft(F.pad(source, (0, fft // 2)), fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
        # White noise with the pulse train's power per frequency. The pulses carry nothing below the pitch, so the
        # noise that replaces them is cut below 90 Hz, or it would rumble.
        breath = spectrum(shaping.breath * audible * math.sqrt(0.5 / c.harmonic_limit)) * torch.sigmoid((frequency - 90.0) / 12.0)[:, None]
        mixed = spectrum(pulses) * keep + breath * torch.sqrt((1 - keep ** 2).clamp_min(0))
        harmonic = torch.istft(mixed * response.transpose(1, 2), fft, hop, window=window, center=True, length=samples + fft // 2)[:, :samples]
        hiss = self._shape(noise, noise_response, c.noise_fft, c.noise_hop, self.noise_window)
        if p.noise_cap is not None:
            # In a vowel the noise branch sometimes whispers the vowel as loudly as the harmonic branch voices it.
            level = lambda wave: wave.unflatten(1, (frames, c.hop)).pow(2).mean(-1)
            gain = (level(harmonic) * 10 ** (p.noise_cap / 10) / level(hiss).clamp_min(1e-20)).clamp(max=1.0).sqrt()
            gain = torch.where(shaping.vowel, gain, torch.ones_like(gain))
            gain = gain.gather(1, torch.minimum(torch.arange(frames, device=gain.device)[None], last))   # padding repeats the last frame
            gain = F.avg_pool1d(F.pad(gain[:, None], (1, 1), mode="replicate"), 3, 1)
            hiss = hiss * F.interpolate(gain, size=samples, mode="linear", align_corners=False)[:, 0]
        wave = (harmonic + hiss) * audible
        if p.treble:
            kernel = treble_kernel(p.treble, c.sample_rate).to(wave.device)
            wave = F.conv1d(wave[:, None], kernel[None, None], padding=len(kernel) // 2)[:, 0] * audible
        return wave


class VibeVoca(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        self.syllable_weight = nn.Embedding(FEATURE_LEVELS, c.channels)    # tone / stress (vibe_voca.text.FEATURE_NAMES)
        self.voices = nn.Embedding(len(c.speakers), c.channels)
        self.encoder = Stack(c, c.encoder_layers)
        # Outputs per token: log(1 + frames) and, for pause-capable tokens, a
        # "has frames at all" logit. A pause is absent more often than present, so
        # one regression would average both cases into a pause that is too short.
        self.duration = Predictor(c, 2)
        self.frame_voices = nn.Embedding(len(c.speakers), c.channels)
        self.frame_position = nn.Linear(3, c.channels)
        self.prosody_stack = Stack(c, c.prosody_layers)
        # Outputs: continuous log(F0/reference), voiced logit, normalized log RMS energy.
        self.prosody_head = nn.Linear(c.channels, 3)
        self.prosody_projection = nn.Linear(3, c.channels)
        self.acoustic_stack = Stack(c, c.acoustic_layers)
        self.vocoder = SourceFilter(c)
        self.register_buffer("adapter_gate", torch.tensor([v in c.adapter_voices for v in c.speakers]), persistent=False)
        phones = torch.zeros(c.vocab_size, dtype=torch.bool)
        phones[sorted(PHONE_IDS)] = True
        self.register_buffer("phone_tokens", phones, persistent=False)
        vowels = torch.zeros(c.vocab_size, dtype=torch.bool)
        vowels[[SYMBOL_TO_ID[v] for v in VOWELS]] = True
        self.register_buffer("vowel_tokens", vowels, persistent=False)
        kinds = torch.zeros(c.vocab_size, dtype=torch.long)       # 0 pause or mark, then the groups ``Polish`` treats alike
        for kind, symbols in enumerate((OPEN, STOPS, HISS, BREATH, MARK), start=1):
            kinds[[SYMBOL_TO_ID[symbol] for symbol in symbols]] = kind
        self.register_buffer("sound_kind", kinds, persistent=False)

    def encode(self, tokens, weight, lengths, speakers):
        if tokens.ndim != 2 or weight.shape != tokens.shape or lengths.shape != (tokens.shape[0],) or speakers.shape != lengths.shape:
            raise ValueError("expected tokens[B,N], weight[B,N], lengths[B], speakers[B]")
        if tokens.shape[1] > self.config.max_tokens or bool((lengths < 1).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("invalid token length")
        mask = length_mask(lengths, tokens.shape[1])
        x = (self.tokens(tokens) + self.syllable_weight(weight) + self.voices(speakers)[:, None]) * mask[..., None]
        return self.encoder(x, mask, self.gate(speakers)), mask

    def gate(self, speakers):
        return self.adapter_gate[speakers] if speakers is not None else None

    def expand(self, hidden, durations, token_mask, speakers):
        """Repeat token states by integer durations (zero is allowed for pause
        tokens) and add each frame's position inside its token."""
        if durations.shape != token_mask.shape or bool((durations < 0).any()) or bool((durations[~token_mask] != 0).any()):
            raise ValueError("invalid durations")
        lengths = durations.sum(1)
        if bool((lengths < 1).any()) or bool((lengths > self.config.max_frames).any()):
            raise ValueError("utterance has no frames or exceeds max_frames; split the input")
        size = int(lengths.max())
        ends = durations.cumsum(1)
        frame = torch.arange(size, device=hidden.device)[None].expand(hidden.shape[0], size).contiguous()
        index = torch.searchsorted(ends, frame, right=True).clamp(max=durations.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        mask = frame < lengths[:, None]
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + self.frame_position(features) + self.frame_voices(speakers)[:, None]
        return expanded * mask[..., None], mask, lengths

    def predict_prosody(self, frames, frame_mask, speakers=None):
        hidden = self.prosody_stack(frames, frame_mask, self.gate(speakers))
        return hidden, self.prosody_head(hidden) * frame_mask[..., None]

    @staticmethod
    def hard_prosody(prediction):
        """What the decoder is conditioned on when it is not teacher-forced."""
        return torch.stack((prediction[..., 0].clamp(-2.0, 1.5), (prediction[..., 1] > 0).to(prediction.dtype),
                            prediction[..., 2].clamp(-4.5, 3.0)), dim=-1)

    def decode(self, prosody_hidden, condition, frame_mask, speakers=None):
        return self.acoustic_stack((prosody_hidden + self.prosody_projection(condition)) * frame_mask[..., None], frame_mask,
                                   self.gate(speakers))

    def frequency(self, log_f0):
        c = self.config
        return (c.pitch_reference_hz * torch.exp(log_f0.float())).clamp(c.f0_min, c.f0_max)

    def predict_durations(self, hidden, mask, tokens, speed=1.0):
        output = self.duration(hidden, mask).float()
        frames = torch.round(torch.expm1(output[..., 0].clamp(0, math.log1p(400))) / speed).long().clamp_min(1)
        # Every phoneme is audible for at least one frame. A pause-capable token
        # (edge, punctuation, word boundary) lasts its predicted length when the
        # presence logit says there is a pause, and vanishes otherwise.
        present = self.phone_tokens[tokens] | (output[..., 1] > 0)
        return torch.where(present, frames, torch.zeros_like(frames)) * mask

    @staticmethod
    def widen_pitch(prosody, frame_mask, pitch_range):
        """Scale the predicted pitch movement around each utterance's voiced mean.
        Regression predicts a flatter contour than the recordings; values above 1
        restore range (1.0 leaves the prediction untouched)."""
        if pitch_range == 1.0:
            return prosody
        voiced = ((prosody[..., 1] > 0) & frame_mask).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    @staticmethod
    def pitch_texture(shape, polish, seed, device):
        """AR(2) pitch fluctuation for the oscillator (log-F0 units), fitted to InfoRe recordings (see ``Polish``)."""
        from scipy.signal import lfilter
        a1, a2, sigma = polish.texture_ar
        warmup = 64                                                     # start the process in its steady state
        # One stream per row (seed + row), so batching or padding never changes a row; row 0 = a single utterance.
        noise = torch.stack([torch.randn(shape[1] + warmup, generator=torch.Generator().manual_seed(seed + 104729 + row))
                             for row in range(shape[0])]).numpy()
        fluctuation = lfilter([sigma], [1.0, -a1, -a2], noise, axis=1)[:, warmup:]
        return torch.from_numpy(fluctuation * polish.texture).to(device, torch.float32)

    @staticmethod
    def pitch_wobble(shape, semitones, seed, device):
        """Small random pitch movement for the oscillator only (log-F0 units).
        Recorded pitch has cycle-to-cycle irregularity that a predicted contour
        lacks; two thirds of the wobble is slow drift, one third frame-level."""
        generator = torch.Generator().manual_seed(seed + 7919)
        fast = torch.randn(shape, generator=generator)
        slow = F.avg_pool1d(torch.randn(shape[0], 1, shape[1] + 16, generator=generator), 17, stride=1)[:, 0]
        slow = slow / slow.std().clamp_min(1e-6)
        return ((2 * slow + fast) / 5 ** 0.5 * semitones * math.log(2) / 12).to(device)

    def lengthen_pauses(self, tokens, durations, mask, pauses):
        """Post-processing at inference: punctuation lasts at least ``pauses[kind]`` ms ("comma" for , and the
        marks the front end writes as ,; "stop" for . ? !). Returns the new durations and the tokens that were
        lengthened (their frames are made silent after synthesis). Longer predicted pauses are kept."""
        c = self.config
        minimum = torch.zeros_like(durations)
        for kind, symbols in (("comma", (",",)), ("stop", (".", "?", "!"))):
            frames = round(float(pauses.get(kind) or 0) / 1000 * c.sample_rate / c.hop)
            if frames > 0:
                ids = torch.tensor([SYMBOL_TO_ID[s] for s in symbols], device=tokens.device)
                minimum = torch.where(torch.isin(tokens, ids), torch.full_like(durations, frames), minimum)
        lengthened = (minimum > durations) & mask
        return torch.where(lengthened, minimum, durations), lengthened

    @staticmethod
    def frame_owner(durations, size):
        """Index of the token each of ``size`` frames belongs to, [B, size]."""
        frame = torch.arange(size, device=durations.device)[None].expand(durations.shape[0], -1).contiguous()
        return torch.searchsorted(durations.cumsum(1), frame, right=True).clamp(max=durations.shape[1] - 1)

    @staticmethod
    def steady_pitch(prosody, frame_lengths, frame_mask, reach):
        """Triangular moving average of the predicted pitch, ``reach`` frames either side. The prediction flickers
        from frame to frame more than a voice does. Each row is extended with its own last value, so padding
        inside a batch never changes it."""
        last = (frame_lengths - 1).clamp_min(0)[:, None]
        pitch = prosody[..., 0].gather(1, torch.minimum(torch.arange(prosody.shape[1], device=prosody.device)[None], last))
        kernel = torch.cat((torch.arange(1, reach + 2), torch.arange(reach, 0, -1))).to(pitch)
        pitch = F.conv1d(F.pad(pitch[:, None], (reach, reach), mode="replicate"), (kernel / kernel.sum())[None, None])[:, 0]
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    def shaping(self, polish, tokens, durations, voiced, frame_lengths, seed):
        """Per-frame controls for the waveform head, from the sound each frame belongs to (see ``Polish``)."""
        kind = self.sound_kind[tokens].gather(1, self.frame_owner(durations, voiced.shape[1]))
        limits = torch.full(kind.shape, ALL_PULSES, device=kind.device)
        limits = torch.where(kind == 1, polish.open_limit, limits)
        limits = torch.where(kind == 2, polish.consonant_limit, limits)
        limits = torch.where(kind == 3, NO_PULSES, limits)
        limits = torch.where(kind == 4, torch.where(voiced, polish.consonant_limit, NO_PULSES), limits)
        limits = torch.where((kind == 5) & ~voiced, NO_PULSES, limits)
        vowel = self.vowel_tokens[tokens].gather(1, self.frame_owner(durations, voiced.shape[1]))
        breath = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=torch.Generator().manual_seed(seed + 7919 + row))
                                            for row, n in enumerate(frame_lengths)], batch_first=True)
        return Shaping(polish, limits, kind == 1, vowel, breath.to(limits.device))

    @torch.inference_mode()
    def synthesize(self, tokens, weight, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, polish=None, pauses=None):
        """Free-running synthesis: predicted durations, pitch, voicing and energy. ``polish`` (a ``Polish``) applies
        the corrections made while speaking; ``None`` gives the sound exactly as the network was trained to make it."""
        if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
            raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")
        hidden, mask = self.encode(tokens, weight, lengths, speakers)
        durations = self.predict_durations(hidden, mask, tokens, speed)
        lengthened = None
        if pauses:
            durations, lengthened = self.lengthen_pauses(tokens, durations, mask, pauses)
        frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers)
        prosody_hidden, prosody = self.predict_prosody(frames, frame_mask, speakers)
        silent = None
        if lengthened is not None and bool(lengthened.any()):
            silent = lengthened.gather(1, self.frame_owner(durations, frames.shape[1])) & frame_mask
            prosody = torch.cat((prosody[..., :1], torch.where(silent, -5.0, prosody[..., 1])[..., None],
                                 torch.where(silent, -4.0, prosody[..., 2])[..., None], prosody[..., 3:]), dim=-1)
        if polish is not None and polish.pitch_smooth:
            prosody = self.steady_pitch(prosody, frame_lengths, frame_mask, polish.pitch_smooth)
        if voiced_vowels:
            # A pitch tracker can miss creaky vowels in the recordings, and the
            # voicing predictor then learns to drop whole vowels (heard as a whispered
            # word). A vowel is voiced by definition, so its frames are kept voiced.
            ends = durations.cumsum(1)
            frame = torch.arange(frames.shape[1], device=tokens.device)[None].expand(len(lengths), -1).contiguous()
            owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
            vowel = self.vowel_tokens[tokens].gather(1, owner) & frame_mask
            prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                                 prosody[..., 2:]), dim=-1)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        acoustic = self.decode(prosody_hidden, self.hard_prosody(prosody), frame_mask, speakers)
        # One noise stream per row, drawn at the row's own length, so batching or
        # padding never changes a row's audio.
        noise = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=torch.Generator().manual_seed(seed + row))
                                           for row, n in enumerate(frame_lengths)], batch_first=True)
        pitch = prosody[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        if polish is not None and polish.texture > 0:
            pitch = pitch + self.pitch_texture(pitch.shape, polish, seed, pitch.device)
        shaping = None if polish is None else self.shaping(polish, tokens, durations, (prosody[..., 1] > 0) & frame_mask, frame_lengths, seed)
        wave = self.vocoder(acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device), shaping=shaping)
        if silent is not None:
            # Lengthened pauses are silence, faded over 10 ms at both ends.
            gain = 1 - silent.to(wave.dtype).repeat_interleave(self.config.hop, dim=1)[:, :wave.shape[1]]
            fade = int(0.01 * self.config.sample_rate) | 1
            gain = F.avg_pool1d(gain[:, None], fade, stride=1, padding=fade // 2, count_include_pad=False)[:, 0]
            wave = wave * gain[:, :wave.shape[1]]
        return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask}
