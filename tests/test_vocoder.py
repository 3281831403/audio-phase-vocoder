"""Behavioural specification for the phase vocoder exercise.

The tests are intentionally written against the public contract only.
They use deterministic signals: no randomness, no sleeping, no I/O.
"""

import cmath
import math

import pytest

import pvoc.vocoder as vocoder
from pvoc.vocoder import (
    consistency,
    instantaneous_freq,
    istft,
    phase_advance,
    pitch_shift,
    roundtrip_error,
    stft,
    stretch,
)

SR = 8000.0


def _tone(length, frequency, amplitude=1.0, phase=0.0):
    return [
        amplitude * math.sin(2.0 * math.pi * frequency * i / SR + phase)
        for i in range(length)
    ]


def _mixed(length):
    return [
        0.31 * math.sin(2.0 * math.pi * 7.0 * i / length + 0.2)
        + 0.17 * math.cos(2.0 * math.pi * 23.0 * i / length - 0.4)
        + 0.05 * math.sin(2.0 * math.pi * 61.0 * i / length)
        for i in range(length)
    ]


def _dominant_bin(signal, n_fft):
    """Peak non-negative DFT bin of the central ``n_fft`` samples."""
    start = (len(signal) - n_fft) // 2
    segment = signal[start : start + n_fft]
    window = [
        0.5 - 0.5 * math.cos(2.0 * math.pi * n / n_fft) for n in range(n_fft)
    ]
    best_bin = 0
    best_magnitude = -1.0
    for k in range(1, n_fft // 2 + 1):
        real = 0.0
        imag = 0.0
        for n in range(n_fft):
            angle = -2.0 * math.pi * k * n / n_fft
            value = segment[n] * window[n]
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        magnitude = real * real + imag * imag
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            best_bin = k
    return best_bin


def _zero_crossing_frequency(signal, skip):
    """Frequency estimate from upward zero crossings in the middle section."""
    segment = signal[skip : len(signal) - skip]
    crossings = 0
    for previous, current in zip(segment, segment[1:]):
        if previous < 0.0 <= current:
            crossings += 1
    duration = (len(segment) - 1) / SR
    return crossings / duration


def _phase_hash(t, k):
    raw = math.sin((t + 1) * 12.9898 + (k + 1) * 78.233) * 43758.5453
    return 2.0 * math.pi * (raw - math.floor(raw))


# ---------------------------------------------------------------------------
# The five cross-cutting tests: round trip, identity, pitch accuracy,
# phase coherence, and batch isolation.
# ---------------------------------------------------------------------------


def test_stft_istft_roundtrip_is_exact():
    n_fft, hop, length = 256, 64, 1024
    samples = _mixed(length)

    frames = stft(samples, n_fft, hop)
    reconstructed = istft(frames, n_fft, hop, length)

    assert len(reconstructed) == length
    maximum_error = max(
        abs(original - restored)
        for original, restored in zip(samples, reconstructed)
    )
    assert maximum_error < 1e-9
    input_energy = sum(value * value for value in samples)
    output_energy = sum(value * value for value in reconstructed)
    assert output_energy == pytest.approx(input_energy, rel=1e-9, abs=1e-12)


def test_stretch_identity_at_rate_one():
    n_fft, hop, length = 128, 32, 512
    samples = _mixed(length)

    output = stretch(samples, 1.0, n_fft, hop)

    assert len(output) == length
    assert max(
        abs(original - restored) for original, restored in zip(samples, output)
    ) < 1e-9


def test_pitch_shift_doubles_sine_frequency():
    n_fft, hop, length = 1024, 256, 4096
    samples = _tone(length, 440.0)

    output = pitch_shift(samples, 2.0, n_fft, hop)

    assert len(output) == length
    peak_bin = _dominant_bin(output, n_fft)
    expected_bin = 2.0 * 440.0 / (SR / n_fft)
    assert abs(peak_bin - expected_bin) <= 1.0


def test_consistency_coherent_near_one_random_near_zero():
    n_fft, n_frames = 32, 8

    coherent = [
        [
            cmath.rect(1.0, 2.0 * math.pi * (k + 1) * t / 16.0)
            for k in range(n_fft)
        ]
        for t in range(n_frames)
    ]
    coherent_score = consistency(coherent)
    assert coherent_score > 0.99
    assert coherent_score <= 1.0

    random_phase = [
        [cmath.rect(1.0, _phase_hash(t, k)) for k in range(n_fft)]
        for t in range(n_frames)
    ]
    random_score = consistency(random_phase)
    assert 0.0 <= random_score < 0.15


def test_batch_roundtrip_order_and_isolation():
    n_fft, hop = 128, 32
    cases = [
        (257, _mixed(257)),
        (512, _tone(512, 123.0)),
        (1000, [((i % 17) - 8) / 8.0 for i in range(1000)]),
    ]

    forward = [
        roundtrip_error(samples, n_fft, hop) for _, samples in cases
    ]
    backward = [
        roundtrip_error(samples, n_fft, hop)
        for _, samples in reversed(cases)
    ]

    assert forward == pytest.approx(list(reversed(backward)), abs=1e-12)
    for error in forward:
        assert error < 1e-9
    for expected_length, samples in cases:
        reconstructed = istft(stft(samples, n_fft, hop), n_fft, hop, expected_length)
        assert len(reconstructed) == expected_length


# ---------------------------------------------------------------------------
# One test for each of the ten hidden invariant defects.
# ---------------------------------------------------------------------------


def test_istft_uses_per_sample_window_sum():
    # Defect 1: a fixed window-sum denominator scales overlap regions by
    # different amounts, so a unit impulse is neither unit-height nor
    # isolated after reconstruction.
    n_fft, hop, length, impulse_at = 64, 16, 256, 37
    samples = [0.0] * length
    samples[impulse_at] = 1.0

    reconstructed = istft(stft(samples, n_fft, hop), n_fft, hop, length)

    assert reconstructed[impulse_at] == pytest.approx(1.0, abs=1e-9)
    for index, value in enumerate(reconstructed):
        if index != impulse_at:
            assert abs(value) < 1e-9
    assert sum(value * value for value in reconstructed) == pytest.approx(
        1.0, abs=1e-9
    )


def test_instantaneous_freq_unwraps_multiple_turns():
    # Defect 2: wrapping the raw phase difference aliases bins whose phase
    # advances by more than pi during one hop.
    n_fft, hop = 32, 5
    previous = [0.0] * n_fft
    phase = [
        (2.0 * math.pi * k * hop / n_fft) % (2.0 * math.pi)
        for k in range(n_fft)
    ]

    frequencies = instantaneous_freq(previous, phase, hop, n_fft)

    for k in range(n_fft):
        assert frequencies[k] == pytest.approx(
            2.0 * math.pi * k / n_fft, abs=1e-9
        )


def test_phase_advance_accumulates_without_modulo():
    # Defect 3: reducing the accumulator each call loses whole turns;
    # 1000 steps of one radian must remain 1000 radians.
    phase = [0.0]
    for _ in range(1000):
        phase = phase_advance([1.0], 1, 8, phase)
    assert phase[0] == pytest.approx(1000.0, abs=1e-9)

    advanced = phase_advance([2.0, -1.0], 3, 8, [3.0, 0.0])
    assert advanced[0] == pytest.approx(9.0, abs=1e-9)
    assert advanced[1] == pytest.approx(-3.0, abs=1e-9)


def test_stretch_length_has_no_cumulative_drift():
    # Defect 4: floor-per-call length arithmetic compounds.  Stretching by
    # e and then by 1/e must return to the original duration, not lose 3
    # samples.
    n_fft, hop, length = 128, 32, 1000
    samples = _mixed(length)

    stretched = stretch(samples, math.e, n_fft, hop)
    assert len(stretched) == int(round(length / math.e))

    restored = stretch(stretched, 1.0 / math.e, n_fft, hop)
    assert len(restored) == int(round(len(stretched) * math.e))
    assert abs(len(restored) - length) <= 1


def test_pitch_shift_compensates_bin_frequency():
    # Defect 5: after bin remapping, resampling without using the
    # destination-bin frequency leaves a static pitch offset.  A zero
    # crossing estimate resolves a fraction of an FFT bin here.
    n_fft, hop, length = 2048, 512, 8192
    source_frequency = 437.0
    ratio = 1.5
    samples = _tone(length, source_frequency)

    output = pitch_shift(samples, ratio, n_fft, hop)

    assert len(output) == length
    measured = _zero_crossing_frequency(output, skip=2048)
    assert measured == pytest.approx(source_frequency * ratio, abs=2.0)


def test_stft_window_is_periodic():
    # Defect 6: the symmetric Hann window differs from the periodic one at
    # n = 3*n_fft/4.  Frame 1 starts at -16, so the impulse at sample 32
    # lands at frame index 48 and its bin-0 coefficient exposes w[48].
    n_fft, hop, length, impulse_at = 64, 16, 256, 32
    samples = [0.0] * length
    samples[impulse_at] = 1.0

    frames = stft(samples, n_fft, hop)

    assert frames[1][0].real == pytest.approx(0.5, abs=1e-12)
    assert frames[1][0].imag == pytest.approx(0.0, abs=1e-12)


def test_consistency_is_not_absolute_phase_difference_sum():
    # Defect 7: alternating 0, pi/2 phases have large absolute adjacent
    # differences but a perfectly inconsistent second difference (cos = -1).
    n_fft, n_frames = 32, 8
    frames = []
    for t in range(n_frames):
        phase = math.pi / 2.0 if t % 2 else 0.0
        frames.append([cmath.rect(1.0, phase) for _ in range(n_fft)])

    score = consistency(frames)

    assert 0.0 <= score < 0.1


def test_roundtrip_error_checks_entire_signal(monkeypatch):
    # Defect 8: comparing only the first hop hides an error placed late.
    length, n_fft, hop = 64, 16, 8
    fake_frames = [[0j] * n_fft]

    def fake_stft(samples, frame_length, frame_hop):
        return fake_frames

    def fake_istft(frames, frame_length, frame_hop, output_length):
        output = [0.0] * output_length
        output[-2] = 0.75
        return output

    monkeypatch.setattr(vocoder, "stft", fake_stft)
    monkeypatch.setattr(vocoder, "istft", fake_istft)

    error = vocoder.roundtrip_error([0.0] * length, n_fft, hop)

    assert error == pytest.approx(0.75, abs=1e-12)


def test_istft_is_aligned_to_sample_zero():
    # Defect 9: trimming a centred-window buffer only at the end shifts the
    # whole signal by half a window.  The impulse must stay at sample 100.
    n_fft, hop, length, impulse_at = 64, 16, 256, 100
    samples = [0.0] * length
    samples[impulse_at] = 1.0

    reconstructed = istft(stft(samples, n_fft, hop), n_fft, hop, length)

    peak_index = max(range(length), key=lambda i: abs(reconstructed[i]))
    assert peak_index == impulse_at
    assert reconstructed[impulse_at] == pytest.approx(1.0, abs=1e-9)
    assert abs(reconstructed[0]) < 1e-9


def test_silent_bins_do_not_advance_phase():
    # Defect 10: phase 0 in both frames is the documented silent-bin
    # sentinel; it must not be turned into the bin-centre oscillator.
    n_fft, hop = 32, 8
    silence = [0.0] * n_fft

    frequencies = instantaneous_freq(silence, silence, hop, n_fft)

    assert frequencies == pytest.approx([0.0] * n_fft, abs=1e-12)
    advanced = phase_advance(frequencies, hop, n_fft, silence)
    assert advanced == pytest.approx([0.0] * n_fft, abs=1e-12)
