"""Phase-vocoder primitives used by the exercise.

All routines are plain Python and use only :mod:`math` / :mod:`cmath`.

Global conventions
------------------
* A frame length ``n_fft`` is even and positive.
* A hop satisfies ``1 <= hop <= n_fft // 2`` so that the periodic Hann
  window has a constant overlap-add denominator away from the boundaries.
* The analysis/synthesis window is the **periodic** Hann window::

      w[n] = 0.5 - 0.5 * cos(2 * pi * n / n_fft),   0 <= n < n_fft

  It is deliberately not the symmetric Hann window that uses
  ``n_fft - 1`` in the denominator.  The half-sample difference matters:
  with the symmetric window, the overlap-add sum is not constant from
  sample to sample even when the hop divides ``n_fft``.
* Frames are centred, not left-aligned.  With ``half = n_fft // 2``,
  frame ``t`` starts at absolute sample ``t * hop - half``.  Samples
  outside ``[0, length)`` are zero.  Sample 0 is therefore the centre
  of frame 0, and :func:`istft` returns absolute sample 0 as output
  sample 0 (no extra half-window shift).
* Angular frequencies are measured in radians per sample, so the bin
  centre for bin ``k`` is ``2 * pi * k / n_fft``.
* Phase values returned by :func:`phase_advance` are **unwrapped
  accumulators**.  Only :func:`instantaneous_freq` performs modular
  reduction, and it does so on the heterodyned residual, never on the
  full phase.
* The reconstruction tolerance for the round-trip tests is ``1e-9``
  (maximum absolute error), although a correct implementation normally
  reaches machine precision.

Every public function currently raises :class:`NotImplementedError`.
Only this module is meant to be edited; ``tests/`` is the specification.
"""

from __future__ import annotations

import cmath
import math

_TWIDDLE_CACHE = {}


def _twiddles(n, inverse):
    """Cached table of the first ``n // 2`` roots of unity for length ``n``."""
    key = (n, inverse)
    table = _TWIDDLE_CACHE.get(key)
    if table is None:
        sign = 1.0 if inverse else -1.0
        table = [
            cmath.exp(complex(0.0, sign * 2.0 * math.pi * k / n))
            for k in range(n // 2)
        ]
        _TWIDDLE_CACHE[key] = table
    return table


def _fft_pow2(values, inverse):
    """Iterative radix-2 Cooley-Tukey transform of a power-of-two length."""
    n = len(values)
    if n == 1:
        return [complex(values[0])]
    out = [complex(value) for value in values]

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            out[i], out[j] = out[j], out[i]

    twiddle = _twiddles(n, inverse)
    size = 2
    while size <= n:
        half = size // 2
        step = n // size
        for start in range(0, n, size):
            for k in range(half):
                w = twiddle[k * step]
                u = out[start + k]
                v = out[start + k + half] * w
                out[start + k] = u + v
                out[start + k + half] = u - v
        size <<= 1

    if inverse:
        scale = 1.0 / n
        out = [value * scale for value in out]
    return out


def _fft_bluestein(values, inverse):
    """Chirp-z (Bluestein) transform for arbitrary lengths."""
    n = len(values)
    if n == 1:
        return [complex(values[0])]
    sign = 1.0 if inverse else -1.0
    size = 1
    while size < 2 * n - 1:
        size <<= 1

    modulus = 2 * n
    chirp = [
        cmath.exp(complex(0.0, sign * math.pi * ((j * j) % modulus) / n))
        for j in range(n)
    ]
    left = [values[j] * chirp[j] for j in range(n)] + [0j] * (size - n)
    right = [0j] * size
    for j in range(n):
        value = chirp[j].conjugate()
        right[j] = value
        if j:
            right[size - j] = value

    left_spectrum = _fft_pow2(left, False)
    right_spectrum = _fft_pow2(right, False)
    product = [left_spectrum[i] * right_spectrum[i] for i in range(size)]
    convolved = _fft_pow2(product, True)

    result = [chirp[k] * convolved[k] for k in range(n)]
    if inverse:
        result = [value / n for value in result]
    return result


def _fft(values, inverse):
    """Discrete Fourier transform with the ``exp(-2j*pi*k*n/N)`` convention."""
    n = len(values)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_pow2(values, inverse)
    return _fft_bluestein(values, inverse)


def _periodic_hann(n_fft):
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / n_fft) for n in range(n_fft)]


def stft(samples, n_fft, hop):
    """Compute a centred short-time Fourier transform with a periodic Hann window.

    Parameters
    ----------
    samples:
        Sequence of real samples; let ``length = len(samples)``.
    n_fft:
        Even, positive FFT/frame length.
    hop:
        Positive frame hop, at most ``n_fft // 2``.

    Returns
    -------
    list[list[complex]]
        ``m`` frames, each containing exactly ``n_fft`` complex bins
        (the full spectrum, not just the non-negative half).

    Exact contract
    --------------
    Let ``half = n_fft // 2`` and let ``w`` be the periodic Hann window
    above.  The number of frames is::

        m = 0                         if length == 0
        m = ceil(length / hop)        otherwise

    Frame ``t`` (0-based) starts at absolute sample ``t * hop - half``.
    Its ``n``-th sample is ``samples[t * hop - half + n]`` when that
    index lies in ``[0, length)`` and ``0.0`` otherwise.  For
    ``0 <= k < n_fft``::

        X[t][k] = sum_{n=0}^{n_fft-1}
                    w[n] * frame_t[n] * exp(-2j * pi * k * n / n_fft)

    In particular, the bin-0 coefficient of a frame containing an
    isolated unit impulse equals the window value at that impulse
    position; that property is how the periodic-vs-symmetric window
    invariant is observed.

    The analysis must be exactly invertible by :func:`istft` (maximum
    absolute reconstruction error below ``1e-9``) for every signal,
    including impulses, constants, and signals shorter than one frame.
    """
    length = len(samples)
    if length == 0:
        return []

    half = n_fft // 2
    window = _periodic_hann(n_fft)
    frame_count = (length + hop - 1) // hop

    result = []
    for t in range(frame_count):
        start = t * hop - half
        frame = [0.0] * n_fft
        lo = max(0, -start)
        hi = min(n_fft, length - start)
        for n in range(lo, hi):
            frame[n] = samples[start + n] * window[n]
        result.append(_fft(frame, inverse=False))
    return result


def istft(frames, n_fft, hop, length):
    """Invert :func:`stft` by overlap-add with per-sample window normalisation.

    Parameters
    ----------
    frames:
        Sequence of complex spectra, each of length ``n_fft``, using the
        same centred-frame convention as :func:`stft`.
    n_fft:
        Frame length used by the analysis.
    hop:
        Frame hop used by the analysis.
    length:
        Exact number of output samples required.

    Returns
    -------
    list[float]
        Exactly ``length`` real samples.  Output index ``i`` is absolute
        sample ``i``; it is not the start of an internally padded
        overlap-add buffer and must not be shifted by ``n_fft // 2``.

    Exact contract
    --------------
    For each frame ``t``, take the full inverse DFT::

        y_t[n] = (1 / n_fft) * sum_{k=0}^{n_fft-1}
                    frames[t][k] * exp(+2j * pi * k * n / n_fft)

    multiply by the same periodic Hann window ``w[n]``, and add it at
    absolute sample ``t * hop - half + n``.  At the same time accumulate
    the **per-sample** denominator ``sum_t w[n] ** 2`` over exactly the
    frames that cover that absolute sample.  The final sample is::

        out[i] = numerator[i] / denominator[i]      if denominator[i] > 1e-12
        out[i] = 0.0                                otherwise

    A single scalar (fixed) window sum is incorrect: it makes the
    amplitude in overlap regions depend on ``hop`` and destroys energy
    conservation.  The denominator has to be accumulated index by index.

    The returned list is truncated or zero-padded to exactly ``length``,
    taking absolute samples starting at index 0.  Therefore
    ``istft(stft(x, n_fft, hop), n_fft, hop, len(x))`` reproduces ``x``
    to within ``1e-9`` at every sample.
    """
    half = n_fft // 2
    window = _periodic_hann(n_fft)
    frame_count = len(frames)

    buffer_length = 0
    if frame_count:
        last_start = (frame_count - 1) * hop - half
        buffer_length = max(buffer_length, last_start + n_fft)
    buffer_length = max(buffer_length, length)
    if buffer_length <= 0:
        return []

    numerator = [0.0] * buffer_length
    denominator = [0.0] * buffer_length

    for t, spectrum in enumerate(frames):
        frame = _fft(spectrum, inverse=True)
        start = t * hop - half
        lo = max(0, -start)
        hi = min(n_fft, buffer_length - start)
        for n in range(lo, hi):
            index = start + n
            weight = window[n]
            numerator[index] += frame[n].real * weight
            denominator[index] += weight * weight

    output = []
    for i in range(length):
        if denominator[i] > 1e-12:
            output.append(numerator[i] / denominator[i])
        else:
            output.append(0.0)
    return output


def instantaneous_freq(prev_phase, phase, hop, n_fft):
    """Estimate per-bin angular frequency in radians/sample.

    Parameters
    ----------
    prev_phase, phase:
        Equal-length sequences of phase values for one previous frame
        and the current frame.  Length is normally ``n_fft``.
    hop:
        Number of samples between the two frames.
    n_fft:
        FFT length that defines the bin centres.

    Returns
    -------
    list[float]
        ``omega[k]`` in radians per sample.  The angle is not restricted
        to the principal interval, so a bin may report a frequency above
        ``pi`` (up to the Nyquist limit handled by the caller).

    Exact contract
    --------------
    For each bin ``k``, remove the expected bin-centre advance before
    unwrapping the phase residual::

        expected = 2 * pi * k * hop / n_fft
        residual = wrap_to_pi(phase[k] - prev_phase[k] - expected)
        omega[k] = 2 * pi * k / n_fft + residual / hop

    ``wrap_to_pi`` maps to ``[-pi, pi)`` (for example with ``atan2`` of
    the sine and cosine of the residual).  Wrapping the *raw* phase
    difference instead is the classic bug: a bin whose phase advances by
    more than ``pi`` in one hop is then assigned a frequency that is off
    by ``2 * pi * m / hop``.  Subtracting ``expected`` first removes the
    known number of full turns and makes the residual unambiguous for
    frequencies within ``pi / hop`` of the bin centre.

    Silence convention: this signature carries no magnitudes, so a
    zero-magnitude bin is represented by the sentinel pair
    ``prev_phase[k] == 0.0`` and ``phase[k] == 0.0``.  Such a bin must
    return ``0.0`` (undefined/no oscillation), not the bin-centre
    frequency.  Treating it as a normal phase difference injects a
    spurious oscillator into silent passages and advances synthesis
    phase incorrectly.

    Precision target: ``1e-9`` absolute for inputs that obey the
    ``pi / hop`` residual bound.
    """
    result = []
    for k in range(len(phase)):
        previous = prev_phase[k]
        current = phase[k]
        if previous == 0.0 and current == 0.0:
            result.append(0.0)
            continue
        expected = 2.0 * math.pi * k * hop / n_fft
        difference = current - previous - expected
        residual = math.atan2(math.sin(difference), math.cos(difference))
        result.append(2.0 * math.pi * k / n_fft + residual / hop)
    return result


def phase_advance(freqs, hop, n_fft, prev_phase):
    """Integrate angular frequencies into an unwrapped phase accumulator.

    Parameters
    ----------
    freqs:
        Per-bin angular frequencies in radians per sample, normally the
        output of :func:`instantaneous_freq`.
    hop:
        Number of samples to advance.
    n_fft:
        FFT length (accepted for API symmetry with the bin-frequency
        convention; it must not be used to wrap the returned phase).
    prev_phase:
        Per-bin previous phase.  These values are unwrapped accumulators,
        not necessarily in ``[-pi, pi)``.

    Returns
    -------
    list[float]
        ``prev_phase[k] + freqs[k] * hop`` for every bin, with **no**
        modulo operation applied to the result.

    Exact contract
    --------------
    Phase continuity across arbitrarily many frames depends on keeping
    the accumulator unwrapped.  If this function reduces every result
    into ``(-pi, pi]``, the fractional part is lost and repeated calls
    accumulate a drift of ``2 * pi * m``; in a phase vocoder that shows
    up as progressive detuning and phase discontinuities at frame
    boundaries.  Callers may reduce the phase only when evaluating a
    sine/cosine, never when storing the accumulator.

    A frequency of ``0.0`` (for example from a silent bin) leaves the
    corresponding previous phase unchanged.  Precision target: ``1e-9``
    absolute after at least 1000 successive calls.
    """
    return [prev_phase[k] + freqs[k] * hop for k in range(len(freqs))]


def stretch(samples, rate, n_fft, hop):
    """Time-stretch without changing pitch.

    Parameters
    ----------
    samples:
        Input real signal of length ``length``.
    rate:
        Positive speed factor.  A value greater than 1 makes the signal
        shorter; a value smaller than 1 makes it longer.
    n_fft, hop:
        Analysis frame length and hop, following the module conventions.

    Returns
    -------
    list[float]
        A real signal of exactly ``round(length / rate)`` samples
        (Python's nearest-integer rounding; for ``length == 0`` the
        result is empty).  This exact nearest-integer length matters:
        repeatedly truncating with ``int(...)`` lets an irrational rate
        accumulate more than one sample of duration error.  The
        conventional "within one sample of ``length / rate``" tolerance
        is implied, but the implementation should target the nearest
        integer, not the floor.

    Musical contract
    ----------------
    * ``rate == 1`` returns a sample-for-sample copy of the input,
      maximum absolute difference ``< 1e-9`` (no edge transient, no
      phase-vocoder colouration).
    * For other rates the duration changes but every sinusoidal partial
      keeps its original frequency within ``1e-9`` (or, in a spectral
      test, within one FFT bin).  This is time stretching, not
      resampling.

    Implementation sketch
    ---------------------
    Use :func:`stft`, :func:`instantaneous_freq` and
    :func:`phase_advance` with an unwrapped phase accumulator.  Advance
    the synthesis read position by ``hop / rate`` samples per output
    frame (a fractional accumulator is fine) and call :func:`istft`
    with the exact output length.  The length must be decided from the
    input length and ``rate`` as a whole, never by summing per-frame
    floors.
    """
    length = len(samples)
    if length == 0:
        return []

    output_length = int(round(length / rate))
    if output_length <= 0:
        return []

    # The identity case must be exact: a phase vocoder always colours the
    # signal slightly, so copy the samples instead of resynthesising them.
    if rate == 1.0 and output_length == length:
        return [float(value) for value in samples]

    analysis = stft(samples, n_fft, hop)
    analysis_count = len(analysis)
    if analysis_count == 0:
        return [0.0] * output_length

    magnitudes = []
    phases = []
    for spectrum in analysis:
        magnitudes.append([abs(value) for value in spectrum])
        phases.append([cmath.phase(value) for value in spectrum])

    bin_centres = [2.0 * math.pi * k / n_fft for k in range(n_fft)]

    synthesis_count = (output_length + hop - 1) // hop
    if synthesis_count <= 0:
        synthesis_count = 1

    accumulated = list(phases[0])
    synthesis = []
    for s in range(synthesis_count):
        position = s * rate
        index = int(math.floor(position))
        if index > analysis_count - 1:
            index = analysis_count - 1
        if index < 0:
            index = 0

        frame_magnitude = magnitudes[index]
        if s == 0:
            synth_phase = list(accumulated)
        else:
            if index > 0:
                freqs = instantaneous_freq(
                    phases[index - 1], phases[index], hop, n_fft
                )
            else:
                freqs = list(bin_centres)
            accumulated = phase_advance(freqs, hop, n_fft, accumulated)
            synth_phase = accumulated

        synthesis.append(
            [
                frame_magnitude[k] * cmath.exp(complex(0.0, synth_phase[k]))
                for k in range(n_fft)
            ]
        )

    return istft(synthesis, n_fft, hop, output_length)


def pitch_shift(samples, ratio, n_fft, hop):
    """Shift pitch while preserving duration.

    Parameters
    ----------
    samples:
        Input real signal of length ``length``.
    ratio:
        Positive frequency multiplier.  ``2.0`` shifts everything up one
        octave; ``0.5`` shifts down one octave.
    n_fft, hop:
        Analysis frame length and hop.

    Returns
    -------
    list[float]
        Exactly ``length`` real samples: the duration never changes.

    Musical contract
    ----------------
    * A pure sinusoid at frequency ``f`` becomes a pure sinusoid at
      ``ratio * f``, within one FFT bin in a peak-bin test; a more
      precise test may require a small fraction of a bin.
    * ``ratio == 1`` reproduces the input within ``1e-9``.
    * The output length is exactly ``len(samples)``.

    Bin-frequency compensation
    --------------------------
    Moving energy from source bin ``k`` to destination bin
    ``k_dst`` is not enough.  The oscillator/phase advance for the moved
    partial must use the destination bin's angular frequency
    ``2 * pi * k_dst / n_fft`` (plus its measured residual), not the
    source bin's frequency.  Resampling the result at the original
    sample rate without that compensation leaves a static frequency
    offset of roughly one bin, so the measured pitch ratio is wrong even
    though the STFT looks plausible.  A typical correct construction is
    a phase-vocoder time stretch followed by resampling back to
    ``length``, with the resampling ratio applied to the already
    bin-compensated partials.
    """
    length = len(samples)
    if length == 0:
        return []

    if ratio == 1.0:
        return [float(value) for value in samples]

    # Stretch by ``1 / ratio``: the partials keep their original
    # frequencies while the duration becomes ``length * ratio``.  Playing
    # that buffer back at the original rate (plain resampling) then
    # multiplies every frequency by ``ratio`` and restores the duration.
    # Because the stretch already placed each partial on the correct
    # (destination) bin, the resampling cannot introduce a static
    # bin-sized offset.
    stretched = stretch(samples, 1.0 / ratio, n_fft, hop)
    if not stretched:
        return [0.0] * length
    return _resample(stretched, length)


def _resample(values, output_length):
    """Band-limited (Hann-windowed sinc) resampling to ``output_length``."""
    input_length = len(values)
    if output_length <= 0:
        return []
    if input_length == 0:
        return [0.0] * output_length
    if output_length == input_length:
        return [float(value) for value in values]

    step = input_length / output_length
    # When decimating (``step > 1``) the reconstruction filter has to be
    # narrowed to the target Nyquist so stretched high partials do not fold
    # back as aliases.
    cutoff = 1.0 if step <= 1.0 else 1.0 / step
    taps = 16
    result = []
    for i in range(output_length):
        position = i * step
        centre = int(math.floor(position))
        total = 0.0
        for j in range(centre - taps + 1, centre + taps + 1):
            if j < 0 or j >= input_length:
                continue
            distance = position - j
            if abs(distance) < 1e-12:
                kernel = cutoff
            else:
                if abs(distance) >= taps:
                    continue
                argument = math.pi * cutoff * distance
                kernel = (
                    math.sin(argument)
                    / argument
                    * cutoff
                    * (0.5 + 0.5 * math.cos(math.pi * distance / taps))
                )
            total += values[j] * kernel
        result.append(total)
    return result


def consistency(frames):
    """Measure phase-progression consistency of a complex STFT.

    Parameters
    ----------
    frames:
        Sequence of complex spectra (``m`` frames by ``n_fft`` bins).

    Returns
    -------
    float
        A score in ``[0.0, 1.0]``.  ``1.0`` means every usable bin has a
        perfectly linear phase progression across frames; ``0.0`` means
        no coherent progression (or too few frames to judge).

    Exact contract
    --------------
    A constant-frequency partial has phase that is linear in frame
    index, so its circular second difference is zero::

        residual = wrap_to_pi(arg(c) - 2 * arg(b) + arg(a))

    for three consecutive frames ``a, b, c``.  Average
    ``cos(residual)`` over all interior frames and all bins whose three
    magnitudes exceed ``1e-12``, weighting each term by
    ``min(|a|, |b|, |c|)``::

        score = max(0.0, weighted_mean(cos(residual)))

    Return ``0.0`` when ``m < 3`` or no bin is usable.  Clamp the result
    to ``[0.0, 1.0]``.

    The score must not be a sum of absolute adjacent phase differences:
    such a sum is large for arbitrary phases and cannot distinguish a
    coherent ramp from noise.  It also must not use signed first
    differences, which can cancel when phases alternate.  Only the
    circular second difference tests that the *phase increment itself*
    (the instantaneous frequency) is consistent from frame to frame.

    Deterministic random phases should score near 0, while a linear
    phase ramp (even one that wraps through many multiples of ``2*pi``)
    should score near 1.  Precision target: ``1e-9`` for constructed
    coherent frames.
    """
    frame_count = len(frames)
    if frame_count < 3:
        return 0.0

    total = 0.0
    weight_total = 0.0
    for t in range(1, frame_count - 1):
        first = frames[t - 1]
        second = frames[t]
        third = frames[t + 1]
        for k in range(len(second)):
            a = first[k]
            b = second[k]
            c = third[k]
            magnitude_a = abs(a)
            magnitude_b = abs(b)
            magnitude_c = abs(c)
            if (
                magnitude_a <= 1e-12
                or magnitude_b <= 1e-12
                or magnitude_c <= 1e-12
            ):
                continue
            residual = cmath.phase(c) - 2.0 * cmath.phase(b) + cmath.phase(a)
            residual = math.atan2(math.sin(residual), math.cos(residual))
            weight = min(magnitude_a, magnitude_b, magnitude_c)
            total += weight * math.cos(residual)
            weight_total += weight

    if weight_total <= 0.0:
        return 0.0

    score = total / weight_total
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def roundtrip_error(samples, n_fft, hop):
    """Return the worst absolute stft->istft reconstruction error.

    Parameters
    ----------
    samples:
        Input real signal of length ``length``.
    n_fft, hop:
        Frame length and hop.

    Returns
    -------
    float
        ``max(abs(samples[i] - reconstructed[i]))`` over **all**
        ``0 <= i < length`` samples, or ``0.0`` for an empty input.

    Exact contract
    --------------
    The function calls the module-level :func:`stft` and :func:`istft`
    (so that both stages use the same documented conventions) and
    compares every sample, not merely the first ``hop`` samples.  Errors
    in overlap regions are exactly the ones this metric exists to catch;
    restricting the comparison to the first hop can hide a broken
    overlap-add or a half-window alignment shift.

    A correct implementation of the other functions makes this value
    below ``1e-9`` (normally at machine precision).  When a
    reconstruction error is deliberately injected late in the signal,
    this function must return that error, not zero.
    """
    length = len(samples)
    if length == 0:
        return 0.0

    reconstructed = istft(stft(samples, n_fft, hop), n_fft, hop, length)
    error = 0.0
    for original, restored in zip(samples, reconstructed):
        difference = abs(original - restored)
        if difference > error:
            error = difference
    return error
