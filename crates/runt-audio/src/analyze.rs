//! Measuring a rendered buffer, because nobody can listen on a CI box.
//!
//! Ported from `spikes/audio/native/src/main.rs`'s `analyze` command, which
//! existed for exactly this reason and is quoted in FINDINGS:
//!
//! > *`native -- analyze` exists because nobody can listen on this box: it
//! > measures the rendered buffer directly.*
//!
//! These are blunt instruments — a Goertzel comb rather than an FFT, normalized
//! autocorrelation rather than a real pitch tracker. That is the right trade: an
//! assertion needs a number that moves **monotonically** with the parameter it is
//! testing, not a number that is right to four decimal places. `tests/patches.rs`
//! turns each of these into a claim about the DSP that would fail loudly if a
//! filter stopped filtering or an envelope stopped decaying.
//!
//! Nothing here needs the `dsp` feature: it is arithmetic over a `&[f32]`, and
//! keeping it out of the feature gate means a test can measure a buffer that
//! came from anywhere.

/// Goertzel magnitude at `freq` over a mono buffer.
///
/// One bin of a DFT for the price of a two-tap recurrence — which is what makes
/// the 96-probe [`spectral_centroid`] below affordable in a debug-build test.
pub fn goertzel(x: &[f32], sample_rate: f32, freq: f32) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let k = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let coeff = 2.0 * k.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / x.len() as f32
}

/// Interleaved stereo → mono.
pub fn to_mono(stereo: &[f32]) -> Vec<f32> {
    stereo.chunks_exact(2).map(|f| (f[0] + f[1]) * 0.5).collect()
}

/// Interleaved stereo → one channel (`0` left, `1` right).
pub fn channel(stereo: &[f32], index: usize) -> Vec<f32> {
    stereo.chunks_exact(2).map(|f| f[index]).collect()
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// Energy-weighted mean frequency over 96 log-spaced probes from 30 Hz to 8 kHz.
///
/// Moves monotonically with filter cutoff, which is the whole requirement. It is
/// *not* a calibrated spectral centroid and the absolute number means little —
/// only its ordering does.
pub fn spectral_centroid(x: &[f32], sample_rate: f32) -> f32 {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for i in 0..96 {
        let f = 30.0 * (8000.0f32 / 30.0).powf(i as f32 / 95.0);
        let m = goertzel(x, sample_rate, f);
        num += f * m;
        den += m;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Total magnitude in `[lo_hz, hi_hz)`, summed over the same log-spaced probe
/// comb [`spectral_centroid`] uses.
///
/// A ratio of two bands is a far sharper filter-cutoff probe than a centroid is:
/// a saw's harmonic amplitudes fall as `1/n`, so most of its *energy* stays in
/// the bottom octave no matter where the cutoff sits, and the centroid barely
/// moves. The high/low ratio moves by an order of magnitude over the same sweep.
pub fn band_energy(x: &[f32], sample_rate: f32, lo_hz: f32, hi_hz: f32) -> f32 {
    let mut total = 0.0f32;
    for i in 0..96 {
        let f = 30.0 * (8000.0f32 / 30.0).powf(i as f32 / 95.0);
        if f >= lo_hz && f < hi_hz {
            total += goertzel(x, sample_rate, f);
        }
    }
    total
}

/// How much of the signal lives above 1 kHz relative to below it. Unitless,
/// monotone in filter cutoff, and the thing a "did the filter open" assertion
/// actually wants.
pub fn brightness(x: &[f32], sample_rate: f32) -> f32 {
    let low = band_energy(x, sample_rate, 30.0, 1000.0);
    let high = band_energy(x, sample_rate, 1000.0, 9000.0);
    high / low.max(1e-12)
}

/// Normalized-autocorrelation fundamental, in Hz.
///
/// Autocorrelation rather than spectral peak-picking because a resonant lowpass
/// can easily make a harmonic louder than the fundamental — the spike hit
/// exactly that and recorded it in FINDINGS. Period-domain estimation does not
/// care which partial is loudest.
///
/// ## The octave trap
///
/// Plain "strongest correlation wins" reports a **sub**harmonic whenever one
/// exists, because a signal that repeats every `L` samples also repeats every
/// `2L`, and the longer lag can score marginally higher through windowing luck.
/// So after picking the best lag, shorter lags that divide it are re-checked and
/// the *shortest* one still correlating within [`OCTAVE_TOLERANCE`] of the best
/// wins. This is the standard fix and without it an 880 Hz note reads as 440.
pub fn detect_pitch(x: &[f32], sample_rate: f32, min_hz: f32, max_hz: f32) -> f32 {
    /// How close a sub-multiple lag has to come to the best correlation before
    /// it is believed over it.
    const OCTAVE_TOLERANCE: f32 = 0.85;

    let min_lag = ((sample_rate / max_hz) as usize).max(1);
    let max_lag = ((sample_rate / min_hz) as usize).min(x.len() / 2);
    if max_lag <= min_lag || x.is_empty() {
        return 0.0;
    }
    let n = x.len() - max_lag;
    let energy: f32 = x[..n].iter().map(|v| v * v).sum();

    let correlation = |lag: usize| {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..n {
            num += x[i] * x[i + lag];
            den += x[i + lag] * x[i + lag];
        }
        num / (energy * den).max(1e-20).sqrt()
    };

    let mut best = (min_lag, f32::MIN);
    for lag in min_lag..max_lag {
        let norm = correlation(lag);
        if norm > best.1 {
            best = (lag, norm);
        }
    }

    // Walk down the sub-multiples, shortest first.
    for divisor in (2..=8).rev() {
        let candidate = best.0 / divisor;
        if candidate >= min_lag && correlation(candidate) > best.1 * OCTAVE_TOLERANCE {
            return sample_rate / candidate as f32;
        }
    }
    sample_rate / best.0 as f32
}

/// How many samples are subnormal, and how many are not finite.
///
/// DESIGN §8's determinism claim leans on both being zero: a subnormal is where
/// flush-to-zero behaviour stops being portable, and a NaN in an IIR is
/// permanent. `tests/determinism.rs` asserts `(0, 0)`.
pub fn anomalies(x: &[f32]) -> (usize, usize) {
    let subnormal = x.iter().filter(|s| s.is_subnormal()).count();
    let nonfinite = x.iter().filter(|s| !s.is_finite()).count();
    (subnormal, nonfinite)
}
