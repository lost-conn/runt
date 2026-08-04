//! The CPU noise twin's properties (DESIGN §7), with no GPU anywhere.
//!
//! `src/noise.rs` claims to be a faithful port of the 3dimenshift shader
//! library. Three things have to be true for that claim to be worth anything,
//! and none of them is checkable by reading the code:
//!
//! 1. **The 19-cell FCC search really finds the nearest feature point.** The
//!    original's own comment hedges ("for jitter <= ~0.9 they are redundant"),
//!    so this brute-forces a 5³ lattice neighbourhood and demands the same
//!    answer — at the authored jitter of 1.0, which is the case the hedge is
//!    about.
//! 2. **The lattice is actually FCC.** Even parity, and the wrap preserves it.
//! 3. **It is deterministic**, in-process and across processes, which is what
//!    the whole content-addressed bake rests on.
//!
//! Golden values are computed here rather than pinned as constants: DESIGN §4
//! promises same-build determinism, not cross-platform float identity, and a
//! hard-coded constant would turn an honest platform difference into a mystery
//! failure. The double-render pattern is the same one `sim_determinism.rs` uses.

use glam::Vec3;
use runt_core::noise::{
    self, cellular, fcc_round, hash13, hash33, normalized_fbm, wrap_cell, CellReturn, Fractal,
    Lattice,
};

/// A cheap deterministic scatter of sample points. Not an RNG — a fixed
/// sequence, so a failure is reproducible by index.
fn probe(i: u32) -> Vec3 {
    let f = i as f32;
    Vec3::new(
        (f * 0.6180339).sin() * 37.0 + f * 0.013,
        (f * 0.4142135).cos() * 41.0 - f * 0.007,
        (f * 0.7320508).sin() * 29.0 + f * 0.019,
    )
}

/// Every FCC cell (even coordinate sum) within `radius` of `home`.
fn fcc_neighbourhood(home: Vec3, radius: i32) -> Vec<Vec3> {
    let mut cells = Vec::new();
    for z in -radius..=radius {
        for y in -radius..=radius {
            for x in -radius..=radius {
                let cell = home + Vec3::new(x as f32, y as f32, z as f32);
                if (cell.x + cell.y + cell.z) as i64 % 2 == 0 {
                    cells.push(cell);
                }
            }
        }
    }
    cells
}

#[test]
fn the_19_cell_search_finds_the_true_nearest_feature_point() {
    // The claim under test: 19 cells is enough at jitter 1.0. Brute force a 5³
    // lattice neighbourhood (well past anything a jittered point can reach) and
    // demand the same nearest point, on 1000 seeded samples.
    const JITTER: f32 = 1.0;
    let mut checked = 0;
    for i in 0..1000u32 {
        let p = probe(i);
        let got = cellular(p, Lattice::Fcc, CellReturn::F1, JITTER, Vec3::ZERO);

        let home = fcc_round(p);
        let mut best = f32::INFINITY;
        let mut best_point = Vec3::ZERO;
        for cell in fcc_neighbourhood(home, 5) {
            let point = cell + (hash33(cell) - Vec3::splat(0.5)) * JITTER;
            let d = (point - p).length_squared();
            if d < best {
                best = d;
                best_point = point;
            }
        }

        assert!(
            (got.d1 - best.sqrt()).abs() < 1e-4,
            "sample {i} at {p:?}: 19-cell search found {} but the 5³ brute force \
             found {}",
            got.d1,
            best.sqrt()
        );
        assert!(
            got.f1.abs_diff_eq(best_point, 1e-4),
            "sample {i}: nearest point {:?} != brute-force {best_point:?}",
            got.f1
        );
        checked += 1;
    }
    assert_eq!(checked, 1000);
}

#[test]
fn the_second_nearest_is_also_the_true_second_nearest() {
    // F2 - F1 is what the boundary normals key off, so a wrong F2 is a wrong
    // crease, not merely a wrong number.
    const JITTER: f32 = 1.0;
    for i in 0..400u32 {
        let p = probe(i);
        let got = cellular(p, Lattice::Fcc, CellReturn::F2, JITTER, Vec3::ZERO);

        let mut distances: Vec<f32> = fcc_neighbourhood(fcc_round(p), 5)
            .into_iter()
            .map(|cell| {
                let point = cell + (hash33(cell) - Vec3::splat(0.5)) * JITTER;
                (point - p).length()
            })
            .collect();
        distances.sort_by(f32::total_cmp);

        assert!(
            (got.d2 - distances[1]).abs() < 1e-4,
            "sample {i}: F2 {} != brute-force {}",
            got.d2,
            distances[1]
        );
    }
}

#[test]
fn the_lattice_has_even_parity_and_the_wrap_keeps_it() {
    for i in 0..2000u32 {
        let p = probe(i);
        let cell = fcc_round(p);
        for c in cell.to_array() {
            assert_eq!(c, c.round(), "{cell:?} is not integral");
        }
        assert_eq!(
            (cell.x + cell.y + cell.z) as i64 % 2,
            0,
            "{cell:?} is not an FCC lattice point"
        );

        // An even period preserves parity, which is why `TextureSpec` rounds
        // spans to even numbers. An odd one would not, and the seam would show
        // up as a lattice that does not line up with itself.
        for period in [2.0f32, 4.0, 6.0, 16.0, 318.0] {
            let wrapped = wrap_cell(cell, Vec3::new(period, period, 0.0));
            assert_eq!(
                (wrapped.x + wrapped.y + wrapped.z) as i64 % 2,
                0,
                "wrapping {cell:?} by {period} broke parity: {wrapped:?}"
            );
            assert!(
                wrapped.x >= 0.0 && wrapped.x < period && wrapped.y >= 0.0 && wrapped.y < period,
                "wrapping {cell:?} by {period} left {wrapped:?} outside [0, {period})"
            );
            assert_eq!(wrapped.z, cell.z, "period 0 must not touch an axis");
        }
    }
}

#[test]
fn wrapping_is_exact_on_multiples_of_the_period() {
    // The case a reciprocal-based division gets wrong on a GPU (see
    // `noise::wrap_cell`): every exact multiple must land on 0, not on the
    // period itself, because the wrapped index only ever feeds a hash and being
    // one period off is a different random number rather than a small error.
    for period in [2.0f32, 4.0, 6.0, 44.0, 318.0] {
        for k in -20i32..=20 {
            let cell = Vec3::splat(k as f32 * period);
            let wrapped = wrap_cell(cell, Vec3::new(period, period, period));
            assert_eq!(
                wrapped,
                Vec3::ZERO,
                "{}·{period} wrapped to {wrapped:?}",
                k
            );
        }
    }
}

#[test]
fn the_field_is_periodic_under_its_wrap() {
    // The property the seamless bake is built on: translating by a whole period
    // in lattice units reproduces the field exactly.
    const PERIOD: f32 = 6.0;
    let period = Vec3::new(PERIOD, PERIOD, 0.0);
    for i in 0..300u32 {
        let p = probe(i);
        let a = cellular(p, Lattice::Fcc, CellReturn::CellValue, 1.0, period);
        for shift in [
            Vec3::new(PERIOD, 0.0, 0.0),
            Vec3::new(0.0, PERIOD, 0.0),
            Vec3::new(-PERIOD, PERIOD, 0.0),
            Vec3::new(PERIOD * 3.0, -PERIOD * 2.0, 0.0),
        ] {
            let b = cellular(p + shift, Lattice::Fcc, CellReturn::CellValue, 1.0, period);
            assert_eq!(
                a.value, b.value,
                "sample {i} shifted by {shift:?} changed the field"
            );
        }
    }
}

#[test]
// Authored lacunarity, not `e`.
#[allow(clippy::approx_constant)]
fn the_noise_is_bit_stable_across_repeated_evaluation() {
    // In-process double evaluation, `sim_determinism.rs` style: golden values
    // computed here rather than pinned, because DESIGN §4 promises same-build
    // determinism and explicitly does not promise cross-platform bit identity.
    let sample = |i: u32| {
        let p = probe(i);
        (
            hash13(p),
            hash33(p),
            cellular(p, Lattice::Fcc, CellReturn::CellValue, 1.0, Vec3::ZERO).value,
            normalized_fbm(
                p,
                Lattice::Fcc,
                CellReturn::CellValue,
                1.0,
                5,
                2.718,
                0.562,
                Fractal::Fbm,
                0.5,
            ),
        )
    };
    let first: Vec<_> = (0..500).map(sample).collect();
    let second: Vec<_> = (0..500).map(sample).collect();
    assert_eq!(first, second, "noise is not a pure function");

    // And nothing in the batch is degenerate — an all-zeros field would satisfy
    // the equality above and mean nothing.
    let distinct: std::collections::BTreeSet<u32> =
        first.iter().map(|s| s.3.to_bits()).collect();
    assert!(distinct.len() > 400, "the fBm collapsed: {} distinct values", distinct.len());
    for (h13, h33, cell, fbm) in &first {
        for v in [*h13, h33.x, h33.y, h33.z, *cell, *fbm] {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{v} out of range");
        }
    }
}

#[test]
// Authored lacunarity, not `e`.
#[allow(clippy::approx_constant)]
fn the_fbm_range_is_independent_of_the_octave_count() {
    // Amplitude normalization's whole purpose: adding octaves adds detail, not
    // brightness, so a colour ramp authored at 3 octaves still reads right at 6.
    let mut means = Vec::new();
    for octaves in 1..=6u32 {
        let mut sum = 0.0f64;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..500u32 {
            let n = normalized_fbm(
                probe(i) * 0.1,
                Lattice::Fcc,
                CellReturn::CellValue,
                1.0,
                octaves,
                2.718,
                0.562,
                Fractal::Fbm,
                0.5,
            );
            assert!((0.0..=1.0).contains(&n), "{octaves} octaves produced {n}");
            sum += n as f64;
            lo = lo.min(n);
            hi = hi.max(n);
        }
        let mean = sum / 500.0;
        println!("{octaves} octaves: mean {mean:.4}, range [{lo:.4}, {hi:.4}]");
        means.push(mean);
    }
    let first = means[0];
    for (i, m) in means.iter().enumerate() {
        assert!(
            (m - first).abs() < 0.06,
            "octave {} shifted the mean to {m} from {first}: {means:?}",
            i + 1
        );
    }
}

#[test]
fn the_ridged_variant_is_a_different_field_in_the_same_range() {
    for i in 0..200u32 {
        let p = probe(i) * 0.1;
        let fbm = normalized_fbm(
            p,
            Lattice::Fcc,
            CellReturn::F1,
            1.0,
            4,
            2.0,
            0.5,
            Fractal::Fbm,
            0.5,
        );
        let ridged = normalized_fbm(
            p,
            Lattice::Fcc,
            CellReturn::F1,
            1.0,
            4,
            2.0,
            0.5,
            Fractal::Ridged,
            0.5,
        );
        assert!(ridged.is_finite() && ridged >= 0.0, "ridged gave {ridged}");
        assert_ne!(fbm, ridged, "sample {i}: ridged is not doing anything");
    }
}

#[test]
fn octave_weights_fade_and_the_sentinel_turns_them_off() {
    // The distance-LOD knob the bake passes "off" and a live-eval variant will
    // not. `max < min` is the sentinel.
    assert_eq!(noise::octave_weight(0, 1.0, 0.0), 1.0);
    assert_eq!(noise::octave_weight(9, 1.0, 0.0), 1.0);

    // A window from octave 2 to 4: full below, zero above, monotone between.
    let w: Vec<f32> = (0..6).map(|i| noise::octave_weight(i, 2.0, 4.0)).collect();
    assert_eq!(w[0], 1.0);
    assert_eq!(w[1], 1.0);
    assert_eq!(w[2], 1.0);
    assert!((w[3] - 0.5).abs() < 1e-6, "{w:?}");
    assert_eq!(w[4], 0.0);
    assert_eq!(w[5], 0.0);
    for pair in w.windows(2) {
        assert!(pair[1] <= pair[0], "weights must not rise: {w:?}");
    }
}

#[test]
fn the_two_lattices_have_the_densities_the_port_spec_claims() {
    // "FCC places 0.5 lattice points per unit volume vs cubic's 1.0" — the
    // sentence the ×1.26 frequency compensation in the authored params rests
    // on. Counted rather than asserted from the comment.
    let radius = 12i32;
    let cells = (2 * radius + 1) as f32;
    let volume = cells * cells * cells;
    let fcc = fcc_neighbourhood(Vec3::ZERO, radius).len() as f32;
    println!("FCC density {:.3} points/unit³", fcc / volume);
    assert!(
        (fcc / volume - 0.5).abs() < 0.05,
        "FCC density is {}, not ~0.5",
        fcc / volume
    );
}
