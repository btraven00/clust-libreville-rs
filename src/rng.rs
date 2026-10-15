//! Counter-based randomness: every draw is a pure function of (seed, stream, step, vertex), so
//! reordering or parallelising work never changes what anyone draws.

use rayon::prelude::*;

pub(crate) const COLOUR_ORDER: u64 = 1;
pub(crate) const REFINE_ORDER: u64 = 2;
pub(crate) const REFINE_DRAW: u64 = 3;

pub(crate) fn mix(mut z: u64) -> u64 {
    // SplitMix64 finaliser
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(crate) fn key(seed: u64, stream: u64, step: u64, v: u64) -> u64 {
    mix(mix(mix(mix(seed) ^ stream) ^ step) ^ v)
}

/// Uniform in [0, 1) from 53 bits.
pub(crate) fn unit(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// exp(x) for x <= 0, as (1 + x/256)^256: IEEE multiplies only, so bit-identical on every
/// platform (no libm). From graspologic-native (MIT, Microsoft; github.com/microsoft/graspologic-native),
/// `approximate_exponent`. An approximation: off from e^x by about e^(-x²/512) (0.82 at x = -10).
pub(crate) fn exp_neg(x: f64) -> f64 {
    if x < -256.0 {
        return 0.0;
    }
    let mut r = 1.0 + x / 256.0;
    for _ in 0..8 {
        r *= r;
    }
    r
}

/// The vertices 0..n in a seeded order.
pub(crate) fn order(n: usize, seed: u64, stream: u64, step: u64) -> Vec<u32> {
    let mut o: Vec<(u64, u32)> = (0..n as u32).map(|v| (key(seed, stream, step, v as u64), v)).collect();
    o.par_sort_unstable(); // (key, v) pairs are distinct, so the result is the same as a sequential sort
    o.into_iter().map(|(_, v)| v).collect()
}
