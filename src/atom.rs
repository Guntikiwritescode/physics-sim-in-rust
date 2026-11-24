//! Hydrogen orbital sampler (from kavan010/Atoms, atom_realtime.cpp).
//! Picks electron positions with the probability |psi_nlm|^2 from the Schrodinger equation:
//! r from r^2 R_nl(r)^2, theta from sin(theta) P_l^|m|(cos theta)^2, phi uniform (atomic units, a0 = 1),
//! by associated Laguerre and Legendre recurrences. Positions are interpolated inside each table step.

use rayon::prelude::*;

/// Radial part R_nl(r) without its normalising constant (it cancels when sampling).
pub fn radial(n: i32, l: i32, r: f64) -> f64 {
    let rho = 2.0 * r / n as f64;
    let k = n - l - 1;
    let alpha = (2 * l + 1) as f64;
    let mut lag = 1.0;
    if k >= 1 {
        let (mut lm2, mut lm1) = (1.0, 1.0 + alpha - rho);
        lag = lm1;
        for j in 2..=k {
            let j = j as f64;
            lag = ((2.0 * j - 1.0 + alpha - rho) * lm1 - (j - 1.0 + alpha) * lm2) / j;
            lm2 = lm1;
            lm1 = lag;
        }
    }
    (-rho / 2.0).exp() * rho.powi(l) * lag
}

/// Associated Legendre P_l^m(x) for m >= 0 (Condon-Shortley sign; the sign cancels in |psi|^2).
pub fn legendre(l: i32, m: i32, x: f64) -> f64 {
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m { pmm *= -fact * somx2; fact += 2.0; }
    }
    if l == m { return pmm; }
    let mut pm1m = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 { return pm1m; }
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = ((2 * ll - 1) as f64 * x * pm1m - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pm1m;
        pm1m = pll;
    }
    pll
}

/// Probability density |psi|^2 up to a constant (for colouring).
pub fn density(n: i32, l: i32, m: i32, r: f64, theta: f64) -> f64 {
    let rr = radial(n, l, r);
    let p = legendre(l, m.abs(), theta.cos());
    rr * rr * p * p
}

/// A table of the running total of a density on a grid, for drawing values by inverse lookup.
struct Cdf { x0: f64, dx: f64, c: Vec<f64> }

impl Cdf {
    fn new(n: usize, x0: f64, x1: f64, pdf: impl Fn(f64) -> f64) -> Cdf {
        let dx = (x1 - x0) / (n - 1) as f64;
        let mut c = vec![0.0; n];
        let mut prev = pdf(x0);
        for i in 1..n {
            let cur = pdf(x0 + i as f64 * dx);
            c[i] = c[i - 1] + 0.5 * (prev + cur) * dx;
            prev = cur;
        }
        let total = c[n - 1];
        for v in &mut c { *v /= total; }
        Cdf { x0, dx, c }
    }
    fn sample(&self, u: f64) -> f64 {
        let i = self.c.partition_point(|&v| v < u).clamp(1, self.c.len() - 1);
        let (a, b) = (self.c[i - 1], self.c[i]);
        let f = if b > a { (u - a) / (b - a) } else { 0.5 };
        self.x0 + (i as f64 - 1.0 + f) * self.dx
    }
}

pub fn clamp_qn(n: i32, l: i32, m: i32) -> (i32, i32, i32) {
    let n = n.max(1);
    let l = l.clamp(0, n - 1);
    (n, l, m.clamp(-l, l))
}

/// One electron position: (r, theta, phi0, colour value 0..1).
pub type Sample = [f32; 4];

/// Draws `count` positions for orbital (n, l, m). Colour value = where the point's density ranks
/// among all points drawn (0 = thinnest, 1 = densest), so the whole colour scale is used.
pub fn sample(n: i32, l: i32, m: i32, count: usize, seed: u64) -> Vec<Sample> {
    let (n, l, m) = clamp_qn(n, l, m);
    let r_max = 10.0 * (n * n) as f64;
    let rc = Cdf::new(16384, 0.0, r_max, |r| { let rr = radial(n, l, r); r * r * rr * rr });
    let tc = Cdf::new(8192, 0.0, std::f64::consts::PI, |t| { let p = legendre(l, m.abs(), t.cos()); t.sin() * p * p });
    let mut out: Vec<Sample> = (0..count).into_par_iter().map(|i| {
        // Small fast hash-based random numbers, one stream per sample.
        let mut s = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let mut rnd = || { s ^= s >> 30; s = s.wrapping_mul(0xBF58476D1CE4E5B9); s ^= s >> 27; s = s.wrapping_mul(0x94D049BB133111EB); s ^= s >> 31; (s >> 11) as f64 / (1u64 << 53) as f64 };
        let r = rc.sample(rnd());
        let t = tc.sample(rnd());
        let p = rnd() * std::f64::consts::TAU;
        [r as f32, t as f32, p as f32, density(n, l, m, r, t) as f32]
    }).collect();
    let mut order: Vec<u32> = (0..out.len() as u32).collect();
    order.par_sort_unstable_by(|&i, &j| out[i as usize][3].total_cmp(&out[j as usize][3]));
    let k = 1.0 / out.len().max(2) as f32;
    for (rank, &i) in order.iter().enumerate() { out[i as usize][3] = 0.15 + 0.85 * rank as f32 * k; }
    out
}

// Light and the atom: energy levels, the colour of each jump, and how fast each excited orbital
// falls back (electric-dipole rule, worked out from the same wave shapes that are drawn).

/// Hydrogen's Rydberg constant (1/m): R_inf = 10973731.568 /m with the electron/proton mass ratio
/// 1/1836.15267 (CODATA), R_H = R_inf / (1 + m_e/m_p).
pub const R_H: f64 = 10_973_731.568 / (1.0 + 1.0 / 1836.15267);
/// Reduced-mass factor: hydrogen levels and rates are those of an infinitely heavy nucleus times this.
pub const MU: f64 = 1.0 / (1.0 + 1.0 / 1836.15267);
const ALPHA: f64 = 7.297_352_5693e-3;   // fine-structure constant (CODATA)
const AU_TIME: f64 = 2.418_884_326_586e-17; // atomic unit of time, s (CODATA)
pub const N_MAX: i32 = 12;               // same top orbital as the keys allow

/// Vacuum wavelength (nm) of the photon for a jump between levels n1 and n2.
pub fn wavelength_nm(n1: i32, n2: i32) -> f64 {
    let (a, b) = ((n1.min(n2) * n1.min(n2)) as f64, (n1.max(n2) * n1.max(n2)) as f64);
    1e9 / (R_H * (1.0 / a - 1.0 / b))
}

/// Integral of R_n1l1 R_n2l2 r^3 dr over normalised radial functions (atomic units, infinite nucleus).
pub fn radial_integral(n1: i32, l1: i32, n2: i32, l2: i32) -> f64 {
    let r_max = 8.0 * (n1.max(n2) * n1.max(n2)) as f64 + 40.0;
    let steps = 40_000usize;
    let h = r_max / steps as f64;
    let (mut s12, mut s11, mut s22) = (0.0, 0.0, 0.0);
    for i in 0..=steps {
        let r = i as f64 * h;
        let w = if i == 0 || i == steps { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        let (a, b) = (radial(n1, l1, r), radial(n2, l2, r));
        s12 += w * a * b * r * r * r;
        s11 += w * a * a * r * r;
        s22 += w * b * b * r * r;
    }
    s12 / (s11 * s22).sqrt()
}

/// Spontaneous fall rate (1/s) from orbital (n, l) to (n2, l2), summed over the final m:
/// A = (4/3) alpha^3 w^3 max(l, l2)/(2l+1) |<r>|^2 in atomic units (zero unless l2 = l +- 1, n2 < n).
/// The reduced mass scales it by MU (w^3 goes as MU^3, <r>^2 as MU^-2).
pub fn rate(n: i32, l: i32, n2: i32, l2: i32) -> f64 {
    if n2 >= n || (l2 - l).abs() != 1 || l2 < 0 || l2 >= n2 { return 0.0; }
    let w = 0.5 * (1.0 / (n2 * n2) as f64 - 1.0 / (n * n) as f64);
    let r = radial_integral(n, l, n2, l2);
    let ang = l.max(l2) as f64 / (2 * l + 1) as f64;
    (4.0 / 3.0) * ALPHA.powi(3) * w.powi(3) * ang * r * r / AU_TIME * MU
}

/// Every allowed fall from (n, l): (n2, l2, rate 1/s).
pub fn falls(n: i32, l: i32) -> Vec<(i32, i32, f64)> {
    let mut out = Vec::new();
    for n2 in 1..n { for l2 in [l - 1, l + 1] { let a = rate(n, l, n2, l2); if a > 0.0 { out.push((n2, l2, a)); } } }
    out
}

/// Mean life (s) of orbital (n, l); infinite if no single-photon fall is allowed (1s, 2s).
pub fn lifetime(n: i32, l: i32) -> f64 {
    let total: f64 = falls(n, l).iter().map(|f| f.2).sum();
    if total > 0.0 { 1.0 / total } else { f64::INFINITY }
}

/// Every allowed climb from (n, l) by absorbing one photon, up to N_MAX: (n2, l2, relative strength).
/// Strength = max(l, l2)/(2l+1) |<r>|^2 (the same dipole factor, seen from the lower orbital).
pub fn climbs(n: i32, l: i32) -> Vec<(i32, i32, f64)> {
    let mut out = Vec::new();
    for n2 in n + 1..=N_MAX { for l2 in [l - 1, l + 1] {
        if l2 < 0 || l2 >= n2 { continue; }
        let r = radial_integral(n, l, n2, l2);
        out.push((n2, l2, l.max(l2) as f64 / (2 * l + 1) as f64 * r * r));
    }}
    out
}

/// Chances of each final m when (l, m) jumps to l2 = l +- 1 (dipole angular factors, sum to 1).
pub fn m_chances(l: i32, m: i32, l2: i32) -> Vec<(i32, f64)> {
    let (lf, mf) = (l as f64, m as f64);
    let w = |dm: i32| -> f64 {
        let s = dm as f64;
        if l2 == l - 1 {
            if dm == 0 { lf * lf - mf * mf } else { 0.5 * (lf - s * mf) * (lf - s * mf - 1.0) }
        } else if dm == 0 { (lf + 1.0).powi(2) - mf * mf } else { 0.5 * (lf + s * mf + 1.0) * (lf + s * mf + 2.0) }
    };
    let v: Vec<(i32, f64)> = [-1, 0, 1].into_iter().filter(|dm| (m + dm).abs() <= l2).map(|dm| (m + dm, w(dm).max(0.0))).collect();
    let t: f64 = v.iter().map(|x| x.1).sum();
    v.into_iter().map(|(m2, x)| (m2, x / t)).collect()
}

/// Approximate colour of light of this wavelength (nm), 0..1 each (Dan Bruton's 1996 table).
/// Ultraviolet comes back as dim violet and infrared as dim dark red, so they still show.
pub fn wavelength_rgb(nm: f64) -> [f32; 3] {
    if nm < 380.0 { return [0.35, 0.2, 0.6]; }
    if nm > 780.0 { return [0.35, 0.05, 0.05]; }
    let (r, g, b) = if nm < 440.0 { (-(nm - 440.0) / 60.0, 0.0, 1.0) }
        else if nm < 490.0 { (0.0, (nm - 440.0) / 50.0, 1.0) }
        else if nm < 510.0 { (0.0, 1.0, -(nm - 510.0) / 20.0) }
        else if nm < 580.0 { ((nm - 510.0) / 70.0, 1.0, 0.0) }
        else if nm < 645.0 { (1.0, -(nm - 645.0) / 65.0, 0.0) }
        else { (1.0, 0.0, 0.0) };
    let f = if nm < 420.0 { 0.3 + 0.7 * (nm - 380.0) / 40.0 } else if nm > 700.0 { 0.3 + 0.7 * (780.0 - nm) / 80.0 } else { 1.0 };
    [(r * f) as f32, (g * f) as f32, (b * f) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_r(n: i32, l: i32, m: i32) -> f64 {
        let s = sample(n, l, m, 400_000, 42);
        s.iter().map(|x| x[0] as f64).sum::<f64>() / s.len() as f64
    }
    fn mean_cos2(n: i32, l: i32, m: i32) -> f64 {
        let s = sample(n, l, m, 400_000, 7);
        s.iter().map(|x| (x[1] as f64).cos().powi(2)).sum::<f64>() / s.len() as f64
    }

    #[test]
    fn average_radius_matches_textbook() {
        for (n, l) in [(1, 0), (2, 0), (2, 1), (3, 2), (4, 1)] {
            let got = mean_r(n, l, 0);
            let exact = 0.5 * (3 * n * n - l * (l + 1)) as f64;
            assert!((got - exact).abs() / exact < 0.01, "n={n} l={l}: {got} vs {exact}");
        }
    }

    #[test]
    fn angular_shape_matches_textbook() {
        for (l, m, exact) in [(1, 0, 3.0 / 5.0), (1, 1, 1.0 / 5.0), (1, -1, 1.0 / 5.0), (2, 0, 11.0 / 21.0)] {
            let got = mean_cos2(3, l, m);
            assert!((got - exact).abs() < 0.005, "l={l} m={m}: {got} vs {exact}");
        }
    }

    #[test]
    fn changing_orbital_changes_the_shape() {
        let a = mean_r(1, 0, 0);
        let b = mean_r(3, 0, 0);
        assert!((a - 1.5).abs() < 0.02 && (b - 13.5).abs() < 0.15, "{a} {b}");
    }

    // NIST, "Some transitions of the main spectral series of hydrogen" (Wiese & Fuhr data):
    const NIST: [(i32, i32, f64, f64, bool); 6] = [
        (2, 1, 4.699e8, 121.567, true), (3, 1, 5.575e7, 102.573, true), (4, 1, 1.278e7, 97.2537, true),
        (3, 2, 4.410e7, 656.280, false), (4, 2, 8.419e6, 486.132, false), (4, 3, 8.986e6, 1875.10, false),
    ];

    #[test]
    fn line_colours_match_nist() {
        for (n, n2, _, nm, vac) in NIST {
            let got = wavelength_nm(n, n2) / if vac { 1.0 } else { 1.000277 };
            assert!((got - nm).abs() / nm < 5e-5, "{n}->{n2}: {got} vs {nm}");
        }
    }

    #[test]
    fn fall_rates_match_nist() {
        // NIST lists shell averages: A(n->n2) = sum over l, l2 of (2l+1) A(nl->n2l2) / n^2.
        for (n, n2, a_nist, _, _) in NIST {
            let mut sum = 0.0;
            for l in 0..n { for l2 in [l - 1, l + 1] { sum += (2 * l + 1) as f64 * rate(n, l, n2, l2); } }
            let avg = sum / (n * n) as f64;
            assert!((avg - a_nist).abs() / a_nist < 2e-3, "{n}->{n2}: {avg} vs {a_nist}");
        }
        assert!((lifetime(2, 1) * 1e9 - 1.596).abs() < 0.005);
        assert!(lifetime(2, 0).is_infinite() && lifetime(1, 0).is_infinite());
    }

    #[test]
    fn m_chances_add_up_and_follow_the_rules() {
        for l in 0..5 { for m in -l..=l { for l2 in [l - 1, l + 1] {
            if l2 < 0 { continue; }
            let c = m_chances(l, m, l2);
            let t: f64 = c.iter().map(|x| x.1).sum();
            assert!((t - 1.0).abs() < 1e-12);
            assert!(c.iter().all(|&(m2, _)| (m2 - m).abs() <= 1 && m2.abs() <= l2));
        }}}
        assert_eq!(m_chances(1, 0, 0), vec![(0, 1.0)]);
    }
}
