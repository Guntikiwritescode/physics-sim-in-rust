//! Dot clouds for the Atoms tab beyond hydrogen: each dot is one place an electron could be found,
//! drawn at random in proportion to the electron density that the solvers give.
//!
//! - Elements (1-118): elements.rs / dft.rs radial tables (scalar-relativistic LDA). The whole atom
//!   is the spherical average the solver works with (one radial function per shell), so its dots
//!   are drawn with no preferred direction. One chosen orbital gets its real angular shape
//!   |P_l^|m|(cos theta)|^2, as hydrogen's does.
//! - Molecules: molecules.rs (Hartree-Fock), density on a cube of points.
//! - Crystals: solids.rs (periodic DFT), valence density only: the pseudopotential holds the core
//!   electrons, so there are no core dots.
//!
//! Every dot is [r, theta, phi, colour] in Bohr about the middle, like atom.rs's samples, so the
//! same drawing code shows them. Colour: where the dot's density ranks among all dots.

use crate::atom::{legendre, Sample};
use rayon::prelude::*;

fn stream(seed: u64, i: usize) -> impl FnMut() -> f64 {
    let mut s = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    move || {
        s ^= s >> 30; s = s.wrapping_mul(0xBF58476D1CE4E5B9);
        s ^= s >> 27; s = s.wrapping_mul(0x94D049BB133111EB);
        s ^= s >> 31;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Running total of weights, normalised to 1 at the end.
fn cdf(w: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(w.len());
    let mut acc = 0.0;
    for &v in w { acc += v.max(0.0); c.push(acc); }
    let t = acc.max(1e-300);
    for v in &mut c { *v /= t; }
    c
}

fn pick(c: &[f64], u: f64) -> usize { c.partition_point(|&v| v < u).min(c.len() - 1) }

/// Colour value = rank of each dot's density among all dots (0.15 thinnest .. 1 densest), as atom.rs.
pub fn rank_colour(out: &mut [Sample]) {
    let mut order: Vec<u32> = (0..out.len() as u32).collect();
    order.par_sort_unstable_by(|&i, &j| out[i as usize][3].total_cmp(&out[j as usize][3]));
    let k = 1.0 / out.len().max(2) as f32;
    for (rank, &i) in order.iter().enumerate() { out[i as usize][3] = 0.15 + 0.85 * rank as f32 * k; }
}

/// Cartesian (Bohr, y up as the drawing's axis) to the dot form.
fn to_sample(p: [f64; 3], dens: f64) -> Sample {
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-9);
    let theta = (p[1] / r).clamp(-1.0, 1.0).acos();
    let phi = p[2].atan2(p[0]);
    [r as f32, theta as f32, phi as f32, dens as f32]
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ElemPart {
    /// All electrons (the spherical average the solver works with).
    All,
    /// One orbital (index into the solved atom's orbitals) with this m.
    Orbital(usize, i32),
}

pub fn element(s: &crate::elements::Solved, part: ElemPart, count: usize, seed: u64) -> Vec<Sample> {
    let g = &s.grid;
    let tau = std::f64::consts::TAU;
    let mut out: Vec<Sample> = match part {
        ElemPart::All => {
            // Electrons between r and r + dr: 4 pi r^2 rho dr, with dr = r h on the log grid.
            let w: Vec<f64> = (0..g.r.len()).map(|i| g.r[i].powi(3) * s.atom.rho[i]).collect();
            let c = cdf(&w);
            (0..count).into_par_iter().map(|i| {
                let mut rnd = stream(seed, i);
                let k = pick(&c, rnd());
                let r = g.r[k] * (g.h * (rnd() - 0.5)).exp();
                let ct = 1.0 - 2.0 * rnd();
                let st = (1.0 - ct * ct).max(0.0).sqrt();
                let ph = rnd() * tau;
                let mut d = to_sample([r * st * ph.cos(), r * ct, r * st * ph.sin()], 0.0);
                d[3] = s.density_at(r) as f32;
                d
            }).collect()
        }
        ElemPart::Orbital(k, m) => {
            let l = s.atom.shells[k].l as i32;
            let m = m.clamp(-l, l);
            let p = &s.atom.p[k];
            // |P(r)|^2 dr with dr = r h.
            let w: Vec<f64> = (0..g.r.len()).map(|i| p[i] * p[i] * g.r[i]).collect();
            let rc = cdf(&w);
            let nt = 4096;
            let tw: Vec<f64> = (0..nt).map(|i| {
                let t = (i as f64 + 0.5) / nt as f64 * std::f64::consts::PI;
                let a = legendre(l, m.abs(), t.cos());
                t.sin() * a * a
            }).collect();
            let tc = cdf(&tw);
            (0..count).into_par_iter().map(|i| {
                let mut rnd = stream(seed, i);
                let j = pick(&rc, rnd());
                let r = g.r[j] * (g.h * (rnd() - 0.5)).exp();
                let t = (pick(&tc, rnd()) as f64 + rnd()) / nt as f64 * std::f64::consts::PI;
                let ph = rnd() * tau;
                let a = legendre(l, m.abs(), t.cos());
                let pr = p[j] / g.r[j];
                [r as f32, t as f32, ph as f32, (pr * pr * a * a) as f32]
            }).collect()
        }
    };
    rank_colour(&mut out);
    out
}

/// Dots drawn from density values on a cube of `n` points a side (x fastest, then y, then z),
/// half side `half` (Bohr), centred on the middle of the drawing. Each dot lands at a random spot
/// inside the little box round its grid point.
pub fn from_grid(vals: &[f64], n: usize, half: f64, count: usize, seed: u64) -> Vec<Sample> {
    let h = 2.0 * half / (n - 1) as f64;
    let c = cdf(vals);
    let mut out: Vec<Sample> = (0..count).into_par_iter().map(|i| {
        let mut rnd = stream(seed, i);
        let q = pick(&c, rnd());
        let (x, y, z) = (q % n, (q / n) % n, q / (n * n));
        let p = [
            -half + (x as f64 + rnd() - 0.5) * h,
            -half + (y as f64 + rnd() - 0.5) * h,
            -half + (z as f64 + rnd() - 0.5) * h,
        ];
        to_sample(p, vals[q])
    }).collect();
    rank_colour(&mut out);
    out
}

/// Cube points (x fastest) round `centre`.
pub fn cube_points(n: usize, centre: [f64; 3], half: f64) -> Vec<[f64; 3]> {
    let h = 2.0 * half / (n - 1) as f64;
    let mut pts = Vec::with_capacity(n * n * n);
    for z in 0..n { for y in 0..n { for x in 0..n {
        pts.push([centre[0] - half + x as f64 * h, centre[1] - half + y as f64 * h, centre[2] - half + z as f64 * h]);
    } } }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_dots_follow_density() {
        let s = crate::elements::solve(10).unwrap();
        let g = &s.grid;
        let pi4 = 4.0 * std::f64::consts::PI;
        let n = g.integrate(|i| pi4 * g.r[i] * g.r[i] * s.atom.rho[i]);
        let r1 = g.integrate(|i| pi4 * g.r[i].powi(3) * s.atom.rho[i]) / n;
        let d = element(&s, ElemPart::All, 1 << 20, 7);
        let m = d.iter().map(|p| p[0] as f64).sum::<f64>() / d.len() as f64;
        assert!((m / r1 - 1.0).abs() < 0.01, "dots {m} solver {r1}");
        assert!((n - 10.0).abs() < 1e-3);
    }

    #[test]
    fn orbital_dots_have_shape() {
        let s = crate::elements::solve(10).unwrap();
        let k = s.atom.shells.iter().position(|o| o.n == 2 && o.l == 1).unwrap();
        let d = element(&s, ElemPart::Orbital(k, 0), 1 << 18, 3);
        let c2 = d.iter().map(|p| (p[1] as f64).cos().powi(2)).sum::<f64>() / d.len() as f64;
        assert!((c2 - 0.6).abs() < 0.01);
    }

    #[test]
    fn grid_dots_land_where_density_is() {
        let n = 11;
        let mut v = vec![0.0; n * n * n];
        v[7 + n * (2 + n * 5)] = 1.0;
        let d = from_grid(&v, n, 5.0, 1000, 1);
        for p in &d {
            let (r, t, ph) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let (x, y, z) = (r * t.sin() * ph.cos(), r * t.cos(), r * t.sin() * ph.sin());
            assert!((x - 2.0).abs() <= 0.5 + 1e-4 && (y + 3.0).abs() <= 0.5 + 1e-4 && z.abs() <= 0.5 + 1e-4, "{x} {y} {z}");
        }
    }
}

#[cfg(test)]
mod crystal_check {
    #[test]
    #[ignore]
    fn silicon_bond_is_densest() {
        let s = &crate::solids::SOLIDS[0];
        let c = crate::solids::crystal(s, s.a_lda, 2, 120.0).unwrap();
        let a = s.a_lda * crate::solids::BOHR_PER_ANGSTROM;
        let (at, mid) = (c.density_at([0.0; 3]), c.density_at([a / 8.0; 3]));
        let far = c.density_at([a / 2.0, a / 2.0, a / 2.0]);
        assert!(mid > at && mid > far);
    }
}
