//! Atoms of any element: density functional theory for a spherical atom (Kohn-Sham equations on a
//! radial grid), the scalar-relativistic method (ScRLDA) of NIST Standard Reference Database 141
//! (Kotochigova, Levine, Shirley, Stiles, Clark: "Atomic reference data for electronic structure
//! calculations").
//!
//! - Electrons fill shells (n, l) with the occupations given; each shell's electrons share one
//!   radial function (the density is spherical).
//! - Exchange and correlation: Slater exchange with the MacDonald-Vosko relativistic correction and
//!   the Vosko-Wilk-Nusair correlation (Can. J. Phys. 58, 1200 (1980)).
//! - Radial equation: Koelling-Harmon, as a pair (P, Q) on a uniform grid in x = ln r, implicit
//!   Adams-Moulton steps; energy by node counting plus Illinois steps on the mismatch.
//! - Hartree potential: Poisson's equation by Numerov's method in summed form; integrals by the
//!   trapezoid rule in x.
//! - Self-consistency: Pulay (DIIS) mixing of the potential.
//!
//! Units: Hartree atomic units (lengths in Bohr radii, energies in Hartree).

use std::f64::consts::PI;

/// The radial grid: r_i = exp(x0 + i h).
#[derive(Clone)]
pub struct Grid {
    pub r: Vec<f64>,
    pub h: f64,
}

impl Grid {
    pub fn new(r_min: f64, r_max: f64, n: usize) -> Grid {
        let (x0, x1) = (r_min.ln(), r_max.ln());
        let h = (x1 - x0) / (n - 1) as f64;
        Grid { r: (0..n).map(|i| (x0 + i as f64 * h).exp()).collect(), h }
    }
    pub fn standard(z: f64) -> Grid { Grid::new(1e-8 / z.max(1.0), 60.0, 16001) }
    /// A coarser grid for pictures.
    pub fn display(z: f64) -> Grid { Grid::new(1e-8 / z.max(1.0), 60.0, 4001) }
    /// Integral of f(r) dr = sum of f r h (trapezoid in x).
    pub fn integrate(&self, f: impl Fn(usize) -> f64) -> f64 {
        let n = self.r.len();
        let mut s = 0.0;
        for i in 0..n {
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            s += w * f(i) * self.r[i];
        }
        s * self.h
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shell { pub n: u32, pub l: u32, pub occ: f64 }

/// Hartree potential of a spherical density (electrons per Bohr^3).
pub fn hartree(g: &Grid, rho: &[f64]) -> Vec<f64> {
    let m = g.r.len();
    let h2 = g.h * g.h / 12.0;
    // U = r V_H, U'' = -4 pi r rho (in r); with U = sqrt(r) w: w'' = w/4 - 4 pi r^(5/2) rho (in x).
    let s: Vec<f64> = (0..m).map(|i| -4.0 * PI * g.r[i].powf(2.5) * rho[i]).collect();
    let a = 0.25;
    let fa = 1.0 - h2 * a;
    let mut w = vec![0.0; m];
    // Near the nucleus U ~ -(2 pi / 3) rho(0) r^3 (plus c r, added below).
    w[0] = -2.0 * PI / 3.0 * rho[0] * g.r[0].powf(2.5);
    w[1] = -2.0 * PI / 3.0 * rho[0] * g.r[1].powf(2.5);
    // Numerov in summed form: y = w - h^2/12 (a w + s), y[i+1] - 2 y[i] + y[i-1] = h^2 (a w + s)[i]. The
    // step d = y[i+1] - y[i] is carried, so the small change per step is not lost to rounding.
    let yv = |i: usize, w: f64| w - h2 * (a * w + s[i]);
    let mut d = yv(1, w[1]) - yv(0, w[0]);
    let mut y = yv(1, w[1]);
    for i in 1..m - 1 {
        d += 12.0 * h2 * (a * w[i] + s[i]);
        y += d;
        w[i + 1] = (y + h2 * s[i + 1]) / fa;
    }
    let q = g.integrate(|i| 4.0 * PI * g.r[i] * g.r[i] * rho[i]);
    let u_end = w[m - 1] * g.r[m - 1].sqrt();
    let c = (q - u_end) / g.r[m - 1];
    (0..m).map(|i| (w[i] * g.r[i].sqrt() + c * g.r[i]) / g.r[i]).collect()
}

/// Speed of light in atomic units, the value SRD 141 used (CODATA 1986).
pub const C_LIGHT: f64 = 137.0359895;

/// The pair (P, Q) obeys d/dx (P, Q) = A (P, Q) with x = ln r. Returns A for energy e at grid point i.
fn pair_matrix(l: u32, r: f64, v: f64, e: f64) -> [[f64; 2]; 2] {
    let c = C_LIGHT;
    let m = 1.0 + (e - v) / (2.0 * c * c);
    let ll = (l * (l + 1)) as f64;
    [[1.0, 2.0 * m * c * r], [ll / (2.0 * m * c * r) + r * (v - e) / c, -1.0]]
}

/// Bound state (n, l) in potential v (Hartree, on the grid): node counting brackets the energy,
/// then the mismatch of Q/P at the matching point is driven to zero (secant steps inside the
/// bracket). Steps: implicit Adams-Moulton, 4th order (exact 2x2 solve per step, so it is stable
/// for any step). Returns (energy, P) with P alone normalised, as in SRD 141. None if no state
/// with the right number of nodes was found.
pub fn radial_state(g: &Grid, v: &[f64], z: f64, n: u32, l: u32) -> Option<(f64, Vec<f64>)> {
    let m = g.r.len();
    let c = C_LIGHT;
    let h = g.h;
    let nodes_want = (n - l - 1) as usize;
    let lh = l as f64 + 0.5;
    let mut vmin = f64::INFINITY;
    for i in 0..m { vmin = vmin.min(v[i] + lh * lh / (2.0 * g.r[i] * g.r[i])); }
    let mut e_lo = (-z * z / (2.0 * (n * n) as f64)).min(vmin) * 1.4 - 1.0;
    let mut e_hi = 0.0;
    let mut e = -z * z / (2.0 * (n * n) as f64);
    let gamma = ((l * (l + 1)) as f64 + 1.0 - (z / c).powi(2)).sqrt();
    let ratio = (gamma - 1.0) * c / z;
    let mut pp = vec![0.0; m];
    let mut qq = vec![0.0; m];
    let mut f_prev: Option<(f64, f64)> = None;   // (e, mismatch) of the last trial with the right nodes
    let mut fl: Option<(f64, f64)> = None;       // bracket ends with the right nodes and their mismatch
    let mut fh: Option<(f64, f64)> = None;
    let mut last_side = 0;
    for _ in 0..300 {
        let a: Vec<[[f64; 2]; 2]> = (0..m).map(|i| pair_matrix(l, g.r[i], v[i], e)).collect();
        let Some(mut t) = (0..m).rev().find(|&i| v[i] + (l * (l + 1)) as f64 / (2.0 * g.r[i] * g.r[i]) < e) else {
            e_lo = e; e = 0.5 * (e_lo + e_hi); continue;
        };
        t = t.clamp(8, m - 8);
        // Outward: P ~ r^gamma, Q = ratio P near the nucleus.
        for i in 0..3 { pp[i] = g.r[i].powf(gamma); qq[i] = ratio * pp[i]; }
        let mut nodes = 0;
        am_steps(&a, &mut pp, &mut qq, h, 2, t + 1, &mut nodes, t);
        if nodes != nodes_want {
            if nodes > nodes_want { e_hi = e; } else { e_lo = e; }
            e = 0.5 * (e_lo + e_hi);
            continue;
        }
        let (po, qo) = (pp[t], qq[t]);
        // Inward from where the state has died away (about e^-45 of its size at t).
        let lam = |i: usize| -> f64 {
            let k2 = 2.0 * (v[i] - e) + (l * (l + 1)) as f64 / (g.r[i] * g.r[i]);
            k2.max(0.0).sqrt()
        };
        let mut acc = 0.0;
        let mut end = m - 1;
        for i in t..m { acc += lam(i) * g.r[i] * h; if acc > 45.0 { end = i; break; } }
        end = end.max(t + 6).min(m - 1);
        for i in end - 2..=end {
            let ll = lam(i).max(1e-6);
            pp[i] = 1e-30 * (-(ll * (g.r[i] - g.r[end]))).exp();
            qq[i] = -ll * pp[i] / (2.0 * (1.0 + (e - v[i]) / (2.0 * c * c)) * c) - pp[i] / g.r[i] / (2.0 * c);
        }
        for k in end + 1..m { pp[k] = 0.0; qq[k] = 0.0; }
        am_steps_in(&a, &mut pp, &mut qq, h, end - 2, t);
        let (pi, qi) = (pp[t], qq[t]);
        let mis = qo / po - qi / pi;
        // Scale the outer part to meet the inner one at t.
        let s = pi / po;
        // Bracket by the sign of the mismatch (it falls as the energy rises). Illinois rule: when
        // the same end is replaced twice in a row, the other end's value is halved, so the secant
        // steps cannot creep toward the root from one side only.
        if mis < 0.0 {
            if last_side == 1 { if let Some((a1, f1)) = fl { fl = Some((a1, 0.5 * f1)); } }
            e_hi = e; fh = Some((e, mis)); last_side = 1;
        } else {
            if last_side == -1 { if let Some((a1, f1)) = fh { fh = Some((a1, 0.5 * f1)); } }
            e_lo = e; fl = Some((e, mis)); last_side = -1;
        }
        // done when the energy is pinned to about 1e-12
        if (e_hi - e_lo).abs() < 1e-12 * e.abs().max(1.0) || mis.abs() < 1e-13 {
            for k in 0..t { pp[k] *= s; qq[k] *= s; }
            let norm = g.integrate(|i| pp[i] * pp[i]);
            let kk = 1.0 / norm.sqrt();
            for k in 0..m { pp[k] *= kk; }
            return Some((e, pp));
        }
        // Next energy: secant through the two bracket ends if both are known, else through the
        // last two trials, else bisection; always kept inside the bracket.
        let mut e_new = match (fl, fh, f_prev) {
            (Some((a1, f1)), Some((a2, f2)), _) if f1 != f2 => a1 - f1 * (a2 - a1) / (f2 - f1),
            (_, _, Some((a1, f1))) if f1 != mis => e - mis * (e - a1) / (mis - f1),
            _ => 0.5 * (e_lo + e_hi),
        };
        if !(e_new > e_lo && e_new < e_hi) { e_new = 0.5 * (e_lo + e_hi); }
        f_prev = Some((e, mis));
        e = e_new;
    }
    None
}

/// Adams-Moulton 4th order, outward from index `from` (needs from-2..=from filled) up to `to`.
fn am_steps(a: &[[[f64; 2]; 2]], p: &mut [f64], q: &mut [f64], h: f64, from: usize, to: usize, nodes: &mut usize, t: usize) {
    let fx = |i: usize, p: &[f64], q: &[f64]| (a[i][0][0] * p[i] + a[i][0][1] * q[i], a[i][1][0] * p[i] + a[i][1][1] * q[i]);
    for i in from..to {
        let (f0p, f0q) = fx(i, p, q);
        let (f1p, f1q) = fx(i - 1, p, q);
        let (f2p, f2q) = fx(i - 2, p, q);
        let rp = p[i] + h / 24.0 * (19.0 * f0p - 5.0 * f1p + f2p);
        let rq = q[i] + h / 24.0 * (19.0 * f0q - 5.0 * f1q + f2q);
        let k = 9.0 * h / 24.0;
        let b = a[i + 1];
        // (I - k B) y = r
        let (m00, m01, m10, m11) = (1.0 - k * b[0][0], -k * b[0][1], -k * b[1][0], 1.0 - k * b[1][1]);
        let det = m00 * m11 - m01 * m10;
        p[i + 1] = (m11 * rp - m01 * rq) / det;
        q[i + 1] = (m00 * rq - m10 * rp) / det;
        if i + 1 <= t && p[i + 1] * p[i] < 0.0 { *nodes += 1; }
        if p[i + 1].abs() > 1e150 { for k2 in 0..=i + 1 { p[k2] *= 1e-150; q[k2] *= 1e-150; } }
    }
}

/// Adams-Moulton 4th order, inward from `from` (needs from..=from+2 filled) down to `to`.
fn am_steps_in(a: &[[[f64; 2]; 2]], p: &mut [f64], q: &mut [f64], h: f64, from: usize, to: usize) {
    let fx = |i: usize, p: &[f64], q: &[f64]| (a[i][0][0] * p[i] + a[i][0][1] * q[i], a[i][1][0] * p[i] + a[i][1][1] * q[i]);
    let hh = -h;
    let mut i = from;
    while i > to {
        let (f0p, f0q) = fx(i, p, q);
        let (f1p, f1q) = fx(i + 1, p, q);
        let (f2p, f2q) = fx(i + 2, p, q);
        let rp = p[i] + hh / 24.0 * (19.0 * f0p - 5.0 * f1p + f2p);
        let rq = q[i] + hh / 24.0 * (19.0 * f0q - 5.0 * f1q + f2q);
        let k = 9.0 * hh / 24.0;
        let b = a[i - 1];
        let (m00, m01, m10, m11) = (1.0 - k * b[0][0], -k * b[0][1], -k * b[1][0], 1.0 - k * b[1][1]);
        let det = m00 * m11 - m01 * m10;
        p[i - 1] = (m11 * rp - m01 * rq) / det;
        q[i - 1] = (m00 * rq - m10 * rp) / det;
        if p[i - 1].abs() > 1e150 { let n = p.len(); for k2 in i - 1..n { p[k2] *= 1e-150; q[k2] *= 1e-150; } }
        i -= 1;
    }
}

/// VWN interpolation pieces: the fit G(x) with its parameters, and dG/dx.
fn vwn_fit(x: f64, a: f64, x0: f64, b: f64, c: f64) -> (f64, f64) {
    let xx = |y: f64| y * y + b * y + c;
    let q = (4.0 * c - b * b).sqrt();
    let xx0 = xx(x0);
    let at = (q / (2.0 * x + b)).atan();
    let g = a * ((x * x / xx(x)).ln() + 2.0 * b / q * at
        - b * x0 / xx0 * (((x - x0) * (x - x0) / xx(x)).ln() + 2.0 * (b + 2.0 * x0) / q * at));
    let d = q * q + (2.0 * x + b) * (2.0 * x + b);
    let dg = a * (2.0 / x - (2.0 * x + b) / xx(x) - 4.0 * b / d
        - b * x0 / xx0 * (2.0 / (x - x0) - (2.0 * x + b) / xx(x) - 4.0 * (b + 2.0 * x0) / d));
    (g, dg)
}

/// Exchange-correlation energy per electron and potential at density rho: Slater exchange with
/// the MacDonald-Vosko relativistic correction, VWN correlation.
pub fn xc(rho: f64) -> (f64, f64) {
    if rho <= 1e-300 { return (0.0, 0.0); }
    let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
    let x = rs.sqrt();
    let mut ex = -0.75 * (3.0 / PI * rho).powf(1.0 / 3.0);
    let mut vx = -(6.0 / PI * (0.5 * rho)).powf(1.0 / 3.0);
    // beta = p_F / (m c)
    let beta = (3.0 * PI * PI * rho).powf(1.0 / 3.0) / C_LIGHT;
    let mu = (1.0 + beta * beta).sqrt();
    let asb = (beta + mu).ln();
    let rr = 1.0 - 1.5 * ((beta * mu - asb) / (beta * beta)).powi(2);
    let ss = 1.5 * asb / (beta * mu) - 0.5;
    ex *= rr;
    vx *= ss;
    let (ec, dec_dx) = vwn_fit(x, 0.0310907, -0.10498, 3.72744, 12.9352);
    // v = e - (rs/3) de/drs, with de/drs = de/dx / (2x)
    let vc = ec - x / 6.0 * dec_dx;
    (ex + ec, vx + vc)
}

#[derive(Clone, Debug)]
pub struct Atom {
    pub shells: Vec<Shell>,
    /// Energy of each shell's orbital.
    pub eps: Vec<f64>,
    /// Radial function P of each shell.
    pub p: Vec<Vec<f64>>,
    pub rho: Vec<f64>,
    pub e_tot: f64,
}

/// Solve the atom with nuclear charge z and these shells; total energy converged to about `tol`.
pub fn solve(g: &Grid, z: f64, shells: &[Shell], tol: f64) -> Option<Atom> {
    let m = g.r.len();
    let mut v_in = thomas_fermi(g, z);
    let mut hist_v: Vec<Vec<f64>> = Vec::new();
    let mut hist_r: Vec<Vec<f64>> = Vec::new();
    let mut e_prev = 0.0;
    let mut eps = vec![0.0; shells.len()];
    let mut ps = vec![vec![0.0; m]; shells.len()];
    let mut calm = 0;
    for it in 0..400 {
        for (k, o) in shells.iter().enumerate() {
            let (e, p) = radial_state(g, &v_in, z, o.n, o.l)?;
            eps[k] = e;
            ps[k] = p;
        }
        // the density comes from P alone, as SRD 141 does
        let mut rho = vec![0.0; m];
        for (k, o) in shells.iter().enumerate() {
            for i in 0..m { rho[i] += o.occ * (ps[k][i] * ps[k][i]) / (4.0 * PI * g.r[i] * g.r[i]); }
        }
        let vh = hartree(g, &rho);
        let mut exc = vec![0.0; m];
        let mut v_out = vec![0.0; m];
        for i in 0..m {
            let (e, v) = xc(rho[i]);
            exc[i] = e;
            v_out[i] = -z / g.r[i] + vh[i] + v;
        }
        let w4 = |i: usize| 4.0 * PI * g.r[i] * g.r[i] * rho[i];
        let sum_eps: f64 = shells.iter().zip(&eps).map(|(o, e)| o.occ * e).sum();
        let e_kin = sum_eps - g.integrate(|i| w4(i) * v_in[i]);
        let e_coul = 0.5 * g.integrate(|i| w4(i) * vh[i]);
        let e_enuc = -z * g.integrate(|i| w4(i) / g.r[i]);
        let e_xc = g.integrate(|i| w4(i) * exc[i]);
        let e_tot = e_kin + e_coul + e_enuc + e_xc;
        let res: Vec<f64> = (0..m).map(|i| v_out[i] - v_in[i]).collect();
        let rnorm = g.integrate(|i| w4(i) * res[i].powi(2)).sqrt();
        calm = if (e_tot - e_prev).abs() < tol { calm + 1 } else { 0 };
        if it > 2 && (calm >= 3 || rnorm < 1e-9) && rnorm < 1e-6 {
            return Some(Atom { shells: shells.to_vec(), eps, p: ps, rho, e_tot });
        }
        e_prev = e_tot;
        // Pulay mixing of the potential; restart the history if the residual jumps.
        if let Some(last) = hist_r.last() {
            let lastn = g.integrate(|i| w4(i) * last[i].powi(2)).sqrt();
            if rnorm > 10.0 * lastn { hist_v.clear(); hist_r.clear(); }
        }
        hist_v.push(v_in.clone());
        hist_r.push(res);
        if hist_v.len() > 6 { hist_v.remove(0); hist_r.remove(0); }
        let k = hist_v.len();
        let beta = 0.3;
        let dot = |a: &[f64], b: &[f64]| g.integrate(|i| 4.0 * PI * g.r[i] * g.r[i] * (rho[i] + 1e-6) * a[i] * b[i]);
        let mut coef = vec![0.0; k];
        if k == 1 { coef[0] = 1.0; } else {
            let n = k + 1;
            let mut a = vec![vec![0.0; n + 1]; n];
            for i in 0..k { for j in 0..k { a[i][j] = dot(&hist_r[i], &hist_r[j]); } a[i][k] = 1.0; a[k][i] = 1.0; }
            a[k][n] = 1.0;
            match gauss(a) { Some(x) => coef.copy_from_slice(&x[..k]), None => { coef = vec![0.0; k]; coef[k - 1] = 1.0; } }
        }
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..k { s += coef[j] * (hist_v[j][i] + beta * hist_r[j][i]); }
            v_in[i] = s;
        }
    }
    None
}

/// Thomas-Fermi potential as a starting guess (Latter's fit to the screening function).
fn thomas_fermi(g: &Grid, z: f64) -> Vec<f64> {
    g.r.iter().map(|&r| {
        let x = r * z.powf(1.0 / 3.0) / 0.8853;
        let s = x.sqrt();
        let phi = 1.0 / (1.0 + 0.02747 * s + 1.243 * x - 0.1486 * x * s + 0.2302 * x * x + 0.007298 * x * x * s + 0.006944 * x * x * x);
        -(z * phi).max(1.0) / r
    }).collect()
}

/// Gaussian elimination with partial pivoting on an augmented matrix; None if singular.
fn gauss(mut a: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let n = a.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 { return None; }
        a.swap(c, p);
        for i in 0..n {
            if i == c { continue; }
            let f = a[i][c] / a[c][c];
            for j in c..=n { a[i][j] -= f * a[c][j]; }
        }
    }
    Some((0..n).map(|i| a[i][n] / a[i][i]).collect())
}

const LETTERS: &str = "spdfghi";

/// Parse "1s^2 2s^2 2p^6" (noble-gas cores like "[Ar]" allowed).
pub fn parse_config(s: &str) -> Vec<Shell> {
    let mut out: Vec<Shell> = Vec::new();
    for tok in s.split_whitespace() {
        if let Some(core) = tok.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
            let cfg = match core {
                "He" => "1s^2",
                "Ne" => "[He] 2s^2 2p^6",
                "Ar" => "[Ne] 3s^2 3p^6",
                "Kr" => "[Ar] 3d^10 4s^2 4p^6",
                "Xe" => "[Kr] 4d^10 5s^2 5p^6",
                "Rn" => "[Xe] 4f^14 5d^10 6s^2 6p^6",
                _ => continue,
            };
            out.extend(parse_config(cfg));
            continue;
        }
        let (nl, occ) = tok.split_once('^').unwrap_or((tok, "1"));
        let n: u32 = nl[..nl.len() - 1].parse().unwrap_or(1);
        let l = LETTERS.find(nl.chars().last().unwrap_or('s')).unwrap_or(0) as u32;
        out.push(Shell { n, l, occ: occ.parse().unwrap_or(1.0) });
    }
    out
}

pub fn shell_name(s: &Shell) -> String { format!("{}{}", s.n, LETTERS.as_bytes()[s.l as usize] as char) }

const NIST_CONFIGS: &str = include_str!("../data/nist_configurations.txt");

/// The ground configuration SRD 141 used for element z (1-92).
pub fn nist_config(z: u32) -> Vec<Shell> {
    for line in NIST_CONFIGS.lines() {
        let t = line.trim_start();
        let Some((zs, rest)) = t.split_once(char::is_whitespace) else { continue };
        if zs.parse::<u32>().ok() != Some(z) { continue; }
        // "Symbol   neutral...   cation...": the neutral atom's is the first of the two.
        let rest = rest.trim_start();
        let (_, cfgs) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let cfgs = cfgs.replace("</B>", "");
        let s = cfgs.trim().split("  ").map(|s| s.trim()).find(|s| !s.is_empty()).unwrap_or("");
        return parse_config(s);
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hartree_of_hydrogen() {
        let g = Grid::standard(1.0);
        let rho: Vec<f64> = g.r.iter().map(|r| (-2.0 * r).exp() / PI).collect();
        let v = hartree(&g, &rho);
        for (i, &r) in g.r.iter().enumerate().step_by(997) {
            let exact = 1.0 / r - (1.0 + 1.0 / r) * (-2.0 * r).exp();
            assert!((v[i] - exact).abs() < 1e-9 * (1.0 + exact.abs()), "r {r}: {} vs {exact}", v[i]);
        }
    }

    const SRD141: &str = include_str!("../data/nist_srd141.txt");

    fn reference(z: u32) -> (f64, Vec<(String, f64)>) {
        let start = SRD141.find(&format!("@ ScRLDA neutral {z} ")).unwrap();
        let (mut e_tot, mut levels) = (0.0, Vec::new());
        for line in SRD141[start..].lines().skip(1) {
            if line.starts_with('@') { break; }
            match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["Etot", "=", v] => e_tot = v.parse().unwrap(),
                [name, v] => levels.push((name.to_string(), v.parse().unwrap())),
                _ => {}
            }
        }
        (e_tot, levels)
    }

    const TOL_E: f64 = 1.5e-6;
    const TOL_LEVEL: f64 = 2.5e-6;

    fn check(z: u32) {
        let (e_tot, levels) = reference(z);
        let a = solve(&Grid::standard(z as f64), z as f64, &nist_config(z), 1e-9).unwrap_or_else(|| panic!("Z={z} did not converge"));
        assert!((a.e_tot - e_tot).abs() < TOL_E, "Z={z}: Etot {} vs {e_tot}", a.e_tot);
        let mut found = 0;
        for (o, e) in a.shells.iter().zip(&a.eps) {
            let lab = shell_name(o);
            let alt = if lab == "3d" { "32".to_string() } else { lab.clone() };
            if let Some((_, v)) = levels.iter().find(|(nm, _)| *nm == lab || *nm == alt) {
                found += 1;
                assert!((e - v).abs() < TOL_LEVEL, "Z={z}: {lab} {e} vs {v}");
            }
        }
        assert!(found >= levels.len(), "Z={z}: only {found} of {} levels matched", levels.len());
    }

    #[test]
    fn nist_srd141_sample() {
        for z in [1, 2, 8, 26, 29, 54, 64, 79, 92] { check(z); }
    }

    #[test]
    #[ignore]
    fn nist_srd141_all() {
        use rayon::prelude::*;
        (1..=92u32).into_par_iter().for_each(check);
    }
}
