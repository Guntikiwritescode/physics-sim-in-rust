//! CPU copy of the black hole ray maths in rt_hole.wgsl (f64), for the accuracy tests against
//! exact results (shadow edge, bending angle).
//! Spinning (Kerr) black hole in Cartesian Kerr-Schild coordinates, following the equations
//! in the GRay2 paper (Chan et al. 2018, ApJ 867, 59). Outgoing form (l_t = -1), so rays traced
//! backwards from the camera cross the horizon smoothly.
//! Units: r_s = 1, so the mass M = 0.5. Spin axis = +Z of these coordinates.
//! Hamiltonian H = 1/2 g^{mu nu} p_mu p_nu with g^{mu nu} = eta - f l^mu l^nu.

pub type V3 = [f64; 3];
pub const M: f64 = 0.5;
/// Step = STEP_K * r, halved inside r = 3 (as rt_hole.wgsl).
pub const STEP_K: f64 = 0.15;

pub fn add(a: V3, b: V3, s: f64) -> V3 { [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]] }
pub fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn scale(a: V3, s: f64) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn len(a: V3) -> f64 { dot(a, a).sqrt() }

/// Horizon radius for spin length a (= spin * M).
pub fn r_plus(a: f64) -> f64 { M + (M * M - a * a).max(0.0).sqrt() }

/// Innermost stable circular orbit, prograde (Bardeen, Press & Teukolsky 1972). spin in 0..1.
pub fn isco(spin: f64) -> f64 {
    let s = spin;
    let z1 = 1.0 + (1.0 - s * s).cbrt() * ((1.0 + s).cbrt() + (1.0 - s).cbrt());
    let z2 = (3.0 * s * s + z1 * z1).sqrt();
    M * (3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt())
}

/// Kerr-Schild radius r (implicit: r^4 - (R^2 - a^2) r^2 - a^2 z^2 = 0).
pub fn ks_r(x: V3, a: f64) -> f64 {
    let a2 = a * a;
    let aa = dot(x, x) - a2;
    let q = (aa * aa + 4.0 * a2 * x[2] * x[2]).sqrt();
    (0.5 * (aa + q)).max(1e-12).sqrt()
}

struct Geo { r: f64, f: f64, l: V3, grad_r: V3, d: f64, s: f64 }

fn geo(x: V3, a: f64) -> Geo {
    let a2 = a * a;
    let aa = dot(x, x) - a2;
    let q = (aa * aa + 4.0 * a2 * x[2] * x[2]).sqrt().max(1e-12);
    let r2 = (0.5 * (aa + q)).max(1e-12);
    let r = r2.sqrt();
    let s = r2 * r2 + a2 * x[2] * x[2];
    let f = 2.0 * M * r * r2 / s;
    let d = r2 + a2;
    // Outgoing form = time-reversed ingoing form, which flips the spin; using -a in l keeps the
    // spin along +Z.
    let ao = -a;
    let l = [(r * x[0] + ao * x[1]) / d, (r * x[1] - ao * x[0]) / d, x[2] / r];
    let k = 1.0 / (r * q);
    let grad_r = [x[0] * r2 * k, x[1] * r2 * k, (x[2] * r2 + a2 * x[2]) * k];
    Geo { r, f, l, grad_r, d, s }
}

#[allow(dead_code)] // used by the tests
/// Covariant photon momentum: pt (conserved, -energy) and spatial p.
pub fn hamiltonian(x: V3, pt: f64, p: V3, a: f64) -> f64 {
    let g = geo(x, a);
    let ll = pt + dot(g.l, p);
    0.5 * (-pt * pt + dot(p, p) - g.f * ll * ll)
}

/// (dx/dlambda, dp/dlambda).
pub fn derivs(x: V3, pt: f64, p: V3, a: f64) -> (V3, V3) {
    let g = geo(x, a);
    let (r, d) = (g.r, g.d);
    let ll = pt + dot(g.l, p);
    let xdot = add(p, g.l, -g.f * ll);
    // grad f
    let r2 = r * r;
    let cf = 2.0 * M * r2 * (3.0 * g.s - 4.0 * r2 * r2) / (g.s * g.s);
    let mut gf = scale(g.grad_r, cf);
    gf[2] += -4.0 * M * r2 * r * a * a * x[2] / (g.s * g.s);
    // grad (l . p) with p held fixed
    let c = (x[0] * p[0] + x[1] * p[1]) / d - 2.0 * r * (g.l[0] * p[0] + g.l[1] * p[1]) / d - p[2] * x[2] / r2;
    let mut gl = scale(g.grad_r, c);
    let ao = -a; // see geo()
    gl[0] += (r * p[0] - ao * p[1]) / d;
    gl[1] += (r * p[1] + ao * p[0]) / d;
    gl[2] += p[2] / r;
    let pdot = add(scale(gf, 0.5 * ll * ll), gl, g.f * ll);
    (xdot, pdot)
}

/// 4-vector as (t, spatial). Metric dot product g_{mu nu} u^mu v^nu, g = eta + f l l, l_t = -1.
pub fn gdot(x: V3, a: f64, u: (f64, V3), v: (f64, V3)) -> f64 {
    let g = geo(x, a);
    let lu = -u.0 + dot(g.l, u.1);
    let lv = -v.0 + dot(g.l, v.1);
    -u.0 * v.0 + dot(u.1, v.1) + g.f * lu * lv
}

/// Lower an index: covariant (t, spatial) from contravariant.
pub fn lower(x: V3, a: f64, u: (f64, V3)) -> (f64, V3) {
    let g = geo(x, a);
    let lu = -u.0 + dot(g.l, u.1);
    (-u.0 - g.f * lu, add(u.1, g.l, g.f * lu))
}

/// Zero-angular-momentum observer (does not orbit relative to distant space); static when a = 0.
pub fn zamo(x: V3, a: f64) -> (f64, V3) {
    let xt = (1.0, [0.0; 3]);
    let xp = (0.0, [-x[1], x[0], 0.0]);
    let w = -gdot(x, a, xt, xp) / gdot(x, a, xp, xp).max(1e-12);
    let v = (1.0, scale(xp.1, w));
    let n = 1.0 / (-gdot(x, a, v, v)).max(1e-12).sqrt();
    (n, scale(v.1, n))
}

/// Backward ray for a camera at x looking along coordinate direction `dir` (up hint `up`).
/// Returns covariant (pt, p) of the time-reversed photon. Photon energy seen by the camera = 1.
pub fn camera_ray(x: V3, a: f64, dir: V3, up: V3) -> (f64, V3) {
    let u = zamo(x, a);
    let proj = |w: V3, basis: &[(f64, V3)]| -> (f64, V3) {
        let mut v = (0.0, w);
        let k = gdot(x, a, v, u);
        v = (v.0 + k * u.0, add(v.1, u.1, k));
        for e in basis {
            let k = gdot(x, a, v, *e);
            v = (v.0 - k * e.0, add(v.1, e.1, -k));
        }
        let n = gdot(x, a, v, v).max(1e-30).sqrt();
        (v.0 / n, scale(v.1, 1.0 / n))
    };
    let ef = proj(dir, &[]);
    let _eu = proj(up, &[ef]);
    // p' = -u + n  (n = ef here: the ray goes straight out of the camera)
    lower(x, a, (-u.0 + ef.0, add(scale(u.1, -1.0), ef.1, 1.0)))
}

#[derive(Debug, PartialEq)]
pub enum Fate { Captured, Escaped(V3), OutOfSteps }

/// Same step rules as the shader: spatial step = k * r (2x finer inside r < 3), RK4 in lambda.
pub fn trace(x0: V3, pt: f64, p0: V3, a: f64, k_far: f64, r_far: f64, max_steps: usize) -> (Fate, usize) {
    let (mut x, mut p) = (x0, p0);
    let rh = r_plus(a);
    for i in 0..max_steps {
        let r = ks_r(x, a);
        let (k1x, k1p) = derivs(x, pt, p, a);
        if r < rh + 0.01 { return (Fate::Captured, i); }
        let outward = dot(x, k1x) > 0.0;
        if len(x) > r_far && outward { return (Fate::Escaped(scale(k1x, 1.0 / len(k1x))), i); }
        let mut ds = k_far * r;
        if r < 3.0 { ds *= 0.5; }
        let h = ds / len(k1x).max(1e-12);
        let (k2x, k2p) = derivs(add(x, k1x, 0.5 * h), pt, add(p, k1p, 0.5 * h), a);
        let (k3x, k3p) = derivs(add(x, k2x, 0.5 * h), pt, add(p, k2p, 0.5 * h), a);
        let (k4x, k4p) = derivs(add(x, k3x, h), pt, add(p, k3p, h), a);
        for c in 0..3 {
            x[c] += h / 6.0 * (k1x[c] + 2.0 * k2x[c] + 2.0 * k3x[c] + k4x[c]);
            p[c] += h / 6.0 * (k1p[c] + 2.0 * k2p[c] + 2.0 * k3p[c] + k4p[c]);
        }
    }
    (Fate::OutOfSteps, max_steps)
}

/// Ray from far away on the -X side, heading +X, offset y0 in the disk plane (Z = 0).
pub fn far_ray(y0: f64, a: f64, far: f64) -> (V3, f64, V3) {
    let x = [-far, y0, 0.0];
    let (pt, p) = camera_ray(x, a, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    (x, pt, p)
}

/// Edge of the shadow on one side, by bisection on the offset |y0| (sign picks the side).
/// side = +1: photon going round with the spin (prograde); -1: against it.
pub fn critical_b(a: f64, side: f64, k_far: f64) -> f64 {
    let (mut lo, mut hi) = (0.5, 5.0);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        let (x, pt, p) = far_ray(side * m, a, 2000.0);
        let (f, _) = trace(x, pt, p, a, k_far, 2001.0, 100_000);
        if f == Fate::Captured { lo = m } else { hi = m }
    }
    lo
}

/// Bend angle of a ray aimed at distance b from very far away (a = 0).
pub fn deflection(b: f64, k_far: f64) -> f64 {
    let far = 20_000.0;
    let (x, pt, p) = far_ray(b, 0.0, far);
    match trace(x, pt, p, 0.0, k_far, far, 100_000).0 {
        Fate::Escaped(v) => -v[1].atan2(v[0]),
        f => panic!("ray at b={b} did not escape: {f:?}"),
    }
}

/// Weak-field bend to second order: 2/b + (15 pi / 16) / b^2  (r_s = 1 units).
pub fn deflection_expected(b: f64) -> f64 {
    2.0 / b + 15.0 * std::f64::consts::PI / 16.0 / (b * b)
}

/// Exact shadow-edge aim distance in the disk plane (circular photon orbits, Bardeen 1973).
/// prograde = true: with the spin. Returns |b|.
pub fn critical_b_exact(spin: f64, prograde: bool) -> f64 {
    let a = spin * M;
    if a == 0.0 { return 3.0 * 3f64.sqrt() * M; }
    let sgn = if prograde { -1.0 } else { 1.0 };
    let r = 2.0 * M * (1.0 + ((2.0 / 3.0) * (sgn * spin).acos()).cos());
    (-(r * r * r - 3.0 * M * r * r + a * a * r + a * a * M) / (a * (r - M))).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    const K: f64 = STEP_K;

    #[test]
    fn photon_capture_limit_no_spin() {
        let exact = 1.5 * 3f64.sqrt();
        for side in [1.0, -1.0] {
            let b = critical_b(0.0, side, K);
            assert!((b - exact).abs() / exact < 0.01, "b={b} exact={exact}");
        }
    }

    #[test]
    fn shadow_edges_with_spin() {
        for spin in [0.5, 0.9] {
            for (side, pro) in [(1.0, true), (-1.0, false)] {
                let b = critical_b(spin * M, side, K);
                let exact = critical_b_exact(spin, pro);
                assert!((b - exact).abs() / exact < 0.01, "spin {spin} pro {pro}: b={b} exact={exact}");
            }
        }
    }

    #[test]
    fn bend_far_rays() {
        for b in [100.0, 500.0] {
            let got = deflection(b, K);
            let exp = deflection_expected(b);
            assert!((got - exp).abs() / exp < 0.01, "b={b} got={got} exp={exp}");
        }
    }

    #[test]
    fn known_radii() {
        assert!((r_plus(0.9 * M) - 0.7179).abs() < 1e-4);
        assert!((isco(0.0) - 3.0).abs() < 1e-9);
        assert!((isco(0.9) - 1.1604).abs() < 1e-4);
        assert!((critical_b_exact(0.9, true) - 1.4222).abs() < 1e-4);
        assert!((critical_b_exact(0.9, false) - 3.4162).abs() < 1e-4);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let a = 0.45;
        let (pt, p) = (0.8, [0.3, -0.5, 0.4]);
        for x in [[3.0, 1.0, 0.7], [0.4, 1.1, -0.9], [-6.0, 2.0, 0.1]] {
            let (_, pd) = derivs(x, pt, p, a);
            for i in 0..3 {
                let e = 1e-6;
                let mut xp = x; xp[i] += e;
                let mut xm = x; xm[i] -= e;
                let fd = -(hamiltonian(xp, pt, p, a) - hamiltonian(xm, pt, p, a)) / (2.0 * e);
                assert!((fd - pd[i]).abs() < 1e-6 * (1.0 + fd.abs()), "x={x:?} i={i} fd={fd} an={}", pd[i]);
            }
        }
    }

    #[test]
    fn conserved_along_a_strong_bend() {
        let a = 0.9 * M;
        let (x0, pt, p0) = far_ray(-3.6, a, 60.0);
        let lz = |x: V3, p: V3| x[0] * p[1] - x[1] * p[0];
        let l0 = lz(x0, p0);
        let (mut x, mut p) = (x0, p0);
        for _ in 0..2000 {
            let r = ks_r(x, a);
            let (k1x, k1p) = derivs(x, pt, p, a);
            if len(x) > 61.0 && dot(x, k1x) > 0.0 { break; }
            assert!(r > r_plus(a), "ray fell in");
            let ds = K * r * if r < 3.0 { 0.5 } else { 1.0 };
            let h = ds / len(k1x);
            let (k2x, k2p) = derivs(add(x, k1x, 0.5 * h), pt, add(p, k1p, 0.5 * h), a);
            let (k3x, k3p) = derivs(add(x, k2x, 0.5 * h), pt, add(p, k2p, 0.5 * h), a);
            let (k4x, k4p) = derivs(add(x, k3x, h), pt, add(p, k3p, h), a);
            for c in 0..3 {
                x[c] += h / 6.0 * (k1x[c] + 2.0 * k2x[c] + 2.0 * k3x[c] + k4x[c]);
                p[c] += h / 6.0 * (k1p[c] + 2.0 * k2p[c] + 2.0 * k3p[c] + k4p[c]);
            }
        }
        let (h1, l1) = (hamiltonian(x, pt, p, a), lz(x, p));
        assert!(h1.abs() < 1e-4, "H drifted to {h1}");
        assert!((l1 - l0).abs() / l0.abs() < 1e-4, "Lz drifted {l0} -> {l1}");
    }
}
