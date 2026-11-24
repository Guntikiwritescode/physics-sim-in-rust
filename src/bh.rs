//! Barnes-Hut tree for gravity. Far groups of bodies pull
//! as one mass plus its quadrupole; near ones pull one by one. A group counts as far when its box
//! size s and the distance d from its centre of mass satisfy s / (d - offset) < THETA, where offset is
//! how far the centre of mass sits from the box centre (keeps the test safe for lopsided boxes).
//! The same tree finds touching pairs (boxes remember their largest body radius).

use rayon::prelude::*;

/// Opening angle: smaller is more exact and slower.
pub const THETA: f64 = 0.4;
const LEAF: usize = 8;
const NONE: u32 = u32::MAX;

#[derive(Clone)]
struct Node {
    c: [f64; 3],     // box centre
    h: f64,          // half the box size
    com: [f64; 3],   // centre of mass
    gm: f64,         // G * mass
    q: [f64; 6],     // G * quadrupole about com: xx, yy, zz, xy, xz, yz (traceless)
    off: f64,        // |com - c|
    rmax: f64,       // largest body radius inside
    child: [u32; 8], // children (NONE = empty); all NONE for a leaf
    start: u32,      // leaf: bodies idx[start..end]
    end: u32,
}

pub struct Tree {
    nodes: Vec<Node>,
    idx: Vec<u32>,
}

impl Tree {
    /// `pos`, `gm` (G * mass) and `rad` (radius) per body.
    pub fn build(pos: &[[f64; 3]], gm: &[f64], rad: &[f64]) -> Tree {
        let n = pos.len();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in pos { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
        let c = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
        let h = (0..3).map(|k| 0.5 * (hi[k] - lo[k])).fold(0.0, f64::max).max(1e-30) * 1.0001;
        let mut t = Tree { nodes: Vec::with_capacity(2 * n / LEAF + 16), idx: (0..n as u32).collect() };
        if n > 0 { t.make(0, n, c, h, pos, gm, rad, 0); }
        t
    }

    #[allow(clippy::too_many_arguments)]
    fn make(&mut self, s: usize, e: usize, c: [f64; 3], h: f64, pos: &[[f64; 3]], gm: &[f64], rad: &[f64], depth: u32) -> u32 {
        let me = self.nodes.len() as u32;
        self.nodes.push(Node { c, h, com: [0.0; 3], gm: 0.0, q: [0.0; 6], off: 0.0, rmax: 0.0, child: [NONE; 8], start: s as u32, end: e as u32 });
        if e - s > LEAF && depth < 48 {
            // Sort this range into the 8 octants (counting sort), then build each.
            let oct = |p: &[f64; 3]| ((p[0] >= c[0]) as usize) | (((p[1] >= c[1]) as usize) << 1) | (((p[2] >= c[2]) as usize) << 2);
            let mut count = [0usize; 8];
            for &i in &self.idx[s..e] { count[oct(&pos[i as usize])] += 1; }
            let mut at = [0usize; 8];
            let mut acc = s;
            for k in 0..8 { at[k] = acc; acc += count[k]; }
            let mut sorted = vec![0u32; e - s];
            let mut fill = at;
            for &i in &self.idx[s..e] { let k = oct(&pos[i as usize]); sorted[fill[k] - s] = i; fill[k] += 1; }
            self.idx[s..e].copy_from_slice(&sorted);
            let hh = 0.5 * h;
            for k in 0..8 {
                if count[k] == 0 { continue; }
                let cc = [c[0] + if k & 1 != 0 { hh } else { -hh }, c[1] + if k & 2 != 0 { hh } else { -hh }, c[2] + if k & 4 != 0 { hh } else { -hh }];
                let ch = self.make(at[k], at[k] + count[k], cc, hh, pos, gm, rad, depth + 1);
                self.nodes[me as usize].child[k] = ch;
            }
            // Mass, centre of mass and quadrupole from the children.
            let kids: Vec<u32> = self.nodes[me as usize].child.iter().copied().filter(|&x| x != NONE).collect();
            let m: f64 = kids.iter().map(|&k| self.nodes[k as usize].gm).sum();
            let mut com = [0.0; 3];
            if m > 0.0 { for &k in &kids { let n = &self.nodes[k as usize]; for d in 0..3 { com[d] += n.gm * n.com[d] / m; } } }
            else { com = c; }
            let mut q = [0.0; 6];
            let mut rmax: f64 = 0.0;
            for &k in &kids {
                let n = &self.nodes[k as usize];
                let d = [n.com[0] - com[0], n.com[1] - com[1], n.com[2] - com[2]];
                add_quad(&mut q, n.gm, d);
                for j in 0..6 { q[j] += n.q[j]; }
                rmax = rmax.max(n.rmax);
            }
            let nd = &mut self.nodes[me as usize];
            nd.gm = m; nd.com = com; nd.q = q; nd.rmax = rmax;
            nd.off = ((com[0] - c[0]).powi(2) + (com[1] - c[1]).powi(2) + (com[2] - c[2]).powi(2)).sqrt();
        } else {
            let ids = &self.idx[s..e];
            let m: f64 = ids.iter().map(|&i| gm[i as usize]).sum();
            let mut com = [0.0; 3];
            if m > 0.0 { for &i in ids { for d in 0..3 { com[d] += gm[i as usize] * pos[i as usize][d] / m; } } } else { com = c; }
            let mut q = [0.0; 6];
            let mut rmax: f64 = 0.0;
            for &i in ids {
                let p = pos[i as usize];
                add_quad(&mut q, gm[i as usize], [p[0] - com[0], p[1] - com[1], p[2] - com[2]]);
                rmax = rmax.max(rad[i as usize]);
            }
            let nd = &mut self.nodes[me as usize];
            nd.gm = m; nd.com = com; nd.q = q; nd.rmax = rmax;
            nd.off = ((com[0] - c[0]).powi(2) + (com[1] - c[1]).powi(2) + (com[2] - c[2]).powi(2)).sqrt();
        }
        me
    }

    /// Accelerations of all bodies (in parallel) and the touching pairs (i < j).
    /// Walks the tree once per leaf (a group of up to 8 bodies) instead of once per body: a box is
    /// far for the whole group when it passes the test from the nearest point of the group's box.
    pub fn accelerations(&self, pos: &[[f64; 3]], gm: &[f64], rad: &[f64]) -> (Vec<[f64; 3]>, Vec<(usize, usize)>) {
        self.accelerations_with(pos, gm, rad, THETA)
    }

    /// The same with another opening angle.
    pub fn accelerations_with(&self, pos: &[[f64; 3]], gm: &[f64], rad: &[f64], theta: f64) -> (Vec<[f64; 3]>, Vec<(usize, usize)>) {
        let leaves: Vec<u32> = (0..self.nodes.len() as u32).filter(|&k| self.nodes[k as usize].child.iter().all(|&c| c == NONE)).collect();
        let parts: Vec<Vec<(usize, [f64; 3], Vec<(usize, usize)>)>> = leaves.par_iter().map(|&lk| {
            let leaf = &self.nodes[lk as usize];
            let ids = &self.idx[leaf.start as usize..leaf.end as usize];
            // The group's own tight box.
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            let mut rmax: f64 = 0.0;
            for &i in ids { let p = pos[i as usize]; for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } rmax = rmax.max(rad[i as usize]); }
            let gc = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
            let gh = [0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2])];
            let mut far: Vec<u32> = Vec::with_capacity(256);
            let mut near: Vec<u32> = Vec::with_capacity(64);
            let mut stack: Vec<u32> = Vec::with_capacity(64);
            stack.push(0);
            while let Some(k) = stack.pop() {
                let n = &self.nodes[k as usize];
                let gap = box_box(gc, gh, n.c, n.h);
                let may_touch = gap <= rmax + n.rmax;
                let dmin = box_dist_v(n.com, gc, gh);   // nearest group point to the centre of mass
                if !may_touch && dmin > n.off && 2.0 * n.h < theta * (dmin - n.off) { far.push(k); continue; }
                if n.child.iter().all(|&c| c == NONE) { near.push(k); } else { for &c in &n.child { if c != NONE { stack.push(c); } } }
            }
            ids.iter().map(|&i| {
                let i = i as usize;
                let p = pos[i];
                let ri = rad[i];
                let mut a = [0.0; 3];
                let mut hits = Vec::new();
                for &k in &far {
                    let n = &self.nodes[k as usize];
                    let x = [p[0] - n.com[0], p[1] - n.com[1], p[2] - n.com[2]];
                    let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    let r = r2.sqrt();
                    let (r3, r5) = (r2 * r, r2 * r2 * r);
                    let r7 = r5 * r2;
                    let q = &n.q;
                    let qx = [q[0] * x[0] + q[3] * x[1] + q[4] * x[2], q[3] * x[0] + q[1] * x[1] + q[5] * x[2], q[4] * x[0] + q[5] * x[1] + q[2] * x[2]];
                    let xqx = x[0] * qx[0] + x[1] * qx[1] + x[2] * qx[2];
                    for c in 0..3 { a[c] += -n.gm * x[c] / r3 + qx[c] / r5 - 2.5 * xqx * x[c] / r7; }
                }
                for &k in &near {
                    let n = &self.nodes[k as usize];
                    for &j in &self.idx[n.start as usize..n.end as usize] {
                        let j = j as usize;
                        if j == i { continue; }
                        let dx = [pos[j][0] - p[0], pos[j][1] - p[1], pos[j][2] - p[2]];
                        let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                        let t = ri + rad[j];
                        if d2 < t * t && i < j { hits.push((i, j)); }
                        let dd = d2.max(0.25 * t * t);
                        let kk = gm[j] / (dd * dd.sqrt());
                        for c in 0..3 { a[c] += kk * dx[c]; }
                    }
                }
                (i, a, hits)
            }).collect()
        }).collect();
        let mut acc = vec![[0.0; 3]; pos.len()];
        let mut hits = Vec::new();
        for part in parts { for (i, a, h) in part { acc[i] = a; hits.extend(h); } }
        (acc, hits)
    }
}

/// Gap between two boxes (centre, half sizes per axis / half size); 0 if they overlap.
fn box_box(c1: [f64; 3], h1: [f64; 3], c2: [f64; 3], h2: f64) -> f64 {
    let mut s = 0.0;
    for k in 0..3 { let e = ((c1[k] - c2[k]).abs() - h1[k] - h2).max(0.0); s += e * e; }
    s.sqrt()
}

/// Distance from point p to a box with per-axis half sizes; 0 inside.
fn box_dist_v(p: [f64; 3], c: [f64; 3], h: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for k in 0..3 { let e = ((p[k] - c[k]).abs() - h[k]).max(0.0); s += e * e; }
    s.sqrt()
}

/// Add the quadrupole of mass gm at offset d: gm * (3 d d^T - |d|^2 I).
fn add_quad(q: &mut [f64; 6], gm: f64, d: [f64; 3]) {
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    q[0] += gm * (3.0 * d[0] * d[0] - r2);
    q[1] += gm * (3.0 * d[1] * d[1] - r2);
    q[2] += gm * (3.0 * d[2] * d[2] - r2);
    q[3] += gm * 3.0 * d[0] * d[1];
    q[4] += gm * 3.0 * d[0] * d[2];
    q[5] += gm * 3.0 * d[1] * d[2];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let mut rnd = move || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 11) as f64 / (1u64 << 53) as f64 };
        let mut pos = vec![[0.0; 3]];
        let mut gm = vec![1e20];
        let mut rad = vec![1e6];
        for i in 1..n {
            let (r, a) = (1e7 * (0.2 + rnd()), std::f64::consts::TAU * rnd());
            let z = if i % 5 == 0 { 3e6 * (rnd() - 0.5) * 4.0 } else { 2e5 * (rnd() - 0.5) };
            pos.push([r * a.cos() + if i % 7 == 0 { 2e6 } else { 0.0 }, z, r * a.sin()]);
            gm.push(1e10 * (0.1 + rnd()));
            rad.push(2e3 + 4e4 * rnd());
        }
        (pos, gm, rad)
    }

    fn direct(pos: &[[f64; 3]], gm: &[f64], rad: &[f64]) -> (Vec<[f64; 3]>, Vec<(usize, usize)>) {
        let mut acc = vec![[0.0; 3]; pos.len()];
        let mut hits = Vec::new();
        for i in 0..pos.len() { for j in 0..pos.len() {
            if i == j { continue; }
            let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let t = rad[i] + rad[j];
            if d2 < t * t && i < j { hits.push((i, j)); }
            let dd = d2.max(0.25 * t * t);
            let k = gm[j] / (dd * dd.sqrt());
            for c in 0..3 { acc[i][c] += k * d[c]; }
        }}
        (acc, hits)
    }

    #[test]
    fn tree_matches_direct_sum() {
        for n in [600, 5000] {
            let (pos, gm, rad) = cloud(n, 7 + n as u64);
            let t = Tree::build(&pos, &gm, &rad);
            let (a, mut h) = t.accelerations(&pos, &gm, &rad);
            let (b, mut hd) = direct(&pos, &gm, &rad);
            let mut err: Vec<f64> = a.iter().zip(&b).map(|(x, y)| {
                let e = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt();
                e / (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt()
            }).collect();
            err.sort_by(f64::total_cmp);
            let (med, p99) = (err[err.len() / 2], err[err.len() * 99 / 100]);
            assert!(med < 2e-4 && p99 < 1e-3, "median {med}, p99 {p99}");
            h.sort(); hd.sort();
            assert_eq!(h, hd);
        }
    }

    #[test]
    fn tree_matches_direct_sum_in_a_cluster() {
        let n = 4000;
        let mut s = 11u64;
        let mut rnd = move || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 11) as f64 / (1u64 << 53) as f64 };
        let mut pos = Vec::new();
        for _ in 0..n {
            let r = 1e10 / (rnd().max(1e-9).powf(-2.0 / 3.0) - 1.0).sqrt();
            let (u, ph) = (2.0 * rnd() - 1.0, std::f64::consts::TAU * rnd());
            let st = (1.0 - u * u).sqrt();
            pos.push([r * st * ph.cos(), r * st * ph.sin(), r * u]);
        }
        let gm = vec![1e20; n];
        let rad = vec![1e3; n];
        let t = Tree::build(&pos, &gm, &rad);
        let (a, _) = t.accelerations(&pos, &gm, &rad);
        let (b, _) = direct(&pos, &gm, &rad);
        let mut err: Vec<f64> = a.iter().zip(&b).map(|(x, y)| {
            let e = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt();
            e / (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt()
        }).collect();
        err.sort_by(f64::total_cmp);
        let (med, p99) = (err[n / 2], err[n * 99 / 100]);
        assert!(med < 2e-4 && p99 < 1e-3, "median {med}, p99 {p99}");
    }

    fn direct_par(pos: &[[f64; 3]], gm: &[f64], rad: &[f64]) -> Vec<[f64; 3]> {
        (0..pos.len()).into_par_iter().map(|i| {
            let mut a = [0.0; 3];
            for j in 0..pos.len() {
                let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let t = rad[i] + rad[j];
                let dd = d2.max(0.25 * t * t);
                let k = if i == j { 0.0 } else { gm[j] / (dd * dd.sqrt()) };
                for c in 0..3 { a[c] += k * d[c]; }
            }
            a
        }).collect()
    }

    #[test]
    fn tree_drift_matches_direct_sum() {
        let n = 5000;
        let mut s = 99u64;
        let mut rnd = move || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 11) as f64 / (1u64 << 53) as f64 };
        let dir = |rnd: &mut dyn FnMut() -> f64| loop {
            let v = [2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 1e-3 && l <= 1.0 { break [v[0] / l, v[1] / l, v[2] / l]; }
        };
        let pos0: Vec<[f64; 3]> = (0..n).map(|_| { let d = dir(&mut rnd); let r = ((rnd() * 0.99).max(1e-9).powf(-2.0 / 3.0) - 1.0).powf(-0.5); [d[0] * r, d[1] * r, d[2] * r] }).collect();
        let gm = vec![1.0 / n as f64; n];
        let rad = vec![0.005; n];
        let energy = |pos: &[[f64; 3]], vel: &[[f64; 3]]| -> f64 {
            let pe: f64 = (0..n).into_par_iter().map(|i| {
                let mut e = 0.0;
                for j in i + 1..n {
                    let d = ((pos[j][0] - pos[i][0]).powi(2) + (pos[j][1] - pos[i][1]).powi(2) + (pos[j][2] - pos[i][2]).powi(2)).sqrt();
                    let sft = 0.5 * (rad[i] + rad[j]);
                    e -= gm[i] * gm[j] * if d >= sft { 1.0 / d } else { (3.0 * sft * sft - d * d) / (2.0 * sft * sft * sft) };
                }
                e
            }).sum();
            let ke: f64 = (0..n).map(|i| 0.5 * gm[i] * (vel[i][0].powi(2) + vel[i][1].powi(2) + vel[i][2].powi(2))).sum();
            pe + ke
        };
        let w = energy(&pos0, &vec![[0.0; 3]; n]);
        let mut vel0: Vec<[f64; 3]> = (0..n).map(|_| { let d = dir(&mut rnd); let v = rnd(); [d[0] * v, d[1] * v, d[2] * v] }).collect();
        let mean = [0, 1, 2].map(|c| vel0.iter().map(|v| v[c]).sum::<f64>() / n as f64);
        for v in vel0.iter_mut() { for c in 0..3 { v[c] -= mean[c]; } }
        let k: f64 = vel0.iter().map(|v| 0.5 / n as f64 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])).sum();
        let f = (-0.5 * w / k).sqrt();
        for v in vel0.iter_mut() { for c in 0..3 { v[c] *= f; } }
        let run = |tree: bool| {
            let (mut pos, mut vel) = (pos0.clone(), vel0.clone());
            let acc = |p: &[[f64; 3]]| if tree { Tree::build(p, &gm, &rad).accelerations(p, &gm, &rad).0 } else { direct_par(p, &gm, &rad) };
            let dt = 0.005;
            let mut a = acc(&pos);
            for _ in 0..200 {
                for i in 0..n { for c in 0..3 { vel[i][c] += 0.5 * dt * a[i][c]; pos[i][c] += dt * vel[i][c]; } }
                a = acc(&pos);
                for i in 0..n { for c in 0..3 { vel[i][c] += 0.5 * dt * a[i][c]; } }
            }
            let p = [0, 1, 2].map(|c| (0..n).map(|i| gm[i] * vel[i][c]).sum::<f64>());
            (energy(&pos, &vel), (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        };
        let e0 = energy(&pos0, &vel0);
        let (ed, pd) = run(false);
        let (et, pt) = run(true);
        let pscale: f64 = (0..n).map(|i| gm[i] * (vel0[i][0].powi(2) + vel0[i][1].powi(2) + vel0[i][2].powi(2)).sqrt()).sum();
        let (dd, dt_) = ((ed - e0) / e0.abs(), (et - e0) / e0.abs());
        assert!((dt_ - dd).abs() < 1e-4, "tree energy drift {dt_} against exact {dd}");
        assert!(pt / pscale < 1e-4 && pd / pscale < 1e-12, "momentum: tree {pt}, exact {pd}");
    }
}
