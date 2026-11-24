//! Gravity sim physics (from kavan010/gravity_sim). Every body pulls every other (Newton, run on
//! all CPU cores), leapfrog steps. Touching bodies merge, bounce or shatter depending on how hard
//! they hit; each keeps total mass and momentum. SI units.

use rayon::prelude::*;

pub const G: f64 = 6.6743e-11;
/// Bodies are drawn and collide at 1/SIZE_RATIO of the size their density gives (gravity_sim's
/// sizeRatio).
pub const SIZE_RATIO: f64 = 30.0;
/// Above this many bodies gravity uses the Barnes-Hut tree instead of the exact pair sum.
pub const TREE_FROM: usize = 4096;

/// Collision outcome by impact speed v compared with the pair's escape speed v_esc
/// = sqrt(2 G (m1 + m2) / (r1 + r2)) (the measure Leinhardt & Stewart 2012 use):
/// v < v_esc: merge.  v_esc <= v < SHATTER_SPEED v_esc: bounce.  Faster: shatter.
/// The published model has 8 outcomes; this keeps 3.
pub const SHATTER_SPEED: f64 = 3.0;
/// Bounciness (coefficient of restitution, as REBOUND's hard-sphere method): share of the
/// approach speed along the line of centres kept after a bounce.
pub const RESTITUTION: f64 = 0.5;
/// A shatter makes one big remnant (half the mass) plus up to 6 equal pieces; pieces lighter
/// than this merge instead (REBOUND's fragmentation module has a minimum fragment mass too).
pub const MIN_PIECE_KG: f64 = 1e19;
/// Shatters stop (pairs merge instead) above this many bodies, so the count can't blow up.
pub const MAX_BODIES: usize = 6000;
/// Share of the pair's relative kinetic energy the pieces fly apart with (the rest is heat).
pub const SHATTER_KEEP: f64 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome { Merge, Bounce, Shatter }

/// What a body is. A black hole swallows anything that touches its horizon; its size is the
/// horizon, r+ = G M / c^2 (1 + sqrt(1 - a*^2)), not mass / density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind { Rock, Star { temp_k: f64 }, Hole { spin: f64 } }

pub const C_LIGHT: f64 = 299_792_458.0;
/// Spin limit for a hole spun up by infalling matter (Thorne 1974).
pub const SPIN_MAX: f64 = 0.998;

#[derive(Clone, Debug)]
pub struct Body {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub density: f64, // kg/m^3
    pub color: [f32; 3],
    pub glow: bool,
    pub kind: Kind,
    /// Drawn and collides at 1/shrink of its true size (SIZE_RATIO in gravity_sim's scene,
    /// 1 in the true-size world).
    pub shrink: f64,
}

impl Body {
    pub fn radius(&self) -> f64 {
        if let Kind::Hole { spin } = self.kind {
            let m = G * self.mass / (C_LIGHT * C_LIGHT);
            return m * (1.0 + (1.0 - spin * spin).max(0.0).sqrt());
        }
        (3.0 * self.mass / (4.0 * std::f64::consts::PI * self.density)).cbrt() / self.shrink
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }

pub struct World {
    pub bodies: Vec<Body>,
    cache: Option<Vec<[f64; 3]>>, // accelerations from the end of the last step (if no body changed)
    pub bounces: usize,
    pub shatters: usize,
    pub restitution: f64,   // bounciness (starts at RESTITUTION; the panel can change it)
    pub shatter_speed: f64, // shatter above this many escape speeds (starts at SHATTER_SPEED)
    /// Wall-clock time allowed per `advance` call in ms (0 = no limit). With a limit, steps keep
    /// their safe length and the time that doesn't fit is not simulated (the world runs slower than
    /// asked instead of less accurately); `last_done` says how much was simulated.
    pub budget_ms: f64,
    pub last_done: f64,
}

impl World {
    pub fn new(bodies: Vec<Body>) -> World { World { bodies, cache: None, bounces: 0, shatters: 0, restitution: RESTITUTION, shatter_speed: SHATTER_SPEED, budget_ms: 0.0, last_done: 0.0 } }

    /// What a touching pair does: merge, bounce or shatter. None if they are already moving apart.
    pub fn outcome(&self, i: usize, j: usize) -> Option<Outcome> {
        let (a, b) = (&self.bodies[i], &self.bodies[j]);
        let d = sub(b.pos, a.pos);
        let vr = sub(b.vel, a.vel);
        if matches!(a.kind, Kind::Hole { .. }) || matches!(b.kind, Kind::Hole { .. }) { return Some(Outcome::Merge); }
        if dot(vr, d) >= 0.0 { return None; }
        let v = dot(vr, vr).sqrt();
        let vesc = (2.0 * G * (a.mass + b.mass) / (a.radius() + b.radius())).sqrt();
        Some(if v < vesc { Outcome::Merge } else if v < self.shatter_speed * vesc { Outcome::Bounce } else { Outcome::Shatter })
    }

    /// Bounce i and j off each other along the line of centres (hard-sphere impulse), then move
    /// them apart to just touching. Momentum and centre of mass unchanged.
    fn bounce(&mut self, i: usize, j: usize) {
        let (ma, mb) = (self.bodies[i].mass, self.bodies[j].mass);
        let d = sub(self.bodies[j].pos, self.bodies[i].pos);
        let dl = dot(d, d).sqrt().max(1e-9);
        let n = d.map(|x| x / dl);
        let vn = dot(sub(self.bodies[j].vel, self.bodies[i].vel), n); // < 0: approaching
        let mu = ma * mb / (ma + mb);
        let jimp = -(1.0 + self.restitution) * mu * vn;
        let gap = self.bodies[i].radius() + self.bodies[j].radius() - dl;
        for c in 0..3 {
            self.bodies[i].vel[c] -= jimp / ma * n[c];
            self.bodies[j].vel[c] += jimp / mb * n[c];
            if gap > 0.0 {
                self.bodies[i].pos[c] -= gap * mb / (ma + mb) * n[c];
                self.bodies[j].pos[c] += gap * ma / (ma + mb) * n[c];
            }
        }
        self.bounces += 1;
    }

    /// Pieces from a shatter of i and j: one remnant with half the mass at the centre of mass,
    /// plus 2, 4 or 6 equal pieces in opposite pairs (so momentum and centre of mass are exact).
    /// None if the pieces would be lighter than MIN_PIECE_KG.
    fn shatter(&self, i: usize, j: usize) -> Option<Vec<Body>> {
        let (a, b) = (&self.bodies[i], &self.bodies[j]);
        let m = a.mass + b.mass;
        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        let mut col = [0.0f32; 3];
        for c in 0..3 {
            pos[c] = (a.mass * a.pos[c] + b.mass * b.pos[c]) / m;
            vel[c] = (a.mass * a.vel[c] + b.mass * b.vel[c]) / m;
            col[c] = ((a.mass * a.color[c] as f64 + b.mass * b.color[c] as f64) / m) as f32;
        }
        let density = m / (a.mass / a.density + b.mass / b.density);
        let pairs = [3usize, 2, 1].into_iter().find(|&p| 0.5 * m / (2 * p) as f64 >= MIN_PIECE_KG)?;
        let piece = 0.5 * m / (2 * pairs) as f64;
        let vr = sub(b.vel, a.vel);
        let ke_rel = 0.5 * a.mass * b.mass / m * dot(vr, vr);
        // All pieces at the same speed u: sum of 0.5 piece u^2 = SHATTER_KEEP * ke_rel.
        let u = (SHATTER_KEEP * ke_rel / (piece * pairs as f64)).sqrt();
        // Directions: the pair's approach line and two at right angles to it.
        let vl = dot(vr, vr).sqrt();
        let e0 = vr.map(|x| x / vl);
        let helper = if e0[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let cross = |p: [f64; 3], q: [f64; 3]| [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
        let e1 = { let c = cross(e0, helper); let l = dot(c, c).sqrt(); c.map(|x| x / l) };
        let e2 = cross(e0, e1);
        let big = Body { pos, vel, mass: 0.5 * m, density, color: col, glow: a.glow || b.glow, kind: if a.mass >= b.mass { a.kind } else { b.kind }, shrink: a.shrink };
        let small = Body { pos, vel, mass: piece, density, color: col, glow: false, kind: Kind::Rock, shrink: a.shrink };
        let gap = 1.2 * (big.radius() + small.radius());
        let mut out = vec![big];
        for e in [e1, e2, e0].iter().take(pairs) {
            for s in [1.0, -1.0] {
                let mut p = small.clone();
                for c in 0..3 { p.pos[c] += s * gap * e[c]; p.vel[c] += s * u * e[c]; }
                out.push(p);
            }
        }
        Some(out)
    }

    /// Settle all touching pairs: bounces and shatters first (each body at most once per call),
    /// then merges (in groups). Returns true if any body was added, removed or changed speed.
    pub fn resolve(&mut self, hits: &[(usize, usize)]) -> bool {
        if hits.is_empty() { return false; }
        let n = self.bodies.len();
        let mut used = vec![false; n];
        let mut dead = vec![false; n];
        let mut merges = Vec::new();
        let mut pieces = Vec::new();
        let mut changed = false;
        for &(i, j) in hits {
            match self.outcome(i, j) {
                None => {}
                Some(Outcome::Merge) => merges.push((i, j)),
                Some(_) if used[i] || used[j] => {}
                Some(Outcome::Bounce) => { self.bounce(i, j); used[i] = true; used[j] = true; changed = true; }
                Some(Outcome::Shatter) => {
                    let room = n + pieces.len() + 6 <= MAX_BODIES;
                    match self.shatter(i, j).filter(|_| room) {
                        Some(p) => { pieces.extend(p); dead[i] = true; dead[j] = true; self.shatters += 1; }
                        None => merges.push((i, j)),
                    }
                    used[i] = true; used[j] = true; changed = true;
                }
            }
        }
        let merges: Vec<(usize, usize)> = merges.into_iter().filter(|&(i, j)| !dead[i] && !dead[j]).collect();
        if !pieces.is_empty() {
            // Drop shattered bodies, fixing up the merge pairs' indices.
            let mut new_idx = vec![usize::MAX; n];
            let mut k = 0;
            for i in 0..n { if !dead[i] { new_idx[i] = k; k += 1; } }
            let mut keep = dead.iter();
            self.bodies.retain(|_| !*keep.next().unwrap());
            let merges: Vec<(usize, usize)> = merges.iter().map(|&(i, j)| (new_idx[i], new_idx[j])).collect();
            self.merge(&merges);
            self.bodies.extend(pieces);
            return true;
        }
        self.merge(&merges) > 0 || changed
    }

    /// Acceleration on every body, plus the pairs that touch (i < j). Up to TREE_FROM bodies: the
    /// exact sum over all pairs; above: the Barnes-Hut tree (bh.rs).
    pub fn accelerations(&self) -> (Vec<[f64; 3]>, Vec<(usize, usize)>) {
        if self.bodies.len() > TREE_FROM {
            let b = &self.bodies;
            let pos: Vec<[f64; 3]> = b.iter().map(|x| x.pos).collect();
            let gm: Vec<f64> = b.iter().map(|x| G * x.mass).collect();
            let rad: Vec<f64> = b.iter().map(|x| x.radius()).collect();
            return crate::bh::Tree::build(&pos, &gm, &rad).accelerations(&pos, &gm, &rad);
        }
        self.accelerations_direct()
    }

    pub fn accelerations_direct(&self) -> (Vec<[f64; 3]>, Vec<(usize, usize)>) {
        // Flat arrays (not the Body structs) so the inner loop reads memory in order.
        let b = &self.bodies;
        let (px, py, pz): (Vec<f64>, Vec<f64>, Vec<f64>) = (b.iter().map(|x| x.pos[0]).collect(), b.iter().map(|x| x.pos[1]).collect(), b.iter().map(|x| x.pos[2]).collect());
        let gm: Vec<f64> = b.iter().map(|x| G * x.mass).collect();
        let rad: Vec<f64> = b.iter().map(|x| x.radius()).collect();
        let out: Vec<([f64; 3], Vec<(usize, usize)>)> = (0..b.len()).into_par_iter().map(|i| {
            let (xi, yi, zi, ri) = (px[i], py[i], pz[i], rad[i]);
            let (mut ax, mut ay, mut az) = (0.0f64, 0.0f64, 0.0f64);
            let mut hits = Vec::new();
            for j in 0..px.len() {
                let (dx, dy, dz) = (px[j] - xi, py[j] - yi, pz[j] - zi);
                let d2 = dx * dx + dy * dy + dz * dz;
                let touch = ri + rad[j];
                let t2 = touch * touch;
                if d2 < t2 && i < j { hits.push((i, j)); }
                // Inside each other the pull is capped at half the touching distance (they merge
                // this step anyway). j == i gives d = 0, so it adds nothing.
                let dd = d2.max(0.25 * t2);
                let k = gm[j] / (dd * dd.sqrt());
                ax += k * dx; ay += k * dy; az += k * dz;
            }
            ([ax, ay, az], hits)
        }).collect();
        let mut acc = Vec::with_capacity(out.len());
        let mut hits = Vec::new();
        for (a, h) in out { acc.push(a); hits.extend(h); }
        (acc, hits)
    }

    /// Shortest time scale in the system: sqrt(d^3 / (G (m1 + m2))) over close pairs is costly,
    /// so use each body's acceleration: sqrt(r_typ / |a|) with r_typ = its radius * 10.
    fn safe_dt(&self, acc: &[[f64; 3]]) -> f64 {
        self.bodies.iter().zip(acc).map(|(b, a)| {
            let am = dot(*a, *a).sqrt().max(1e-30);
            (10.0 * b.radius() / am).sqrt()
        }).fold(f64::INFINITY, f64::min)
    }

    /// Merge touching bodies: mass and momentum kept exactly, position at the centre of mass,
    /// volume added (density from total mass / total volume), colour mass-weighted.
    pub fn merge(&mut self, hits: &[(usize, usize)]) -> usize {
        if hits.is_empty() { return 0; }
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
        for &(i, j) in hits {
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            if a != b { parent[a.max(b)] = a.min(b); }
        }
        let mut merged = 0;
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for i in 0..n { let r = find(&mut parent, i); groups.entry(r).or_default().push(i); }
        let mut out = Vec::with_capacity(groups.len());
        for (_, g) in groups {
            if g.len() == 1 { out.push(self.bodies[g[0]].clone()); continue; }
            merged += g.len() - 1;
            let m: f64 = g.iter().map(|&i| self.bodies[i].mass).sum();
            let vol: f64 = g.iter().map(|&i| self.bodies[i].mass / self.bodies[i].density).sum();
            let mut pos = [0.0; 3];
            let mut vel = [0.0; 3];
            let mut col = [0.0f64; 3];
            for &i in &g {
                let b = &self.bodies[i];
                let w = b.mass / m;
                for c in 0..3 { pos[c] += w * b.pos[c]; vel[c] += w * b.vel[c]; col[c] += w * b.color[c] as f64; }
            }
            let heaviest = g.iter().copied().max_by(|&a, &b| self.bodies[a].mass.total_cmp(&self.bodies[b].mass)).unwrap();
            // A hole in the group swallows the rest: it stays a hole, at its own place; its spin
            // (about the y axis, as in the black hole sim) changes by the angular momentum brought in.
            let hole = g.iter().copied().find(|&i| matches!(self.bodies[i].kind, Kind::Hole { .. }));
            let (kind, pos, color, glow) = match hole {
                Some(h) => {
                    let hb = &self.bodies[h];
                    let Kind::Hole { spin } = hb.kind else { unreachable!() };
                    let mut j = spin * G * hb.mass * hb.mass / C_LIGHT;
                    for &i in &g {
                        if i == h { continue; }
                        let b = &self.bodies[i];
                        let (r, v) = (sub(b.pos, hb.pos), sub(b.vel, hb.vel));
                        j += b.mass * (r[2] * v[0] - r[0] * v[2]);
                    }
                    let a = (j * C_LIGHT / (G * m * m)).clamp(-SPIN_MAX, SPIN_MAX);
                    (Kind::Hole { spin: a }, hb.pos, hb.color, false)
                }
                None => (self.bodies[heaviest].kind, pos, col.map(|c| c as f32), self.bodies[heaviest].glow),
            };
            out.push(Body {
                pos, vel, mass: m, density: m / vol,
                color, glow, kind, shrink: self.bodies[heaviest].shrink,
            });
        }
        self.bodies = out;
        merged
    }

    /// Advance by `total` seconds in leapfrog (kick-drift-kick) steps no longer than `dt_max`
    /// and no longer than the safe step; at most `max_steps` steps (cap on work per frame).
    /// Returns the number of steps taken.
    pub fn advance(&mut self, total: f64, dt_max: f64, max_steps: usize) -> usize {
        let mut left = total;
        let mut steps = 0;
        let t0 = std::time::Instant::now();
        let floor = if self.budget_ms > 0.0 { 0.0 } else { total / max_steps as f64 };
        // Reuse last step's accelerations when nothing was added or removed since.
        let (mut acc, mut hits) = match self.cache.take() {
            Some(a) if a.len() == self.bodies.len() => (a, Vec::new()),
            _ => self.accelerations(),
        };
        while left > 0.0 && steps < max_steps {
            if self.resolve(&hits) { (acc, _) = self.accelerations(); }
            let dt = left.min(dt_max).min(0.2 * self.safe_dt(&acc)).max(floor);
            let dt = dt.min(left);
            for (b, a) in self.bodies.iter_mut().zip(&acc) {
                for c in 0..3 { b.vel[c] += 0.5 * dt * a[c]; b.pos[c] += dt * b.vel[c]; }
            }
            (acc, hits) = self.accelerations();
            for (b, a) in self.bodies.iter_mut().zip(&acc) {
                for c in 0..3 { b.vel[c] += 0.5 * dt * a[c]; }
            }
            self.bodies.retain(|b| b.pos.iter().chain(&b.vel).all(|v| v.is_finite()));
            if self.bodies.len() != acc.len() { (acc, hits) = self.accelerations(); }
            left -= dt;
            steps += 1;
            if self.budget_ms > 0.0 && t0.elapsed().as_secs_f64() * 1e3 > self.budget_ms { break; }
        }
        self.last_done = total - left;
        if !self.resolve(&hits) { self.cache = Some(acc); }
        steps
    }

    pub fn energy(&self) -> f64 {
        let b = &self.bodies;
        let ke: f64 = b.iter().map(|x| 0.5 * x.mass * dot(x.vel, x.vel)).sum();
        let mut pe = 0.0;
        for i in 0..b.len() { for j in i + 1..b.len() {
            let d = sub(b[i].pos, b[j].pos);
            pe -= G * b[i].mass * b[j].mass / dot(d, d).sqrt();
        }}
        ke + pe
    }

    pub fn momentum(&self) -> [f64; 3] {
        let mut p = [0.0; 3];
        for b in &self.bodies { for c in 0..3 { p[c] += b.mass * b.vel[c]; } }
        p
    }
}

/// gravity_sim's scene: a star and two planets (positions and masses from gravity_sim.cpp, in km)
/// on circular orbits.
pub fn preset_original() -> Vec<Body> {
    let star_m = 1.989e25;
    let planet_m = 5.97219e22;
    let star = [0.0, 0.0, -350e3];
    let mut out = vec![Body { pos: star, vel: [0.0; 3], mass: star_m, density: 5515.0, color: [1.0, 0.929, 0.176], glow: true, kind: Kind::Rock, shrink: SIZE_RATIO }];
    for (x, s) in [(-5000e3, 1.0), (5000e3, -1.0)] {
        let pos = [x, 650e3, -350e3];
        let r = dot(sub(pos, star), sub(pos, star)).sqrt();
        let v = (G * star_m / r).sqrt();
        out.push(Body { pos, vel: [0.0, 0.0, s * v], mass: planet_m, density: 5515.0, color: [0.0, 1.0, 1.0], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO });
    }
    out
}

/// Bigger: the same star with a disk of `n` rocks on near-circular orbits. They bump, merge and
/// grow into planets.
pub fn preset_disk(n: usize, seed: u64) -> Vec<Body> {
    let star_m = 1.989e25;
    let mut out = vec![Body { pos: [0.0; 3], vel: [0.0; 3], mass: star_m, density: 5515.0, color: [1.0, 0.929, 0.176], glow: true, kind: Kind::Rock, shrink: SIZE_RATIO }];
    let mut s = seed;
    let mut rnd = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 11) as f64 / (1u64 << 53) as f64 };
    for _ in 0..n {
        let r = 1500e3 + 10500e3 * rnd().sqrt();
        let th = rnd() * std::f64::consts::TAU;
        let y = (rnd() - 0.5) * 0.02 * r;
        let pos = [r * th.cos(), y, r * th.sin()];
        let v = (G * star_m / r).sqrt() * (1.0 + (rnd() - 0.5) * 0.04);
        let vel = [-v * th.sin(), 0.0, v * th.cos()];
        let mass = 10f64.powf(20.0 + 1.5 * rnd());
        let t = rnd() as f32;
        let b = Body { pos, vel, mass, density: 3000.0 + 3000.0 * rnd(), color: [0.55 + 0.4 * t, 0.5 + 0.2 * t, 0.45 - 0.2 * t], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO };
        // No rock starts inside another (they would merge before the first frame).
        let clear = out.iter().all(|o| (0..3).map(|c| (o.pos[c] - b.pos[c]).powi(2)).sum::<f64>().sqrt() > o.radius() + b.radius());
        if clear { out.push(b); }
    }
    out
}

/// Eclipse: the star, a planet, and two moons behind the planet, one fully in its shadow and one
/// half in it. Nothing moves.
pub fn preset_eclipse() -> Vec<Body> {
    let star = Body { pos: [0.0; 3], vel: [0.0; 3], mass: 1.989e25, density: 5515.0, color: [1.0, 0.929, 0.176], glow: true, kind: Kind::Rock, shrink: SIZE_RATIO };
    let planet = Body { pos: [6000e3, 0.0, 0.0], vel: [0.0; 3], mass: 5.97219e22, density: 5515.0, color: [0.0, 1.0, 1.0], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO };
    let moon = |x: f64, y: f64| Body { pos: [x, y, 0.0], vel: [0.0; 3], mass: 7.3e21, density: 3344.0, color: [0.9, 0.9, 0.9], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO };
    let rp = planet.radius();
    // One moon fully in the planet's shadow, one half in it (further back, so the two don't touch).
    vec![star, planet, moon(6150e3, 0.0), moon(6300e3, rp * 1.1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular_pair() -> World {
        let m = 1.989e25;
        let r = 5e6;
        let v = (G * m / r).sqrt();
        World::new(vec![
            Body { pos: [0.0; 3], vel: [0.0; 3], mass: m, density: 1e12, color: [1.0; 3], glow: true, kind: Kind::Rock, shrink: SIZE_RATIO },
            Body { pos: [r, 0.0, 0.0], vel: [0.0, 0.0, v], mass: 1.0, density: 1e12, color: [1.0; 3], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO },
        ])
    }

    #[test]
    fn circular_orbit_closes_after_one_period() {
        let mut w = circular_pair();
        let (m, r) = (w.bodies[0].mass, 5e6);
        let period = std::f64::consts::TAU * (r * r * r / (G * m)).sqrt();
        let e0 = w.energy();
        w.advance(period, period / 2000.0, 1_000_000);
        let p = w.bodies[1].pos;
        let err = dot(sub(p, [r, 0.0, 0.0]), sub(p, [r, 0.0, 0.0])).sqrt() / r;
        let de = ((w.energy() - e0) / e0).abs();
        assert!(err < 1e-4 && de < 1e-8);
    }

    #[test]
    fn merging_keeps_mass_and_momentum() {
        let mut w = World::new(vec![
            Body { pos: [0.0; 3], vel: [1000.0, 0.0, 0.0], mass: 2e22, density: 3000.0, color: [1.0, 0.0, 0.0], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO },
            Body { pos: [1.0, 0.0, 0.0], vel: [-500.0, 200.0, 0.0], mass: 1e22, density: 6000.0, color: [0.0, 0.0, 1.0], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO },
        ]);
        let (m0, p0) = (w.bodies.iter().map(|b| b.mass).sum::<f64>(), w.momentum());
        let (_, hits) = w.accelerations();
        assert_eq!(w.merge(&hits), 1);
        assert_eq!(w.bodies.len(), 1);
        let p1 = w.momentum();
        assert!((w.bodies[0].mass - m0).abs() / m0 < 1e-15);
        for c in 0..3 { assert!((p1[c] - p0[c]).abs() < 1e-6 * p0[0].abs()); }
        let vol = w.bodies[0].mass / w.bodies[0].density;
        assert!((vol - (2e22 / 3000.0 + 1e22 / 6000.0)).abs() / vol < 1e-12);
    }

    #[test]
    fn swallowed_rock_changes_the_spin() {
        let run = |spin: f64, v: f64| {
            let hole = Body { pos: [0.0; 3], vel: [0.0; 3], mass: 1e31, density: 1e18, color: [0.0; 3], glow: false, kind: Kind::Hole { spin }, shrink: 1.0 };
            let rock = Body { pos: [0.0, 0.0, 1e6], vel: [v, 0.0, 0.0], mass: 1e30, density: 3000.0, color: [1.0; 3], glow: false, kind: Kind::Rock, shrink: 1.0 };
            let mut w = World::new(vec![hole, rock]);
            let (_, hits) = w.accelerations();
            assert_eq!(w.merge(&hits), 1);
            let Kind::Hole { spin: a } = w.bodies[0].kind else { panic!("the hole is gone") };
            let m = 1.1e31;
            let j = spin * G * 1e31 * 1e31 / C_LIGHT + 1e30 * (1e6 * v);
            (a, (j * C_LIGHT / (G * m * m)).clamp(-SPIN_MAX, SPIN_MAX), w.bodies[0].mass)
        };
        let (a, want, m) = run(0.5, 2e6);
        assert!((a - want).abs() < 1e-12 && a > 0.5 * (1.0 / 1.1f64).powi(2), "spin {a}, want {want}");
        assert!((m - 1.1e31).abs() < 1e16);
        let (a, want, _) = run(0.5, -2e6);
        assert!((a - want).abs() < 1e-12 && a < 0.5 * (1.0 / 1.1f64).powi(2), "spin {a}, want {want}");
        let (a, _, _) = run(0.9, 1e9);
        assert_eq!(a, SPIN_MAX);
        let (a, _, _) = run(-0.9, -1e9);
        assert_eq!(a, -SPIN_MAX);
    }

    #[test]
    fn original_scene_runs_a_minute_cleanly() {
        let mut w = World::new(preset_original());
        let e0 = w.energy();
        for _ in 0..3600 { w.advance(200.0 / 60.0, 5.0, 1000); }
        assert_eq!(w.bodies.len(), 3, "the planets should orbit, not crash");
        let de = ((w.energy() - e0) / e0).abs();
        assert!(de < 1e-4);
    }

    #[test]
    fn disk_keeps_mass_and_momentum_while_merging() {
        let mut w = World::new(preset_disk(400, 7));
        let m0: f64 = w.bodies.iter().map(|b| b.mass).sum();
        let p0 = w.momentum();
        for _ in 0..600 { w.advance(200.0 / 60.0, 5.0, 1000); }
        let m1: f64 = w.bodies.iter().map(|b| b.mass).sum();
        let p1 = w.momentum();
        let pscale = w.bodies.iter().map(|b| b.mass * dot(b.vel, b.vel).sqrt()).sum::<f64>();
        assert!((m1 - m0).abs() / m0 < 1e-12);
        for c in 0..3 { assert!((p1[c] - p0[c]).abs() / pscale < 1e-9, "momentum {c}: {} vs {}", p1[c], p0[c]); }
    }

    fn pair(v: f64) -> World {
        let a = Body { pos: [0.0; 3], vel: [0.0, 0.0, 0.0], mass: 2e21, density: 3000.0, color: [1.0; 3], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO };
        let mut b = Body { pos: [0.0; 3], vel: [-v, 0.3 * v, 0.0], mass: 1e21, density: 4000.0, color: [0.5; 3], glow: false, kind: Kind::Rock, shrink: SIZE_RATIO };
        let touch = 0.999 * (a.radius() + b.radius());
        b.pos = [touch * 0.8, touch * 0.6, 0.0];
        World::new(vec![a, b])
    }
    fn ke(w: &World) -> f64 { w.bodies.iter().map(|b| 0.5 * b.mass * dot(b.vel, b.vel)).sum() }
    fn com(w: &World) -> [f64; 3] {
        let m: f64 = w.bodies.iter().map(|b| b.mass).sum();
        let mut c = [0.0; 3];
        for b in &w.bodies { for k in 0..3 { c[k] += b.mass * b.pos[k] / m; } }
        c
    }
    fn vesc(w: &World) -> f64 {
        let (a, b) = (&w.bodies[0], &w.bodies[1]);
        (2.0 * G * (a.mass + b.mass) / (a.radius() + b.radius())).sqrt()
    }

    #[test]
    fn slow_hit_merges_medium_bounces_fast_shatters() {
        let v = vesc(&pair(1.0));
        for (k, want) in [(0.5, Outcome::Merge), (2.0, Outcome::Bounce), (5.0, Outcome::Shatter)] {
            let w = pair(k * v / 1.044);
            assert_eq!(w.outcome(0, 1), Some(want), "at {k} x escape speed");
        }
    }

    #[test]
    fn bounce_keeps_momentum_and_loses_the_right_energy() {
        let mut w = pair(2.0 * vesc(&pair(1.0)));
        let (p0, ke0, c0) = (w.momentum(), ke(&w), com(&w));
        let (a, b) = (&w.bodies[0], &w.bodies[1]);
        let d = sub(b.pos, a.pos);
        let n = d.map(|x| x / dot(d, d).sqrt());
        let vn = dot(sub(b.vel, a.vel), n);
        let mu = a.mass * b.mass / (a.mass + b.mass);
        let (_, hits) = w.accelerations();
        assert!(w.resolve(&hits));
        assert_eq!((w.bodies.len(), w.bounces), (2, 1));
        let (p1, c1) = (w.momentum(), com(&w));
        let scale = w.bodies.iter().map(|b| b.mass * dot(b.vel, b.vel).sqrt()).sum::<f64>();
        for k in 0..3 { assert!((p1[k] - p0[k]).abs() / scale < 1e-14); assert!((c1[k] - c0[k]).abs() < 1e-6); }
        let want = 0.5 * (1.0 - RESTITUTION * RESTITUTION) * mu * vn * vn;
        let lost = ke0 - ke(&w);
        assert!((lost - want).abs() / want < 1e-9);
        assert!(w.outcome(0, 1).is_none(), "they should now be moving apart");
    }

    #[test]
    fn shatter_keeps_mass_momentum_and_centre() {
        let mut w = pair(5.0 * vesc(&pair(1.0)));
        let (m0, p0, c0, ke0) = (w.bodies.iter().map(|b| b.mass).sum::<f64>(), w.momentum(), com(&w), ke(&w));
        let (_, hits) = w.accelerations();
        assert!(w.resolve(&hits));
        assert_eq!((w.bodies.len(), w.shatters), (7, 1));
        let m1: f64 = w.bodies.iter().map(|b| b.mass).sum();
        let (p1, c1) = (w.momentum(), com(&w));
        let scale = w.bodies.iter().map(|b| b.mass * dot(b.vel, b.vel).sqrt()).sum::<f64>();
        assert!((m1 - m0).abs() / m0 < 1e-15);
        for k in 0..3 { assert!((p1[k] - p0[k]).abs() / scale < 1e-14); assert!((c1[k] - c0[k]).abs() < 1e-6); }
        assert!(ke(&w) < ke0, "a shatter can't make energy");
        let (_, hits) = w.accelerations();
        assert!(hits.is_empty());
    }
}
