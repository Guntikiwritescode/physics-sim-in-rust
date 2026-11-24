//! The black hole tab's world: free bodies (stars, black holes, rocks) at true size, SI units
//! (metres, kg, seconds). Every one pulls every other (Newton), leapfrog steps; touching bodies
//! merge, bounce or shatter by impact speed; a black hole swallows whatever touches its horizon and
//! its spin changes (nbody.rs). Tides: a star that comes inside a much heavier body's Roche limit
//! is torn into pieces.

use crate::nbody::{self, Body, Kind, G};
use glam::DVec3;

pub struct World {
    pub free: nbody::World,
    /// Simulated seconds per real second.
    pub time_scale: f64,
    pub paused: bool,
    /// Simulated seconds of the last step (for motion of free bodies between frames).
    pub last_dt: f64,
    pub torn: usize,
    /// Share of the set speed the free bodies actually ran at lately (1 = keeping up).
    pub speed_share: f64,
}

/// Wall-clock time per frame for the free bodies' gravity (ms).
pub const FREE_BUDGET_MS: f64 = 12.0;

fn v3(a: [f64; 3]) -> DVec3 { DVec3::from(a) }

/// Roche limit (fluid body): d = 2.44 R_M (rho_M / rho_m)^(1/3).
pub fn roche_limit(big: &Body, small: &Body) -> f64 {
    if let Kind::Hole { .. } = big.kind {
        // A hole has no density; use the same formula written with its mass:
        // d = 2.44 (3 M / (4 pi rho_m))^(1/3).
        return 2.44 * (3.0 * big.mass / (4.0 * std::f64::consts::PI * small.density)).cbrt();
    }
    2.44 * big.radius() * big.shrink * (big.density / small.density).cbrt()
}

impl World {
    pub fn new(bodies: Vec<Body>) -> World {
        World { free: nbody::World::new(bodies), time_scale: 1.0, paused: false, last_dt: 0.0, torn: 0, speed_share: 1.0 }
    }

    pub fn step(&mut self, dt: f64) {
        let total = if self.paused { 0.0 } else { dt.min(0.1) * self.time_scale };
        self.last_dt = total;
        if total <= 0.0 { return; }
        // Safe-length steps within a time budget per frame. When the budget runs out, the rest of
        // the frame's time is not simulated (shown as "gravity keeps up with x% of the set speed").
        self.free.budget_ms = FREE_BUDGET_MS;
        self.free.advance(total, total, 1_000_000);
        let done = self.free.last_done;
        self.speed_share = 0.8 * self.speed_share + 0.2 * (done / total);
        self.last_dt = done;
        self.tides();
    }

    /// Tides: a body (not a hole) inside the Roche limit of a much heavier one is pulled apart
    /// into a remnant and pieces spread along the line to the heavy body (the stretch direction).
    fn tides(&mut self) {
        let n = self.free.bodies.len();
        if n > nbody::MAX_BODIES - 8 { return; }
        let mut torn = None;
        'outer: for i in 0..n {
            let s = &self.free.bodies[i];
            // Only whole stars are torn (once): the pieces are hot gas clumps.
            if !matches!(s.kind, Kind::Star { .. }) || s.mass < 4.0 * nbody::MIN_PIECE_KG { continue; }
            for (j, b) in self.free.bodies.iter().enumerate() {
                if i == j || b.mass < 10.0 * s.mass { continue; }
                let d = (v3(b.pos) - v3(s.pos)).length();
                if d < roche_limit(b, s) && d > b.radius() + s.radius() {
                    torn = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = torn else { return };
        let s = self.free.bodies[i].clone();
        let b = &self.free.bodies[j];
        let dir = (v3(b.pos) - v3(s.pos)).normalize();
        // Pieces: half the mass stays as a remnant, the rest in 4 pieces along the stretch line;
        // nearer pieces orbit faster (their speed differs by the tidal spread, as in a real
        // disruption), so they string out into a stream. Mass and momentum are kept.
        let (px, pv) = (v3(s.pos), v3(s.vel));
        let piece = 0.125 * s.mass;
        let sep = 2.5 * s.radius();
        let dv = (G * b.mass / (v3(b.pos) - px).length()).sqrt() * sep / (v3(b.pos) - px).length();
        // Pieces: glowing gas (glow = true), each with the star's density.
        let piece_of = |m: f64, pos: DVec3, vel: DVec3| Body { pos: pos.into(), vel: vel.into(), mass: m, kind: Kind::Rock, glow: true, ..s.clone() };
        let mut out = vec![piece_of(0.5 * s.mass, px, pv)];
        for (k, f) in [-2.0, -1.0, 1.0, 2.0].into_iter().enumerate() {
            let pos = px + dir * (f * sep);
            let vel = pv - dir.cross(DVec3::Y).normalize_or(DVec3::X) * (f * dv * 0.5) + dir * (f * dv * 0.1);
            let _ = k;
            out.push(piece_of(piece, pos, vel));
        }
        // Remnant keeps the momentum balance: pieces at +-f cancel in position and velocity.
        self.free.bodies.remove(i);
        self.free.bodies.extend(out);
        self.torn += 1;
    }

    pub fn momentum(&self) -> DVec3 { v3(self.free.momentum()) }
}

pub const SUN_R: f64 = 6.957e8;
pub const AU: f64 = 1.496e11;

pub fn ball_density(m: f64, r: f64) -> f64 { m / (4.0 / 3.0 * std::f64::consts::PI * r * r * r) }

pub fn star(pos: DVec3, vel: DVec3, m: f64, r: f64, t: f64) -> Body {
    Body { pos: pos.into(), vel: vel.into(), mass: m, density: ball_density(m, r), color: [1.0, 0.9, 0.7], glow: true, kind: Kind::Star { temp_k: t }, shrink: 1.0 }
}
pub fn hole(pos: DVec3, m: f64, spin: f64) -> Body {
    Body { pos: pos.into(), vel: [0.0; 3], mass: m, density: 1e18, color: [0.0; 3], glow: false, kind: Kind::Hole { spin }, shrink: 1.0 }
}

/// kavan010's black hole scene: Sagittarius A*-like hole (8.54e36 kg, spin 0.9) and two Sun-sized
/// stars 4e11 m out on circular orbits, just outside the hole's Roche limit for a Sun (2.8e11 m).
pub fn setup_black_hole() -> World {
    let m = 8.54e36;
    let h = hole(DVec3::ZERO, m, 0.9);
    let mut bodies = vec![h];
    for (p, c) in [(DVec3::new(4e11, 0.0, 0.0), [1.0, 0.85, 0.35]), (DVec3::new(0.0, 0.0, 4e11), [1.0, 0.3, 0.15])] {
        let v = (G * m / p.length()).sqrt();
        let dir = DVec3::Y.cross(p).normalize();
        let mut s = star(p, dir * v, 1.98892e30, SUN_R, 5000.0);
        s.color = c;
        bodies.push(s);
    }
    let mut w = World::new(bodies);
    w.time_scale = 2000.0;
    w
}

/// kavan010's gravity_sim scenes (nbody.rs presets) at true size, the glowing star a real light.
/// gravity_sim draws bodies at 1/30 of their size; here every length is 30x larger instead, speeds
/// 1/sqrt(30) (orbits keep their shape, v ~ sqrt(GM/r)) and time runs 30^1.5 = 164x faster, so the
/// picture and motion on screen match.
pub const GRAV_SCALE: f64 = nbody::SIZE_RATIO;
fn from_gravity_preset(bodies: Vec<Body>) -> World {
    let mut bodies = bodies;
    for b in &mut bodies {
        if b.glow { b.kind = Kind::Star { temp_k: 5772.0 }; }
        let k = GRAV_SCALE * b.shrink / nbody::SIZE_RATIO;   // (presets all use SIZE_RATIO)
        for c in 0..3 { b.pos[c] *= k; b.vel[c] /= k.sqrt(); }
        b.shrink = 1.0;
    }
    let mut w = World::new(bodies);
    w.time_scale = 200.0 * GRAV_SCALE.powf(1.5);
    w
}
pub fn setup_gravity_disk() -> World { from_gravity_preset(nbody::preset_disk(2000, 1)) }
pub fn setup_eclipse() -> World { let mut w = from_gravity_preset(nbody::preset_eclipse()); w.paused = true; w }
pub fn setup_gravity_original() -> World { from_gravity_preset(nbody::preset_original()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hole_swallows_star_mass_and_momentum_kept() {
        let mut w = World::new(vec![hole(DVec3::ZERO, 8.54e36, 0.5), star(DVec3::new(2e11, 0.0, 0.0), DVec3::new(0.0, 0.0, 1e5), 2e30, 1e9, 5000.0)]);
        let m0 = w.free.bodies.iter().map(|b| b.mass).sum::<f64>();
        let p0 = w.momentum();
        w.time_scale = 1e4;
        for _ in 0..4000 { w.step(1.0 / 60.0); if w.free.bodies.len() == 1 { break; } }
        let m1 = w.free.bodies.iter().map(|b| b.mass).sum::<f64>();
        let p1 = w.momentum();
        assert!((m1 - m0).abs() / m0 < 1e-12);
        assert!((p1 - p0).length() <= 1e-9 * (m0 * 1e5));
        assert!(w.torn > 0, "a sun-like star passing this close should be torn apart");
    }

    #[test]
    fn roche_limit_matches_textbook() {
        // Moon at Earth: fluid Roche limit 2.44 R_E (rho_E / rho_M)^(1/3) = about 18,400 km.
        let ball = |m: f64, r: f64| Body { pos: [0.0; 3], vel: [0.0; 3], mass: m, density: ball_density(m, r), color: [1.0; 3], glow: false, kind: Kind::Rock, shrink: 1.0 };
        let e = ball(5.972e24, 6.371e6);
        let m = ball(7.342e22, 1.7374e6);
        let d = roche_limit(&e, &m);
        assert!((d - 1.84e7).abs() / 1.84e7 < 0.02, "{d}");
    }

    #[test]
    fn true_size_gravity_scenes_move_like_the_originals() {
        let k = GRAV_SCALE;
        let mut small = nbody::World::new(nbody::preset_original());
        let mut big = setup_gravity_original();
        for b in &big.free.bodies { assert_eq!(b.shrink, 1.0); }
        let t = 3600.0;
        small.advance(t, 10.0, 1_000_000);
        big.free.advance(t * k.powf(1.5), 10.0 * k.powf(1.5), 1_000_000);
        let mut worst: f64 = 0.0;
        for (a, b) in small.bodies.iter().zip(&big.free.bodies) {
            let d = ((0..3).map(|c| (b.pos[c] / k - a.pos[c]).powi(2)).sum::<f64>()).sqrt();
            let r = (0..3).map(|c| a.pos[c].powi(2)).sum::<f64>().sqrt().max(1.0);
            worst = worst.max(d / r);
        }
        assert!(worst < 1e-9, "{worst}");
        for setup in [setup_gravity_disk(), setup_eclipse(), setup_gravity_original()] {
            let b = &setup.free.bodies;
            for i in 0..b.len() { for j in i + 1..b.len() {
                let d = ((0..3).map(|c| (b[i].pos[c] - b[j].pos[c]).powi(2)).sum::<f64>()).sqrt();
                assert!(d > b[i].radius() + b[j].radius(), "bodies {i} and {j} start inside each other");
            }}
        }
    }
}
