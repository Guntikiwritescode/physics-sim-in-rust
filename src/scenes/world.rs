//! The Black hole tab: a Kerr black hole, stars and rocks at true size, Newton's gravity, tides
//! and merging (world.rs), drawn by the path tracer (rt.rs), the orbit camera (orbit.rs).

use crate::camera::{look_at, Camera};
use crate::orbit::OrbitCam;
use crate::frame::{Frame, Post};
use crate::host::{Input, Scene};
use crate::nbody::{Body, Kind, C_LIGHT, G};
use crate::rt::{ObjGpu, Opts, Renderer};
use crate::world::{self, World};
use crate::kerr;
use glam::DVec3;
use std::collections::HashMap;
use std::time::Instant;
use winit::keyboard::KeyCode;

/// Frames a still view keeps getting cleaner for (then the GPU rests).
const STILL_FRAMES: u32 = 400;
/// Radiance of a Sun-like star surface in the renderer's units.
const SUN_RADIANCE: f32 = 1.2e5;
/// Outer edge of a black hole's gas disk, in Schwarzschild radii.
const DISK_OUT: f64 = 14.0;

pub const SETUPS: [&str; 5] = ["(unused)", "Black hole and two stars", "Gravity disk (2,000 rocks)", "Eclipse", "kavan's gravity scene"];
pub const SETUP_ORDER: [usize; 4] = [1, 2, 4, 3];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Follow { Body(usize), Nothing }

pub struct WorldScene {
    /// Screen seconds for the disk's swirl, stopped while paused, wrapped at 64 x 8 s (rt_hole.wgsl
    /// SWIRL_P), where its pattern repeats exactly.
    swirl_clock: f64,
    pub world: World,
    pub setup: usize,
    cam: Camera,
    ctl: OrbitCam,
    follow: Follow,
    follow_pos: DVec3,
    /// World position the camera's numbers are measured from: the followed body's centre (it
    /// moves with it, so the camera rides along). World positions near 1 AU only have about 30
    /// micrometre steps in 64-bit numbers; numbers measured from a nearby origin keep every
    /// scale smooth.
    origin: DVec3,
    tracer: Option<Renderer>,
    hover: Option<usize>,
    /// Seconds the mouse has rested where it is (the name label waits for it).
    mouse_still: f64,
    /// Frames drawn (the first one only says "Loading").
    frames: u64,
    scale: f64,
    acc_frames: u32,
    last_view: Vec<u8>,
    adapt_t: Instant,
    adapt_frames: u32,
    last_t: Instant,
    pub ev: f32,
    mouse: (f64, f64),
    pub disk: bool,
    /// Views to go back / forward to (Backspace, the panel's Back / Forward).
    back: Vec<Mark>,
    fwd: Vec<Mark>,
    /// Text typed in the panel's finder.
    find: String,
    /// The setup's start view (Home).
    home: Mark,
}

/// A remembered view: what was followed, where the middle was relative to it, how far and from
/// which side the camera looked.
#[derive(Clone, Copy)]
struct Mark { follow: Follow, rel: DVec3, dist: f64, dir: DVec3 }

fn temp_rgb(t: f64) -> [f32; 3] {
    // Colour of a hot surface (blackbody) as RGB, normalised to the brightest channel. Fit by
    // Tanner Helland (2012) to Mitchell Charity's blackbody table; good to a few percent.
    let t = (t / 100.0).clamp(10.0, 400.0);
    let r = if t <= 66.0 { 255.0 } else { 329.698727446 * (t - 60.0).powf(-0.1332047592) };
    let g = if t <= 66.0 { 99.4708025861 * t.ln() - 161.1195681661 } else { 288.1221695283 * (t - 60.0).powf(-0.0755148492) };
    let b = if t >= 66.0 { 255.0 } else if t <= 19.0 { 0.0 } else { 138.5177312231 * (t - 10.0).ln() - 305.0447927307 };
    let c = [r, g, b].map(|x: f64| (x.clamp(0.0, 255.0) / 255.0) as f32);
    let m = c[0].max(c[1]).max(c[2]).max(1e-6);
    c.map(|x| x / m)
}

fn fmt_len(m: f64) -> String {
    let a = m.abs();
    if a < 1e-6 { format!("{:.1} nm", m * 1e9) }
    else if a < 1e-3 { format!("{:.1} um", m * 1e6) }
    else if a < 1.0 { format!("{:.1} mm", m * 1e3) }
    else if a < 1e3 { format!("{:.1} m", m) }
    else if a < 1e8 { format!("{:.0} km", m / 1e3) }
    else if a < 1.5e10 { format!("{:.2} million km", m / 1e9) }
    else if a < 1e13 { format!("{:.2} AU", m / world::AU) }
    else { format!("{:.2} light years", m / 9.4607e15) }
}
/// Speed factor, short: 1, 10, 3.2 thousand, 1.0 million.
fn fmt_speed(x: f64) -> String {
    if x < 10.0 { format!("{x:.2}") } else if x < 1e3 { format!("{x:.0}") } else if x < 1e6 { format!("{:.1} thousand", x / 1e3) } else { format!("{:.1} million", x / 1e6) }
}
impl WorldScene {
    pub fn new(setup: usize) -> WorldScene {
        let setup = if SETUP_ORDER.contains(&setup) { setup } else { SETUP_ORDER[0] };
        let (world, mut cam, target, follow) = Self::make(setup);
        let follow_pos = match follow { Follow::Body(i) => DVec3::from(world.free.bodies[i].pos), Follow::Nothing => DVec3::ZERO };
        // Camera numbers from here on are measured from what it follows.
        cam.pos -= follow_pos;
        let ctl = OrbitCam::new(&cam, target - follow_pos, DVec3::Y, 1e-11, 1e14);
        WorldScene {
            swirl_clock: 0.0, world, setup, cam, ctl, follow, follow_pos, origin: follow_pos, tracer: None, hover: None, mouse_still: 0.0,
            frames: 0, scale: 0.5, acc_frames: 0, last_view: Vec::new(), adapt_t: Instant::now(), adapt_frames: 0,
            last_t: Instant::now(), ev: 0.0, mouse: (0.0, 0.0), disk: true,
            back: Vec::new(), fwd: Vec::new(), find: String::new(),
            home: Mark { follow, rel: DVec3::ZERO, dist: 1.0, dir: DVec3::Z },
        }
        .with_home()
    }

    fn with_home(mut self) -> Self { self.home = self.mark(); self }

    /// Back to the setup's start view (remembered for Back).
    pub fn go_home(&mut self) {
        let m = self.mark();
        self.remember(m);
        let h = self.home;
        self.go_mark(h);
    }

    fn make(setup: usize) -> (World, Camera, DVec3, Follow) {
        // gravity_sim's camera places, km -> m, times the true-size scale.
        let k = 1e3 * world::GRAV_SCALE;
        let grav = |w: World, pos: DVec3, target: DVec3| {
            let cam = Camera { pos: pos * k, rot: look_at(pos * k, target * k, DVec3::Y), fov_y: 50f64.to_radians() };
            (w, cam, target * k, Follow::Body(0))
        };
        match setup {
            2 => grav(world::setup_gravity_disk(), DVec3::new(0.0, 9000.0, 22000.0), DVec3::ZERO),
            3 => grav(world::setup_eclipse(), DVec3::new(6000.0, 40.0, 260.0), DVec3::new(6100.0, 20.0, 0.0)),
            4 => grav(world::setup_gravity_original(), DVec3::new(0.0, 2000.0, 10000.0), DVec3::new(0.0, 0.0, -350.0)),
            _ => {
                let w = world::setup_black_hole();
                // 26 Schwarzschild radii out, 8 degrees above the disk.
                let rs = 2.0 * G * w.free.bodies[0].mass / (C_LIGHT * C_LIGHT);
                let (el, az) = (82f64.to_radians(), 2.4f64);
                let pos = DVec3::new(el.sin() * az.cos(), el.cos(), el.sin() * az.sin()) * (26.0 * rs);
                let cam = Camera { pos, rot: look_at(pos, DVec3::ZERO, DVec3::Y), fov_y: 60f64.to_radians() };
                (w, cam, DVec3::ZERO, Follow::Body(0))
            }
        }
    }

    fn to_cam(&self, w: DVec3) -> DVec3 { w - self.origin }
    /// Measure camera numbers from a new origin (the view doesn't move).
    fn set_origin(&mut self, new: DVec3) {
        let d = self.origin - new;
        if d == DVec3::ZERO { return; }
        self.origin = new;
        self.ctl.shift(&mut self.cam, d);
    }
    /// World position of what `f` follows (None: gone).
    fn base_of(&self, f: Follow) -> Option<DVec3> {
        match f {
            Follow::Body(i) => self.world.free.bodies.get(i).map(|b| DVec3::from(b.pos)),
            Follow::Nothing => Some(self.origin),
        }
    }
    /// Start following `f` (camera numbers then measured from it).
    fn follow_now(&mut self, f: Follow) {
        let Some(b) = self.base_of(f) else { return };
        self.follow = f;
        self.follow_pos = b;
        self.set_origin(b);
    }
    fn mark(&self) -> Mark {
        let dir = (self.cam.pos - self.ctl.target).normalize_or(DVec3::Z);
        Mark { follow: self.follow, rel: self.ctl.target, dist: (self.cam.pos - self.ctl.target).length(), dir }
    }
    /// Remember a view for Back (taken before a trip).
    fn remember(&mut self, m: Mark) {
        self.back.push(m);
        if self.back.len() > 64 { self.back.remove(0); }
        self.fwd.clear();
    }
    fn start_trip(&mut self, target: DVec3, dist: f64, dir: Option<DVec3>) {
        self.ctl.go_to(target, dist);
        if let Some(d) = dir { self.ctl.look_from(d); }
    }
    /// Keep the camera out of every body (a star can pass where the camera is), and
    /// the zoom range right for what is in focus.
    fn keep_out(&mut self) {
        if let Follow::Body(i) = self.follow {
            if let Some(b) = self.world.free.bodies.get(i) { self.ctl.min_dist = Self::near_limit(b); }
        }
        for b in &self.world.free.bodies {
            let c = self.to_cam(DVec3::from(b.pos));
            let r = Self::near_limit(b) * 0.9;
            if (self.cam.pos - c).length() < r * 1.5 { self.ctl.keep_out(&mut self.cam, c, r); }
        }
    }
    /// How close the camera may come to a body's centre: a black hole's shadow is about 2.6 times
    /// its horizon across, so 4 horizons; anything else 1.3 radii.
    fn near_limit(b: &Body) -> f64 {
        if matches!(b.kind, Kind::Hole { .. }) { 4.0 * b.radius() } else { 1.3 * b.radius() }
    }

    fn go_mark(&mut self, m: Mark) {
        let f = match m.follow { Follow::Body(i) if i >= self.world.free.bodies.len() => Follow::Nothing, f => f };
        self.follow_now(f);
        self.start_trip(m.rel, m.dist, Some(m.dir));
    }
    pub fn go_back(&mut self) {
        let Some(m) = self.back.pop() else { return };
        let now = self.mark();
        self.fwd.push(now);
        self.go_mark(m);
    }
    pub fn go_forward(&mut self) {
        let Some(m) = self.fwd.pop() else { return };
        let now = self.mark();
        self.back.push(now);
        self.go_mark(m);
    }

    /// Throw a rock from the camera, toward the middle of the picture: a bit in front, at half the
    /// circular-orbit speed round the heaviest body, plus the speed of what the camera follows.
    pub fn throw(&mut self) {
        let (f, _, _) = self.cam.axes();
        let b = &self.world.free.bodies;
        let Some(heavy) = (0..b.len()).max_by(|&i, &j| b[i].mass.total_cmp(&b[j].mass)) else { return };
        let d = (self.ctl.target - self.cam.pos).length();
        let p = self.origin + self.cam.pos + f * (0.2 * d);
        let r = (p - DVec3::from(b[heavy].pos)).length();
        let v = f * 0.5 * (G * b[heavy].mass / r).sqrt() + self.follow_vel();
        let rad = 0.004 * d;
        let m = 3000.0 * 4.0 / 3.0 * std::f64::consts::PI * rad * rad * rad;
        self.world.free.bodies.push(Body { pos: p.into(), vel: v.into(), mass: m, density: 3000.0, color: [0.8, 0.75, 0.7],
            glow: false, kind: Kind::Rock, shrink: 1.0 });
    }

    fn follow_vel(&self) -> DVec3 {
        match self.follow {
            Follow::Body(i) => self.world.free.bodies.get(i).map(|b| DVec3::from(b.vel)).unwrap_or_default(),
            Follow::Nothing => DVec3::ZERO,
        }
    }

    /// Ride along with what the camera follows. Free bodies get renumbered when they merge, so
    /// the followed body is found again as the one nearest where it should be now.
    fn ride(&mut self) {
        let now = match self.follow {
            Follow::Body(_) => {
                let b = &self.world.free.bodies;
                let want = self.follow_pos;
                let Some(i) = (0..b.len()).min_by(|&i, &j| (DVec3::from(b[i].pos) - want).length().total_cmp(&(DVec3::from(b[j].pos) - want).length())) else { return };
                self.follow = Follow::Body(i);
                DVec3::from(b[i].pos)
            }
            Follow::Nothing => {
                // Flying free: re-centre the numbers when the camera is far from the origin
                // compared with what it looks at.
                if self.cam.pos.length() > 1e6 * self.ctl.dist.max(1e-9) { let w = self.origin + self.cam.pos; self.set_origin(w); }
                return;
            }
        };
        // Camera numbers are measured from what is followed: moving the origin moves the camera
        // and a trip under way along with it.
        self.follow_pos = now;
        self.origin = now;
    }

    fn predict_follow(&mut self) {
        if let Follow::Body(i) = self.follow {
            if let Some(b) = self.world.free.bodies.get(i) {
                self.follow_pos += DVec3::from(b.vel) * self.world.last_dt;
            }
        }
    }

    /// The body under screen point `m` and its distance (bodies at least about 8 pixels wide, so
    /// far ones can be clicked).
    fn pick(&self, m: (f64, f64), screen: (u32, u32)) -> Option<(usize, f64)> {
        let (o, d) = self.cam.ray(m.0, m.1, screen.0.max(1) as f64, screen.1.max(1) as f64);
        let mut best: Option<(usize, f64)> = None;
        let px = 2.0 * (0.5 * self.cam.fov_y).tan() / screen.1.max(1) as f64;
        for (i, b) in self.world.free.bodies.iter().enumerate() {
            let c = self.to_cam(DVec3::from(b.pos)) - o;
            let t = c.dot(d);
            if t <= 0.0 { continue; }
            let r = b.radius().max(8.0 * px * t);
            let q = c - d * t;
            if q.length() < r {
                let th = t - (r * r - q.length_squared()).max(0.0).sqrt();
                if best.is_none_or(|(_, bt)| th < bt) { best = Some((i, th)); }
            }
        }
        best
    }

    /// Focus on body i: the camera eases to 4 radii from it (or the closest allowed) and follows it.
    fn fly_to(&mut self, i: usize) {
        let m = self.mark();
        self.follow_now(Follow::Body(i));
        let b = &self.world.free.bodies[i];
        self.ctl.min_dist = Self::near_limit(b);
        let dist = (4.0 * b.radius()).max(self.ctl.min_dist);
        self.remember(m);
        self.start_trip(DVec3::ZERO, dist, None);
    }

    /// Everything as the renderer wants it: positions relative to the camera, last frame's
    /// position for motion, lamps (stars) and light-bending bodies (holes).
    fn gpu_objects(&self) -> (Vec<ObjGpu>, Vec<u32>, Vec<u32>) {
        // Camera-relative positions are worked out from the origin (never through world numbers,
        // which near 1 AU only have ~30 micrometre steps).
        let f3 = |r: DVec3| [r.x as f32, r.y as f32, r.z as f32];
        let mut objs = Vec::new();
        let mut lamps = Vec::new();
        let mut lens = Vec::new();
        let dt = self.world.last_dt;
        for b in &self.world.free.bodies {
            let i = objs.len() as u32;
            let cc = self.to_cam(DVec3::from(b.pos)) - self.cam.pos;   // centre, from the camera
            let v = DVec3::from(b.vel);
            let r = b.radius();
            let (mat, color, lensgm) = match b.kind {
                Kind::Star { temp_k } => {
                    lamps.push(i);
                    let k = SUN_RADIANCE * ((temp_k / 5772.0).powi(4)) as f32;
                    (3.0, temp_rgb(temp_k).map(|x| x * k), 0.0)
                }
                Kind::Hole { spin } => {
                    // Black hole (rt_hole.wgsl): a = spin M, disk from the innermost stable orbit
                    // out to DISK_OUT, all in Schwarzschild radii.
                    lens.push(i);
                    let a = spin * kerr::M;
                    let rout = if self.disk { DISK_OUT } else { 0.0 };
                    (5.0, [a as f32, kerr::isco(spin) as f32, rout as f32], (G * b.mass / (C_LIGHT * C_LIGHT)) as f32)
                }
                // Hot gas from a torn star: glows (seen directly; only stars light other things).
                Kind::Rock if b.glow => (3.0, b.color.map(|x| x * 0.05 * SUN_RADIANCE), 0.0),
                Kind::Rock => (0.0, b.color, 0.0),
            };
            // Horizon radius in Schwarzschild radii (holes only; read by the shader from ppos.w).
            let rh = if let Kind::Hole { spin } = b.kind { kerr::r_plus(spin * kerr::M) as f32 } else { 0.0 };
            if r > 1e3 {
                // Big ball: stored by its surface point nearest the camera (see rt.wgsl hit_big).
                let n = (-cc).normalize_or(DVec3::Y);
                let s = cc + n * r;
                let (sr, pr) = (f3(s), f3(s - v * dt));
                objs.push(ObjGpu {
                    pos: [sr[0], sr[1], sr[2], 3.0], rot: [n.x as f32, n.y as f32, n.z as f32, 0.0],
                    size: [r as f32, 0.0, 0.0, lensgm], mat: [color[0], color[1], color[2], mat],
                    ppos: [pr[0], pr[1], pr[2], rh],
                });
            } else {
                let (cr, pr) = (f3(cc), f3(cc - v * dt));
                objs.push(ObjGpu {
                    pos: [cr[0], cr[1], cr[2], 0.0], rot: [0.0, 0.0, 0.0, 1.0],
                    size: [r as f32, 0.0, 0.0, lensgm], mat: [color[0], color[1], color[2], mat],
                    ppos: [pr[0], pr[1], pr[2], rh],
                });
            }
        }
        (objs, lamps, lens)
    }

    /// Names for the free bodies, in list order: kind and number by size ("Star 1", "Rock 12").
    fn body_names(&self) -> Vec<String> {
        let b = &self.world.free.bodies;
        let kind = |k: Kind| match k { Kind::Star { .. } => "Star", Kind::Hole { .. } => "Black hole", Kind::Rock => "Rock" };
        let mut order: Vec<usize> = (0..b.len()).collect();
        order.sort_by(|&i, &j| b[j].mass.total_cmp(&b[i].mass));
        let mut names = vec![String::new(); b.len()];
        let mut count: HashMap<&str, usize> = HashMap::new();
        let total: HashMap<&str, usize> = b.iter().fold(HashMap::new(), |mut m, x| { *m.entry(kind(x.kind)).or_insert(0) += 1; m });
        for i in order {
            let k = kind(b[i].kind);
            let n = count.entry(k).or_insert(0);
            *n += 1;
            names[i] = if total[k] == 1 { k.to_string() } else { format!("{k} {n}") };
        }
        names
    }

    /// Finding things: a ring and name on stars and holes when they are too small to see, and an
    /// arrow at the edge pointing to the ones out of view. Rocks too when there are only a few.
    fn markers(&self, p: &egui::Painter) {
        let rect = p.clip_rect();
        let (f, r, u) = self.cam.axes();
        let t = (0.5 * self.cam.fov_y).tan();
        let aspect = (rect.width() / rect.height().max(1.0)) as f64;
        let names = self.body_names();
        let few = self.world.free.bodies.len() <= 30;
        let mut list: Vec<(DVec3, f64, String)> = Vec::new();
        for (i, b) in self.world.free.bodies.iter().enumerate() {
            if b.kind == Kind::Rock && !few { continue; }
            list.push((self.to_cam(DVec3::from(b.pos)) - self.cam.pos, b.radius(), names[i].clone()));
        }
        // A black hole bends the light of what is behind it: put the mark where the picture shows
        // the thing. Same turn as the tracer (rt.wgsl trace(): one turn of 4GM/b times the share
        // of the pass, at the closest point); the camera direction whose turned ray hits the
        // thing's centre is found by halving the interval.
        let holes: Vec<(DVec3, f64)> = self.world.free.bodies.iter().filter(|b| matches!(b.kind, Kind::Hole { .. }))
            .map(|b| (self.to_cam(DVec3::from(b.pos)) - self.cam.pos, G * b.mass / (C_LIGHT * C_LIGHT))).collect();
        let seen = |rel: DVec3| -> DVec3 {
            let mut out = rel;
            for &(h, gm) in &holes {
                let (dl, ds) = (h.length(), rel.length());
                if ds <= dl || h.dot(rel) <= 0.0 { continue; }
                let e1 = h / dl;
                let side = rel - e1 * rel.dot(e1);
                if side.length() < 1e-12 * ds { continue; }
                let e2 = side.normalize();
                let (sx, sy) = (rel.dot(e1), rel.dot(e2));
                let beta = sy.atan2(sx);
                // Miss angle for a camera ray at angle th from the hole (in the plane hole-thing).
                let miss = |th: f64| -> f64 {
                    let (tc, b) = (dl * th.cos(), dl * th.sin());
                    let f = 0.5 * (1.0 + tc / (tc * tc + b * b).sqrt());
                    let ang = 4.0 * gm / b * f * (1.0 + 15.0 * std::f64::consts::PI / 16.0 * gm / b * f);
                    let (cx, cy) = (tc * th.cos(), tc * th.sin());
                    (sy - cy).atan2(sx - cx) - (th - ang)
                };
                let (mut lo, mut hi) = (beta.max(1e-9), std::f64::consts::FRAC_PI_2);
                if miss(lo) <= 0.0 || miss(hi) >= 0.0 { continue; }
                for _ in 0..60 { let mid = 0.5 * (lo + hi); if miss(mid) > 0.0 { lo = mid; } else { hi = mid; } }
                let th = 0.5 * (lo + hi);
                out = (e1 * th.cos() + e2 * th.sin()) * ds;
            }
            out
        };
        let grey = egui::Color32::from_rgb(170, 200, 255);
        let font = egui::FontId::proportional(12.0);
        let c = rect.center();
        let (hw, hh) = (rect.width() * 0.5, rect.height() * 0.5);
        // (ring?, where, direction for arrows, label): marks that land on the same spot (a moon
        // and its planet seen from far away) share one mark and one label.
        let mut marks: Vec<(bool, egui::Pos2, egui::Vec2, String)> = Vec::new();
        let mut add = |ring: bool, at: egui::Pos2, dir: egui::Vec2, label: String| {
            match marks.iter_mut().find(|m| m.0 == ring && (m.1 - at).length() < 16.0) {
                Some(m) => { m.3 = format!("{} / {label}", m.3); }
                None => marks.push((ring, at, dir, label)),
            }
        };
        for (rel, rad, name) in list {
            // Big in the sky (or the ground under you): nothing to find.
            if rad > 0.25 * rel.length() { continue; }
            let rel = seen(rel);
            let z = rel.dot(f);
            let (x, y) = (rel.dot(r), rel.dot(u));
            if z > 0.0 && (x / (z * t * aspect)).abs() < 1.0 && (y / (z * t)).abs() < 1.0 {
                let px = rad / (z * t) * hh as f64;
                if px >= 3.0 { continue; }   // big enough to see
                let at = egui::pos2(c.x + (x / (z * t * aspect)) as f32 * hw, c.y - (y / (z * t)) as f32 * hh);
                add(true, at, egui::Vec2::ZERO, name);
            } else {
                // Out of view (or behind): an arrow at the edge, toward it.
                let (dx, dy) = (x / aspect, y);
                let m = dx.abs().max(dy.abs()).max(1e-300);
                let (ex, ey) = ((dx / m) as f32, (dy / m) as f32);
                // 18 points in from the edges, 56 from the bottom: clear of the scale bar.
                let at = egui::pos2(c.x + ex * (hw - 18.0), c.y - 19.0 - ey * (hh - 37.0));
                let dir = egui::vec2(ex, -ey).normalized();
                let away = (rel.length() - rad).max(0.0);
                add(false, at, dir, format!("{name} ({})", fmt_len(away)));
            }
        }
        for (ring, at, dir, label) in marks {
            if ring {
                p.circle_stroke(at, 7.0, egui::Stroke::new(1.2, grey));
                p.text(at + egui::vec2(10.0, -1.0), egui::Align2::LEFT_CENTER, label, font.clone(), grey);
            } else {
                let side = egui::vec2(-dir.y, dir.x);
                let tip = at + dir * 8.0;
                p.add(egui::Shape::convex_polygon(vec![tip, at - dir * 4.0 + side * 6.0, at - dir * 4.0 - side * 6.0], grey, egui::Stroke::NONE));
                let (ex, ey) = (dir.x, -dir.y);
                let align = egui::Align2([if ex > 0.5 { egui::Align::Max } else if ex < -0.5 { egui::Align::Min } else { egui::Align::Center },
                                          if ey > 0.5 { egui::Align::Min } else if ey < -0.5 { egui::Align::Max } else { egui::Align::Center }]);
                p.text(at - dir * 14.0, align, label, font.clone(), grey);
            }
        }
    }

    fn reset(&mut self, setup: usize) {
        let ev = self.ev;
        *self = WorldScene::new(setup);
        self.ev = ev;
    }
}

impl Scene for WorldScene {
    fn name(&self) -> &'static str { "Black hole" }
    fn help(&self) -> &'static str {
        "A black hole (Kerr, spin 0.9) with stars and rocks at true size. Everything pulls everything \
        (Newton); light bends near the hole (exact Kerr paths). Throw rocks in (1, or the panel) and watch the \
        hole grow and its spin change; stars that come too close are torn apart by tides; bodies that hit \
        merge, bounce or shatter. Other start setups: 2,000 rocks round kavan's star (they merge into planets), \
        an eclipse (a planet's shadow on two moons; starts paused) and kavan's own gravity scene.\n\
        Keys: 1 throw a rock, P pause, [ / ] time slower / faster, R restart, Home start view."
    }
    fn key(&mut self, k: KeyCode, down: bool) {
        if !down { return; }
        match k {
            KeyCode::Digit1 => self.throw(),
            KeyCode::KeyP => self.world.paused = !self.world.paused,
            KeyCode::BracketLeft => self.world.time_scale = (self.world.time_scale / 10.0).max(0.01),
            KeyCode::BracketRight => self.world.time_scale = (self.world.time_scale * 10.0).min(1e8),
            KeyCode::KeyR => self.reset(self.setup),
            KeyCode::Backspace => self.go_back(),
            KeyCode::Home => self.go_home(),
            _ => {}
        }
    }
    fn update(&mut self, dt: f64, input: &Input) {
        if !self.world.paused { self.swirl_clock = (self.swirl_clock + dt) % 512.0; }
        self.mouse_still = if input.mouse == self.mouse && !input.left && !input.right { self.mouse_still + dt } else { 0.0 };
        self.mouse = input.mouse;
        let screen = input.screen;
        if let Some(p) = input.double { if let Some((i, _)) = self.pick(p, screen) { self.fly_to(i); } }
        // Physics, then the camera rides along with what it follows.
        self.world.step(dt);
        self.predict_follow();
        self.ride();
        self.ctl.apply(&mut self.cam, input, dt);
        self.keep_out();
        self.hover = self.pick(input.mouse, screen).map(|x| x.0);
    }
    fn ui(&mut self, ui: &mut egui::Ui) {
        // What to look at, and time: the two things used most.
        let mut pick = None;
        egui::ComboBox::from_id_salt("wsetup").width(ui.available_width() - 8.0).selected_text(SETUPS[self.setup]).show_ui(ui, |ui| {
            for i in SETUP_ORDER {
                if ui.selectable_label(self.setup == i, SETUPS[i]).clicked() { pick = Some(i); }
            }
        });
        if let Some(i) = pick { self.reset(i); }
        ui.horizontal(|ui| {
            if ui.button(if self.world.paused { "Play" } else { "Pause" }).on_hover_text("P").clicked() { self.world.paused = !self.world.paused; }
            let mut lg = self.world.time_scale.log10();
            if ui.add(egui::Slider::new(&mut lg, -2.0..=8.0).custom_formatter(|v, _| format!("{}x", fmt_speed(10f64.powf(v)))))
                .on_hover_text("Speed, times real time ([ and ] change it 10 times)").changed() { self.world.time_scale = 10f64.powf(lg); }
        });
        ui.add_space(4.0);
        self.go_to_ui(ui);
        ui.add_space(4.0);
        egui::CollapsingHeader::new("More").default_open(false).show(ui, |ui| self.more_ui(ui));
    }

    /// Only what the panel doesn't already show, so usually empty and no box.
    fn status(&self) -> Vec<String> {
        let w = &self.world;
        let mut v = Vec::new();
        if !w.paused && w.speed_share < 0.95 { v.push(format!("Gravity keeps up with {:.0}% of the set speed", 100.0 * w.speed_share)); }
        v
    }
    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, frame: &mut Frame, screen: (u32, u32), fps: f64) {
        // First frame: nothing traced yet, so the window shows "Loading ..." (overlay).
        if self.frames == 0 && fps.is_finite() { self.frames = 1; return; }
        self.frames += 1;
        if self.tracer.is_none() { self.tracer = Some(Renderer::new(device)); }
        let moving = !self.world.paused && !self.world.free.bodies.is_empty();
        let c = &self.cam;
        let mut view = Vec::new();
        for x in [c.pos.x, c.pos.y, c.pos.z, c.rot.x, c.rot.y, c.rot.z, c.rot.w] { view.extend_from_slice(&x.to_le_bytes()); }
        let cam_moved = view != self.last_view;
        if cam_moved || moving { self.acc_frames = 0; }
        self.last_view = view;
        let still = !cam_moved && !moving;
        // Trace size for about 40-55 fps, checked every quarter second (a new size restarts the
        // clean-up's history): down quickly below 40, up slowly above 55, at least 12% of the
        // window's width. The picture shown is built at window size from several frames.
        self.adapt_frames += 1;
        let el = self.adapt_t.elapsed().as_secs_f64();
        if fps.is_infinite() { self.scale = 1.0; }   // a screenshot: full size
        else if el >= 0.25 {
            let f = self.adapt_frames as f64 / el;
            self.adapt_t = Instant::now();
            self.adapt_frames = 0;
            if f < 40.0 { self.scale = (self.scale * (f / 45.0).sqrt().clamp(0.7, 0.95)).max(0.12); }
            else if f > 55.0 { self.scale = (self.scale * 1.04).min(1.0); }
        }
        // The trace size moves in steps of 10% (each new size restarts the clean-up's history).
        let size = Renderer::trace_size(screen, crate::rt::scale_step(self.scale));
        let out = Renderer::out_size(screen);
        self.acc_frames += 1;
        if still && self.acc_frames > STILL_FRAMES { return; }
        let dt = self.last_t.elapsed().as_secs_f64().min(0.1) as f32;
        self.last_t = Instant::now();
        let (objs, lamps, lens) = self.gpu_objects();
        let opts = Opts {
            max_bounces: if still { 8 } else { 3 },
            lens_bounces: if still { 2 } else { 1 },
            history_min: if still { 1.0 / 255.0 } else { 0.1 },
            ev: self.ev,
            dt: dt.max(1.0 / 120.0),
            // The disk's swirl runs on its own clock (screen seconds, stopped while paused), so it
            // doesn't jump when the speed changes. Kept small (it only needs to be continuous
            // modulo 64 swirl periods).
            time: self.swirl_clock as f32,
            min_eps: (1e-4 * self.ctl.dist).clamp(1e-15, 1e-3) as f32,
        };
        frame.write_post(queue, &Post { v: [1.0, 0.3, 0.2, 1.0] });
        let cam = self.cam.clone();
        self.tracer.as_mut().unwrap().render(device, queue, enc, frame, &cam, &objs, &lamps, &lens, size, out, &opts);
    }
    fn overlay(&mut self, p: &egui::Painter, ppp: f32, origin: egui::Pos2) {
        if self.frames <= 1 {
            p.text(p.clip_rect().center(), egui::Align2::CENTER_CENTER, "Loading ...", egui::FontId::proportional(18.0), egui::Color32::from_gray(200));
            return;
        }
        // Scale bar (bottom left): a round length about 120 points long at the middle's distance.
        let d = (self.ctl.target - self.cam.pos).length();
        let h_pts = p.clip_rect().height().max(1.0) as f64;
        let per_pt = 2.0 * (0.5 * self.cam.fov_y).tan() * d / h_pts;
        let want = 120.0 * per_pt;
        let e = want.log10().floor();
        let base = 10f64.powf(e);
        let len = [1.0, 2.0, 5.0, 10.0].iter().map(|k| k * base).filter(|&l| l <= want).last().unwrap_or(base);
        let w = (len / per_pt) as f32;
        let y = p.clip_rect().bottom() - 24.0;
        let x0 = p.clip_rect().left() + 20.0;
        let st = egui::Stroke::new(2.0, egui::Color32::WHITE);
        p.line_segment([egui::pos2(x0, y), egui::pos2(x0 + w, y)], st);
        p.line_segment([egui::pos2(x0, y - 5.0), egui::pos2(x0, y + 5.0)], st);
        p.line_segment([egui::pos2(x0 + w, y - 5.0), egui::pos2(x0 + w, y + 5.0)], st);
        p.text(egui::pos2(x0, y - 8.0), egui::Align2::LEFT_BOTTOM, fmt_len(len), egui::FontId::proportional(14.0), egui::Color32::WHITE);
        self.markers(p);
        // The name of what is under the mouse, once the mouse has rested there half a second (a
        // label that follows every move covers what is being looked at).
        let Some(i) = self.hover else { return };
        if self.mouse_still < 0.5 { return; }
        let names = self.body_names();
        let Some(name) = names.get(i) else { return };
        let at = origin + egui::vec2(self.mouse.0 as f32 / ppp + 16.0, self.mouse.1 as f32 / ppp + 16.0);
        let font = egui::FontId::proportional(13.0);
        let galley = p.layout_no_wrap(format!("{name}  (double-click to focus on it)"), font, egui::Color32::from_gray(230));
        let r = egui::Rect::from_min_size(at, galley.size()).expand(4.0);
        p.rect_filled(r, 4.0, egui::Color32::from_black_alpha(170));
        p.galley(at, galley, egui::Color32::WHITE);
    }
}

impl WorldScene {
    /// Panel: find a body by name and focus on it (the list shows only what matches what is typed).
    fn go_to_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::TextEdit::singleline(&mut self.find).desired_width(f32::INFINITY).hint_text("Focus on: type a name ..."))
            .on_hover_text("Type a name, then click it to focus on it. Double-clicking a body in the picture does the same.");
        let want = self.find.trim().to_lowercase();
        if want.is_empty() { return; }
        let names = self.body_names();
        let mut go = None;
        egui::ScrollArea::vertical().id_salt("finder").max_height(160.0).show(ui, |ui| {
            let b = &self.world.free.bodies;
            let mut order: Vec<usize> = (0..b.len()).collect();
            let rank = |k: Kind| match k { Kind::Hole { .. } => 0, Kind::Star { .. } => 1, Kind::Rock => 2 };
            order.sort_by(|&i, &j| rank(b[i].kind).cmp(&rank(b[j].kind)).then(b[j].mass.total_cmp(&b[i].mass)));
            let mut shown = 0;
            for i in order {
                if !names[i].to_lowercase().contains(want.as_str()) || shown >= 40 { continue; }
                shown += 1;
                if ui.selectable_label(self.follow == Follow::Body(i), names[i].as_str()).clicked() { go = Some(i); }
            }
            if shown == 0 { ui.weak("nothing by that name"); }
        });
        if let Some(i) = go { self.fly_to(i); self.find.clear(); }
    }

    /// Panel, "More": Back / Forward / Start view, a rock to throw, brightness, the gas disk.
    fn more_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.back.is_empty(), egui::Button::new("Back")).on_hover_text("Backspace").clicked() { self.go_back(); }
            if ui.add_enabled(!self.fwd.is_empty(), egui::Button::new("Forward")).clicked() { self.go_forward(); }
            if ui.button("Start view").on_hover_text("Home").clicked() { self.go_home(); }
        });
        if ui.button("Throw a rock (1)").on_hover_text("From the camera, toward the middle of the picture").clicked() { self.throw(); }
        ui.add(egui::Slider::new(&mut self.ev, -4.0..=4.0).custom_formatter(|v, _| format!("{v:.1}")).text("brightness"))
            .on_hover_text("On top of the automatic exposure; +1 is twice as bright");
        if self.world.free.bodies.iter().any(|b| matches!(b.kind, Kind::Hole { .. })) { ui.checkbox(&mut self.disk, "Black hole gas disk"); }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_colour_is_near_white_and_cool_stars_are_red() {
        let s = temp_rgb(5772.0);
        assert!(s[0] == 1.0 && s[1] > 0.9 && s[2] > 0.8, "{s:?}");
        let r = temp_rgb(3000.0);
        assert!(r[0] == 1.0 && r[2] < 0.5, "{r:?}");
        let b = temp_rgb(15000.0);
        assert!(b[2] == 1.0 && b[0] < 0.9, "{b:?}");
    }

    #[test]
    fn length_labels() {
        assert_eq!(fmt_len(1.5), "1.5 m");
        assert_eq!(fmt_len(world::AU), "1.00 AU");
        assert_eq!(fmt_len(6.371e6), "6371 km");
    }

    fn inp(m: (f64, f64)) -> Input { Input { screen: (1280, 720), mouse: m, ..Default::default() } }

    #[test]
    fn corner_box_is_empty_at_the_start() {
        let mut w = WorldScene::new(0);
        w.world.paused = true;
        w.update(1.0 / 60.0, &inp((640.0, 360.0)));
        assert!(w.status().is_empty(), "{:?}", w.status());
    }

    fn on_screen(c: &Camera, p: DVec3) -> (f64, f64) {
        let (f, r, u) = c.axes();
        let v = p - c.pos;
        let t = (0.5 * c.fov_y).tan();
        let (x, y) = (v.dot(r) / v.dot(f) / (t * 1280.0 / 720.0), v.dot(u) / v.dot(f) / t);
        ((x + 1.0) * 640.0, (1.0 - y) * 360.0)
    }

    #[test]
    fn black_hole_setup_opens_on_the_hole() {
        let mut w = WorldScene::new(SETUP_ORDER[0]);
        w.update(1.0 / 60.0, &inp((640.0, 360.0)));
        let h = w.world.free.bodies.iter().position(|b| matches!(b.kind, Kind::Hole { .. })).expect("a hole");
        let s = on_screen(&w.cam, w.to_cam(DVec3::from(w.world.free.bodies[h].pos)));
        assert!((s.0 - 640.0).abs() < 1.0 && (s.1 - 360.0).abs() < 1.0, "hole at {s:?}");
    }

    #[test]
    fn camera_never_inside_a_body() {
        let mut seed = 11u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f64 / (1u64 << 31) as f64 };
        for setup in SETUP_ORDER {
            let mut w = WorldScene::new(setup);
            w.world.time_scale *= 10.0;
            for f in 0..2000 {
                let mut i = inp((1280.0 * rnd(), 720.0 * rnd()));
                let r = rnd();
                if r < 0.3 { i.swipe = (200.0 * (rnd() - 0.5), 200.0 * (rnd() - 0.5)); }
                else if r < 0.5 { i.pinch = 0.3 * (rnd() - 0.4); }
                else if r < 0.55 { i.held.insert(KeyCode::ShiftLeft); i.swipe = (100.0 * (rnd() - 0.5), 100.0 * (rnd() - 0.5)); }
                else if r < 0.6 { i.left = true; i.drag_left = (60.0 * (rnd() - 0.5), 60.0 * (rnd() - 0.5)); }
                else if r < 0.61 { i.double = Some(i.mouse); }
                else if r < 0.62 { w.throw(); }
                w.update(1.0 / 60.0, &i);
                assert!(w.cam.pos.is_finite() && w.ctl.dist.is_finite(), "setup {setup} frame {f}: broken numbers");
                for b in &w.world.free.bodies {
                    let d = (w.cam.pos - w.to_cam(DVec3::from(b.pos))).length();
                    assert!(d >= b.radius() * 0.999, "setup {setup} frame {f}: camera inside a body ({d} m from its centre, radius {})", b.radius());
                }
            }
        }
    }

    #[test]
    fn double_click_focuses_on_a_star() {
        let mut w = WorldScene::new(SETUP_ORDER[0]);
        w.world.paused = true;
        w.update(1.0 / 60.0, &inp((640.0, 360.0)));
        let i = w.world.free.bodies.iter().position(|b| matches!(b.kind, Kind::Star { .. })).unwrap();
        w.fly_to(i);
        for _ in 0..240 { w.update(1.0 / 60.0, &inp((640.0, 360.0))); assert!(w.cam.pos.is_finite()); }
        assert_eq!(w.follow, Follow::Body(i));
        let r = w.world.free.bodies[i].radius();
        let d = (w.cam.pos - w.to_cam(DVec3::from(w.world.free.bodies[i].pos))).length();
        assert!((d / (4.0 * r) - 1.0).abs() < 0.01, "ended {d} m from the star, want {}", 4.0 * r);
        w.go_back();
        for _ in 0..240 { w.update(1.0 / 60.0, &inp((640.0, 360.0))); }
        assert!(matches!(w.follow, Follow::Body(_)));
    }

    #[test]
    fn drawing_the_panel_changes_nothing() {
        for setup in SETUP_ORDER {
            let mut w = WorldScene::new(setup);
            w.ctl.dist *= 1.0123;
            let (d, ts, ev) = (w.ctl.dist, w.world.time_scale, w.ev);
            let ctx = egui::Context::default();
            for _ in 0..5 {
                let _ = ctx.run_ui(egui::RawInput::default(), |ui| { egui::Panel::left("p").show(ui, |ui| w.ui(ui)); });
            }
            assert_eq!(w.ctl.dist, d, "setup {setup}: view distance changed");
            assert_eq!(w.world.time_scale, ts, "setup {setup}: time speed changed");
            assert_eq!(w.ev, ev);
        }
    }
}

