//! Atoms tab (from kavan010/Atoms, atom_realtime.cpp): millions of sampled electron
//! positions of a hydrogen orbital, moving along the probability current. "Show" also draws the
//! 118 elements, the molecules and the crystals the same way (dots from their solvers' densities,
//! clouds.rs).
//! Light: a photon tuned to an allowed jump up is absorbed and the electron jumps; later it falls
//! back on its own and gives off photons of the right colours (rates from the same wave shapes).

use crate::atom::{clamp_qn, climbs, falls, lifetime, m_chances, sample, wavelength_nm, wavelength_rgb, Sample, N_MAX};
use crate::clouds::{self, ElemPart};
use crate::elements;
use crate::molecules as mc;
use crate::solids as so;
use rayon::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use crate::camera::Camera;
use crate::orbit::OrbitCam;
use crate::frame::{Frame, Post, DEPTH, HDR};
use crate::host::{Input, Scene};
use glam::DVec3;
use winit::keyboard::KeyCode;

const START_COUNT: usize = 1 << 20;
const MAX_COUNT: usize = 1 << 20;
const FLOW_RATE: f64 = 30.0;          // atomic time units per second
const SLOW_START: f64 = 1e9;          // screen seconds per real second: a 1.6 ns fall shows as 1.6 s
const MORPH_SECS: f64 = 0.8;          // how long the cloud takes to slide into a new orbital
const PHOTON_SECS: f64 = 1.2;         // photon flight time from the edge of the view to the atom

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_pos: [f32; 4],
    misc: [f32; 4],
    misc2: [f32; 4],
    ph: [[f32; 4]; 16],
    spec: [[f32; 4]; 16],
}

/// What the GPU needs besides the camera: the morph between orbitals, photons and the spectrum.
struct Extras { blend: f32, old_m: f32, old_rad: f32, grid_half: f32, ph: [[f32; 4]; 16], spec: [[f32; 4]; 16] }

#[allow(clippy::too_many_arguments)]
fn pipe(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, vs: &str, fs: &str,
        topo: wgpu::PrimitiveTopology, depth_write: bool, blend: Option<wgpu::BlendState>, overlay: bool) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs), layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState { topology: topo, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH, depth_write_enabled: Some(depth_write),
            depth_compare: Some(if overlay { wgpu::CompareFunction::Always } else { wgpu::CompareFunction::Less }),
            stencil: Default::default(), bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module, entry_point: Some(fs), compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: HDR, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None, cache: None,
    })
}

struct Draw {
    pt_pipe: wgpu::RenderPipeline,
    grid_pipe: wgpu::RenderPipeline,
    ph_pipe: wgpu::RenderPipeline,
    spec_pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    ubuf: wgpu::Buffer,
    bg: Option<wgpu::BindGroup>,
    cur: Option<wgpu::Buffer>,
    count: u32,
    old_count: u32,
}

impl Draw {
    fn new(device: &wgpu::Device) -> Draw {
        use crate::frame::{storage_entry, ubo_entry};
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atoms"), source: wgpu::ShaderSource::Wgsl(include_str!("../atoms.wgsl").into()),
        });
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[ubo_entry(0, vf), storage_entry(1, vf), storage_entry(2, vf)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let strip = wgpu::PrimitiveTopology::TriangleStrip;
        let pt_pipe = pipe(device, &module, &layout, "vs_pt", "fs_pt", strip, true, None, false);
        let grid_pipe = pipe(device, &module, &layout, "vs_grid", "fs_grid", wgpu::PrimitiveTopology::LineList, false,
            Some(wgpu::BlendState::ALPHA_BLENDING), false);
        let ph_pipe = pipe(device, &module, &layout, "vs_ph", "fs_ph", strip, false, Some(crate::frame::ADD), false);
        let spec_pipe = pipe(device, &module, &layout, "vs_spec", "fs_spec", strip, false, None, true);
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Draw { pt_pipe, grid_pipe, ph_pipe, spec_pipe, bgl, ubuf, bg: None, cur: None, count: 0, old_count: 0 }
    }

    /// New set of points: one GPU buffer, made once per orbital change (not per frame). The
    /// previous set is kept so the dots can slide from the old orbital to the new one.
    fn set_points(&mut self, device: &wgpu::Device, pts: &[Sample]) {
        use wgpu::util::DeviceExt;
        let sbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"), contents: bytemuck::cast_slice(pts), usage: wgpu::BufferUsages::STORAGE,
        });
        let old = self.cur.take().unwrap_or_else(|| sbuf.clone());
        self.old_count = if self.count == 0 { pts.len() as u32 } else { self.count };
        self.bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: old.as_entire_binding() },
            ],
        }));
        self.cur = Some(sbuf);
        self.count = pts.len() as u32;
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, frame: &mut Frame, cam: &Camera,
              dot_r: f64, m: i32, t: f64, cutaway: bool, grid: bool, aspect: f64, pix_h: u32, x: &Extras, light: bool) {
        let (_, right, up) = cam.axes();
        // Near plane follows the camera's distance, so small things (a heavy atom's core, 0.01 a0)
        // can be looked at closely.
        let near = (cam.pos.length() * 1e-3).clamp(1e-4, 0.05);
        let u = U {
            view_proj: cam.view_proj_rel(aspect, near, 5000.0),
            cam_right: [right.x as f32, right.y as f32, right.z as f32, (2.0 * (0.5 * cam.fov_y).tan() / pix_h as f64) as f32],
            cam_up: [up.x as f32, up.y as f32, up.z as f32, dot_r as f32],
            cam_pos: [cam.pos.x as f32, cam.pos.y as f32, cam.pos.z as f32, t as f32],
            misc: [m as f32, if cutaway { 1.0 } else { 0.0 }, x.blend, x.old_m],
            misc2: [self.old_count as f32, x.old_rad, aspect as f32, x.grid_half],
            ph: x.ph,
            spec: x.spec,
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        {
            let mut rp = frame.begin_scene(enc, true);
            if let Some(bg) = &self.bg {
                rp.set_bind_group(0, bg, &[]);
                if grid { rp.set_pipeline(&self.grid_pipe); rp.draw(0..12, 0..1); }
                rp.set_pipeline(&self.pt_pipe);
                rp.draw(0..4, 0..self.count);
                if light { rp.set_pipeline(&self.ph_pipe); rp.draw(0..4, 0..8); }
                rp.set_pipeline(&self.spec_pipe);
                rp.draw(0..4, 0..1);
            }
        }
        frame.encode_post(enc);
    }
}

/// How a non-hydrogen cloud is drawn.
pub struct Other {
    /// What it is, for the panel and status line.
    pub title: String,
    /// More lines for the panel (method, checks, what is left out).
    pub notes: Vec<String>,
    pub dot_r: f64,
    /// Drift: an element's single orbital with m != 0 carries a current like hydrogen's; else 0.
    pub m: i32,
    /// Radius holding 95 percent of the dots (a0).
    pub r95: f64,
    /// Lines for the spectrum strip (nm, strength): an element's NIST lines; empty otherwise.
    pub spec: Vec<(f64, f64)>,
}

impl Other {
    pub fn view_dist(&self) -> f64 { 3.2 * self.r95 }
}

pub struct Photon { pub pos: DVec3, pub vel: DVec3, pub nm: f64, incoming: bool, checked: bool, age: f64 }

pub struct Sim {
    pub n: i32, pub l: i32, pub m: i32,
    pub count: usize,
    pub slow: f64,                 // screen seconds per real second
    pub tune: i32,                 // photon tuned to the jump n -> tune (0 = off-tune)
    pub photons: Vec<Photon>,
    pub fall_in: Option<f64>,      // screen seconds until the electron falls (None: no allowed fall)
    morph: f64,                // 0..1 slide from the old orbital
    old_look: (f64, i32),      // dot radius (a0) and flow m of the cloud before the last change
    shown_look: (f64, i32),    // the same for the dots on screen (a new orbital's may still be on their way)
    /// Set when the dots are not a hydrogen orbital (an element, molecule or crystal).
    pub other: Option<Other>,
    pub spectrum: Vec<(f64, f64)>, // (nm, count) of photons given off
    rng: u64,
    pts: Vec<Sample>,
    dirty: bool,
    t: f64,
    paused: bool,
    cutaway: bool,
    grid: bool,
    pub cam: Camera,
    dist_target: f64,
    /// Dots of a new hydrogen orbital being drawn on another thread, so the picture doesn't stop
    /// at every jump.
    pending: Option<Receiver<(Vec<Sample>, i32, i32)>>,
    /// Going back to hydrogen: the other cloud stays until hydrogen's dots are ready.
    leaving_other: bool,
}

impl Sim {
    pub fn new(n: i32, l: i32, m: i32, count: usize) -> Sim {
        let mut s = Sim { n, l, m, count, slow: SLOW_START, tune: 0, photons: Vec::new(), fall_in: None, morph: 1.0,
            old_look: (0.05 * n as f64 / 3.0, m), shown_look: (0.05 * n as f64 / 3.0, m), other: None, spectrum: Vec::new(), rng: 0x9E3779B97F4A7C15,
            pts: Vec::new(), dirty: true, t: 0.0, paused: false, cutaway: true, grid: true,
            cam: Camera::new(DVec3::new(50.0, 0.0, 0.1), DVec3::ZERO), dist_target: 50.0,
            pending: None, leaving_other: false };
        s.resample();
        s.wait_points();
        s.morph = 1.0;
        s.tune = s.n + 1;
        s.schedule_fall(); // an excited start state falls back too
        s
    }
    fn rnd(&mut self) -> f64 {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<T: Copy>(&mut self, v: &[(T, f64)]) -> T {
        let total: f64 = v.iter().map(|x| x.1).sum();
        let mut u = self.rnd() * total;
        for &(x, w) in v { if u < w { return x; } u -= w; }
        v.last().unwrap().0
    }
    /// Size of the cloud (average radius, a0).
    pub fn cloud(&self) -> f64 { 0.5 * (3 * self.n * self.n - self.l * (self.l + 1)) as f64 }
    /// Start the wait for the next fall: random, with the orbital's real mean life (slowed down).
    pub fn schedule_fall(&mut self) {
        let tau = lifetime(self.n, self.l);
        self.fall_in = if tau.is_finite() { Some(-(1.0 - self.rnd()).ln() * tau * self.slow) } else { None };
    }
    /// Jump to a new orbital because of light: slide the dots, frame the new cloud, wait for the next fall.
    fn jump(&mut self, n: i32, l: i32, m: i32) {
        (self.n, self.l, self.m) = (n, l, m);
        self.resample();
        self.dist_target = 4.0 * self.cloud() + 10.0;
        self.schedule_fall();
        if self.tune <= self.n { self.tune = (self.n + 1).min(N_MAX); }
    }
    fn photon_rgb(nm: f64) -> [f32; 3] { wavelength_rgb(nm) }
    /// Wavelength of the photon the player would fire now.
    pub fn tuned_nm(&self) -> f64 {
        if self.tune > self.n { wavelength_nm(self.n, self.tune) }
        // Off-tune: halfway between the first two lines up, so no jump matches it.
        else { 0.5 * (wavelength_nm(self.n, self.n + 1) + wavelength_nm(self.n, (self.n + 2).min(N_MAX + 1))) }
    }
    pub fn fire(&mut self) {
        let (_, right, up) = self.cam.axes();
        let d = 4.0 * self.cloud() + 10.0;
        let from = -right * d + up * 0.15 * d;
        let nm = self.tuned_nm();
        self.photons.push(Photon { pos: from, vel: -from / PHOTON_SECS, nm, incoming: true, checked: false, age: 0.0 });
        self.photons.truncate(8);
    }
    /// Dot radius (a0) and the m that sets the drift of the dots on screen.
    pub fn look(&self) -> (f64, i32) { self.shown_look }
    /// Show other dots (an element, molecule or crystal): they slide in from the current ones.
    pub fn show_other(&mut self, pts: Vec<Sample>, o: Other) {
        self.pending = None;
        self.leaving_other = false;
        self.old_look = self.shown_look;
        self.shown_look = (o.dot_r, o.m);
        self.pts = pts;
        self.dirty = true;
        self.morph = 0.0;
        self.dist_target = o.view_dist();
        self.other = Some(o);
        self.photons.clear();
        self.fall_in = None;
    }
    pub fn show_hydrogen(&mut self) {
        if self.other.is_none() { return; }
        self.leaving_other = true;
        self.resample();
    }
    /// Take in a new hydrogen cloud drawn on its thread, if it is ready.
    pub fn take_points(&mut self) {
        let got = match &self.pending {
            Some(rx) => match rx.try_recv() {
                Ok(p) => Some(p),
                // The drawing thread stopped without an answer: draw here instead.
                Err(mpsc::TryRecvError::Disconnected) => Some((self.sample_now(), self.n, self.m)),
                Err(mpsc::TryRecvError::Empty) => None,
            },
            None => None,
        };
        if let Some(p) = got { self.apply_points(p); }
    }
    fn sample_now(&self) -> Vec<Sample> {
        sample(self.n, self.l, self.m, self.count, 0x5EED ^ (self.n * 100 + self.l * 10 + self.m) as u64)
    }
    /// Wait for the cloud being drawn.
    pub fn wait_points(&mut self) {
        let got = self.pending.as_ref().map(|rx| rx.recv().unwrap_or_else(|_| (self.sample_now(), self.n, self.m)));
        if let Some(p) = got { self.apply_points(p); }
    }
    fn apply_points(&mut self, (pts, n, m): (Vec<Sample>, i32, i32)) {
        self.pending = None;
        if self.leaving_other {
            self.leaving_other = false;
            self.other = None;
            self.schedule_fall();
            self.dist_target = 4.0 * self.cloud() + 10.0;
        } else if self.other.is_some() {
            return; // an element, molecule or crystal was picked meanwhile
        }
        self.pts = pts;
        self.dirty = true;
        self.morph = 0.0;
        self.old_look = self.shown_look;
        self.shown_look = (0.05 * n as f64 / 3.0, m);
    }
    pub fn update_light(&mut self, dt: f64) {
        self.take_points();
        if self.morph < 1.0 { self.morph = (self.morph + dt / MORPH_SECS).min(1.0); }
        if self.other.is_some() { return; } // photons and jumps are hydrogen's (atom.rs)
        // Photons fly; an incoming one reaching the middle is absorbed if its colour fits a jump up.
        let mut arrived = None;
        for p in &mut self.photons {
            p.pos += p.vel * dt;
            p.age += dt;
            if p.incoming && !p.checked && p.pos.dot(p.vel) >= 0.0 { p.checked = true; arrived = Some(p.nm); }
        }
        if let Some(nm) = arrived {
            let ups: Vec<((i32, i32), f64)> = climbs(self.n, self.l).into_iter()
                .filter(|&(n2, _, _)| (wavelength_nm(self.n, n2) - nm).abs() < 1e-6 * nm)
                .map(|(n2, l2, w)| ((n2, l2), w)).collect();
            if !ups.is_empty() {
                let (n2, l2) = self.pick(&ups);
                let m2 = self.pick(&m_chances(self.l, self.m, l2));
                self.photons.retain(|p| !(p.incoming && p.checked && (p.nm - nm).abs() < 1e-9));
                self.jump(n2, l2, m2);
            }
        }
        // Falling back: pick where to by the real rates, give off a photon of that colour.
        if let Some(t) = self.fall_in.as_mut() {
            *t -= dt;
            if *t <= 0.0 {
                let f = falls(self.n, self.l);
                let (n2, l2) = self.pick(&f.iter().map(|&(n2, l2, a)| ((n2, l2), a)).collect::<Vec<_>>());
                let m2 = self.pick(&m_chances(self.l, self.m, l2));
                let nm = wavelength_nm(self.n, n2);
                let th = self.rnd() * std::f64::consts::TAU;
                let z = 2.0 * self.rnd() - 1.0;
                let dir = DVec3::new((1.0 - z * z).sqrt() * th.cos(), z, (1.0 - z * z).sqrt() * th.sin());
                let speed = (4.0 * self.cloud() + 10.0) / PHOTON_SECS;
                self.photons.insert(0, Photon { pos: DVec3::ZERO, vel: dir * speed, nm, incoming: false, checked: true, age: 0.0 });
                self.photons.truncate(8);
                match self.spectrum.iter_mut().find(|s| (s.0 - nm).abs() < 1e-6 * nm) {
                    Some(s) => s.1 += 1.0,
                    None => { self.spectrum.push((nm, 1.0)); self.spectrum.truncate(16); }
                }
                self.jump(n2, l2, m2);
            }
        }
        self.photons.retain(|p| p.age < 2.5 * PHOTON_SECS);
    }
    fn extras(&self) -> Extras {
        let mut ph = [[0.0f32; 4]; 16];
        let size = (0.04 * (4.0 * self.cloud() + 10.0)) as f32;
        for (k, p) in self.photons.iter().take(8).enumerate() {
            let c = Self::photon_rgb(p.nm);
            ph[2 * k] = [p.pos.x as f32, p.pos.y as f32, p.pos.z as f32, size];
            ph[2 * k + 1] = [c[0], c[1], c[2], 1.0];
        }
        let mut spec = [[0.0f32; 4]; 16];
        let lines = match &self.other { Some(o) => &o.spec, None => &self.spectrum };
        let top = lines.iter().map(|s| s.1).fold(if self.other.is_some() { 0.0 } else { 1.0 }, f64::max).max(1e-300);
        for (k, &(nm, cnt)) in lines.iter().take(16).enumerate() {
            let c = wavelength_rgb(nm);
            let b = (0.25 + 0.75 * cnt / top) as f32;
            spec[k] = [nm as f32, c[0] * b, c[1] * b, c[2] * b];
        }
        // Grid: 500 a0 square for hydrogen (well past the camera); for the others 10
        // times the cloud's 95 percent radius, so it too reaches past the camera (3.2 times).
        let grid_half = match &self.other { Some(o) => 10.0 * o.r95, None => 250.0 } as f32;
        Extras { blend: self.morph as f32, old_m: self.old_look.1 as f32, old_rad: self.old_look.0 as f32, grid_half, ph, spec }
    }
    /// Draw the dots of orbital (n, l, m) on another thread; they slide in when ready
    /// (take_points). A newer request replaces an older one still being drawn.
    pub fn resample(&mut self) {
        (self.n, self.l, self.m) = clamp_qn(self.n, self.l, self.m);
        let (n, l, m, count) = (self.n, self.l, self.m, self.count);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send((sample(n, l, m, count, 0x5EED ^ (n * 100 + l * 10 + m) as u64), n, m));
        });
        self.pending = Some(rx);
    }
    pub fn key(&mut self, k: KeyCode) {
        if self.other.is_some() {
            // The orbital and light keys are hydrogen's; the view keys work for every cloud.
            match k {
                KeyCode::KeyC => self.cutaway = !self.cutaway,
                KeyCode::KeyH => self.grid = !self.grid,
                KeyCode::KeyP => self.paused = !self.paused,
                _ => {}
            }
            return;
        }
        let before = (self.n, self.l, self.m, self.count);
        match k {
            // Z steps down the same ring X steps up: off-tune, n+1, ..., N_MAX, off-tune.
            KeyCode::KeyZ => { self.tune = if self.tune == 0 { N_MAX } else { self.tune - 1 }; if self.tune <= self.n { self.tune = 0; } }
            KeyCode::KeyX => { self.tune = if self.tune == 0 { self.n + 1 } else { self.tune + 1 }; if self.tune > N_MAX { self.tune = 0; } }
            KeyCode::Space => self.fire(),
            KeyCode::BracketLeft | KeyCode::BracketRight => {
                let f = if k == KeyCode::BracketLeft { 10.0 } else { 0.1 };
                self.slow = (self.slow * f).clamp(1e6, 1e12);
                if let Some(t) = self.fall_in.as_mut() { *t *= f; }
            }
            KeyCode::KeyW => self.n += 1,
            KeyCode::KeyS => self.n -= 1,
            KeyCode::KeyE => self.l += 1,
            KeyCode::KeyD => self.l -= 1,
            KeyCode::KeyR => self.m += 1,
            KeyCode::KeyF => self.m -= 1,
            KeyCode::KeyT => self.count = (self.count * 2).min(MAX_COUNT),
            KeyCode::KeyG => self.count = (self.count / 2).max(1024),
            KeyCode::KeyC => self.cutaway = !self.cutaway,
            KeyCode::KeyH => self.grid = !self.grid,
            KeyCode::KeyP => self.paused = !self.paused,
            _ => {}
        }
        self.n = self.n.clamp(1, 12);
        let (n, l, m) = clamp_qn(self.n, self.l, self.m);
        if (n, l, m, self.count) != before {
            self.n = n; self.l = l; self.m = m;
            self.resample();
            self.schedule_fall();
            if self.tune != 0 && self.tune <= self.n { self.tune = (self.n + 1).min(N_MAX); }
        }
        else { (self.l, self.m) = (l, m); }
    }
}

pub fn lname(l: i32) -> char { ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o'][l as usize] }

/// What the dots show.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Subject {
    /// Hydrogen orbital (n, l, m) with light.
    Hydrogen,
    /// Element Z, all electrons or one orbital.
    Element(u32, ElemPart),
    /// One of molecules::MOLECULES: all electrons (None) or one orbital (alpha set).
    Molecule(usize, Option<usize>),
    /// One of solids::SOLIDS.
    Crystal(usize),
}

/// A solved atom or molecule, kept so another of its orbitals is quick to draw.
#[derive(Clone)]
enum Solved { Elem(Arc<elements::Solved>), Mol(usize, Arc<mc::Solved>) }

enum JobMsg {
    Progress(String),
    Done(Vec<Sample>, Other, Option<Solved>),
    Failed(String),
}

/// Work done off the window's thread (solving and drawing dots can take seconds).
struct Job { what: Subject, rx: Receiver<JobMsg> }

pub struct AtomsScene {
    pub sim: Sim,
    draw: Option<Draw>,
    ctl: OrbitCam,
    pub subject: Subject,
    job: Option<Job>,
    pub progress: String,
    pub failed: Option<String>,
    /// The last solved element (kept so picking another of its orbitals is quick).
    elem: Option<Arc<elements::Solved>>,
    /// The last solved molecule (index into MOLECULES).
    mol: Option<(usize, Arc<mc::Solved>)>,
    /// What the dots on screen are (the subject can change before the thread is done).
    shown: Subject,
    /// The element slider's value while it is being dragged.
    z_pick: u32,
    /// The m slider's value while it is being dragged.
    m_pick: Option<i32>,
}

impl AtomsScene {
    pub fn new() -> AtomsScene {
        let sim = Sim::new(2, 1, 0, START_COUNT);
        let ctl = OrbitCam::new(&sim.cam, DVec3::ZERO, DVec3::Y, 1.0, 2000.0);
        let mut a = AtomsScene { sim, draw: None, ctl, subject: Subject::Hydrogen, job: None, progress: String::new(), failed: None, elem: None, mol: None, shown: Subject::Hydrogen, z_pick: 0, m_pick: None };
        a.home();
        a
    }

    /// Show something else. Hydrogen switches at once; the rest are worked out on a thread and
    /// slide in when ready (the old dots stay until then).
    pub fn choose(&mut self, what: Subject) {
        self.subject = what;
        self.failed = None;
        if what == Subject::Hydrogen {
            self.job = None;
            self.progress.clear();
            self.sim.show_hydrogen();
            self.shown = Subject::Hydrogen;
            self.frame_view();
            return;
        }
        let (tx, rx) = mpsc::channel();
        let count = self.sim.count;
        let cache = match what {
            Subject::Element(..) => self.elem.clone().map(Solved::Elem),
            Subject::Molecule(..) => self.mol.clone().map(|(i, m)| Solved::Mol(i, m)),
            _ => None,
        };
        self.progress = "working ...".into();
        std::thread::spawn(move || {
            // Two cores are left to the window.
            let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1);
            let run = || make_cloud(what, count, cache, &tx);
            // a solver that panics must still answer
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
                Ok(p) => p.install(run), Err(_) => run() }))
                .unwrap_or_else(|_| Err("the solver stopped with an error".into()));
            let _ = tx.send(match r { Ok((pts, o, e)) => JobMsg::Done(pts, o, e), Err(e) => JobMsg::Failed(e) });
        });
        self.job = Some(Job { what, rx });
    }

    /// Take in what the thread sent.
    fn poll(&mut self) {
        let Some(job) = &self.job else { return };
        let mut done = None;
        let mut fail = None;
        loop {
            match job.rx.try_recv() {
                Ok(JobMsg::Progress(t)) => self.progress = t,
                Ok(JobMsg::Done(p, o, e)) => { done = Some((p, o, e)); break; }
                Ok(JobMsg::Failed(e)) => { fail = Some(e); break; }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => { fail = Some("the solver stopped without an answer".to_string()); break; }
            }
        }
        if let Some(e) = fail {
            // Back to what is on screen, so the panel matches the dots.
            self.failed = Some(e);
            self.progress.clear();
            self.job = None;
            self.subject = self.shown;
            return;
        }
        if let Some((pts, o, e)) = done {
            if job.what == self.subject {
                match e { Some(Solved::Elem(a)) => self.elem = Some(a), Some(Solved::Mol(i, m)) => self.mol = Some((i, m)), None => {} }
                // A new thing (not another orbital of the same one) is looked at from a slant, as
                // hydrogen's home view: straight from the side, a ring or a flat molecule shows only its edge.
                let new_thing = std::mem::discriminant(&self.shown) != std::mem::discriminant(&job.what) || match (self.shown, job.what) {
                    (Subject::Element(a, _), Subject::Element(b, _)) => a != b,
                    (Subject::Molecule(a, _), Subject::Molecule(b, _)) => a != b,
                    (Subject::Crystal(a), Subject::Crystal(b)) => a != b,
                    _ => false,
                };
                self.shown = job.what;
                self.sim.show_other(pts, o);
                // Framed (and looked at from a slant) only for a new thing; another orbital or dot
                // count of the same thing keeps the user's zoom.
                if new_thing { self.frame_view(); self.ctl.look_from(DVec3::new(0.0, 0.35, 1.0).normalize()); } else { self.set_range(); }
            }
            self.progress.clear();
            self.job = None;
        }
    }

    /// Camera range and distance to fit what is shown.
    fn frame_view(&mut self) {
        self.set_range();
        self.ctl.go_to(DVec3::ZERO, self.sim.dist_target);
    }
    /// How near and far the camera may go, for the size of what is shown.
    fn set_range(&mut self) {
        let d = self.sim.dist_target;
        self.ctl.min_dist = (0.02 * d).min(1.0);
        self.ctl.max_dist = (50.0 * d).max(2000.0);
    }
}

/// Radius holding 95 percent of the dots.
fn r95(pts: &[Sample]) -> f64 {
    let mut r: Vec<f32> = pts.iter().map(|p| p[0]).collect();
    if r.is_empty() { return 1.0; }
    let k = ((0.95 * r.len() as f64) as usize).min(r.len() - 1);
    let (_, v, _) = r.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *v as f64
}

/// Dot size in proportion to the cloud: 0.05 a0 for the 3s cloud, which holds 95 percent of its
/// dots inside 21.4 a0.
fn dot_size(r95: f64) -> f64 { 0.05 * r95 / 21.4 }

/// Solve and draw the dots for `what` (runs on a thread).
fn make_cloud(what: Subject, count: usize, cache: Option<Solved>, tx: &Sender<JobMsg>)
    -> Result<(Vec<Sample>, Other, Option<Solved>), String> {
    let say = |t: String| { let _ = tx.send(JobMsg::Progress(t)); };
    let seed = 0x5EED_u64;
    match what {
        Subject::Hydrogen => Err("hydrogen is drawn on the window's thread".into()),
        Subject::Element(z, part) => {
            let sym = elements::SYMBOLS[(z - 1) as usize];
            let name = elements::NAMES[(z - 1) as usize];
            let s = match cache {
                Some(Solved::Elem(e)) if e.z == z => e,
                _ => {
                    say(format!("solving {name} (all {z} electrons) ..."));
                    Arc::new(elements::solve(z).ok_or_else(|| format!("{name}: the solver did not settle"))?)
                }
            };
            if let ElemPart::Orbital(k, _) = part { if k >= s.atom.shells.len() { return Err(format!("{name} has {} orbitals", s.atom.shells.len())); } }
            say(format!("drawing {count} dots ..."));
            let pts = clouds::element(&s, part, count, seed ^ z as u64);
            let r = r95(&pts);
            let cfg = elements::config_text(&elements::config(z));
            let (title, m, mut notes) = match part {
                ElemPart::All => (format!("{name} ({sym}, Z = {z}): all {z} electrons"), 0, vec![
                    "Every electron's cloud added up. The solver uses the average over all directions \
                     (one radial shape per shell), so the dots have no preferred direction. Pick one \
                     orbital under Part to see its shape.".to_string()]),
                ElemPart::Orbital(k, m) => {
                    let o = &s.atom.shells[k];
                    let m = m.clamp(-(o.l as i32), o.l as i32);
                    (format!("{name} ({sym}): one {} electron, m = {m}", crate::dft::shell_name(o)), m, vec![
                        "That orbital's radial shape from the solver, times the angular shape of m \
                         (as hydrogen's). The dots drift along its current when m is not 0.".to_string()])
                }
            };
            notes.push(format!("Configuration {cfg}. Scalar-relativistic LDA, the method of NIST SRD 141."));
            let (kind, lines) = elements::lines(z);
            let mut spec: Vec<(f64, f64)> = lines.iter()
                .map(|l| (l.nm, if kind == elements::LineData::WithRates { l.a } else { l.intensity })).filter(|x| x.1 > 0.0).collect();
            spec.sort_by(|a, b| b.1.total_cmp(&a.1));
            spec.truncate(16);
            notes.push(if spec.is_empty() { "Spectrum strip: NIST ASD has no 200-1000 nm lines for this atom.".into() }
                       else { format!("Spectrum strip: its {} strongest lines, 200-1000 nm (NIST ASD).", spec.len()) });
            Ok((pts, Other { title, notes, dot_r: dot_size(r), m, r95: r, spec }, Some(Solved::Elem(s))))
        }
        Subject::Molecule(i, orb) => {
            let info = &mc::MOLECULES[i];
            let (sol, steps) = match cache {
                Some(Solved::Mol(j, m)) if j == i => (m, None),
                _ => {
                    let start = mc::rough_shape(info.key).ok_or("no such molecule")?;
                    say(format!("finding the shape of {} (lowest energy) ...", info.name));
                    let (shape, _, steps) = mc::optimise(&start)?;
                    say("solving its electrons ...".into());
                    (Arc::new(mc::solve(&shape)?), Some(steps))
                }
            };
            if orb.is_some_and(|k| k >= sol.n_orbitals) { return Err(format!("{} has {} orbitals", info.name, sol.n_orbitals)); }
            let (centre, half) = mc::cube_for(&sol.mol, 5.0);
            say(format!("values on {GRID_N}^3 points ..."));
            let cube = clouds::cube_points(GRID_N, centre, half);
            let vals: Vec<f64> = match orb {
                None => mc::density_values(&sol, &cube),
                Some(k) => mc::orbital_values(&sol, k, &cube).into_iter().map(|v| v * v).collect(),
            };
            let h = 2.0 * half / (GRID_N - 1) as f64;
            let on_grid: f64 = vals.iter().sum::<f64>() * h * h * h;
            say(format!("drawing {count} dots ..."));
            let pts = clouds::from_grid(&vals, GRID_N, half, count, seed ^ i as u64 ^ (orb.unwrap_or(999) as u64) << 8);
            let r = r95(&pts);
            let ne = 2 * sol.n_occ;
            let shape = match steps { Some(n) => format!("Shape found by lowest energy in {n} steps; "), None => String::new() };
            let mut notes = vec![format!("Hartree-Fock, STO-3G basis. {shape}energy {:.6} hartree.", sol.energy)];
            let title = match orb {
                None => {
                    notes.push(format!("Electrons counted on the grid: {on_grid:.3} of {ne}."));
                    format!("{} ({}): all {ne} electrons", info.name, info.key)
                }
                Some(k) => {
                    notes.push(format!("One orbital: dots where psi^2 is large (the sign of psi is not shown). Energy {:.4} hartree. Counted on the grid: {on_grid:.3} of 1.",
                        sol.orbital_energies[k]));
                    format!("{} ({}): {}", info.name, info.key, mol_orbital_name(&sol, k))
                }
            };
            notes.push("The dots do not drift: these states carry no current.".into());
            Ok((pts, Other { title, notes, dot_r: dot_size(r), m: 0, r95: r, spec: Vec::new() }, Some(Solved::Mol(i, sol))))
        }
        Subject::Crystal(i) => {
            let info = &so::SOLIDS[i];
            let a = info.a0;
            say(format!("solving {} (periodic, {k}x{k}x{k} k-points; can take a minute) ...", info.name, k = so::PIC_K));
            let c = so::crystal(info, a, so::PIC_K, so::PIC_ECUT)?;
            let ab = a * so::BOHR_PER_ANGSTROM;
            let half = so::CELLS * ab / 2.0;
            say(format!("density on {GRID_N}^3 points ..."));
            // Points spaced a whole number per cell (the block is GRID_N spacings wide, the points
            // sit in the middle of each), so every cell of the crystal is sampled the same way.
            let hg = half * (GRID_N - 1) as f64 / GRID_N as f64;
            let pts3 = clouds::cube_points(GRID_N, [half, half, half], hg);
            let vals: Vec<f64> = pts3.par_iter().map(|&p| c.density_at(p)).collect();
            say(format!("drawing {count} dots ..."));
            let pts = clouds::from_grid(&vals, GRID_N, hg, count, seed ^ (100 + i) as u64);
            let r = r95(&pts);
            let mut notes = notes_crystal(info, a, &c);
            if info.key == "Si" {
                notes.push("The bright spots are the bonds: the valence density is highest halfway between bonded \
                    atoms (0.075 electrons per Bohr^3 there, 0.0001 at an atom's centre; test clouds::crystal_check).".into());
            }
            Ok((pts, Other {
                title: format!("{} crystal: {} x {} x {} cubic cells", info.name, so::CELLS, so::CELLS, so::CELLS),
                notes,
                dot_r: dot_size(r), m: 0, r95: r, spec: Vec::new() }, None))
        }
    }
}

fn notes_crystal(info: &so::SolidInfo, a: f64, c: &so::Crystal) -> Vec<String> {
    vec![
        "Valence electrons only: the pseudopotential stands in for the inner electrons, so the \
         atoms' cores have no dots.".into(),
        format!("Periodic LDA (GTH pseudopotentials, {} basis), lattice constant {a:.4} A, {:.3} electrons per cell counted ({} expected).",
            info.basis, c.n_elec, so::valence(info)),
    ]
}

/// "orbital 5 (HOMO, filled)": counted from 1, lowest energy first.
fn mol_orbital_name(s: &mc::Solved, k: usize) -> String {
    let tag = if k + 1 == s.n_occ { " (HOMO: highest filled)" } else if k == s.n_occ { " (LUMO: lowest empty)" } else if k < s.n_occ { " (filled)" } else { " (empty)" };
    format!("orbital {}{tag}", k + 1)
}

/// Points a side of the cube the density is worked out on (molecules and crystals).
const GRID_N: usize = 112;
impl Default for AtomsScene { fn default() -> Self { Self::new() } }

fn orbital_name(n: i32, l: i32) -> String { format!("{n}{}", lname(l)) }

impl Scene for AtomsScene {
    fn name(&self) -> &'static str { "Atoms" }
    fn help(&self) -> &'static str {
        "A hydrogen atom: each dot is one place the electron could be found (the cloud is the \
         probability, sampled millions of times). The dots drift along the probability current.\n\
         Light: pick a photon colour and fire it. If the colour matches an allowed jump up, the \
         electron takes it; later it falls back by itself and gives off light (the spectrum bar).\n\
         Keys: W/S n up/down, E/D l, R/F m, T/G more/fewer dots, C cut-away, H grid, P pause flow, \
         Z/X photon colour, Space fire, [ / ] time slower/faster.\n\
         Show: an element (1-118), a molecule or a crystal instead, drawn the same way from its own \
         solved electron density; the whole thing or one orbital."
    }
    fn key(&mut self, k: KeyCode, down: bool) {
        if !down { return; }
        if k == KeyCode::Home { self.home(); return; }
        self.sim.key(k);
    }
    fn update(&mut self, dt: f64, input: &Input) {
        self.poll();
        let s = &mut self.sim;
        let before = s.dist_target;
        s.update_light(dt);
        if !s.paused { s.t += dt * FLOW_RATE; }
        // A jump re-frames the cloud (sets dist_target); the trackpad and mouse zoom change it.
        if s.dist_target != before { self.ctl.go_to(DVec3::ZERO, s.dist_target); }
        // The letter keys belong to the atom (n, l, m, photons): only the arrows and +/- move
        // the camera here.
        let mut inp = input.clone();
        inp.held.retain(|k| !matches!(k, KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD));
        self.ctl.apply(&mut s.cam, &inp, dt);
        s.dist_target = self.ctl.want_dist;
    }
    fn ui(&mut self, ui: &mut egui::Ui) {
        if self.subject_ui(ui) { return; }
        let s = &mut self.sim;
        ui.label(format!("Orbital: {}  (m = {})", orbital_name(s.n, s.l), s.m));
        let (mut n, mut l, mut m) = (s.n, s.l, s.m);
        ui.add(egui::Slider::new(&mut n, 1..=12).text("n (shell, size)"));
        ui.add(egui::Slider::new(&mut l, 0..=(n - 1).max(0)).text("l (shape)"));
        ui.add(egui::Slider::new(&mut m, -l..=l).text("m (twist)"));
        if (n, l, m) != (s.n, s.l, s.m) {
            let (n2, l2, m2) = clamp_qn(n, l, m);
            (s.n, s.l, s.m) = (n2, l2, m2);
            s.resample();
            s.schedule_fall();
            s.dist_target = 4.0 * s.cloud() + 10.0;
            self.ctl.go_to(DVec3::ZERO, s.dist_target);
            if s.tune != 0 && s.tune <= s.n { s.tune = (s.n + 1).min(N_MAX); }
        }
        let mut lg = (s.count as f64).log2().round() as i32;
        if ui.add(egui::Slider::new(&mut lg, 10..=20).text("dots (2^x)")).changed() {
            s.count = 1usize << lg;
            s.resample();
        }
        ui.label(format!("{} dots", s.count));
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.cutaway, "Cut-away (C)");
            ui.checkbox(&mut s.grid, "Grid (H)");
        });
        if ui.button(if s.paused { "Play flow (P)" } else { "Pause flow (P)" }).clicked() { s.paused = !s.paused; }
        ui.separator();
        ui.label("Light");
        let c = wavelength_rgb(s.tuned_nm());
        let col = egui::Color32::from_rgb((c[0].min(1.0) * 255.0) as u8, (c[1].min(1.0) * 255.0) as u8, (c[2].min(1.0) * 255.0) as u8);
        ui.horizontal(|ui| {
            ui.label("Photon:");
            egui::ComboBox::from_id_salt("tune").selected_text(if s.tune > s.n { format!("to n = {}", s.tune) } else { "off-tune".into() })
                .show_ui(ui, |ui| {
                    for t in (s.n + 1)..=N_MAX { ui.selectable_value(&mut s.tune, t, format!("to n = {t}  ({:.1} nm)", wavelength_nm(s.n, t))); }
                    ui.selectable_value(&mut s.tune, 0, "off-tune (no jump fits)");
                });
        });
        ui.horizontal(|ui| {
            ui.colored_label(col, "\u{25A0}");
            ui.label(format!("{:.1} nm{}", s.tuned_nm(), band(s.tuned_nm())));
        });
        if ui.button("Fire photon (Space)").clicked() { s.fire(); }
        let mut lg = s.slow.log10();
        if ui.add(egui::Slider::new(&mut lg, 6.0..=12.0).text("slow-down (10^x)")).changed() {
            let f = 10f64.powf(lg) / s.slow;
            s.slow *= f;
            if let Some(t) = s.fall_in.as_mut() { *t *= f; }
        }
        ui.label(format!("1 ns of real time = {:.2} s on screen", s.slow * 1e-9));
        if ui.button("Clear spectrum").clicked() { s.spectrum.clear(); }
        ui.separator();
        if ui.button("Reset view (Home)").clicked() { self.home(); }
    }
    fn status(&self) -> Vec<String> {
        let s = &self.sim;
        vec![match &s.other {
            Some(o) => format!("{}   {} dots", o.title, s.pts.len()),
            None => format!("orbital {} m={}   {} dots", orbital_name(s.n, s.l), s.m, s.count),
        }]
    }
    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, frame: &mut Frame, screen: (u32, u32), _fps: f64) {
        let d = self.draw.get_or_insert_with(|| Draw::new(device));
        let s = &mut self.sim;
        if s.dirty { d.set_points(device, &s.pts); s.dirty = false; }
        frame.set_size(device, screen);
        frame.write_post(queue, &Post { v: [1.0, 0.25, 0.3, 1.0] });
        let x = s.extras();
        let (dot_r, m) = s.look();
        d.encode(queue, enc, frame, &s.cam, dot_r, m, s.t, s.cutaway, s.grid, screen.0 as f64 / screen.1.max(1) as f64, screen.1, &x, s.other.is_none());
    }
}

impl AtomsScene {
    /// The "Show" picker and, for anything but hydrogen, its whole panel. True when it drew the
    /// panel (hydrogen's own controls follow otherwise).
    fn subject_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let kind = |w: &Subject| match w { Subject::Hydrogen => 0, Subject::Element(..) => 1, Subject::Molecule(..) => 2, Subject::Crystal(_) => 3 };
        let names = ["Hydrogen orbital (with light)", "Element (1-118)", "Molecule", "Crystal"];
        let mut k = kind(&self.subject);
        egui::ComboBox::from_label("Show").selected_text(names[k]).show_ui(ui, |ui| {
            for (i, n) in names.iter().enumerate() { ui.selectable_value(&mut k, i, *n); }
        });
        if k != kind(&self.subject) {
            self.z_pick = 0;
            let z = self.elem.as_ref().map(|e| e.z).unwrap_or(6);
            self.choose(match k { 1 => Subject::Element(z, ElemPart::All), 2 => Subject::Molecule(3, None), 3 => Subject::Crystal(0), _ => Subject::Hydrogen });
        }
        if self.subject == Subject::Hydrogen { return false; }
        let mut want = self.subject;
        match &mut want {
            Subject::Element(z, part) => {
                // The atom is solved once the slider is let go (heavy ones take a moment).
                if self.z_pick == 0 { self.z_pick = *z; }
                let name = elements::NAMES[(self.z_pick - 1) as usize];
                let r = ui.add(egui::Slider::new(&mut self.z_pick, 1..=118).text(format!("Z ({name})")));
                if (r.drag_stopped() || (r.changed() && !r.dragged())) && self.z_pick != *z { *z = self.z_pick; *part = ElemPart::All; }
                // The orbitals of the solved atom (once it is solved).
                if let Some(e) = self.elem.as_ref().filter(|e| e.z == *z) {
                    let label = |p: &ElemPart| match p { ElemPart::All => "whole atom".to_string(), ElemPart::Orbital(k, _) => format!("one {} electron", crate::dft::shell_name(&e.atom.shells[*k])) };
                    egui::ComboBox::from_label("Part").selected_text(label(part)).show_ui(ui, |ui| {
                        ui.selectable_value(part, ElemPart::All, "whole atom");
                        for (k, o) in e.atom.shells.iter().enumerate() {
                            let keep_m = if let ElemPart::Orbital(_, m) = *part { m.clamp(-(o.l as i32), o.l as i32) } else { 0 };
                            ui.selectable_value(part, ElemPart::Orbital(k, keep_m), format!("one {} electron", crate::dft::shell_name(o)));
                        }
                    });
                    if let ElemPart::Orbital(k, m) = part {
                        let l = e.atom.shells[*k].l as i32;
                        if l > 0 {
                            // new dots once the slider is let go
                            let mut v = self.m_pick.unwrap_or(*m).clamp(-l, l);
                            let r = ui.add(egui::Slider::new(&mut v, -l..=l).text("m (twist)"));
                            if r.dragged() { self.m_pick = Some(v); } else { self.m_pick = None; *m = v; }
                        }
                    }
                }
            }
            Subject::Molecule(i, orb) => {
                let before = *i;
                egui::ComboBox::from_label("Molecule").selected_text(format!("{} ({})", mc::MOLECULES[*i].name, mc::MOLECULES[*i].key))
                    .show_ui(ui, |ui| { for (j, m) in mc::MOLECULES.iter().enumerate() { ui.selectable_value(i, j, format!("{} ({})", m.name, m.key)); } });
                if *i != before { *orb = None; }
                // Its orbitals once solved: all filled ones and the first few empty ones.
                if let Some((_, s)) = self.mol.as_ref().filter(|m| m.0 == *i) {
                    let label = |o: &Option<usize>| match o { None => "all electrons".to_string(), Some(k) => mol_orbital_name(s, *k) };
                    egui::ComboBox::from_label("Part").selected_text(label(orb)).show_ui(ui, |ui| {
                        ui.selectable_value(orb, None, "all electrons");
                        for k in 0..(s.n_occ + 3).min(s.n_orbitals) { ui.selectable_value(orb, Some(k), mol_orbital_name(s, k)); }
                    });
                }
            }
            Subject::Crystal(i) => {
                egui::ComboBox::from_label("Crystal").selected_text(so::SOLIDS[*i].name)
                    .show_ui(ui, |ui| { for (j, c) in so::SOLIDS.iter().enumerate() { ui.selectable_value(i, j, c.name); } });
            }
            Subject::Hydrogen => {}
        }
        if want != self.subject { self.choose(want); }
        let s = &mut self.sim;
        if !self.progress.is_empty() { ui.label(format!("\u{23F3} {}", self.progress)); }
        if let Some(e) = &self.failed { ui.colored_label(egui::Color32::from_rgb(255, 120, 100), format!("Failed: {e}")); }
        if let Some(o) = &s.other {
            ui.separator();
            ui.strong(&o.title);
            for n in &o.notes { ui.label(n); }
        }
        ui.separator();
        let mut lg = (s.count as f64).log2().round() as i32;
        let mut redo = false;
        // New dots once the slider is let go (each set of dots takes a moment to draw).
        let r = ui.add(egui::Slider::new(&mut lg, 10..=20).text("dots (2^x)"));
        if r.changed() { s.count = 1usize << lg; }
        if r.drag_stopped() || (r.changed() && !r.dragged()) { redo = true; }
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.cutaway, "Cut-away (C)");
            ui.checkbox(&mut s.grid, "Grid (H)");
        });
        if s.other.as_ref().is_some_and(|o| o.m != 0) && ui.button(if s.paused { "Play flow (P)" } else { "Pause flow (P)" }).clicked() { s.paused = !s.paused; }
        if redo { let w = self.subject; self.choose(w); }
        ui.separator();
        if ui.button("Reset view (Home)").clicked() { self.home(); }
        true
    }

    fn home(&mut self) {
        let d = match &self.sim.other { Some(o) => o.view_dist(), None => 4.0 * self.sim.cloud() + 10.0 };
        self.ctl.go_to(DVec3::ZERO, d);
        self.ctl.look_from(DVec3::new(0.0, 0.35, 1.0).normalize());
    }
}

fn band(nm: f64) -> &'static str { if nm < 380.0 { " (ultraviolet, invisible)" } else if nm > 750.0 { " (infrared, invisible)" } else { " (visible)" } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slide_starts_from_the_dots_on_screen() {
        let mut s = Sim::new(3, 2, 0, 1 << 12);
        let near = |a: (f64, i32), b: (f64, i32)| (a.0 - b.0).abs() < 1e-12 && a.1 == b.1;
        assert!(near(s.look(), (0.05, 0)));
        s.key(KeyCode::KeyR);
        s.key(KeyCode::KeyR);
        s.wait_points();
        assert!(near(s.old_look, (0.05, 0)), "slide must start from m = 0, the dots that were shown: {:?}", s.old_look);
        assert!(near(s.look(), (0.05, 2)));
    }

    #[test]
    fn atoms_shader_validates() {
        let src = include_str!("../atoms.wgsl");
        let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m).unwrap_or_else(|e| panic!("{e:?}"));
    }

    #[test]
    fn tuning_keys_go_round_both_ways() {
        let mut s = Sim::new(2, 1, 0, 1 << 10);
        let mut up = vec![s.tune];
        for _ in 0..(N_MAX - s.n + 1) { s.key(KeyCode::KeyX); up.push(s.tune); }
        assert_eq!(up.first(), up.last(), "X goes round: {up:?}");
        let mut down = vec![s.tune];
        for _ in 0..(N_MAX - s.n + 1) { s.key(KeyCode::KeyZ); down.push(s.tune); }
        down.reverse();
        assert_eq!(up, down, "Z goes round the same ring backwards");
        assert!(up.iter().all(|&t| t == 0 || (t > s.n && t <= N_MAX)));
    }

    #[test]
    fn a_lost_solver_is_reported() {
        let mut a = AtomsScene::new();
        let (tx, rx) = mpsc::channel::<JobMsg>();
        drop(tx);
        a.subject = Subject::Crystal(0);
        a.job = Some(Job { what: Subject::Crystal(0), rx });
        a.poll();
        assert!(a.job.is_none() && a.failed.is_some());
        assert_eq!(a.subject, Subject::Hydrogen);
    }

    #[test]
    fn switching_what_is_shown() {
        let mut a = AtomsScene::new();
        a.sim.count = 1 << 14;
        let wait = |a: &mut AtomsScene| {
            for _ in 0..600 { a.poll(); if a.job.is_none() { return; } std::thread::sleep(std::time::Duration::from_millis(50)); }
            panic!("job did not finish");
        };
        a.choose(Subject::Element(26, ElemPart::All));
        wait(&mut a);
        assert!(a.failed.is_none(), "{:?}", a.failed);
        assert!(a.sim.other.as_ref().unwrap().title.starts_with("iron"));
        assert_eq!(a.sim.pts.len(), 1 << 14);
        let k = a.elem.as_ref().unwrap().atom.shells.iter().position(|o| o.n == 3 && o.l == 2).unwrap();
        a.choose(Subject::Element(26, ElemPart::Orbital(k, 2)));
        wait(&mut a);
        assert_eq!(a.sim.look().1, 2);
        a.choose(Subject::Molecule(3, Some(4)));
        wait(&mut a);
        let o = a.sim.other.as_ref().unwrap();
        assert!(o.title.contains("HOMO"), "{}", o.title);
        a.sim.key(KeyCode::KeyW);
        assert!(a.sim.other.is_some());
        a.choose(Subject::Hydrogen);
        assert!(a.sim.other.is_some());
        a.sim.wait_points();
        assert!(a.sim.other.is_none());
        assert_eq!(a.sim.look(), (0.05 * a.sim.n as f64 / 3.0, a.sim.m));
    }
}
