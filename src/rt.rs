//! Renderer: path tracer (rt.wgsl) and clean-up (denoise.wgsl, rt_copy.wgsl). Objects are found
//! through a tree of boxes (BVH) built every frame.

use crate::camera::Camera;
use crate::frame::{fullscreen_pipe, storage_entry, ubo_entry, Frame};

/// Most pixels traced per frame (memory: about 140 bytes per traced pixel for all the images).
pub const MAX_TRACE_PIXELS: u32 = 1_600_000;
/// The picture scale used for the trace: `s` rounded to a power of 1.1 (at most 1). The scale
/// follows the frame rate in small steps; rounding means the trace (and the clean-up's history)
/// is made again only when it has moved by about 10%.
pub fn scale_step(s: f64) -> f64 {
    let k = (s.max(1e-3).ln() / 1.1f64.ln()).round();
    1.1f64.powf(k).min(1.0)
}

/// Most pixels in the window-size picture (the screen's last scale-up is a plain filter).
pub const MAX_OUT_PIXELS: u32 = 1_600_000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjGpu {
    pub pos: [f32; 4],   // centre relative to the camera, w = shape (0 ball, 3 big ball: pos = surface point nearest the camera)
    pub rot: [f32; 4],   // big ball: xyz = outward direction at pos (ball: unused)
    pub size: [f32; 4],  // x = radius, w = lens G M / c^2 (m)
    pub mat: [f32; 4],   // rgb, w = material (0 matte, 3 lamp, 5 black hole: rgb = spin a, inner and outer disk edge)
    pub ppos: [f32; 4],  // pos last frame, w = horizon radius in Schwarzschild radii (black hole)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtU {
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    info: [f32; 4],
    info2: [f32; 4],
    prev_pos: [f32; 4],
    prev_fwd: [f32; 4],
    prev_right: [f32; 4],
    prev_up: [f32; 4],
    jitter: [f32; 4],
    misc: [f32; 4],
    lamps: [[u32; 4]; 2],
    lens: [[u32; 4]; 4],
    up: [f32; 4],
}

#[derive(Clone, Copy)]
pub struct Opts {
    pub max_bounces: u32,
    /// Bounces along which light bending is followed (each bent piece needs a second search).
    pub lens_bounces: u32,
    /// Smallest weight of the new frame in the running blend: 0.1 while things move (old light
    /// fades fast), about 1/255 when still (the picture keeps getting cleaner).
    pub history_min: f32,
    /// Exposure shift in stops (panel slider).
    pub ev: f32,
    pub dt: f32,
    /// Seconds (black hole gas swirl).
    pub time: f32,
    /// Smallest offset for new rays, metres (1 mm normally; less when zoomed in close).
    pub min_eps: f32,
}

const COMMON: &str = include_str!("rt_common.wgsl");

/// Common code + the shared top of `src` + its section `name` (sections start with `//@@ name`).
fn section(src: &str, name: &str) -> String {
    let head = src.split("//@@").next().unwrap_or("");
    let body = src.split("//@@").find(|p| p.trim_start().starts_with(name)).unwrap_or("");
    format!("{COMMON}\n{head}\n//{body}")
}

/// Every shader module the renderer builds: (name, source).
pub fn sources() -> Vec<(&'static str, String)> {
    let dn = include_str!("denoise.wgsl");
    let cp = include_str!("rt_copy.wgsl");
    vec![
        ("trace", format!("{COMMON}\n{}\n{}", include_str!("rt.wgsl"), include_str!("rt_hole.wgsl"))),
        ("temporal", section(dn, "temporal")),
        ("atrous", section(dn, "atrous")),
        ("compose", section(dn, "compose")),
        ("up", section(dn, "up")),
        ("expo", section(cp, "expo")),
        ("copy", section(cp, "copy")),
    ]
}

pub struct Bvh { pub nodes: Vec<[f32; 4]>, pub prims: Vec<u32> }

/// Box round one object (min, max).
pub fn aabb(o: &ObjGpu) -> ([f32; 3], [f32; 3]) {
    let b = bound(o);
    let pad = 1e-4 * (1.0 + b[3]);
    ([0, 1, 2].map(|k| b[k] - b[3] - pad), [0, 1, 2].map(|k| b[k] + b[3] + pad))
}

/// Ball holding the object (centre, radius), for the shader's cheap first check.
pub fn bound(o: &ObjGpu) -> [f32; 4] {
    match o.pos[3] as i32 {
        3 => [o.pos[0] - o.size[0] * o.rot[0], o.pos[1] - o.size[0] * o.rot[1], o.pos[2] - o.size[0] * o.rot[2], o.size[0] * 1.0001],
        _ => [o.pos[0], o.pos[1], o.pos[2], o.size[0] * 1.0001],
    }
}

/// Tree: split the objects in half along the longest side of their centres' spread, until 2 or
/// fewer are left. Children of a node are stored next to each other.
pub fn build_bvh(objs: &[ObjGpu]) -> Bvh {
    let boxes: Vec<([f32; 3], [f32; 3])> = objs.iter().map(aabb).collect();
    let cen = |i: u32| { let (a, b) = boxes[i as usize]; [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])] };
    let mut prims: Vec<u32> = (0..objs.len() as u32).collect();
    let mut nodes: Vec<[[f32; 4]; 2]> = vec![[[0.0; 4]; 2]];
    if objs.is_empty() { return Bvh { nodes: vec![[0.0; 4]; 2], prims: vec![0] }; }
    let mut work = vec![(0usize, 0usize, objs.len())];
    while let Some((ni, s, e)) = work.pop() {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut clo = [f32::INFINITY; 3];
        let mut chi = [f32::NEG_INFINITY; 3];
        for &i in &prims[s..e] {
            let (a, b) = boxes[i as usize];
            let c = cen(i);
            for k in 0..3 { lo[k] = lo[k].min(a[k]); hi[k] = hi[k].max(b[k]); clo[k] = clo[k].min(c[k]); chi[k] = chi[k].max(c[k]); }
        }
        if e - s <= 2 {
            nodes[ni] = [[lo[0], lo[1], lo[2], f32::from_bits(s as u32)], [hi[0], hi[1], hi[2], f32::from_bits((e - s) as u32)]];
            continue;
        }
        let ext = [chi[0] - clo[0], chi[1] - clo[1], chi[2] - clo[2]];
        let ax = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
        let mid = (s + e) / 2;
        prims[s..e].select_nth_unstable_by(mid - s, |&a, &b| cen(a)[ax].total_cmp(&cen(b)[ax]));
        let left = nodes.len();
        nodes.push([[0.0; 4]; 2]);
        nodes.push([[0.0; 4]; 2]);
        nodes[ni] = [[lo[0], lo[1], lo[2], f32::from_bits(left as u32)], [hi[0], hi[1], hi[2], f32::from_bits(0)]];
        work.push((left, s, mid));
        work.push((left + 1, mid, e));
    }
    Bvh { nodes: nodes.into_iter().flatten().collect(), prims }
}

fn halton(mut i: u32, b: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 { f /= b as f32; r += f * (i % b) as f32; i /= b; }
    r
}

#[allow(dead_code)] // the textures are kept with their views for as long as the bind groups use them
struct Img { tex: wgpu::Texture, view: wgpu::TextureView }

#[allow(dead_code)]
struct Targets {
    size: (u32, u32),
    imgs: Vec<Img>,
    bg_trace: [wgpu::BindGroup; 2],
    bg_temporal: [wgpu::BindGroup; 2],
    bg_atrous: [[wgpu::BindGroup; 5]; 2],
    bg_compose: [wgpu::BindGroup; 2],
    bg_expo: [wgpu::BindGroup; 2],
}

/// The window-size picture (cs_up's history, two so one is read while the other is written) and
/// the bind groups that join it to the trace's images (made again when either changes size).
#[allow(dead_code)]
struct UpTargets {
    size: (u32, u32),
    trace_size: (u32, u32),
    imgs: [Img; 2],
    bg_up: [wgpu::BindGroup; 2],
    bg_copy: [wgpu::BindGroup; 2],   // copies up[p]
}

// Image slots in Targets::imgs.
const ILLUM: usize = 0;
const EMIT: usize = 1;
const ALB: usize = 2;
const GEO: usize = 3;     // 3, 4
const HIST: usize = 5;    // 5, 6
const MOM: usize = 7;     // 7, 8
const EHIST: usize = 9;   // 9, 10
const A: usize = 11;
const B: usize = 12;
const COL: usize = 13;

struct Layouts {
    trace: wgpu::BindGroupLayout,
    temporal: wgpu::BindGroupLayout,
    atrous: wgpu::BindGroupLayout,
    compose: wgpu::BindGroupLayout,
    up: wgpu::BindGroupLayout,
    expo: wgpu::BindGroupLayout,
    copy: wgpu::BindGroupLayout,
}

pub struct Renderer {
    lay: Layouts,
    p_trace: wgpu::ComputePipeline,
    p_temporal: wgpu::ComputePipeline,
    p_atrous: wgpu::ComputePipeline,
    p_compose: wgpu::ComputePipeline,
    p_up: wgpu::ComputePipeline,
    p_expo_sum: wgpu::ComputePipeline,
    p_expo_set: wgpu::ComputePipeline,
    p_copy: wgpu::RenderPipeline,
    ubuf: wgpu::Buffer,
    steps: Vec<wgpu::Buffer>,
    expo: wgpu::Buffer,
    objs: wgpu::Buffer,
    bvh: wgpu::Buffer,
    prims: wgpu::Buffer,
    bounds: wgpu::Buffer,
    caps: (usize, usize, usize),
    targets: Option<Targets>,
    up: Option<UpTargets>,
    /// Frames since the window-size picture was last started again.
    up_frames: u32,
    parity: usize,
    frame_no: u32,
    prev: Option<Camera>,
    /// Filtering sampler for the window picture's history.
    smp: wgpu::Sampler,
}

/// 1 + the index of the ball the camera is inside (0: none). Positions are camera-relative. The
/// camera's own rays look through it (a thing thrown through the camera): seen from inside, a
/// solid would fill the picture with black.
fn inside_solid(objs: &[ObjGpu]) -> f32 {
    objs.iter().position(|o| {
        o.pos[3] == 0.0 && o.mat[3] != 5.0   // a black hole: its own rules
            && o.pos[0] * o.pos[0] + o.pos[1] * o.pos[1] + o.pos[2] * o.pos[2] < o.size[0] * o.size[0]
    }).map(|i| i as f32 + 1.0).unwrap_or(0.0)
}

const CS: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

fn tex_e(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
        },
        count: None,
    }
}
fn store_e(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: CS,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D2 },
        count: None,
    }
}
fn ftex_e(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: CS,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}
fn rw_e(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: CS,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn bgl(device: &wgpu::Device, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })
}
fn compute(device: &wgpu::Device, src: &str, entry: &str, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(entry), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None,
    })
}

const F16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const F32: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const U8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const OBJ_BYTES: usize = std::mem::size_of::<ObjGpu>();

impl Renderer {
    pub fn new(device: &wgpu::Device) -> Renderer {
        let src: std::collections::HashMap<&str, String> = sources().into_iter().collect();
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let lay = Layouts {
            trace: bgl(device, &[ubo_entry(0, CS), storage_entry(1, CS), storage_entry(2, CS), storage_entry(3, CS),
                store_e(4, F16), store_e(5, F16), store_e(6, U8), store_e(7, F32), storage_entry(8, CS)]),
            temporal: bgl(device, &[ubo_entry(0, CS), storage_entry(1, CS), tex_e(2, CS), tex_e(3, CS), tex_e(4, CS), tex_e(5, CS),
                tex_e(6, CS), tex_e(7, CS), tex_e(8, CS), store_e(9, F16), store_e(10, F16), store_e(11, F16)]),
            atrous: bgl(device, &[ubo_entry(0, CS), ubo_entry(1, CS), tex_e(2, CS), tex_e(3, CS), store_e(4, F16)]),
            compose: bgl(device, &[ubo_entry(0, CS), tex_e(1, CS), tex_e(2, CS), tex_e(3, CS), store_e(4, F16)]),
            up: bgl(device, &[ubo_entry(0, CS), storage_entry(1, CS), tex_e(2, CS), tex_e(3, CS), ftex_e(4), store_e(5, F16),
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: CS, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }]),
            expo: bgl(device, &[ubo_entry(0, CS), rw_e(1), tex_e(2, CS), tex_e(3, CS), tex_e(4, CS)]),
            copy: bgl(device, &[ubo_entry(0, vf), storage_entry(1, wgpu::ShaderStages::FRAGMENT), tex_e(2, wgpu::ShaderStages::FRAGMENT)]),
        };
        let p_trace = compute(device, &src["trace"], "cs_trace", &lay.trace);
        let p_temporal = compute(device, &src["temporal"], "cs_temporal", &lay.temporal);
        let p_atrous = compute(device, &src["atrous"], "cs_atrous", &lay.atrous);
        let p_compose = compute(device, &src["compose"], "cs_compose", &lay.compose);
        let p_up = compute(device, &src["up"], "cs_up", &lay.up);
        let p_expo_sum = compute(device, &src["expo"], "cs_expo_sum", &lay.expo);
        let p_expo_set = compute(device, &src["expo"], "cs_expo_set", &lay.expo);
        let copy_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("copy"), source: wgpu::ShaderSource::Wgsl(src["copy"].clone().into()) });
        let p_copy = fullscreen_pipe(device, &copy_mod, &lay.copy, "fs_copy", crate::frame::HDR, None);
        use wgpu::util::DeviceExt;
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt u"), size: std::mem::size_of::<RtU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let steps = (0..5).map(|i| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step"), contents: bytemuck::cast_slice(&[(1u32 << i) as f32, 0.0, 0.0, 0.0]), usage: wgpu::BufferUsages::UNIFORM,
        })).collect();
        let expo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("expo"), contents: bytemuck::cast_slice(&[0u32; 4 + 128]), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let smp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("window picture"), address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge, mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        let (objs, bvh, prims, bounds) = (Self::sbuf(device, 64 * OBJ_BYTES), Self::sbuf(device, 128 * 32), Self::sbuf(device, 64 * 4), Self::sbuf(device, 64 * 16));
        Renderer {
            lay, p_trace, p_temporal, p_atrous, p_compose, p_up, p_expo_sum, p_expo_set, p_copy, ubuf, steps, expo,
            objs, bvh, prims, bounds, caps: (64 * OBJ_BYTES, 128 * 32, 64 * 4), targets: None, up: None, up_frames: 0, parity: 0, frame_no: 0, prev: None,
            smp,
        }
    }

    fn sbuf(device: &wgpu::Device, bytes: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: bytes.max(16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        })
    }

    /// Biggest trace size with the window's shape, `scale` of the window's pixels, under the cap.
    pub fn trace_size(view: (u32, u32), scale: f64) -> (u32, u32) {
        let (w, h) = (view.0.max(16) as f64, view.1.max(16) as f64);
        let s = scale.min((MAX_TRACE_PIXELS as f64 / (w * h)).sqrt());
        (((w * s).round() as u32).max(16), ((h * s).round() as u32).max(16))
    }

    fn make_targets(&self, device: &wgpu::Device, size: (u32, u32)) -> Targets {
        let img = |fmt| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: None, size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: fmt,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            Img { tex, view }
        };
        let fmts = [F16, F16, U8, F32, F32, F16, F16, F16, F16, F16, F16, F16, F16, F16];
        let imgs: Vec<Img> = fmts.iter().map(|&f| img(f)).collect();
        let v = |i: usize| wgpu::BindingResource::TextureView(&imgs[i].view);
        let bg = |layout: &wgpu::BindGroupLayout, res: Vec<wgpu::BindingResource>| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout,
            entries: &res.into_iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r }).collect::<Vec<_>>(),
        });
        let ub = || self.ubuf.as_entire_binding();
        let per = |p: usize| {
            let q = 1 - p;
            let trace = bg(&self.lay.trace, vec![ub(), self.objs.as_entire_binding(), self.bvh.as_entire_binding(), self.prims.as_entire_binding(),
                v(ILLUM), v(EMIT), v(ALB), v(GEO + p), self.bounds.as_entire_binding()]);
            let temporal = bg(&self.lay.temporal, vec![ub(), self.objs.as_entire_binding(), v(ILLUM), v(EMIT), v(GEO + p), v(GEO + q),
                v(HIST + q), v(MOM + q), v(EHIST + q), v(A), v(MOM + p), v(EHIST + p)]);
            // Blur passes: A -> hist[p] (kept as next frame's history) -> B -> A -> B -> A.
            let chain = [(A, HIST + p), (HIST + p, B), (B, A), (A, B), (B, A)];
            let atrous: [wgpu::BindGroup; 5] = std::array::from_fn(|i| bg(&self.lay.atrous,
                vec![ub(), self.steps[i].as_entire_binding(), v(GEO + p), v(chain[i].0), v(chain[i].1)]));
            let compose = bg(&self.lay.compose, vec![ub(), v(ALB), v(A), v(EHIST + p), v(COL)]);
            let expo = bg(&self.lay.expo, vec![ub(), self.expo.as_entire_binding(), v(COL), v(GEO + p), v(EMIT)]);
            (trace, temporal, atrous, compose, expo)
        };
        let (t0, m0, a0, c0, e0) = per(0);
        let (t1, m1, a1, c1, e1) = per(1);
        Targets {
            size, bg_trace: [t0, t1], bg_temporal: [m0, m1], bg_atrous: [a0, a1], bg_compose: [c0, c1],
            bg_expo: [e0, e1], imgs,
        }
    }

    /// The window-size pictures (`size`), joined to the trace's images (made when the window or the
    /// trace changes size; the pictures are kept when only the trace changed, `keep`).
    fn make_up(&self, device: &wgpu::Device, size: (u32, u32), keep: Option<UpTargets>) -> UpTargets {
        let t = self.targets.as_ref().expect("targets before window pictures");
        let imgs = match keep {
            Some(k) if k.size == size => k.imgs,
            _ => std::array::from_fn(|_| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("window picture"), size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: F16,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                Img { tex, view }
            }),
        };
        fn tv(i: &Img) -> wgpu::BindingResource<'_> { wgpu::BindingResource::TextureView(&i.view) }
        let lv = |i: usize| wgpu::BindingResource::TextureView(&t.imgs[i].view);
        let bg = |layout: &wgpu::BindGroupLayout, res: Vec<wgpu::BindingResource>| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout,
            entries: &res.into_iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r }).collect::<Vec<_>>(),
        });
        let bg_up = std::array::from_fn(|p| bg(&self.lay.up, vec![self.ubuf.as_entire_binding(), self.objs.as_entire_binding(),
            lv(GEO + p), lv(COL), tv(&imgs[1 - p]), tv(&imgs[p]), wgpu::BindingResource::Sampler(&self.smp)]));
        let bg_copy = std::array::from_fn(|p| bg(&self.lay.copy, vec![self.ubuf.as_entire_binding(), self.expo.as_entire_binding(), tv(&imgs[p])]));
        UpTargets { size, trace_size: t.size, imgs, bg_up, bg_copy }
    }

    /// Window-size picture for a window of `view` pixels (the window itself, under a cap).
    pub fn out_size(view: (u32, u32)) -> (u32, u32) {
        let (w, h) = (view.0.max(16) as f64, view.1.max(16) as f64);
        let s = (MAX_OUT_PIXELS as f64 / (w * h)).sqrt().min(1.0);
        (((w * s).round() as u32).max(16), ((h * s).round() as u32).max(16))
    }

    /// Trace and clean one frame of `objs` (positions relative to `cam.pos`) seen by `cam` at `size` pixels, build the picture at
    /// `out` pixels from it (the window size) into `frame`, then run frame.rs's glow and
    /// tone map (the caller writes frame.rs's Post: weight 1 for a live view). `lamps`: indices of lamp balls (lit directly, up to 8).
    /// `lens`: indices of heavy objects that bend light (up to 16).
    #[allow(clippy::too_many_arguments)]
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, frame: &mut Frame,
                  cam: &Camera, objs: &[ObjGpu], lamps: &[u32], lens: &[u32], size: (u32, u32), out: (u32, u32), o: &Opts) {
        // Buffers big enough (bind groups point at them, so they're rebuilt after a resize).
        let bvh = build_bvh(objs);
        let need = (objs.len().max(1) * OBJ_BYTES, bvh.nodes.len() * 16, bvh.prims.len().max(1) * 4);
        let mut rebuild = self.targets.as_ref().is_none_or(|t| t.size != size);
        if need.0 > self.caps.0 {
            self.caps.0 = need.0.next_power_of_two();
            self.objs = Self::sbuf(device, self.caps.0);
            self.bounds = Self::sbuf(device, (self.caps.0 / OBJ_BYTES + 1) * 16);   // 16 bytes per object
            rebuild = true;
        }
        if need.1 > self.caps.1 { self.caps.1 = need.1.next_power_of_two(); self.bvh = Self::sbuf(device, self.caps.1); rebuild = true; }
        if need.2 > self.caps.2 { self.caps.2 = need.2.next_power_of_two(); self.prims = Self::sbuf(device, self.caps.2); rebuild = true; }
        // A new trace size restarts the clean-up's history; the window picture keeps its own
        // unless the window changed size.
        let mut trace_fresh = self.prev.is_none();
        if rebuild {
            if self.targets.as_ref().is_none_or(|t| t.size != size) { trace_fresh = true; }
            self.targets = Some(self.make_targets(device, size));
        }
        let mut up_fresh = self.prev.is_none();
        if rebuild || self.up.as_ref().is_none_or(|u| u.size != out || u.trace_size != size) {
            let keep = self.up.take();
            if keep.as_ref().is_none_or(|k| k.size != out) { up_fresh = true; }
            self.up = Some(self.make_up(device, out, keep));
        }
        if up_fresh { self.up_frames = 0; }
        frame.set_size(device, out);
        if !objs.is_empty() {
            queue.write_buffer(&self.objs, 0, bytemuck::cast_slice(objs));
            let b: Vec<[f32; 4]> = objs.iter().map(bound).collect();
            queue.write_buffer(&self.bounds, 0, bytemuck::cast_slice(&b));
        }
        queue.write_buffer(&self.bvh, 0, bytemuck::cast_slice(&bvh.nodes));
        queue.write_buffer(&self.prims, 0, bytemuck::cast_slice(&bvh.prims));

        self.frame_no = self.frame_no.wrapping_add(1);
        let (f, r, up) = cam.axes();
        let aspect = size.0 as f64 / size.1 as f64;
        let tanh = (0.5 * cam.fov_y).tan();
        let v4 = |v: glam::DVec3, w: f32| [v.x as f32, v.y as f32, v.z as f32, w];
        let pc = self.prev.clone().unwrap_or_else(|| cam.clone());
        let pok = if trace_fresh { 0.0 } else { 1.0 };
        let (pf, pr, pu) = pc.axes();
        let k = (self.frame_no % 16) + 1;
        let jit = [halton(k, 2) - 0.5, halton(k, 3) - 0.5];
        let mut lamps_a = [[0u32; 4]; 2];
        for (i, &l) in lamps.iter().take(8).enumerate() { lamps_a[i / 4][i % 4] = l; }
        let mut lens_a = [[0u32; 4]; 4];
        for (i, &l) in lens.iter().take(16).enumerate() { lens_a[i / 4][i % 4] = l; }
        let u = RtU {
            cam_pos: [0.0, 0.0, 0.0, tanh as f32],
            cam_fwd: v4(f, aspect as f32),
            cam_right: v4(r, (self.frame_no % 65536) as f32),
            cam_up: v4(up, objs.len() as f32),
            info: [size.0 as f32, size.1 as f32, 0.0, lamps.len().min(8) as f32],
            info2: [o.max_bounces as f32, lens.len().min(16) as f32, o.history_min, o.ev],
            prev_pos: v4(pc.pos - cam.pos, 0.0),
            prev_fwd: v4(pf, 0.0),
            prev_right: v4(pr, pok),
            prev_up: v4(pu, o.dt),
            jitter: [jit[0], jit[1], if objs.is_empty() { 0.0 } else { (bvh.nodes.len() / 2) as f32 }, o.lens_bounces as f32],
            misc: [o.time, o.min_eps, 0.0, inside_solid(objs)],
            lamps: lamps_a,
            lens: lens_a,
            up: [out.0 as f32, out.1 as f32, if up_fresh { 0.0 } else { 1.0 }, self.up_frames.min(100_000) as f32],
        };
        self.up_frames = self.up_frames.saturating_add(1);
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));

        let p = self.parity;
        let groups = (size.0.div_ceil(8), size.1.div_ceil(8));
        let t = self.targets.as_ref().expect("targets");
        let ut = self.up.as_ref().expect("window pictures");
        let mut pass = |steps: &[(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32))]| {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for (pipe, bg, g) in steps {
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, *bg, &[]);
                cp.dispatch_workgroups(g.0, g.1, 1);
            }
        };
        pass(&[(&self.p_trace, &t.bg_trace[p], groups)]);
        pass(&[(&self.p_temporal, &t.bg_temporal[p], groups)]);
        // The first blur pass also writes next frame's history.
        let blur: Vec<_> = (0..5).map(|i| (&self.p_atrous, &t.bg_atrous[p][i], groups)).collect();
        pass(&blur);
        pass(&[(&self.p_compose, &t.bg_compose[p], groups)]);
        pass(&[(&self.p_up, &ut.bg_up[p], (out.0.div_ceil(8), out.1.div_ceil(8))), (&self.p_expo_sum, &t.bg_expo[p], (size.0.div_ceil(16), size.1.div_ceil(16))),
               (&self.p_expo_set, &t.bg_expo[p], (1, 1))]);
        {
            let mut rp = frame.begin_final(enc);
            rp.set_pipeline(&self.p_copy);
            rp.set_bind_group(0, &ut.bg_copy[p], &[]);
            rp.draw(0..3, 0..1);
        }
        frame.encode_post(enc);
        self.parity = 1 - p;
        self.prev = Some(cam.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray_box(o: [f32; 3], d: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> bool {
        let (mut tn, mut tf) = (0.0f32, f32::INFINITY);
        for k in 0..3 {
            let inv = 1.0 / if d[k].abs() < 1e-9 { 1e-9 } else { d[k] };
            let (a, b) = ((lo[k] - o[k]) * inv, (hi[k] - o[k]) * inv);
            tn = tn.max(a.min(b));
            tf = tf.min(a.max(b));
        }
        tn <= tf
    }

    #[test]
    fn tree_finds_every_box_a_ray_crosses() {
        let mut s = 99u64;
        let mut r = move || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 40) as f32 / (1u64 << 24) as f32 };
        let mut objs = Vec::new();
        for i in 0..500 {
            let n = { let v = [r() - 0.5, r() - 0.5, r() - 0.5]; let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); v.map(|x| x / l) };
            objs.push(ObjGpu { pos: [r() * 40.0 - 20.0, r() * 10.0, r() * 40.0 - 20.0, if i % 2 == 0 { 0.0 } else { 3.0 }], rot: [n[0], n[1], n[2], 0.0],
                size: [0.2 + r(), 0.0, 0.0, 0.0], ..Default::default() });
        }
        for o in &objs {
            let (lo, hi) = aabb(o);
            let c = if o.pos[3] == 3.0 { [0, 1, 2].map(|k| o.pos[k] - o.size[0] * o.rot[k]) } else { [o.pos[0], o.pos[1], o.pos[2]] };
            for k in 0..3 { for sg in [-1.0, 1.0] {
                let w = c[k] + sg * o.size[0];
                assert!(w >= lo[k] && w <= hi[k], "ball point outside its box");
            } }
        }
        let t = build_bvh(&objs);
        for _ in 0..300 {
            let o = [r() * 60.0 - 30.0, r() * 20.0 - 5.0, r() * 60.0 - 30.0];
            let d = { let v = [r() - 0.5, r() - 0.5, r() - 0.5]; let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); v.map(|x| x / n) };
            let mut want: Vec<u32> = (0..objs.len() as u32).filter(|&i| { let (a, b) = aabb(&objs[i as usize]); ray_box(o, d, a, b) }).collect();
            let mut got = Vec::new();
            let mut stack = vec![0usize];
            while let Some(n) = stack.pop() {
                let (lo, hi) = (t.nodes[2 * n], t.nodes[2 * n + 1]);
                if !ray_box(o, d, [lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]) { continue; }
                let (first, count) = (lo[3].to_bits() as usize, hi[3].to_bits() as usize);
                if count > 0 {
                    for k in 0..count {
                        let i = t.prims[first + k];
                        let (a, b) = aabb(&objs[i as usize]);
                        if ray_box(o, d, a, b) { got.push(i); }
                    }
                } else { stack.push(first); stack.push(first + 1); }
            }
            want.sort();
            got.sort();
            assert_eq!(want, got);
        }
    }

    #[test]
    fn shaders_validate() {
        for (name, src) in sources() {
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn camera_inside_a_ball_is_found() {
        let ball = |pos: [f32; 3], r: f32, mat: f32| ObjGpu { pos: [pos[0], pos[1], pos[2], 0.0], size: [r, 0.0, 0.0, 0.0], mat: [0.5, 0.5, 0.5, mat], ..Default::default() };
        assert_eq!(inside_solid(&[ball([1.0, 0.0, 0.0], 0.9, 0.0)]), 0.0);
        assert_eq!(inside_solid(&[ball([5.0, 0.0, 0.0], 1.0, 0.0), ball([0.5, 0.0, 0.0], 0.6, 0.0)]), 2.0);
        assert_eq!(inside_solid(&[ball([0.1, 0.0, 0.0], 1.0, 5.0)]), 0.0);
    }

    #[test]
    fn trace_size_keeps_shape_and_cap() {
        let (w, h) = Renderer::trace_size((2560, 1600), 1.0);
        assert!(w * h <= MAX_TRACE_PIXELS + 4000, "{w}x{h}");
        assert!(((w as f64 / h as f64) - 1.6).abs() < 0.01);
        assert_eq!(Renderer::trace_size((1000, 500), 0.5), (500, 250));
    }


    pub fn bend_steps(mut p: [f32; 3], mut d: [f32; 3], lenses: &[([f32; 3], f32)], len: f32) -> [f32; 3] {
        let mut travelled = 0.0;
        while travelled < len {
            let near = lenses.iter().map(|(c, _)| { let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt() })
                .fold(f32::INFINITY, f32::min);
            let ds = (0.02 * near).clamp(0.002, 0.5).min(len - travelled);
            let mid = [p[0] + 0.5 * ds * d[0], p[1] + 0.5 * ds * d[1], p[2] + 0.5 * ds * d[2]];
            let mut a = [0.0f32; 3];
            for (c, gm) in lenses {
                let r = [mid[0] - c[0], mid[1] - c[1], mid[2] - c[2]];
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                let k = -2.0 * gm / (r2 * r2.sqrt());
                for i in 0..3 { a[i] += k * r[i]; }
            }
            for i in 0..3 { p[i] += ds * d[i]; d[i] += ds * a[i]; }
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for x in &mut d { *x /= l; }
            travelled += ds;
        }
        d
    }

    pub fn bend_thin(p: [f32; 3], d: [f32; 3], lenses: &[([f32; 3], f32)], len: f32) -> [f32; 3] {
        let mut kick = [0.0f32; 3];
        for (c, gm) in lenses {
            let tc = (0..3).map(|i| (c[i] - p[i]) * d[i]).sum::<f32>();
            let bv: Vec<f32> = (0..3).map(|i| c[i] - (p[i] + tc * d[i])).collect();
            let b = (bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2]).sqrt();
            if b < 1e-4 { continue; }
            let te = len - tc;
            let f = 0.5 * (te / (te * te + b * b).sqrt() + tc / (tc * tc + b * b).sqrt());
            let ang = 4.0 * gm / b * f * (1.0 + 15.0 * std::f32::consts::PI / 16.0 * gm / b * f);
            for i in 0..3 { kick[i] += ang * bv[i] / b; }
        }
        let k = (kick[0] * kick[0] + kick[1] * kick[1] + kick[2] * kick[2]).sqrt();
        if k < 1e-4 { return d; }
        let mut out = [0.0f32; 3];
        for i in 0..3 { out[i] = k.cos() * d[i] + k.sin() * kick[i] / k; }
        let l = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
        out.map(|x| x / l)
    }

    #[test]
    fn light_bends_by_4gm_over_b() {
        let gm = 0.005;
        for b in [1.0f32, 2.0, 4.0] {
            let want = 4.0 * gm / b;
            for (name, d) in [("steps", bend_steps([-200.0, b, 0.0], [1.0, 0.0, 0.0], &[([0.0; 3], gm)], 400.0)),
                              ("shader", bend_thin([-200.0, b, 0.0], [1.0, 0.0, 0.0], &[([0.0; 3], gm)], 400.0))] {
                let turn = (-d[1]).atan2(d[0]);
                assert!((turn - want).abs() / want < 0.03, "b={b} {name}: {turn} vs {want}");
            }
        }
    }

    #[test]
    fn short_pieces_of_path_bend_like_step_by_step() {
        let gm = 0.045;
        for (start, len) in [(-2.0f32, 4.0f32), (-0.5, 6.0), (-6.0, 6.5), (-1.0, 1.5), (-3.0, 2.0), (0.5, 3.0)] {
            let p = [start, 0.9, 0.0];
            let (a, b) = (bend_steps(p, [1.0, 0.0, 0.0], &[([0.0; 3], gm)], len), bend_thin(p, [1.0, 0.0, 0.0], &[([0.0; 3], gm)], len));
            let (ta, tb) = ((-a[1]).atan2(a[0]), (-b[1]).atan2(b[0]));
            assert!((ta - tb).abs() < 0.05 * ta.abs(), "start {start} len {len}: {ta} vs {tb}");
        }
    }
}
