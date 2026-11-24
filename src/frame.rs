//! HDR picture chain shared by both tabs: scene target (+ depth) -> running average
//! (anti-aliasing) -> bloom chain -> tone-mapped picture. The scene itself is drawn by the
//! caller in the pass from `begin_scene` or `begin_final`.

use wgpu::util::DeviceExt;

pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const BLOOM_LEVELS: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Post { pub v: [f32; 4] } // exposure, bloom strength, vignette, blend weight of the new scene

struct Targets {
    size: (u32, u32),
    scene_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    acc_views: [wgpu::TextureView; 2],
    bloom_views: Vec<wgpu::TextureView>,
    bg_accum: [wgpu::BindGroup; 2], // [i]: new scene + acc[1-i] -> acc[i]
    bg_down0: [wgpu::BindGroup; 2], // reads acc[i]
    bg_down: Vec<wgpu::BindGroup>,  // level i >= 1 reads bloom[i-1]
    bg_up: Vec<wgpu::BindGroup>,
    bg_comp: [wgpu::BindGroup; 2],  // acc[i] + bloom
}

pub struct Frame {
    accum_pipe: wgpu::RenderPipeline,
    down_first_pipe: wgpu::RenderPipeline,
    down_pipe: wgpu::RenderPipeline,
    up_pipe: wgpu::RenderPipeline,
    comp_pipe: wgpu::RenderPipeline,
    post_buf: wgpu::Buffer,
    tex_bgl: wgpu::BindGroupLayout,
    comp_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    targets: Option<Targets>,
    cur: usize, // which running-average texture is newest
    /// This frame's picture was drawn straight into the next running-average texture
    /// (begin_final): encode_post skips the blend.
    direct: bool,
}

pub fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
        },
        count: None,
    }
}
pub fn smp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None,
    }
}
pub fn ubo_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
pub fn storage_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub fn fullscreen_pipe(
    device: &wgpu::Device, module: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout,
    fs: &str, fmt: wgpu::TextureFormat, blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(bgl)], immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module, entry_point: Some(fs), compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: fmt, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

pub const ADD: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

impl Frame {
    /// `out_format` must be a non-sRGB format: the composite pass encodes sRGB itself.
    pub fn new(device: &wgpu::Device, out_format: wgpu::TextureFormat) -> Frame {
        let post_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"), source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });
        let post_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post"), contents: bytemuck::bytes_of(&Post { v: [1.0, 0.45, 0.35, 1.0] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[tex_entry(0), smp_entry(1)],
        });
        let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[tex_entry(0), tex_entry(1), smp_entry(2), ubo_entry(3, wgpu::ShaderStages::FRAGMENT)],
        });
        let accum_pipe = fullscreen_pipe(device, &post_mod, &comp_bgl, "fs_accum", HDR, None);
        let down_first_pipe = fullscreen_pipe(device, &post_mod, &tex_bgl, "fs_down_first", HDR, None);
        let down_pipe = fullscreen_pipe(device, &post_mod, &tex_bgl, "fs_down", HDR, None);
        let up_pipe = fullscreen_pipe(device, &post_mod, &tex_bgl, "fs_up", HDR, Some(ADD));
        let comp_pipe = fullscreen_pipe(device, &post_mod, &comp_bgl, "fs_composite", out_format, None);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        Frame { accum_pipe, down_first_pipe, down_pipe, up_pipe, comp_pipe, post_buf, tex_bgl, comp_bgl, sampler, targets: None, cur: 0, direct: false }
    }
    /// Size of the scene picture. Returns true if the targets were (re)made.
    pub fn set_size(&mut self, device: &wgpu::Device, size: (u32, u32)) -> bool {
        let size = (size.0.max(1), size.1.max(1));
        if self.targets.as_ref().map(|t| t.size) == Some(size) { return false; }
        let tex = |w: u32, h: u32| device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: HDR,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        }).create_view(&Default::default());
        let scene_view = tex(size.0, size.1);
        let depth_view = device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        }).create_view(&Default::default());
        let acc_views = [tex(size.0, size.1), tex(size.0, size.1)];
        let bloom_views: Vec<_> = (0..BLOOM_LEVELS).map(|i| tex(size.0 >> (i + 1), size.1 >> (i + 1))).collect();
        let tb = |v: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let cb = |a: &wgpu::TextureView, b: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.comp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.post_buf.as_entire_binding() },
            ],
        });
        let bg_accum = [cb(&scene_view, &acc_views[1]), cb(&scene_view, &acc_views[0])];
        let bg_down0 = [tb(&acc_views[0]), tb(&acc_views[1])];
        let bg_down: Vec<_> = (0..BLOOM_LEVELS).map(|i| tb(&bloom_views[i.max(1) - 1])).collect();
        let bg_up: Vec<_> = (0..BLOOM_LEVELS).map(|i| tb(&bloom_views[i])).collect();
        let bg_comp = [cb(&acc_views[0], &bloom_views[0]), cb(&acc_views[1], &bloom_views[0])];
        self.targets = Some(Targets { size, scene_view, depth_view, acc_views, bloom_views, bg_accum, bg_down0, bg_down, bg_up, bg_comp });
        true
    }

    pub fn write_post(&self, queue: &wgpu::Queue, post: &Post) { queue.write_buffer(&self.post_buf, 0, bytemuck::bytes_of(post)); }

    /// Starts a pass that draws the scene (cleared to transparent black, depth cleared to 1).
    pub fn begin_scene<'e>(&self, enc: &'e mut wgpu::CommandEncoder, with_depth: bool) -> wgpu::RenderPass<'e> {
        let t = self.targets.as_ref().expect("set_size first");
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &t.scene_view, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: if with_depth { Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }) } else { None },
            ..Default::default()
        })
    }

    /// Starts a pass that draws the finished picture straight into the next running-average
    /// texture, for a picture that replaces the last one (blend weight 1): saves the scene picture
    /// and the blend pass (each a full read or write of the picture).
    pub fn begin_final<'e>(&mut self, enc: &'e mut wgpu::CommandEncoder) -> wgpu::RenderPass<'e> {
        self.direct = true;
        let t = self.targets.as_ref().expect("set_size first");
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("final"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &t.acc_views[1 - self.cur], depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        })
    }

    /// After the scene: blend it into the running average (weight from Post.v[3]; 1 = replace)
    /// and rebuild the bloom chain.
    pub fn encode_post(&mut self, enc: &mut wgpu::CommandEncoder) {
        let next = 1 - self.cur;
        let t = self.targets.as_ref().expect("set_size first");
        let pass = |enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, load: wgpu::LoadOp<wgpu::Color>,
                    pipe: &wgpu::RenderPipeline, bg: &wgpu::BindGroup| {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, bg, &[]);
            rp.draw(0..3, 0..1);
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
        if !self.direct { pass(enc, &t.acc_views[next], clear, &self.accum_pipe, &t.bg_accum[next]); }
        self.direct = false;
        for i in 0..BLOOM_LEVELS {
            if i == 0 { pass(enc, &t.bloom_views[0], clear, &self.down_first_pipe, &t.bg_down0[next]); }
            else { pass(enc, &t.bloom_views[i], clear, &self.down_pipe, &t.bg_down[i]); }
        }
        for i in (1..BLOOM_LEVELS).rev() {
            pass(enc, &t.bloom_views[i - 1], wgpu::LoadOp::Load, &self.up_pipe, &t.bg_up[i]);
        }
        self.cur = next;
    }
    /// Draws the final picture into `rect` = [x, y, w, h] (pixels) of `out`; the rest is black.
    pub fn encode_screen(&self, enc: &mut wgpu::CommandEncoder, out: &wgpu::TextureView, r: [f32; 4]) {
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("screen"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        // Nothing drawn yet (a scene's first frame, while it loads): the window stays black, with the
        // panel and the scene's overlay on top.
        let Some(t) = self.targets.as_ref() else { return };
        rp.set_viewport(r[0], r[1], r[2].max(1.0), r[3].max(1.0), 0.0, 1.0);
        rp.set_pipeline(&self.comp_pipe);
        rp.set_bind_group(0, &t.bg_comp[self.cur], &[]);
        rp.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn post_shader_validates() {
        let src = include_str!("post.wgsl");
        let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m).unwrap_or_else(|e| panic!("{e:?}"));
    }
}
