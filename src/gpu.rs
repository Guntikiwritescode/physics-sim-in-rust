use std::sync::Arc;
use winit::window::Window;

const MEM_LIMIT_MB: u64 = 1500;

/// This program's memory in MB (from `ps`). Quits if it is over the limit.
pub fn memory_guard() -> u64 {
    let pid = std::process::id().to_string();
    let mb = std::process::Command::new("ps").args(["-o", "rss=", "-p", &pid]).output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0);
    if mb > MEM_LIMIT_MB {
        eprintln!("memory {mb} MB is over the {MEM_LIMIT_MB} MB limit, quitting");
        std::process::exit(2);
    }
    mb
}

pub struct Gpu {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Gpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).expect("device");
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Non-sRGB format: the composite pass encodes sRGB itself.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            // two frames may wait for the screen, so one frame's CPU work overlaps the last one's GPU work
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        Gpu { window, surface, device, queue, config }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.config.width = w.max(1);
        self.config.height = h.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// The screen image to draw on. Get it before queueing any GPU work: if there is none
    /// (window covered, minimised, resizing), queue nothing and wait 50 ms, or the queued work piles up.
    pub fn acquire(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => Some(f),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                std::thread::sleep(std::time::Duration::from_millis(50));
                None
            }
            _ => { std::thread::sleep(std::time::Duration::from_millis(50)); None }
        }
    }

    pub fn finish(&self, enc: wgpu::CommandEncoder, frame: wgpu::SurfaceTexture) {
        self.queue.submit([enc.finish()]);
        self.window.pre_present_notify();
        frame.present();
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}
