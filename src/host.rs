//! One window for every scene: a scene menu, an on-screen control panel, live labels and help
//! (egui 0.35, MIT/Apache), and mouse/keyboard input passed to the scene only when the panel
//! doesn't want it.

use crate::frame::Frame;
use crate::gpu::{memory_guard, Gpu};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// What the mouse and keyboard did since the last frame (only what the panel didn't take).
#[derive(Default, Clone)]
pub struct Input {
    pub held: HashSet<KeyCode>,
    pub mouse: (f64, f64),              // physical pixels from the top left of the picture area
    pub left: bool,
    pub right: bool,
    pub drag_left: (f64, f64),          // movement while held, this frame (pixels)
    pub drag_right: (f64, f64),
    pub scroll: f64,                    // mouse-wheel notches, + = away from you (zoom in)
    pub swipe: (f64, f64),              // trackpad two-finger swipe this frame (pixels, like a drag)
    pub pinch: f64,                     // trackpad pinch this frame (magnification change, + = apart)
    pub pressed_left: Option<(f64, f64)>,
    pub released_left: bool,
    pub double: Option<(f64, f64)>,
    pub screen: (u32, u32),
}

impl Input {
    pub fn shift(&self) -> bool { self.held.contains(&KeyCode::ShiftLeft) || self.held.contains(&KeyCode::ShiftRight) }
    pub fn axis(&self, plus: KeyCode, minus: KeyCode) -> f64 {
        (self.held.contains(&plus) as i32 - self.held.contains(&minus) as i32) as f64
    }
    fn end_frame(&mut self) {
        self.drag_left = (0.0, 0.0);
        self.drag_right = (0.0, 0.0);
        self.scroll = 0.0;
        self.swipe = (0.0, 0.0);
        self.pinch = 0.0;
        self.pressed_left = None;
        self.released_left = false;
        self.double = None;
    }
}

pub trait Scene {
    fn name(&self) -> &'static str;
    /// Controls, one per line, shown in the help box.
    fn help(&self) -> &'static str;
    fn key(&mut self, _k: KeyCode, _down: bool) {}
    fn update(&mut self, dt: f64, input: &Input);
    /// Buttons and sliders for the side panel.
    fn ui(&mut self, ui: &mut egui::Ui);
    /// Lines for the small box at the bottom right.
    fn status(&self) -> Vec<String>;
    /// Draw into frame (call frame.set_size / write_post / encode_post as needed).
    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder,
              frame: &mut Frame, screen: (u32, u32), fps: f64);
    /// Draw on top of the picture (arrows, labels). Scene pixel (x, y) is at
    /// `origin + (x, y) / ppp` in panel points.
    fn overlay(&mut self, _p: &egui::Painter, _ppp: f32, _origin: egui::Pos2) {}
}

pub type Factory = (&'static str, fn() -> Box<dyn Scene>);

pub const CAMERA_HELP: &str = "Trackpad: two-finger swipe turns round the focus, pinch zooms, Shift + swipe slides\nMouse: left-drag turns, right-drag (or Shift + left-drag) slides, wheel zooms\nDouble-click a body: focus on it\nArrow keys: turn   +/-: zoom (in the Black hole tab also A/D and W/S)\nHome: start view   Tab: hide/show the panel   F1: help";

struct Gfx {
    gpu: Gpu,
    frame: Frame,
    ctx: egui::Context,
    state: egui_winit::State,
    rend: egui_wgpu::Renderer,
}

struct App {
    gfx: Option<Gfx>,
    scenes: Vec<Factory>,
    idx: usize,
    scene: Box<dyn Scene>,
    switch_to: Option<usize>,
    input: Input,
    last_click: Option<Instant>,
    last_frame: Instant,
    fps_t0: Instant,
    fps_frames: u32,
    fps: f64,
    mem: u64,
    show_panel: bool,
    show_help: bool,
    view: [f32; 4],    // picture area in the window (physical pixels): right of the panel, below the menu
    raw_mouse: (f64, f64),
}

impl App {
    fn render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().min(0.1);
        self.last_frame = now;
        if let Some(i) = self.switch_to.take() {
            self.idx = i;
            self.scene = (self.scenes[i].1)();
            self.input.held.clear();
        }
        if let Some(g) = self.gfx.as_ref() {
            if self.view[2] < 1.0 { self.view = [0.0, 0.0, g.gpu.config.width as f32, g.gpu.config.height as f32]; }
            self.input.screen = (self.view[2] as u32, self.view[3] as u32);
        }
        // Update first, then wait for a screen image, so the update overlaps the GPU's work on the
        // last frame.
        self.scene.update(dt, &self.input);
        self.input.end_frame();

        let Some(g) = self.gfx.as_mut() else { return };
        let Some(surf) = g.gpu.acquire() else { return };
        let screen = (g.gpu.config.width, g.gpu.config.height);
        let view = surf.texture.create_view(&Default::default());
        let mut enc = g.gpu.device.create_command_encoder(&Default::default());
        let pic = (self.view[2] as u32, self.view[3] as u32);
        self.scene.render(&g.gpu.device, &g.gpu.queue, &mut enc, &mut g.frame, pic, self.fps);
        g.frame.encode_screen(&mut enc, &view, self.view);

        let raw = g.state.take_egui_input(&g.gpu.window);
        let (scenes, idx) = (&self.scenes, self.idx);
        let scene = &mut self.scene;
        let mut switch = None;
        let (show_panel, show_help) = (&mut self.show_panel, &mut self.show_help);
        let (fps, mem) = (self.fps, self.mem);
        let mut area = egui::Rect::NOTHING;
        let mut out = g.ctx.run_ui(raw, |ui| area = panel(ui, scenes, idx, scene.as_mut(), show_panel, show_help, fps, mem, &mut switch));
        let ppp = g.ctx.pixels_per_point();
        let new_view = view_px(area, ppp, screen);
        if new_view != self.view {
            self.view = new_view;
            self.input.mouse = (self.raw_mouse.0 - new_view[0] as f64, self.raw_mouse.1 - new_view[1] as f64);
        }
        if let Some(i) = switch { self.switch_to = Some(i); }
        g.state.handle_platform_output(&g.gpu.window, std::mem::take(&mut out.platform_output));
        paint_panel(&g.ctx, &mut g.rend, &g.gpu.device, &g.gpu.queue, &mut enc, &view, out, screen);
        g.gpu.finish(enc, surf);

        self.fps_frames += 1;
        let el = (now - self.fps_t0).as_secs_f64();
        if el >= 1.0 {
            self.mem = memory_guard();
            self.fps = self.fps_frames as f64 / el;
            self.fps_t0 = now;
            self.fps_frames = 0;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() { return; }
        let attrs = Window::default_attributes().with_title("Physics sims (Rust)")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 860.0));
        let window = Arc::new(el.create_window(attrs).expect("window"));
        let gpu = Gpu::new(window.clone());
        let frame = Frame::new(&gpu.device, gpu.config.format);
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &*window, Some(window.scale_factor() as f32), None, None);
        let rend = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, egui_wgpu::RendererOptions::default());
        self.gfx = Some(Gfx { gpu, frame, ctx, state, rend });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = self.gfx.as_mut() else { return };
        let resp = g.state.on_window_event(&g.gpu.window, &event);
        let ui_mouse = g.ctx.egui_wants_pointer_input();
        let ui_keys = g.ctx.egui_wants_keyboard_input();
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => g.gpu.resize(s.width, s.height),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::CursorMoved { position, .. } => {
                self.raw_mouse = (position.x, position.y);
                let (x, y) = (position.x - self.view[0] as f64, position.y - self.view[1] as f64);
                let (dx, dy) = (x - self.input.mouse.0, y - self.input.mouse.1);
                if self.input.left { self.input.drag_left.0 += dx; self.input.drag_left.1 += dy; }
                if self.input.right { self.input.drag_right.0 += dx; self.input.drag_right.1 += dy; }
                self.input.mouse = (x, y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if down && !ui_mouse && !resp.consumed {
                            self.input.left = true;
                            self.input.pressed_left = Some(self.input.mouse);
                            let now = Instant::now();
                            if self.last_click.is_some_and(|t| (now - t).as_secs_f64() < 0.35) {
                                self.input.double = Some(self.input.mouse);
                                self.last_click = None;   // a third click starts a new pair
                            } else {
                                self.last_click = Some(now);
                            }
                        } else if !down && self.input.left {
                            self.input.left = false;
                            self.input.released_left = true;
                        }
                    }
                    MouseButton::Right | MouseButton::Middle => {
                        if down && !ui_mouse { self.input.right = true; } else if !down { self.input.right = false; }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !ui_mouse {
                    // A mouse wheel sends notches; a trackpad sends pixels (a two-finger swipe), which
                    // turn the view.
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => self.input.scroll += y as f64,
                        MouseScrollDelta::PixelDelta(p) => { self.input.swipe.0 += p.x; self.input.swipe.1 += p.y; }
                    }
                }
            }
            // Trackpad pinch: the change in magnification (the delta may be NaN, winit docs).
            WindowEvent::PinchGesture { delta, .. } => {
                if !ui_mouse && delta.is_finite() { self.input.pinch += delta; }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    if !down { self.input.held.remove(&k); }
                    if ui_keys { return; }
                    if down { self.input.held.insert(k); }
                    if !event.repeat {
                        match (k, down) {
                            // Escape belongs to the scene
                            (KeyCode::Tab, true) => self.show_panel = !self.show_panel,
                            (KeyCode::F1, true) => self.show_help = !self.show_help,
                            _ => self.scene.key(k, down),
                        }
                    }
                }
            }
            WindowEvent::Focused(false) => { self.input.held.clear(); self.input.left = false; self.input.right = false; }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = self.gfx.as_ref() { g.gpu.window.request_redraw(); }
    }
}

/// The on-screen panel: scene menu on top, controls on the left, live labels bottom right, help box.
#[allow(clippy::too_many_arguments)]
fn panel(ui: &mut egui::Ui, scenes: &[Factory], idx: usize, scene: &mut dyn Scene, show_panel: &mut bool, show_help: &mut bool,
         fps: f64, mem: u64, switch: &mut Option<usize>) -> egui::Rect {
    let ctx = ui.ctx().clone();
    egui::Panel::top("scenes").show(ui, |ui| {
        ui.horizontal(|ui| {
            for (i, (name, _)) in scenes.iter().enumerate() {
                let t = egui::RichText::new(*name).size(16.0);
                if ui.selectable_label(i == idx, t).clicked() && i != idx { *switch = Some(i); }
            }
        });
    });
    let area = ui.available_rect_before_wrap();
    // The controls float over the picture, top left, only as tall as they need.
    if *show_panel {
        egui::Window::new("controls").title_bar(false).resizable(false).collapsible(false)
            // placed below the tabs so it does not cover them
            .anchor(egui::Align2::LEFT_TOP, [10.0, area.min.y + 8.0]).default_width(260.0).max_height(area.height() - 20.0)
            .frame(egui::Frame::window(&ctx.global_style()).fill(egui::Color32::from_black_alpha(200)))
            .show(&ctx, |ui| {
                ui.set_width(260.0);
                ui.horizontal(|ui| {
                    if ui.small_button("Help").on_hover_text("F1").clicked() { *show_help = !*show_help; }
                    if ui.small_button("Hide").on_hover_text("Tab").clicked() { *show_panel = false; }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical().show(ui, |ui| { scene.ui(ui); });
            });
    }
    let ppp = ctx.pixels_per_point();
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("overlay"))).with_clip_rect(area);
    scene.overlay(&painter, ppp, area.min);
    if !*show_panel {
        egui::Area::new(egui::Id::new("show_panel")).fixed_pos(area.min + egui::vec2(8.0, 8.0)).show(&ctx, |ui| {
            if ui.button("Menu").on_hover_text("Tab").clicked() { *show_panel = true; }
        });
    }
    let lines = scene.status();
    if !lines.is_empty() {
        egui::Window::new("Now").anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -10.0]).title_bar(false)
            .resizable(false).collapsible(false).show(&ctx, |ui| {
                for l in &lines { ui.label(l); }
            });
    }
    if *show_help {
        egui::Window::new("Help").open(show_help).default_pos([300.0, 60.0]).default_width(460.0).show(&ctx, |ui| {
            ui.strong(scene.name());
            ui.label(scene.help());
            ui.separator();
            ui.strong("Camera");
            ui.label(CAMERA_HELP);
            ui.separator();
            ui.weak(format!("{fps:.0} frames a second, {mem} MB"));
        });
    }
    area
}

/// Picture area in physical pixels [x, y, w, h], from the panel's free area in points.
fn view_px(area: egui::Rect, ppp: f32, screen: (u32, u32)) -> [f32; 4] {
    if !area.is_positive() { return [0.0, 0.0, screen.0 as f32, screen.1 as f32]; }
    let x = (area.min.x * ppp).round().clamp(0.0, screen.0 as f32 - 16.0);
    let y = (area.min.y * ppp).round().clamp(0.0, screen.1 as f32 - 16.0);
    [x, y, (area.width() * ppp).round().clamp(16.0, screen.0 as f32 - x), (area.height() * ppp).round().clamp(16.0, screen.1 as f32 - y)]
}

/// Draw the panel's shapes on top of the picture in `view`.
#[allow(clippy::too_many_arguments)]
fn paint_panel(ctx: &egui::Context, rend: &mut egui_wgpu::Renderer, device: &wgpu::Device, queue: &wgpu::Queue,
               enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, out: egui::FullOutput, screen: (u32, u32)) {
    let ppp = out.pixels_per_point;
    let jobs = ctx.tessellate(out.shapes, ppp);
    let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [screen.0, screen.1], pixels_per_point: ppp };
    for (id, delta) in &out.textures_delta.set { rend.update_texture(device, queue, *id, delta); }
    rend.update_buffers(device, queue, enc, &jobs, &sd);
    {
        let rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("panel"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        let mut rp = rp.forget_lifetime();
        rend.render(&mut rp, &jobs, &sd);
    }
    for id in &out.textures_delta.free { rend.free_texture(id); }
}

/// No window: run scene `idx` for `secs`, then save what the window would show, panel included,
/// as a PPM picture of w x h pixels (ppp 2).
pub fn screenshot(scenes: Vec<Factory>, idx: usize, out: &str, secs: f64, w: u32, h: u32) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let mut frame = Frame::new(&device, fmt);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());
    let mut rend = egui_wgpu::Renderer::new(&device, fmt, egui_wgpu::RendererOptions::default());
    let mut scene = (scenes[idx].1)();
    let (mut show_panel, mut show_help) = (true, false);
    let dt = 1.0 / 60.0;
    let frames = ((secs / dt) as usize).max(1);
    let mut pic = [0.0, 0.0, w as f32, h as f32];
    for f in 0..frames {
        if f % 60 == 0 { memory_guard(); }
        let input = Input { screen: (pic[2] as u32, pic[3] as u32), mouse: (pic[2] as f64 * 0.6, pic[3] as f64 * 0.5), ..Default::default() };
        scene.update(dt, &input);
        let mut enc = device.create_command_encoder(&Default::default());
        scene.render(&device, &queue, &mut enc, &mut frame, (pic[2] as u32, pic[3] as u32), f64::INFINITY);
        frame.encode_screen(&mut enc, &view, pic);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 / 2.0, h as f32 / 2.0))),
            time: Some(f as f64 * dt),
            predicted_dt: dt as f32,
            ..Default::default()
        };
        ctx.set_pixels_per_point(2.0);
        let mut switch = None;
        let mut area = egui::Rect::NOTHING;
        let out = ctx.run_ui(raw, |ui| area = panel(ui, &scenes, idx, scene.as_mut(), &mut show_panel, &mut show_help, 60.0, 0, &mut switch));
        paint_panel(&ctx, &mut rend, &device, &queue, &mut enc, &view, out, (w, h));
        pic = view_px(area, 2.0, (w, h));
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let mut e = device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([e.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = read.slice(..).get_mapped_range();
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in data.chunks_exact(4) { ppm.extend_from_slice(&px[..3]); }
    std::fs::write(out, ppm).expect("write");
}

pub fn run(scenes: Vec<Factory>, start: usize) {
    let el = EventLoop::new().expect("event loop");
    el.set_control_flow(ControlFlow::Poll);
    let now = Instant::now();
    let scene = (scenes[start].1)();
    let mut app = App {
        gfx: None, scenes, idx: start, scene, switch_to: None, input: Input::default(), last_click: None,
        last_frame: now, fps_t0: now, fps_frames: 0, fps: 0.0, mem: 0, show_panel: true, show_help: false,
        view: [0.0; 4], raw_mouse: (0.0, 0.0),
    };
    el.run_app(&mut app).expect("run");
}
