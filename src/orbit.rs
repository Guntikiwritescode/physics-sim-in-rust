//! The camera: always orbits a focus point (a body, or the atom). Turning is a turntable round a
//! fixed "up", so the view never flips over the poles. Speeds are fixed: half a turn for a swipe or
//! drag the height of the picture, a zoom by e for a pinch that doubles the finger spread, 15% per
//! wheel notch. Every change is eased over about 0.05-0.1 s.

use crate::camera::{look_at, Camera};
use crate::host::Input;
use glam::DVec3;
use winit::keyboard::KeyCode;

/// Half a turn for a drag or swipe the height of the picture.
const TURN_PER_HEIGHT: f64 = std::f64::consts::PI;
/// Smallest angle between the view and straight up or down (radians).
const PITCH_LIMIT: f64 = 0.03;
/// One mouse-wheel notch.
const WHEEL: f64 = 0.85;
/// Easing times (s): turning, zooming, sliding.
const EASE_TURN: f64 = 0.04;
const EASE_ZOOM: f64 = 0.08;
const EASE_MOVE: f64 = 0.12;

#[derive(Clone)]
pub struct OrbitCam {
    /// The point turned round, in camera numbers (moves with what the scene follows).
    pub target: DVec3,
    /// Distance from the target (now) and where it is easing to.
    pub dist: f64,
    pub want_dist: f64,
    pub min_dist: f64,
    pub max_dist: f64,
    /// Turntable angles round `up` (yaw) and up from the plane at right angles to it (pitch).
    yaw: f64,
    pitch: f64,
    want_yaw: f64,
    want_pitch: f64,
    want_target: DVec3,
    pub up: DVec3,
    /// A reference direction in the plane at right angles to `up` (yaw 0).
    ref_dir: DVec3,
}

impl OrbitCam {
    /// Start from the camera as it is, turning round `target` with `up` as up.
    pub fn new(cam: &Camera, target: DVec3, up: DVec3, min_dist: f64, max_dist: f64) -> OrbitCam {
        let up = up.normalize();
        let ref_dir = up.any_orthonormal_vector();
        let mut o = OrbitCam { target, dist: 1.0, want_dist: 1.0, min_dist, max_dist, yaw: 0.0, pitch: 0.0, want_yaw: 0.0, want_pitch: 0.0, want_target: target, up, ref_dir };
        o.take_view(cam);
        o
    }

    /// Take the camera's present place as the view (no easing).
    pub fn take_view(&mut self, cam: &Camera) {
        let off = cam.pos - self.target;
        let d = off.length().max(1e-300);
        let dir = off / d;
        let side = self.up.cross(self.ref_dir);
        self.pitch = dir.dot(self.up).clamp(-1.0, 1.0).asin().clamp(-std::f64::consts::FRAC_PI_2 + PITCH_LIMIT, std::f64::consts::FRAC_PI_2 - PITCH_LIMIT);
        self.yaw = dir.dot(side).atan2(dir.dot(self.ref_dir));
        self.dist = d.clamp(self.min_dist, self.max_dist);
        self.want_dist = self.dist;
        self.want_yaw = self.yaw;
        self.want_pitch = self.pitch;
        self.want_target = self.target;
    }

    /// Unit vector from the target to the camera.
    pub fn dir(&self) -> DVec3 {
        let side = self.up.cross(self.ref_dir);
        let h = self.pitch.cos();
        (self.ref_dir * (h * self.yaw.cos()) + side * (h * self.yaw.sin()) + self.up * self.pitch.sin()).normalize()
    }

    /// Move everything by `d` (the scene's numbers were re-centred; the view doesn't change).
    pub fn shift(&mut self, cam: &mut Camera, d: DVec3) {
        self.target += d;
        self.want_target += d;
        cam.pos += d;
    }

    /// Ease to a new target and distance (double-click, the focus list, Home).
    /// (The distance is kept in range as it eases, against the range at that time: the scene
    /// may change the range for the new focus after this call.)
    pub fn go_to(&mut self, target: DVec3, dist: f64) {
        self.want_target = target;
        self.want_dist = dist;
    }

    /// Ease to a view direction (unit vector from the target to the camera).
    pub fn look_from(&mut self, dir: DVec3) {
        let side = self.up.cross(self.ref_dir);
        self.want_pitch = dir.dot(self.up).clamp(-1.0, 1.0).asin().clamp(-std::f64::consts::FRAC_PI_2 + PITCH_LIMIT, std::f64::consts::FRAC_PI_2 - PITCH_LIMIT);
        let y = dir.dot(side).atan2(dir.dot(self.ref_dir));
        // The short way round.
        let mut dy = y - self.yaw;
        while dy > std::f64::consts::PI { dy -= std::f64::consts::TAU; }
        while dy < -std::f64::consts::PI { dy += std::f64::consts::TAU; }
        self.want_yaw = self.yaw + dy;
    }

    /// Mouse, trackpad and keys for one frame. `screen_h`: picture height in pixels.
    pub fn apply(&mut self, cam: &mut Camera, input: &Input, dt: f64) {
        let h = (input.screen.1.max(1)) as f64;
        let rad_per_px = TURN_PER_HEIGHT / h;
        let shift = input.shift();
        // Turn: left-drag, or a two-finger swipe. Slide: right-drag, Shift+left-drag, Shift+swipe.
        let (mut tx, mut ty, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0);
        if input.left {
            if shift { sx += input.drag_left.0; sy += input.drag_left.1; } else { tx += input.drag_left.0; ty += input.drag_left.1; }
        }
        if input.right { sx += input.drag_right.0; sy += input.drag_right.1; }
        if shift { sx += input.swipe.0; sy += input.swipe.1; } else { tx += input.swipe.0; ty += input.swipe.1; }
        // Arrow keys: a quarter turn a second.
        let kx = input.axis(KeyCode::ArrowRight, KeyCode::ArrowLeft) + input.axis(KeyCode::KeyD, KeyCode::KeyA);
        let ky = input.axis(KeyCode::ArrowDown, KeyCode::ArrowUp);
        let key_px = 0.5 * h * dt;   // a quarter turn per second
        tx += kx * key_px;
        ty += ky * key_px;
        self.want_yaw -= tx * rad_per_px;
        self.want_pitch = (self.want_pitch + ty * rad_per_px).clamp(-std::f64::consts::FRAC_PI_2 + PITCH_LIMIT, std::f64::consts::FRAC_PI_2 - PITCH_LIMIT);
        // Slide: the point at the target's depth follows the fingers or the mouse.
        if sx != 0.0 || sy != 0.0 {
            let (_, r, u) = cam.axes();
            let per_px = 2.0 * (0.5 * cam.fov_y).tan() * self.dist / h;
            self.want_target += (-r * sx + u * sy) * per_px;
        }
        // Zoom: pinch, wheel, keys.
        let mut z = (-input.pinch).exp() * WHEEL.powf(input.scroll);
        let kz = input.axis(KeyCode::KeyS, KeyCode::KeyW) + input.axis(KeyCode::Minus, KeyCode::Equal);
        z *= (kz * dt * 2f64.ln()).exp();   // keys: 2x a second
        if z.is_finite() && z > 0.0 { self.want_dist = (self.want_dist * z).clamp(self.min_dist, self.max_dist); }
        self.step(cam, dt);
    }

    /// Ease toward the wanted view and place the camera.
    pub fn step(&mut self, cam: &mut Camera, dt: f64) {
        let k = |tau: f64| 1.0 - (-dt.max(0.0) / tau).exp();
        let kt = k(EASE_TURN);
        self.yaw += (self.want_yaw - self.yaw) * kt;
        self.pitch += (self.want_pitch - self.pitch) * kt;
        self.want_dist = self.want_dist.clamp(self.min_dist, self.max_dist);
        self.dist = (self.dist.ln() + (self.want_dist.ln() - self.dist.ln()) * k(EASE_ZOOM)).exp();
        self.target += (self.want_target - self.target) * k(EASE_MOVE);
        self.place(cam);
    }

    /// Put the camera where the angles and distance say.
    pub fn place(&self, cam: &mut Camera) {
        cam.pos = self.target + self.dir() * self.dist;
        cam.rot = look_at(cam.pos, self.target, self.up);
    }

    /// Keep the camera out of a ball (centre `c`, radius `r`, camera numbers): moved straight out
    /// to `r` if inside. Done after `apply` every frame for everything near; the view direction
    /// stays on the target.
    pub fn keep_out(&self, cam: &mut Camera, c: DVec3, r: f64) {
        let v = cam.pos - c;
        let d = v.length();
        if d < r {
            cam.pos = c + if d > 0.0 { v / d } else { self.dir() } * r;
            cam.rot = look_at(cam.pos, self.target, self.up);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp() -> Input { Input { screen: (1280, 720), mouse: (640.0, 360.0), ..Default::default() } }
    fn cam_at(p: DVec3) -> Camera { Camera { pos: p, rot: look_at(p, DVec3::ZERO, DVec3::Y), fov_y: 50f64.to_radians() } }

    #[test]
    fn swipe_turns_half_a_turn_per_picture_height() {
        for parts in [1usize, 10, 100] {
            let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
            let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 1.0, 100.0);
            let d0 = o.dir();
            for _ in 0..parts { let mut i = inp(); i.swipe = (720.0 / parts as f64, 0.0); o.apply(&mut cam, &i, 1.0 / 60.0); }
            for _ in 0..120 { o.apply(&mut cam, &inp(), 1.0 / 60.0); }
            let ang = d0.dot(o.dir()).clamp(-1.0, 1.0).acos();
            assert!((ang - std::f64::consts::PI).abs() < 1e-3, "{parts} parts: turned {} deg", ang.to_degrees());
            assert!(((cam.pos.length()) - 10.0).abs() < 1e-9, "distance changed");
        }
    }

    #[test]
    fn pinch_and_wheel_zoom_by_fixed_amounts() {
        for parts in [1usize, 20, 200] {
            let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
            let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 0.01, 100.0);
            for _ in 0..parts { let mut i = inp(); i.pinch = 1.0 / parts as f64; o.apply(&mut cam, &i, 1.0 / 60.0); }
            for _ in 0..120 { o.apply(&mut cam, &inp(), 1.0 / 60.0); }
            assert!((o.dist - 10.0 / std::f64::consts::E).abs() < 1e-6, "pinch in {parts} parts: {}", o.dist);
        }
        let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
        let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 0.01, 100.0);
        for _ in 0..10 { let mut i = inp(); i.scroll = 1.0; o.apply(&mut cam, &i, 1.0 / 60.0); }
        for _ in 0..120 { o.apply(&mut cam, &inp(), 1.0 / 60.0); }
        assert!((o.dist - 10.0 * 0.85f64.powi(10)).abs() < 1e-6, "wheel: {}", o.dist);
    }

    #[test]
    fn zoom_stays_in_range() {
        let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
        let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 2.0, 50.0);
        for _ in 0..200 { let mut i = inp(); i.pinch = 5.0; o.apply(&mut cam, &i, 1.0 / 60.0); assert!(cam.pos.length() >= 2.0 - 1e-9); }
        for _ in 0..200 { let mut i = inp(); i.scroll = -50.0; o.apply(&mut cam, &i, 1.0 / 60.0); assert!(cam.pos.length() <= 50.0 + 1e-9); }
    }

    #[test]
    fn never_rolls_or_flips() {
        let mut seed = 3u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5 };
        let mut cam = cam_at(DVec3::new(0.0, 3.0, 10.0));
        let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 1.0, 100.0);
        for _ in 0..5000 {
            let mut i = inp();
            i.swipe = (400.0 * rnd(), 400.0 * rnd());
            o.apply(&mut cam, &i, 1.0 / 60.0);
            let (_, r, u) = cam.axes();
            assert!(r.dot(DVec3::Y).abs() < 1e-9, "rolled");
            assert!(u.dot(DVec3::Y) > 0.0, "upside down");
        }
    }

    #[test]
    fn shift_swipe_slides_the_target() {
        let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
        let mut o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 1.0, 100.0);
        let mut i = inp();
        i.held.insert(KeyCode::ShiftLeft);
        i.swipe = (100.0, 0.0);
        o.apply(&mut cam, &i, 1.0 / 60.0);
        for _ in 0..120 { o.apply(&mut cam, &inp(), 1.0 / 60.0); }
        let per_px = 2.0 * (25f64.to_radians()).tan() * 10.0 / 720.0;
        assert!((o.target.x + 100.0 * per_px).abs() < 1e-6 && o.target.y.abs() < 1e-9, "target {}", o.target);
    }

    #[test]
    fn keep_out_moves_the_camera_out_of_a_ball() {
        let mut cam = cam_at(DVec3::new(0.0, 0.0, 10.0));
        let o = OrbitCam::new(&cam, DVec3::ZERO, DVec3::Y, 1.0, 100.0);
        o.keep_out(&mut cam, DVec3::new(0.0, 0.0, 9.0), 2.0);
        assert!((cam.pos - DVec3::new(0.0, 0.0, 11.0)).length() < 1e-9);
        let (f, _, _) = cam.axes();
        assert!((f - DVec3::NEG_Z).length() < 1e-9);
    }
}
