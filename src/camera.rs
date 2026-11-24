//! The camera: a position and a turn (quaternion), so it never flips at the poles. orbit.rs moves it.

use glam::{DMat3, DMat4, DQuat, DVec3};

pub fn look_at(pos: DVec3, target: DVec3, up_hint: DVec3) -> DQuat {
    let fwd = (target - pos).normalize();
    let mut right = fwd.cross(up_hint);
    if right.length_squared() < 1e-12 { right = fwd.any_orthonormal_vector(); }
    let right = right.normalize();
    let up = right.cross(fwd);
    DQuat::from_mat3(&DMat3::from_cols(right, up, -fwd)).normalize()
}

#[derive(Clone)]
pub struct Camera {
    pub pos: DVec3,
    pub rot: DQuat,
    pub fov_y: f64, // radians
}

impl Camera {
    pub fn new(pos: DVec3, target: DVec3) -> Camera {
        Camera { pos, rot: look_at(pos, target, DVec3::Y), fov_y: 50f64.to_radians() }
    }
    pub fn axes(&self) -> (DVec3, DVec3, DVec3) { (self.rot * DVec3::NEG_Z, self.rot * DVec3::X, self.rot * DVec3::Y) }

    /// Projection * view (depth 0..1 as wgpu wants), camera-relative so
    /// big world coordinates keep their precision: pass positions relative to `self.pos`.
    pub fn view_proj_rel(&self, aspect: f64, near: f64, far: f64) -> [[f32; 4]; 4] {
        let view = DMat4::from_quat(self.rot.inverse());
        let f = 1.0 / (0.5 * self.fov_y).tan();
        let proj = DMat4::from_cols_array_2d(&[
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ]);
        (proj * view).to_cols_array_2d().map(|c| c.map(|v| v as f32))
    }
}

impl Camera {
    pub fn ray(&self, x: f64, y: f64, w: f64, h: f64) -> (DVec3, DVec3) {
        let (f, r, u) = self.axes();
        let t = (0.5 * self.fov_y).tan();
        let nx = (2.0 * x / w - 1.0) * t * w / h;
        let ny = (1.0 - 2.0 * y / h) * t;
        (self.pos, (f + r * nx + u * ny).normalize())
    }
}
