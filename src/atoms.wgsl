// Atom sim drawing: each sampled electron position is a small sphere (drawn exactly per pixel).
// Positions move along the probability current: phi(t) = phi0 + m t / (r sin theta)^2.

struct U {
    view_proj: mat4x4<f32>,   // camera-relative
    cam_right: vec4<f32>,     // xyz, w = pixel angle (rad)
    cam_up: vec4<f32>,        // xyz, w = sphere radius (a0)
    cam_pos: vec4<f32>,       // xyz camera position, w = flow time (atomic units)
    misc: vec4<f32>,          // x = m, y = cutaway on (1/0), z = morph 0..1 (1 = new orbital), w = old m
    misc2: vec4<f32>,         // x = number of old points, y = old sphere radius, z = screen aspect, w = grid half size
    ph: array<vec4<f32>, 16>, // 8 photons: [2k] = xyz position (a0), w = size; [2k+1] = rgb, w = brightness
    spec: array<vec4<f32>, 16>, // spectrum lines: x = wavelength (nm), yzw = colour * brightness
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<vec4<f32>>;   // r, theta, phi0, colour value
@group(0) @binding(2) var<storage, read> old_pts: array<vec4<f32>>; // the orbital before the last jump

// kavan010/Atoms's "heat/fire" colour stops: black, purple, red, orange, yellow, white.
fn fire(v0: f32) -> vec3<f32> {
    let v = clamp(v0, 0.0, 1.0) * 5.0;
    let i = min(i32(v), 4);
    let t = v - f32(i);
    var c = array<vec3<f32>, 6>(
        vec3(0.0, 0.0, 0.0), vec3(0.5, 0.0, 0.99), vec3(0.8, 0.0, 0.0),
        vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0));
    return mix(c[i], c[i + 1], t);
}

fn world_pos(p: vec4<f32>, m: f32) -> vec3<f32> {
    let st = max(abs(p.x * sin(p.y)), 1e-4);
    let w = m / (st * st);
    // Wrap the angle per particle so large w * t keeps its precision as well as f32 allows.
    let phi = p.z + (w * u.cam_pos.w) % 6.2831853;
    let s = sin(p.y);
    return vec3(p.x * s * cos(phi), p.x * cos(p.y), p.x * s * sin(phi));
}

struct SOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) q: vec2<f32>,      // position on the dot, -1..1
    @location(1) c: vec3<f32>,
};

// Each point is a small round dot facing the camera, shaded like a ball. Depth is the dot's
// centre (no per-pixel depth write), so the GPU can skip hidden dots early: much faster with
// millions of them.
@vertex
fn vs_pt(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SOut {
    let q = pts[ii];
    // After a jump each dot slides from a spot of the old orbital to a spot of the new one
    // (the old points are only read while sliding).
    let k = smoothstep(0.0, 1.0, u.misc.z);
    var q0 = q;
    var w = world_pos(q, u.misc.x);
    if (k < 1.0) {
        q0 = old_pts[ii % u32(max(u.misc2.x, 1.0))];
        w = mix(world_pos(q0, u.misc.w), w, k);
    }
    var o: SOut;
    // Cutaway: hide the quarter with x < 0 and y > 0 to show the inside.
    if (u.misc.y > 0.5 && w.x < 0.0 && w.y > 0.0) {
        o.pos = vec4(2.0, 2.0, 2.0, 1.0);   // outside the screen: dropped
        return o;
    }
    let c = w - u.cam_pos.xyz;
    let rad = max(mix(u.misc2.y, u.cam_up.w, k), 0.8 * length(c) * u.cam_right.w);
    let corner = vec2(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
    let p = c + (corner.x * u.cam_right.xyz + corner.y * u.cam_up.xyz) * rad;
    o.pos = u.view_proj * vec4(p, 1.0);
    o.q = corner;
    o.c = fire(mix(q0.w, q.w, k));
    return o;
}

@fragment
fn fs_pt(i: SOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.q, i.q);
    if (r2 > 1.0) { discard; }
    return vec4(i.c * (0.75 + 0.5 * sqrt(1.0 - r2)), 1.0);
}

// Reference grid: a square in the y = 0 plane, 2 x 2 cells (500 a0 for hydrogen; scaled to the
// cloud for the rest).
struct GOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32) -> GOut {
    let line = vi / 2u;           // 6 lines: 3 along x, 3 along z
    let end = f32(vi & 1u) * 2.0 - 1.0;
    let k = f32(line % 3u) - 1.0;
    let g = u.misc2.w;
    var xz = vec2(end * g, k * g);
    if (line >= 3u) { xz = vec2(k * g, end * g); }
    var o: GOut;
    o.pos = u.view_proj * vec4(vec3(xz.x, 0.0, xz.y) - u.cam_pos.xyz, 1.0);
    return o;
}

@fragment
fn fs_grid() -> @location(0) vec4<f32> { return vec4(0.25, 0.25, 0.25, 0.5); }

struct POut { @builtin(position) pos: vec4<f32>, @location(0) q: vec2<f32>, @location(1) c: vec3<f32> };

@vertex
fn vs_ph(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> POut {
    let a = u.ph[2u * ii];
    let b = u.ph[2u * ii + 1u];
    var o: POut;
    if (b.w <= 0.0) { o.pos = vec4(2.0, 2.0, 2.0, 1.0); return o; }
    let corner = vec2(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
    let c = a.xyz - u.cam_pos.xyz;
    let rad = max(a.w, 6.0 * length(c) * u.cam_right.w);
    let p = c + (corner.x * u.cam_right.xyz + corner.y * u.cam_up.xyz) * rad;
    o.pos = u.view_proj * vec4(p, 1.0);
    o.q = corner;
    o.c = b.rgb * b.w;
    return o;
}

@fragment
fn fs_ph(i: POut) -> @location(0) vec4<f32> {
    let r2 = dot(i.q, i.q);
    if (r2 > 1.0) { discard; }
    let g = exp(-6.0 * r2) * 6.0 + exp(-40.0 * r2) * 20.0;
    return vec4(i.c * g, 0.0);
}

const SPEC_LO: f32 = 90.0;
const SPEC_HI: f32 = 2000.0;

fn wl_rgb(nm: f32) -> vec3<f32> {   // same table as atom::wavelength_rgb
    if (nm < 380.0) { return vec3(0.35, 0.2, 0.6); }
    if (nm > 780.0) { return vec3(0.35, 0.05, 0.05); }
    var c = vec3(1.0, 0.0, 0.0);
    if (nm < 440.0) { c = vec3(-(nm - 440.0) / 60.0, 0.0, 1.0); }
    else if (nm < 490.0) { c = vec3(0.0, (nm - 440.0) / 50.0, 1.0); }
    else if (nm < 510.0) { c = vec3(0.0, 1.0, -(nm - 510.0) / 20.0); }
    else if (nm < 580.0) { c = vec3((nm - 510.0) / 70.0, 1.0, 0.0); }
    else if (nm < 645.0) { c = vec3(1.0, -(nm - 645.0) / 65.0, 0.0); }
    var f = 1.0;
    if (nm < 420.0) { f = 0.3 + 0.7 * (nm - 380.0) / 40.0; } else if (nm > 700.0) { f = 0.3 + 0.7 * (780.0 - nm) / 80.0; }
    return c * f;
}

struct SpOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_spec(@builtin(vertex_index) vi: u32) -> SpOut {
    let corner = vec2(f32(vi & 1u), f32((vi >> 1u) & 1u));
    var o: SpOut;
    o.pos = vec4(-0.95 + 1.9 * corner.x, -0.97 + 0.12 * corner.y, 0.0, 1.0);
    o.uv = corner;
    return o;
}

@fragment
fn fs_spec(i: SpOut) -> @location(0) vec4<f32> {
    let nm = SPEC_LO * pow(SPEC_HI / SPEC_LO, i.uv.x);
    // Faint rainbow where eyes can see (380-750 nm), dark elsewhere.
    var col = vec3(0.02);
    if (nm >= 380.0 && nm <= 750.0) { col = wl_rgb(nm) * 0.12; }
    let px = fwidth(i.uv.x);
    for (var k = 0; k < 16; k = k + 1) {
        let l = u.spec[k];
        if (l.x <= 0.0) { break; }
        let x = log(l.x / SPEC_LO) / log(SPEC_HI / SPEC_LO);
        let d = abs(i.uv.x - x) / px;
        col = col + l.yzw * 6.0 * (1.0 - smoothstep(1.0, 2.5, d));
    }
    return vec4(col, 1.0);
}
