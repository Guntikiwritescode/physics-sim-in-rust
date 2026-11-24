// All positions are relative to the camera (the camera is at 0), so huge world coordinates
// never reach the GPU's 32-bit numbers.
// Shared by the tracer (rt.wgsl) and the clean-up passes (denoise.wgsl, rt_copy.wgsl).

struct U {
    cam_pos: vec4<f32>,     // xyz, w = tan(fov_y / 2)
    cam_fwd: vec4<f32>,     // xyz, w = aspect
    cam_right: vec4<f32>,   // xyz, w = frame number (random seed)
    cam_up: vec4<f32>,      // xyz, w = number of objects
    info: vec4<f32>,        // x, y = traced picture size, w = number of lamps
    info2: vec4<f32>,       // x = max bounces, y = number of heavy objects, z = smallest history weight, w = exposure shift (stops)
    prev_pos: vec4<f32>,    // the camera one frame ago (same layout as above)
    prev_fwd: vec4<f32>,
    prev_right: vec4<f32>,  // w = 1 if last frame's images can be reused
    prev_up: vec4<f32>,     // w = seconds since last frame
    jitter: vec4<f32>,      // xy = sub-pixel offset of this frame's rays (pixels), z = number of tree nodes, w = bounces that follow light bending
    misc: vec4<f32>,        // x = time (s, for the black hole gas swirl), y = smallest ray offset (m), z unused, w = 1 + ball the camera is in (its rays look through it)
    lamps: array<vec4<u32>, 2>,  // up to 8 lamp object indices
    lens: array<vec4<u32>, 4>,   // up to 16 heavy (light-bending) object indices
    up: vec4<f32>,          // shown picture: x, y = size (pixels), z = 1 if its last frame can be reused, w = frames since it was reset
};

struct Obj {
    pos: vec4<f32>,   // centre, w = shape (0 ball, 3 big ball: pos = surface point nearest the camera)
    rot: vec4<f32>,   // big ball: xyz = outward direction at that point (ball: unused)
    size: vec4<f32>,  // x = radius, w = lens G M / c^2 (0 = none)
    mat: vec4<f32>,   // rgb colour (lamp: light), w = material (0 matte, 3 lamp, 5 black hole: see rt_hole.wgsl)
    ppos: vec4<f32>,  // pos one frame ago (black hole: w = horizon radius in Schwarzschild radii)
};

const PI: f32 = 3.14159265;
const NO_HIT: f32 = 1e30;
const SKY_DEPTH: f32 = 1e4;

// Direction of the ray through picture point `px` (pixels, top left = 0).
fn pixel_dir(px: vec2<f32>) -> vec3<f32> {
    let ndc = vec2(px.x / u.info.x * 2.0 - 1.0, 1.0 - px.y / u.info.y * 2.0);
    return normalize(u.cam_fwd.xyz + ndc.x * u.cam_pos.w * u.cam_fwd.w * u.cam_right.xyz + ndc.y * u.cam_pos.w * u.cam_up.xyz);
}

// Where a point (given relative to the previous camera) was on the previous picture, in pixels.
// Returns (-1, -1) if it was behind that camera.
fn prev_pixel(v: vec3<f32>) -> vec2<f32> {
    let z = dot(v, u.prev_fwd.xyz);
    if (z <= 1e-6) { return vec2(-1.0); }
    let x = dot(v, u.prev_right.xyz) / (z * u.cam_pos.w * u.cam_fwd.w);
    let y = dot(v, u.prev_up.xyz) / (z * u.cam_pos.w);
    return vec2((x * 0.5 + 0.5) * u.info.x, (0.5 - y * 0.5) * u.info.y);
}

// Unit normals stored in two numbers (octahedral mapping).
fn oct_enc(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z >= 0.0) { return p; }
    return (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p >= vec2(0.0));
}
fn oct_dec(e: vec2<f32>) -> vec3<f32> {
    var n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) { n = vec3((1.0 - abs(n.yx)) * select(vec2(-1.0), vec2(1.0), n.xy >= vec2(0.0)), n.z); }
    return normalize(n);
}

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
