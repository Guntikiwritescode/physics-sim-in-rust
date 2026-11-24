// Post-processing: running average (anti-aliasing), bloom (down/up chain), ACES tone mapping, sRGB encode.

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4(p[i], 0.0, 1.0);
    o.uv = vec2(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);   // texture coordinates
    return o;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

fn box4(uv: vec2<f32>) -> vec3<f32> {
    let ts = 1.0 / vec2<f32>(textureDimensions(src));
    let a = textureSampleLevel(src, smp, uv + ts * vec2(-1.0, -1.0), 0.0).rgb;
    let b = textureSampleLevel(src, smp, uv + ts * vec2(1.0, -1.0), 0.0).rgb;
    let c = textureSampleLevel(src, smp, uv + ts * vec2(-1.0, 1.0), 0.0).rgb;
    let d = textureSampleLevel(src, smp, uv + ts * vec2(1.0, 1.0), 0.0).rgb;
    return (a + b + c + d) * 0.25;
}

@fragment
fn fs_down_first(in: VOut) -> @location(0) vec4<f32> {
    let c = box4(in.uv);
    // Soft threshold: only the bright parts glow.
    let l = max(c.r, max(c.g, c.b));
    let w = max(l - 0.8, 0.0) / max(l, 1e-4);
    // Cap what one pixel can add to the glow (smoothly, near 4). A star's surface is far brighter
    // than that; uncapped, the square-shaped blur filters turn its glow into a big white square.
    let b = c * w;
    return vec4(b / (1.0 + max(b.r, max(b.g, b.b)) / 4.0), 1.0);
}

@fragment
fn fs_down(in: VOut) -> @location(0) vec4<f32> {
    return vec4(box4(in.uv), 1.0);
}

@fragment
fn fs_up(in: VOut) -> @location(0) vec4<f32> {
    // 3x3 tent filter, added onto the next larger level (additive blending).
    let ts = 1.0 / vec2<f32>(textureDimensions(src));
    var s = vec3(0.0);
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(-1.0, -1.0), 0.0).rgb;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(0.0, -1.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(1.0, -1.0), 0.0).rgb;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(-1.0, 0.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(src, smp, in.uv, 0.0).rgb * 4.0;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(1.0, 0.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(-1.0, 1.0), 0.0).rgb;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(0.0, 1.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(src, smp, in.uv + ts * vec2(1.0, 1.0), 0.0).rgb;
    return vec4(s / 16.0, 1.0);
}

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var other: texture_2d<f32>;    // accumulate: previous result; composite: bloom
@group(0) @binding(2) var csmp: sampler;
@group(0) @binding(3) var<uniform> post: vec4<f32>;   // x = exposure, y = bloom strength, z = vignette, w = blend weight

// Blend the new (jittered) trace into the running average: smooth edges when the view is still.
@fragment
fn fs_accum(in: VOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(hdr, csmp, in.uv, 0.0);
    let prev = textureSampleLevel(other, csmp, in.uv, 0.0);
    return mix(prev, c, post.w);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}
fn srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3(0.0031308));
}

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(hdr, csmp, in.uv, 0.0);
    let b = textureSampleLevel(other, csmp, in.uv, 0.0).rgb;
    var rgb = (c.rgb + b * post.y) * post.x;
    rgb = aces(rgb);
    let dc = in.uv - 0.5;
    rgb = rgb * (1.0 - post.z * dot(dc, dc));
    // Dither: the screen keeps 256 levels, so dark smooth glows show steps. Noise of about one
    // level (two uniform numbers added: triangular, -1..1 levels) hides them.
    let px = vec2<u32>(in.pos.xy);
    let h1 = f32((px.x * 1973u + px.y * 9277u + 26699u) * 0x9E3779B1u >> 8u) / 16777216.0;
    let h2 = f32((px.x * 7919u + px.y * 3301u + 1013u) * 0x85EBCA77u >> 8u) / 16777216.0;
    return vec4(srgb(rgb) + (h1 + h2 - 1.0) / 255.0, 1.0);
}
