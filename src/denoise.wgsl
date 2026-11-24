// Clean-up of the one-path-per-pixel picture, in four steps:
// 1. cs_temporal: reuse last frame's light. Each pixel finds where its surface was on the last
//    picture (camera move + object move), checks it is the same surface (same object, distance,
//    direction), and blends. Also keeps the average brightness and brightness squared, which tell
//    how noisy each pixel still is.
// 2. cs_atrous (5 times, 1, 2, 4, 8, 16 pixels apart, 3x3 taps each): blur the light, but only across pixels on
//    the same smooth surface with similar brightness; noisy pixels are blurred more.
// 3. cs_compose: put the colours back: picture = emit + colour * light.
// 4. cs_up: the picture at window size, built up over time from frames whose rays are shifted a
//    fraction of a pixel (smooth edges; detail finer than the trace when it is smaller than the
//    window), clamped to the nearby colours so moving things don't leave trails.
// Then cs_expo_sum / cs_expo_set measure the average brightness and set the exposure.

@group(0) @binding(0) var<uniform> u: U;

fn size() -> vec2<i32> { return vec2<i32>(i32(u.info.x), i32(u.info.y)); }
fn inside(p: vec2<i32>) -> bool { let s = size(); return p.x >= 0 && p.y >= 0 && p.x < s.x && p.y < s.y; }

//@@ temporal
@group(0) @binding(1) var<storage, read> objs: array<Obj>;
@group(0) @binding(2) var t_illum: texture_2d<f32>;
@group(0) @binding(3) var t_emit: texture_2d<f32>;
@group(0) @binding(4) var t_geo: texture_2d<f32>;
@group(0) @binding(5) var t_geo_prev: texture_2d<f32>;
@group(0) @binding(6) var t_hist_prev: texture_2d<f32>;
@group(0) @binding(7) var t_mom_prev: texture_2d<f32>;
@group(0) @binding(8) var t_ehist_prev: texture_2d<f32>;
@group(0) @binding(9) var o_light: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var o_mom: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var o_ehist: texture_storage_2d<rgba16float, write>;


struct Back { px: vec2<f32>, depth: f32, n: vec3<f32>, id: f32, ok: bool };

fn back(px: vec2<f32>, g: vec4<f32>) -> Back {
    var r: Back;
    let id = g.w;
    let d = pixel_dir(px);
    r.id = id;
    r.n = oct_dec(g.xy);
    if (id < 0.5) {
        // Sky (or nothing): only the camera turn matters.
        r.px = prev_pixel(d);
        r.depth = SKY_DEPTH;
        r.ok = r.px.x >= 0.0;
        return r;
    }
    let p = u.cam_pos.xyz + d * g.z;
    let o = objs[u32(id + 0.5) - 1u];
    let pp = o.ppos.xyz + (p - o.pos.xyz);   // balls are moved, not turned
    let v = pp - u.prev_pos.xyz;
    r.px = prev_pixel(v);
    r.depth = length(v);
    r.ok = r.px.x >= 0.0;
    return r;
}

fn same_surface(b: Back, gp: vec4<f32>) -> bool {
    if (abs(gp.w - b.id) > 0.5) { return false; }
    if (b.id < 0.5) { return true; }
    if (abs(gp.z - b.depth) > 0.05 * b.depth) { return false; }   // relative: works from rocks to stars
    return dot(oct_dec(gp.xy), b.n) > 0.8;
}

@compute @workgroup_size(8, 8)
fn cs_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = vec2<i32>(gid.xy);
    if (!inside(ip)) { return; }
    let px = vec2<f32>(gid.xy) + 0.5;
    let c = textureLoad(t_illum, ip, 0).rgb;
    let e = textureLoad(t_emit, ip, 0).rgb;
    let g = textureLoad(t_geo, ip, 0);
    var hist = vec3(0.0);
    var mom = vec3(0.0);
    var eh = vec3(0.0);
    var wsum = 0.0;
    if (u.prev_right.w > 0.5) {
        let b = back(px, g);
        if (b.ok) {
            // Bilinear: the 4 last-frame pixels round the found point, each kept only if it shows
            // the same surface.
            let q = b.px - 0.5;
            let q0 = vec2<i32>(floor(q));
            let f = q - floor(q);
            for (var k = 0; k < 4; k = k + 1) {
                let o = vec2<i32>(k & 1, k >> 1);
                let s = q0 + o;
                if (!inside(s)) { continue; }
                if (!same_surface(b, textureLoad(t_geo_prev, s, 0))) { continue; }
                let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
                hist = hist + w * textureLoad(t_hist_prev, s, 0).rgb;
                mom = mom + w * textureLoad(t_mom_prev, s, 0).xyz;
                eh = eh + w * textureLoad(t_ehist_prev, s, 0).rgb;
                wsum = wsum + w;
            }
        }
    }
    let l = lum(c);
    var n = 1.0;
    var m1 = l;
    var m2 = l * l;
    var outc = c;
    var oute = e;
    if (wsum > 0.01) {
        hist = hist / wsum; mom = mom / wsum; eh = eh / wsum;
        n = min(mom.z + 1.0, 255.0);
        let a = max(1.0 / n, u.info2.z);
        outc = mix(hist, c, a);
        // Light given off (gas disk, stars, sky) has no surface to check
        // the history against, and it moves and changes on its own: kept unchecked, it leaves
        // trails and rings for seconds. So the old value is first
        // pulled into this frame's local range, mean +- 1.25 standard deviations of the 3x3 round
        // the pixel (variance clipping), and at most the last 32 frames count
        // (still view).
        var e1 = vec3(0.0); var e2 = vec3(0.0); var ec = 0.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let s = ip + vec2(x, y);
                if (!inside(s)) { continue; }
                let v = textureLoad(t_emit, s, 0).rgb;
                e1 = e1 + v; e2 = e2 + v * v; ec = ec + 1.0;
            }
        }
        e1 = e1 / ec;
        let esd = sqrt(max(e2 / ec - e1 * e1, vec3(0.0)));
        // The range always holds this pixel's own value: a lone bright point (a star) lies above
        // mean + 1.25 sd of its 3x3 (at about half its value), so without this its history would be
        // cut to half every frame.
        let ehc = clamp(eh, min(max(e1 - 1.25 * esd, vec3(0.0)), e), max(e1 + 1.25 * esd, e));
        // While the view moves, this frame counts 60%: each frame's history is re-sampled
        // (bilinear), which blurs a little more every frame, and the range cut then takes away
        // the blurred part, so with less weight point-like lights (stars) fade while turning.
        oute = mix(ehc, e, max(a, select(0.6, 1.0 / 32.0, u.info2.z < 0.05)));
        m1 = mix(mom.x, l, a);
        m2 = mix(mom.y, l * l, a);
    }
    var var_ = max(m2 - m1 * m1, 0.0);
    if (n < 4.0) {
        // Too little history to know the noise: use the spread of the 3x3 neighbours instead.
        var s1 = 0.0; var s2 = 0.0; var cnt = 0.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let s = ip + vec2(x, y);
                if (!inside(s)) { continue; }
                let v = lum(textureLoad(t_illum, s, 0).rgb);
                s1 = s1 + v; s2 = s2 + v * v; cnt = cnt + 1.0;
            }
        }
        s1 = s1 / cnt; s2 = s2 / cnt;
        var_ = max(var_, max(s2 - s1 * s1, 0.0) * 4.0 / n);
    }
    textureStore(o_light, ip, vec4(outc, var_));
    textureStore(o_mom, ip, vec4(m1, m2, n, 0.0));
    textureStore(o_ehist, ip, vec4(oute, 1.0));
}

//@@ atrous
@group(0) @binding(1) var<uniform> pass_step: vec4<f32>;   // x = pixel step
@group(0) @binding(2) var a_geo: texture_2d<f32>;
@group(0) @binding(3) var a_in: texture_2d<f32>;
@group(0) @binding(4) var a_out: texture_storage_2d<rgba16float, write>;

fn world_at(ip: vec2<i32>, g: vec4<f32>) -> vec3<f32> {
    return u.cam_pos.xyz + pixel_dir(vec2<f32>(ip) + 0.5) * g.z;
}

@compute @workgroup_size(8, 8)
fn cs_atrous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = vec2<i32>(gid.xy);
    if (!inside(ip)) { return; }
    let c = textureLoad(a_in, ip, 0);
    let g = textureLoad(a_geo, ip, 0);
    if (g.w < 0.5) { textureStore(a_out, ip, c); return; }   // sky: nothing to blur
    let n = oct_dec(g.xy);
    let p = world_at(ip, g);
    // Noise level: the variance, lightly blurred (3x3) so single pixels don't decide it.
    // (Only in the first two passes; later passes have already smoothed it.)
    var v = c.w; var vw = 1.0;
    if (pass_step.x < 2.5) {
        v = 0.0; vw = 0.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let s = ip + vec2(x, y);
                if (!inside(s)) { continue; }
                let w = select(0.25, 0.5, x == 0) * select(0.25, 0.5, y == 0);
                v = v + w * textureLoad(a_in, s, 0).w; vw = vw + w;
            }
        }
    }
    let sd = sqrt(max(v / vw, 0.0));
    let lc = lum(c.rgb);
    let st = i32(pass_step.x);
    var kern = array<f32, 2>(0.5, 0.25);
    var sum = vec4(0.0);
    var wsum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = ip + vec2(x, y) * st;
            if (!inside(s)) { continue; }
            let gq = textureLoad(a_geo, s, 0);
            if (abs(gq.w - g.w) > 0.5) { continue; }   // another object
            let q = textureLoad(a_in, s, 0);
            let nq = oct_dec(gq.xy);
            let wn = pow(max(dot(n, nq), 0.0), 64.0);
            // Same surface: the other point must lie close to this pixel's surface plane.
            let pq = world_at(s, gq);
            let wz = exp(-abs(dot(n, pq - p)) / (0.004 * g.z * f32(st) + 1e-4 * g.z));
            let wl = exp(-abs(lum(q.rgb) - lc) / (4.0 * sd + 1e-4));
            let w = kern[abs(x)] * kern[abs(y)] * wn * wz * wl;
            sum = sum + vec4(q.rgb * w, q.w * w * w);
            wsum = wsum + w;
        }
    }
    if (wsum < 1e-6) { textureStore(a_out, ip, c); return; }
    textureStore(a_out, ip, vec4(sum.rgb / wsum, sum.w / (wsum * wsum)));
}

//@@ compose
@group(0) @binding(1) var c_alb: texture_2d<f32>;
@group(0) @binding(2) var c_light: texture_2d<f32>;   // cleaned light
@group(0) @binding(3) var c_ehist: texture_2d<f32>;
@group(0) @binding(4) var c_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = vec2<i32>(gid.xy);
    if (!inside(ip)) { return; }
    let alb = textureLoad(c_alb, ip, 0).rgb;
    let c = textureLoad(c_ehist, ip, 0).rgb + alb * textureLoad(c_light, ip, 0).rgb;
    textureStore(c_out, ip, vec4(c, 1.0));
}

//@@ up
// The trace is smaller than the window when the GPU can't keep up. Each frame's rays are shifted
// by a different fraction of a pixel (Halton 2, 3), so over a few frames they land on different
// spots of every window pixel. This pass keeps a window-size picture (its history) and each frame
// adds the rays that landed near each window pixel: where a ray landed close, it counts a lot;
// where none did, the history (moved with the camera and objects) is kept. The history is pulled
// into the colour range of the nearby rays so moving things don't leave trails.
@group(0) @binding(1) var<storage, read> t_objs: array<Obj>;
@group(0) @binding(2) var x_geo: texture_2d<f32>;
@group(0) @binding(3) var x_col: texture_2d<f32>;
@group(0) @binding(4) var x_prev: texture_2d<f32>;
@group(0) @binding(5) var x_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var x_smp: sampler;

// Where the surface seen at trace position `px` (trace pixels) with geometry g was on the last
// frame, in trace pixels.
fn back2(px: vec2<f32>, g: vec4<f32>) -> vec2<f32> {
    let d = pixel_dir(px);
    if (g.w < 0.5) { return prev_pixel(d); }
    let p = u.cam_pos.xyz + d * g.z;
    let o = t_objs[u32(g.w + 0.5) - 1u];
    let pp = o.ppos.xyz + (p - o.pos.xyz);
    return prev_pixel(pp - u.prev_pos.xyz);
}

fn osize() -> vec2<i32> { return vec2<i32>(i32(u.up.x), i32(u.up.y)); }

// Last frame's window picture at point p (window pixels), Catmull-Rom (stays sharp when it is
// moved a fraction of a pixel every frame; plain blending blurs a little more each frame). Done
// with 5 blended reads instead of 16 single ones (the four corner pixels, which carry the least
// weight, are left out).
fn history(p: vec2<f32>) -> vec3<f32> {
    let ts = vec2<f32>(osize());
    let sp = p - 0.5;
    let tc = floor(sp) + 0.5;
    let f = sp - floor(sp);
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (tc - 1.0) / ts;
    let t3 = (tc + 2.0) / ts;
    let t12 = (tc + w2 / w12) / ts;
    let a = w12.x * w0.y; let b = w0.x * w12.y; let c = w12.x * w12.y; let d = w3.x * w12.y; let e = w12.x * w3.y;
    let h = a * textureSampleLevel(x_prev, x_smp, vec2(t12.x, t0.y), 0.0).rgb
          + b * textureSampleLevel(x_prev, x_smp, vec2(t0.x, t12.y), 0.0).rgb
          + c * textureSampleLevel(x_prev, x_smp, t12, 0.0).rgb
          + d * textureSampleLevel(x_prev, x_smp, vec2(t3.x, t12.y), 0.0).rgb
          + e * textureSampleLevel(x_prev, x_smp, vec2(t12.x, t3.y), 0.0).rgb;
    return h / (a + b + c + d + e);
}

@compute @workgroup_size(8, 8)
fn cs_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let op = vec2<i32>(gid.xy);
    let osz = vec2<f32>(u.up.xy);
    if (f32(gid.x) >= osz.x || f32(gid.y) >= osz.y) { return; }
    let isz = vec2<f32>(u.info.xy);
    let r = isz / osz;                       // trace pixels per window pixel (1 or less)
    let q = (vec2<f32>(gid.xy) + 0.5) * r;   // this window pixel's centre, in trace pixels
    let jit = u.jitter.xy;
    // Trace pixel i's ray went through i + 0.5 + jitter. The 3x3 round the nearest one.
    let ic = vec2<i32>(round(q - 0.5 - jit));
    var sharp = vec3(0.0); var ws = 0.0;     // rays within about half a window pixel
    var wide = vec3(0.0); var ww = 0.0;      // rays within about a trace pixel (no history)
    var m1 = vec3(0.0); var m2 = vec3(0.0);
    var g_near = vec4(0.0, 0.0, 3.0e38, 0.0);
    var have_surface = false;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = clamp(ic + vec2(x, y), vec2(0), size() - 1);
            let c = textureLoad(x_col, s, 0).rgb;
            let d = (vec2<f32>(s) + 0.5 + jit - q);     // trace pixels
            let dw = d / r;                              // window pixels
            let a = exp(-2.0 * dot(dw, dw));
            let b = exp(-3.0 * dot(d, d));
            sharp = sharp + a * c; ws = ws + a;
            wide = wide + b * c; ww = ww + b;
            // Colour range: weighted like `wide`, so it changes smoothly across the window
            // pixels (an even 3x3 changes in steps of a trace pixel: blocks while moving).
            m2 = m2 + b * c * c;
            // The nearest surface among the middle one and its 4 side neighbours moves this
            // pixel's history (edges of things in front keep their own motion).
            if (x * y == 0) {
                let g = textureLoad(x_geo, s, 0);
                if (g.w > 0.5 && g.z < g_near.z) { g_near = g; have_surface = true; }
            }
        }
    }
    let c_wide = wide / ww;
    if (u.up.z < 0.5) { textureStore(x_out, op, vec4(c_wide, 1.0)); return; }
    var g = g_near;
    if (!have_surface) { g = textureLoad(x_geo, clamp(ic, vec2(0), size() - 1), 0); }
    let pp = back2(q, g) / r;                // last frame's position, window pixels
    if (pp.x < 0.0 || pp.y < 0.0 || pp.x >= osz.x || pp.y >= osz.y) { textureStore(x_out, op, vec4(c_wide, 1.0)); return; }
    m1 = c_wide; m2 = m2 / ww;
    let sd = sqrt(max(m2 - m1 * m1, vec3(0.0)));
    // Still view: a little wider colour range (sharper from the jittered rays). Not wide: things
    // move on their own in a still view too (the gas disk turns, photons fly), and a range of 3
    // standard deviations leaves trails.
    let still = u.info2.z < 0.05;
    let k = select(1.0, 1.5, still);
    // How much this frame counts: an average over the frames since the reset, but at least 15%
    // while moving (3% when still); times how close this frame's rays came to the pixel.
    let a_min = select(0.15, 0.03, still);
    let a = max(1.0 / (u.up.w + 1.0), a_min) * min(ws, 1.0);
    var c = c_wide;
    if (ws > 1e-3) { c = sharp / ws; }
    // The range holds this pixel's own value too (a lone star is brighter than its
    // neighbourhood's mean + k sd).
    let h = clamp(history(pp), min(max(m1 - k * sd, vec3(0.0)), c), max(m1 + k * sd, c));
    textureStore(x_out, op, vec4(mix(h, c, a), 1.0));
}
