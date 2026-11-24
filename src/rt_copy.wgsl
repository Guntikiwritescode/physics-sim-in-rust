// Automatic exposure (measure the picture's average brightness, ease the exposure towards it) and the
// final copy of the cleaned picture into the HDR chain (frame.rs: glow, tone map, screen).

@group(0) @binding(0) var<uniform> u: U;

//@@ expo
// bins: how many surface pixels fall in each half-stop of brightness, from 2^-20 (bin 0) to 2^12.
const EXPO_LOW = 9.5367e-7;   // 2^-20
// all: the same for every pixel (sky too), for the "brightest 5%" limit.
struct Expo { sum: atomic<i32>, cnt: atomic<u32>, value: f32, ready: u32, bins: array<atomic<u32>, 64>, all: array<atomic<u32>, 64> };
@group(0) @binding(1) var<storage, read_write> ex: Expo;
@group(0) @binding(2) var e_img: texture_2d<f32>;
@group(0) @binding(3) var e_geo: texture_2d<f32>;
@group(0) @binding(4) var e_emit: texture_2d<f32>;
var<workgroup> wbins: array<atomic<u32>, 64>;
var<workgroup> wall: array<atomic<u32>, 64>;
fn expo_bin(l: f32) -> u32 { return u32(clamp(floor((log2(l) + 20.0) * 2.0), 0.0, 63.0)); }

// Brightness histogram of the picture's lit surfaces. Left out: sky (in space the picture is
// mostly black, and metering it makes every star a white blob), things that glow (a star's
// own surface is 10,000x brighter than what it lights). Very
// dark surfaces are left out in cs_expo_set.
@compute @workgroup_size(16, 16)
fn cs_expo_sum(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li < 64u) { atomicStore(&wbins[li], 0u); atomicStore(&wall[li], 0u); }
    workgroupBarrier();
    let ip = vec2<i32>(gid.xy);
    if (f32(gid.x) < u.info.x && f32(gid.y) < u.info.y) {
        let l = lum(textureLoad(e_img, ip, 0).rgb);
        if (l > EXPO_LOW) { atomicAdd(&wall[expo_bin(l)], 1u); }
    }
    if (f32(gid.x) < u.info.x && f32(gid.y) < u.info.y && textureLoad(e_geo, ip, 0).w > 0.5) {
        let l = lum(textureLoad(e_img, ip, 0).rgb);
        // "Glows" = light added before the first surface (a lamp's own light, the black hole's
        // gas in front). Left out when that is most of the pixel; a surface seen through a little
        // of it still counts.
        if (l > EXPO_LOW && lum(textureLoad(e_emit, ip, 0).rgb) <= 0.5 * l) {
            atomicAdd(&wbins[expo_bin(l)], 1u);
        }
    }
    workgroupBarrier();
    if (li < 64u) {
        let c = atomicLoad(&wbins[li]);
        if (c > 0u) { atomicAdd(&ex.bins[li], c); }
        let ca = atomicLoad(&wall[li]);
        if (ca > 0u) { atomicAdd(&ex.all[li], ca); }
    }
}

// Exposure that brings the middle surface brightness (median) to mid grey, but not so high that
// the brightest 5% of the picture goes past 1.0 (flat white after tone mapping). Eased over about
// 0.6 s so the picture doesn't flicker when something bright passes. Shift in stops from the
// panel (info2.w).
@compute @workgroup_size(1)
fn cs_expo_set() {
    var n = 0u;
    for (var k = 0u; k < 64u; k = k + 1u) { n = n + atomicLoad(&ex.bins[k]); }
    // Fewer than 1 in 50 pixels show a lit surface: keep the exposure as it is (or 1 at the start).
    if (n * 50u < u32(u.info.x * u.info.y)) {
        if (ex.ready == 0u) { ex.value = exp2(u.info2.w); ex.ready = 1u; }
    } else {
        // Brightest 5% first; then the median of the surfaces within 6 stops of that (a night
        // side next to a day side is left out, or it pulls the exposure up until the day side is
        // flat white). "Dark" is relative, so a dim scene (twilight) is still measured.
        var run = 0u;
        var k95 = 63u;
        for (var k = 0u; k < 64u; k = k + 1u) {
            run = run + atomicLoad(&ex.bins[k]);
            if (run * 20u >= n * 19u) { k95 = k; break; }
        }
        let k0 = max(i32(k95) - 12, 0);
        var m = 0u;
        for (var k = u32(k0); k < 64u; k = k + 1u) { m = m + atomicLoad(&ex.bins[k]); }
        var kmed = k95;
        run = 0u;
        for (var k = u32(k0); k < 64u; k = k + 1u) {
            run = run + atomicLoad(&ex.bins[k]);
            if (run * 2u >= m) { kmed = k; break; }
        }
        // The brightest 5% of the whole picture (sky included) may not go past 1.0: a bright sky
        // over dark ground is not flattened to white.
        var na = 0u;
        for (var k = 0u; k < 64u; k = k + 1u) { na = na + atomicLoad(&ex.all[k]); }
        var a95 = k95;
        run = 0u;
        for (var k = 0u; k < 64u; k = k + 1u) {
            run = run + atomicLoad(&ex.all[k]);
            if (run * 20u >= na * 19u) { a95 = max(k, k95); break; }
        }
        let t = min(0.18 / exp2((f32(kmed) + 0.5) * 0.5 - 20.0), 1.0 / exp2((f32(a95) + 0.5) * 0.5 - 20.0));
        let target_ = clamp(t * exp2(u.info2.w), 1e-6, 1e4);
        if (ex.ready == 0u) {
            ex.value = target_;
            ex.ready = 1u;
        } else {
            let k = 1.0 - exp(-max(u.prev_up.w, 0.0) / 0.6);
            ex.value = exp2(mix(log2(ex.value), log2(target_), k));
        }
    }
    for (var k = 0u; k < 64u; k = k + 1u) { atomicStore(&ex.bins[k], 0u); atomicStore(&ex.all[k], 0u); }
}

//@@ copy
struct ExpoR { sum: i32, cnt: u32, value: f32, ready: u32 };
@group(0) @binding(1) var<storage, read> exr: ExpoR;
@group(0) @binding(2) var k_img: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VOut {
    let p = vec2(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: VOut;
    o.pos = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2(p.x, 1.0 - p.y);
    return o;
}

@fragment
fn fs_copy(i: VOut) -> @location(0) vec4<f32> {
    let c = textureLoad(k_img, vec2<i32>(i.pos.xy), 0).rgb;
    let e = select(exr.value, 1.0, exr.ready == 0u);
    var o = c * e;
    // Soft limit on very bright pixels (a star's surface is 10,000x a lit rock): the glow pass
    // would otherwise spread one star over half the screen. Below 16 nothing changes.
    let l = lum(o);
    if (l > 16.0) { o = o * ((16.0 + log2(l / 16.0) * 8.0) / l); }
    return vec4(o, 1.0);
}
