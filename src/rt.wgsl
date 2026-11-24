// The light: a path tracer (one path per pixel per frame), written so the clean-up passes can work.
// Paths start at the camera and bounce off matte surfaces (random direction, cosine-weighted).
// Lamps (stars) are sampled straight at each matte bounce (next-event estimation). Rays passing
// heavy objects turn toward them (thin-lens method, see trace()); near a black hole they follow
// the exact Kerr path (rt_hole.wgsl). Objects are found through a tree of boxes (BVH).
//
// Besides the light, each pixel stores what the clean-up needs about the first matte surface the
// path meets: its colour, direction, distance and object. The light after that surface is stored
// divided by its colour ("illum"), so the clean-up can blur the light without blurring the
// colours; light seen before reaching it (lamps, gas, sky) is kept apart ("emit").

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> objs: array<Obj>;
@group(0) @binding(2) var<storage, read> bvh: array<vec4<f32>>;   // 2 per node: (min, first), (max, count)
@group(0) @binding(3) var<storage, read> prims: array<u32>;
@group(0) @binding(4) var out_illum: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_alb: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var out_geo: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var<storage, read> bounds: array<vec4<f32>>;   // per object: centre, radius of a ball holding it

// Offset for new rays: 1 mm near the camera, growing with distance so far hits (planets, stars)
// don't hit themselves again through rounding.
fn eps_at(p: vec3<f32>) -> f32 { return max(u.misc.y, 4e-6 * length(p)); }   // misc.y: 1 mm, or less when zoomed in
const NONE: u32 = 0xffffffffu;

var<private> seed: u32;
fn rnd() -> f32 {
    seed = seed * 747796405u + 2891336453u;
    var w = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w) / 4294967296.0;
}

fn hit_ball(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32, tmin: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let q = oc - b * rd;                 // closest-approach offset: stable for big, far balls
    let h = r * r - dot(q, q);
    if (h < 0.0) { return NO_HIT; }
    let s = sqrt(h);
    if (-b - s > tmin) { return -b - s; }
    if (-b + s > tmin) { return -b + s; }
    return NO_HIT;
}

// Big ball (planet, star), stored as s = its surface point nearest the camera, n = outward
// direction there, R = radius. Solving |w + R n + t d|^2 = R^2 with w = ro - s keeps only small
// numbers, so a camera 2 m above a 6,371 km planet still gets millimetre-exact ground.
fn hit_big(ro: vec3<f32>, rd: vec3<f32>, s: vec3<f32>, n: vec3<f32>, r: f32, tmin: f32) -> f32 {
    let w = ro - s;
    let b = dot(rd, w) + r * dot(n, rd);
    let c = dot(w, w) + 2.0 * r * dot(n, w);
    let h = b * b - c;
    if (h < 0.0) { return NO_HIT; }
    let sq = sqrt(h);
    // Roots q and c / q with q = -(b + sign(b) sq): no subtraction of nearly equal numbers.
    let q = -(b + select(-sq, sq, b >= 0.0));
    var t1 = q;
    var t2 = q;
    if (abs(q) > 1e-30) { t2 = c / q; }
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    if (lo > tmin) { return lo; }
    if (hi > tmin) { return hi; }
    return NO_HIT;
}

fn obj_centre(o: Obj) -> vec3<f32> {
    if (i32(o.pos.w) == 3) { return o.pos.xyz - o.size.x * o.rot.xyz; }
    return o.pos.xyz;
}

fn hit_obj(i: u32, ro: vec3<f32>, rd: vec3<f32>, tmin: f32) -> f32 {
    // Only the fields needed are read (not the whole object).
    let pos = objs[i].pos;
    let size = objs[i].size;
    let kind = i32(pos.w);
    if (kind == 0) { return hit_ball(ro, rd, pos.xyz, size.x, tmin); }
    if (kind == 3) { return hit_big(ro, rd, pos.xyz, objs[i].rot.xyz, size.x, tmin); }
    return NO_HIT;
}

// Cheap first check: could the ray reach object i's bounding ball closer than `best`?
fn might_hit(i: u32, ro: vec3<f32>, rd: vec3<f32>, tmin: f32, best: f32) -> bool {
    let b = bounds[i];
    let oc = b.xyz - ro;
    let tca = dot(oc, rd);
    let q = oc - tca * rd;
    return dot(q, q) <= b.w * b.w && tca + b.w > tmin && tca - b.w < best;
}

fn normal_at(i: u32, p: vec3<f32>) -> vec3<f32> {
    let o = objs[i];
    if (i32(o.pos.w) == 3) { return normalize((p - o.pos.xyz) + o.size.x * o.rot.xyz); }
    return normalize(p - o.pos.xyz);
}

struct Hit { t: f32, id: u32 };

// Entry distance into a box (slab method), or NO_HIT.
fn box_t(ro: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let t1 = (lo - ro) * inv;
    let t2 = (hi - ro) * inv;
    let tn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tf = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tf < max(tn, 0.0)) { return NO_HIT; }
    return max(tn, 0.0);
}

// Nearest object along a ray, through the tree: nearer child first, skip boxes beyond the best hit.
fn closest(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Hit {
    var h = Hit(tmax, NONE);
    if (u.jitter.z < 0.5) { return h; }
    let tmin = 0.25 * eps_at(ro);
    // The camera's own rays (they start at 0) pass through a solid the camera is inside.
    let skip = select(NONE, u32(u.misc.w) - 1u, u.misc.w > 0.5 && dot(ro, ro) == 0.0);
    // Few objects: testing them all is cheaper than walking the tree.
    let n = u32(u.cam_up.w);
    if (n <= 32u) {
        for (var i = 0u; i < n; i = i + 1u) {
            if (i == skip || !might_hit(i, ro, rd, tmin, h.t)) { continue; }
            let t = hit_obj(i, ro, rd, tmin);
            if (t < h.t) { h = Hit(t, i); }
        }
        return h;
    }
    let safe = select(rd, vec3(1e-9), abs(rd) < vec3(1e-9));
    let inv = 1.0 / safe;
    var stack: array<u32, 24>;
    var sp = 1;
    stack[0] = 0u;
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let lo = bvh[2u * ni];
        let hi = bvh[2u * ni + 1u];
        if (box_t(ro, inv, lo.xyz, hi.xyz) >= h.t) { continue; }
        let first = bitcast<u32>(lo.w);
        let count = bitcast<u32>(hi.w);
        if (count > 0u) {
            for (var k = 0u; k < count; k = k + 1u) {
                let i = prims[first + k];
                if (i == skip || !might_hit(i, ro, rd, tmin, h.t)) { continue; }
                let t = hit_obj(i, ro, rd, tmin);
                if (t < h.t) { h = Hit(t, i); }
            }
        } else {
            let a = first;
            let b = first + 1u;
            let ta = box_t(ro, inv, bvh[2u * a].xyz, bvh[2u * a + 1u].xyz);
            let tb = box_t(ro, inv, bvh[2u * b].xyz, bvh[2u * b + 1u].xyz);
            // Push the farther child first so the nearer one is opened next.
            if (ta <= tb) {
                if (tb < h.t && sp < 24) { stack[sp] = b; sp = sp + 1; }
                if (ta < h.t && sp < 24) { stack[sp] = a; sp = sp + 1; }
            } else {
                if (ta < h.t && sp < 24) { stack[sp] = a; sp = sp + 1; }
                if (tb < h.t && sp < 24) { stack[sp] = b; sp = sp + 1; }
            }
        }
    }
    return h;
}

// Light passing heavy objects: a straight piece of path from ray_o to the next hit is turned
// toward each heavy object by 2 G M / b * (sin at the end - sin at the start), measured from the
// closest-approach point, times (1 + 15 pi/16 G M / b * share of a full pass) for the next-order
// term. All turns are added and applied once per piece (thin-lens method). A full pass gives 4 G M / b.
var<private> ray_o: vec3<f32>;
var<private> ray_d: vec3<f32>;

fn trace(bend: bool) -> Hit {
    gr_col = vec3(0.0);
    gr_trans = 1.0;
    let h = closest(ray_o, ray_d, NO_HIT);
    // The exact zone of a black hole on the way: follow the Kerr path through it (rt_hole.wgsl).
    let en = hole_entry(h.t);
    if (en.id != NONE) {
        let hg = gr_march(en.id, ray_o + ray_d * en.t, ray_d);
        if (hg.id != NONE) { return hg; }
        return closest(ray_o, ray_d, NO_HIT);   // left the zone: straight on from there
    }
    let nl = u32(u.info2.y);
    if (nl == 0u || !bend) { return h; }
    let tend = min(h.t, 1e15);
    var kick = vec3(0.0);
    var at_sum = 0.0;
    var ang_sum = 0.0;
    for (var k = 0u; k < nl; k = k + 1u) {
        let o = objs[u.lens[k / 4u][k % 4u]];
        let oc = obj_centre(o);
        let tc = dot(oc - ray_o, ray_d);
        let bv = oc - (ray_o + tc * ray_d);
        let b = length(bv);
        if (b < 1e-6 * length(oc - ray_o) + 1e-4) { continue; }
        let te = tend - tc;
        let f = 0.5 * (te / sqrt(te * te + b * b) + tc / sqrt(tc * tc + b * b));
        let ang = 4.0 * o.size.w / b * f * (1.0 + 15.0 * PI / 16.0 * o.size.w / b * f);
        var at = tc;
        if (tc <= 0.0) { at = 0.0; } else if (tc >= tend) { at = 0.5 * tend; }
        kick = kick + ang * (bv / b);
        at_sum = at_sum + ang * at;
        ang_sum = ang_sum + ang;
    }
    let k = length(kick);
    if (k < 1e-4) { return h; }
    ray_o = ray_o + (at_sum / ang_sum) * ray_d;
    ray_d = normalize(cos(k) * ray_d + sin(k) * (kick / k));
    return closest(ray_o, ray_d, NO_HIT);
}

// Sky: the star field (hashed points, a faint Milky Way band).
fn hash3(p: vec3<f32>) -> f32 {
    var q = vec3<u32>(bitcast<vec3<u32>>(p));
    var h = q.x * 0x9E3779B1u ^ q.y * 0x85EBCA77u ^ q.z * 0xC2B2AE3Du;
    h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
    h = h ^ (h >> 12u);
    return f32(h) / 4294967296.0;
}
// Background stars: at most one per cell of a 2048 x 2048 grid on each face of a cube round the
// camera, the grid even in angle (equi-angular cube map: cells 0.044 degrees across, within a
// factor 1.4 everywhere; 1 cell in 1250 has a star). Each star is drawn as a small soft spot of
// fixed total light, at least about one traced pixel wide, found by looking at the cells round
// the ray, so a star doesn't flicker when traced pixels are several cells wide.
const STAR_N: f32 = 2048.0;
const QPI: f32 = 0.78539816;   // pi / 4: half a cube face, as an angle
fn star_face(d: vec3<f32>, ax: i32, sg: f32, sig: f32, r: i32) -> vec3<f32> {
    var m = d.x; var a = d.y; var b = d.z;
    if (ax == 1) { m = d.y; a = d.z; b = d.x; } else if (ax == 2) { m = d.z; a = d.x; b = d.y; }
    m = m * sg;
    if (m <= 1e-3) { return vec3(0.0); }
    let g = (atan(vec2(a, b) / m) / QPI * 0.5 + 0.5) * STAR_N;
    let ci = floor(g);
    let fid = f32(ax * 2) + select(0.0, 1.0, sg > 0.0);
    var c = vec3(0.0);
    let flux = 1.0 / (1500.0 * 1500.0);   // the light of one star of colour value 1
    let k2 = -0.5 / (sig * sig);
    for (var i = -r; i <= r; i = i + 1) {
        for (var j = -r; j <= r; j = j + 1) {
            let cc = ci + vec2(f32(i), f32(j));
            if (cc.x < 0.0 || cc.y < 0.0 || cc.x >= STAR_N || cc.y >= STAR_N) { continue; }
            let h = hash3(vec3(cc, fid));
            if (h <= 0.9992) { continue; }
            let kk = (h - 0.9992) / 0.0008;
            let t = hash3(vec3(cc, fid + 7.0));
            let off = vec2(hash3(vec3(cc, fid + 13.0)), hash3(vec3(cc, fid + 19.0))) * 0.8 + 0.1;
            let uv = tan(((cc + off) / STAR_N * 2.0 - 1.0) * QPI);
            var sd = vec3(sg, uv.x, uv.y);
            if (ax == 1) { sd = vec3(uv.y, sg, uv.x); } else if (ax == 2) { sd = vec3(uv.x, uv.y, sg); }
            let e = 2.0 - 2.0 * dot(d, normalize(sd));   // squared angle to the star
            let col = mix(vec3(1.0, 0.75, 0.55), vec3(0.65, 0.78, 1.0), t) * (0.05 + 1.5 * kk * kk * kk);
            c = c + col * exp(e * k2);
        }
    }
    return c * flux / (6.2831853 * sig * sig);
}
fn stars(d: vec3<f32>) -> vec3<f32> {
    // Spot size: 0.6 traced pixel while the view moves (the trace is coarser than the window),
    // 0.5 window pixel when still (the window picture then builds up from many jittered traces).
    let pa = 2.0 * u.cam_pos.w / max(u.info.y, 1.0);
    let pw = 2.0 * u.cam_pos.w / max(u.up.y, 1.0);
    let cell = 2.0 * QPI / STAR_N;          // a cell's angle along a face's middle lines
    let sig = clamp(select(0.6 * pa, 0.5 * pw, u.info2.z < 0.05), 0.35 * cell, 1.5 * cell);
    // Cells looked at: out to 3 sigma where cells are smallest (0.71 cell, at the corners).
    let r = i32(ceil(3.0 * sig / (0.71 * cell)));
    let ad = abs(d);
    var ax = 0;
    if (ad.y > ad.x && ad.y >= ad.z) { ax = 1; } else if (ad.z > ad.x && ad.z > ad.y) { ax = 2; }
    var c = vec3(0.0);
    // The ray's own cube face, and the faces next to it when the ray is within reach of an edge
    // (so stars near a seam are whole).
    let big = max(max(ad.x, ad.y), ad.z);
    let near = QPI - f32(r + 1) * cell;
    for (var k = 0; k < 3; k = k + 1) {
        let v = select(select(ad.z, ad.y, k == 1), ad.x, k == 0);
        if (k == ax || atan(v / big) > near) {
            let sg = select(-1.0, 1.0, select(select(d.z, d.y, k == 1), d.x, k == 0) > 0.0);
            c = c + star_face(d, k, sg, sig, r);
        }
    }
    // Milky Way: a soft band round a tilted circle.
    let g = dot(d, normalize(vec3(0.3, 0.85, -0.42)));
    c = c + vec3(0.004, 0.004, 0.005) * exp(-g * g * 30.0);
    return c;
}
fn sky(d: vec3<f32>) -> vec3<f32> { return stars(d); }

fn cos_dir(n: vec3<f32>) -> vec3<f32> {
    let a = 2.0 * PI * rnd();
    let r = sqrt(rnd());
    let up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * (r * cos(a)) + b * (r * sin(a)) + n * sqrt(max(1.0 - r * r, 0.0)));
}
// Light reaching point p (normal n) straight from one randomly picked lamp, times the matte rule.
fn lamp_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let nl = i32(u.info.w);
    if (nl == 0) { return vec3(0.0); }
    let k = min(i32(rnd() * f32(nl)), nl - 1);
    return light_from(p, n, u.lamps[k / 4][k % 4], f32(nl));
}

// Light from lamp li on a surface at p facing n (one point on the lamp, with a shadow ray),
// times `share` (the number of lamps when one is picked at random).
fn light_from(p: vec3<f32>, n: vec3<f32>, li: u32, share: f32) -> vec3<f32> {
    let lo = objs[li];
    let to = obj_centre(lo) - p;
    let d2 = dot(to, to);
    let r = lo.size.x;
    if (d2 <= r * r) { return vec3(0.0); }
    let cmax = sqrt(1.0 - r * r / d2);
    let cth = 1.0 - rnd() * (1.0 - cmax);
    let sth = sqrt(max(1.0 - cth * cth, 0.0));
    let a = 2.0 * PI * rnd();
    let w = normalize(to);
    let up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(w.y) > 0.9);
    let t = normalize(cross(up, w));
    let b = cross(w, t);
    let dir = t * (sth * cos(a)) + b * (sth * sin(a)) + w * cth;
    let cosn = dot(dir, n);
    if (cosn <= 0.0) { return vec3(0.0); }
    // Shadow ray: anything else in the way (a black hole too) blocks it.
    let h = closest(p + n * eps_at(p), dir, NO_HIT);
    if (h.id != li) { return vec3(0.0); }
    // (albedo / pi) * L * cos / pdf, pdf = 1 / (2 pi (1 - cmax)); times number of lamps.
    return lo.mat.rgb * cosn * 2.0 * (1.0 - cmax) * share;
}

struct G { illum: vec3<f32>, emit: vec3<f32>, alb: vec3<f32>, n: vec3<f32>, depth: f32, id: u32 };

fn radiance(ro0: vec3<f32>, rd0: vec3<f32>) -> G {
    ray_o = ro0;
    ray_d = rd0;
    var g = G(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), SKY_DEPTH, NONE);
    var found = false;          // first matte surface found (the clean-up's surface)
    var thr = vec3(1.0);        // before `found`: real throughput; after: throughput / its colour
    var count_lamps = true;
    var path = 0.0;             // length of the path to the surface (for depth)
    let maxb = i32(u.info2.x);
    for (var bounce = 0; bounce < maxb; bounce = bounce + 1) {
        let start = ray_o;
        // Light bending is followed for the first few pieces of each path (jitter.w of them):
        // later bounces carry little light, and each bent piece costs a second search.
        let h = trace(f32(bounce) < u.jitter.w);
        // Light from a black hole's hot gas met on the way (and how much it hides behind it).
        if (found) { g.illum = g.illum + thr * gr_col; } else { g.emit = g.emit + thr * gr_col; }
        thr = thr * gr_trans;
        if (h.id == NONE) {
            let c = thr * sky(ray_d);
            if (found) { g.illum = g.illum + c; } else { g.emit = g.emit + c; }
            break;
        }
        let o = objs[h.id];
        let p = ray_o + ray_d * h.t;
        if (!found) { path = path + distance(start, p); }
        let m = i32(o.mat.w);
        if (m == 5) { break; }   // fell into a black hole (or hidden by its opaque gas)
        let n = normal_at(h.id, p);
        if (m == 3) {
            if (count_lamps) {
                let c = thr * o.mat.rgb;
                if (found) { g.illum = g.illum + c; } else { g.emit = g.emit + c; }
            }
            if (!found) { g.n = n; g.depth = path; g.id = h.id; }
            break;
        }
        let nf = select(-n, n, dot(ray_d, n) < 0.0);
        if (!found) {
            found = true;
            g.alb = thr * o.mat.rgb;
            g.n = nf; g.depth = path; g.id = h.id;
            thr = vec3(1.0);
        } else {
            thr = thr * o.mat.rgb;
        }
        // Lamp light is added with the colour already taken out (thr is relative to it).
        g.illum = g.illum + thr * lamp_light(p, nf);
        ray_d = cos_dir(nf);
        count_lamps = false;
        ray_o = p + nf * eps_at(p);
        // Russian roulette after 2 bounces (keeps the average right, saves work).
        if (bounce >= 2) {
            let tm = max(thr.x, max(thr.y, thr.z));
            let keep = select(1.0, clamp(tm, 0.05, 1.0), tm < 0.5);
            if (rnd() > keep) { break; }
            thr = thr / keep;
        }
    }
    return g;
}

fn bad3(c: vec3<f32>) -> bool {
    let b = (bitcast<vec3<u32>>(c) & vec3(0x7f800000u)) == vec3(0x7f800000u);
    return any(b);
}

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (f32(gid.x) >= u.info.x || f32(gid.y) >= u.info.y) { return; }
    let px = vec2<f32>(gid.xy);
    seed = gid.x * 1973u + gid.y * 9277u + u32(u.cam_right.w) * 26699u;
    seed = seed ^ (seed >> 16u);
    let d = pixel_dir(px + 0.5 + u.jitter.xy);
    var g = radiance(u.cam_pos.xyz, d);
    // A NaN or infinity in one pixel would spread through the blur and glow passes: count it as black.
    if (bad3(g.illum)) { g.illum = vec3(0.0); }
    if (bad3(g.emit)) { g.emit = vec3(0.0); }
    g.emit = min(g.emit, vec3(60000.0));   // stored as 16-bit floats (largest 65504)
    // Keep single very bright paths ("fireflies") from blowing out the picture.
    let l = max(g.illum.x, max(g.illum.y, g.illum.z));
    if (l > 32.0) { g.illum = g.illum * (32.0 / l); }
    let id = select(0.0, f32(g.id + 1u), g.id != NONE);
    textureStore(out_illum, gid.xy, vec4(g.illum, 1.0));
    textureStore(out_emit, gid.xy, vec4(g.emit, 1.0));
    textureStore(out_alb, gid.xy, vec4(clamp(g.alb, vec3(0.0), vec3(1.0)), 1.0));
    textureStore(out_geo, gid.xy, vec4(oct_enc(g.n + vec3(0.0, 0.0, 1e-9)), g.depth, id));
}
