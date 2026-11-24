// Black holes inside the one renderer. Far from a hole, light bends by the weak-field formula
// (trace() in rt.wgsl). Inside a sphere of GR_R Schwarzschild radii, the ray instead follows the
// exact Kerr path (Cartesian Kerr-Schild coordinates, Hamiltonian form from GRay2, Chan et al.
// 2018; CPU copy in kerr.rs),
// including the hot gas disk (thick, Doppler-shifted, lit by its own heat).
//
// A hole is stored as a big ball whose radius is the horizon, with mat = (a, inner disk edge,
// outer disk edge, 5), all in Schwarzschild radii (a = spin * M, M = 0.5), size.w = G M / c^2 in
// metres, ppos.w = horizon radius in Schwarzschild radii. Its spin axis is the world y axis.

const GR_R: f32 = 20.0;
const GR_STEPS: i32 = 220;
const KM: f32 = 0.5;            // M in Schwarzschild radii
const STEP_K: f32 = 0.15;       // step = 0.15 r
const DISK_T: f32 = 5200.0;     // display temperature at the hottest ring (K)
const DISK_HR: f32 = 0.05;      // thickness: scale height / radius
const DISK_BRIGHT: f32 = 1.1;
const SWIRL_P: f32 = 8.0;       // seconds each copy of the swirl turns before it is faded out

// Light picked up from disk gas on the last exact stretch, and how much light from behind still
// gets through (read by radiance()).
var<private> gr_col: vec3<f32>;
var<private> gr_trans: f32;

// World axes (y up) <-> hole axes (Z = spin axis).
fn to_ks(v: vec3<f32>) -> vec3<f32> { return vec3(v.z, v.x, v.y); }
fn from_ks(v: vec3<f32>) -> vec3<f32> { return vec3(v.y, v.z, v.x); }

fn ks_r(x: vec3<f32>, a: f32) -> f32 {
    let a2 = a * a;
    let aa = dot(x, x) - a2;
    let q = sqrt(aa * aa + 4.0 * a2 * x.z * x.z);
    return sqrt(max(0.5 * (aa + q), 1e-12));
}
fn ks_l(x: vec3<f32>, r: f32, a: f32) -> vec3<f32> {
    let d = r * r + a * a;
    return vec3((r * x.x - a * x.y) / d, (r * x.y + a * x.x) / d, x.z / r);
}
fn ks_f(x: vec3<f32>, r: f32, a: f32) -> f32 {
    let r2 = r * r;
    return 2.0 * KM * r * r2 / (r2 * r2 + a * a * x.z * x.z);
}
fn gdot(x: vec3<f32>, a: f32, u4: vec4<f32>, v4: vec4<f32>) -> f32 {
    let r = ks_r(x, a);
    let l = ks_l(x, r, a);
    let f = ks_f(x, r, a);
    let lu = -u4.w + dot(l, u4.xyz);
    let lv = -v4.w + dot(l, v4.xyz);
    return -u4.w * v4.w + dot(u4.xyz, v4.xyz) + f * lu * lv;
}
fn orbit4(x: vec3<f32>, a: f32, w: f32) -> vec4<f32> {
    let v = vec4(-x.y * w, x.x * w, 0.0, 1.0);
    return v / sqrt(max(-gdot(x, a, v, v), 1e-12));
}
struct Deriv { dx: vec3<f32>, dp: vec3<f32> };
fn derivs(x: vec3<f32>, pt: f32, p: vec3<f32>, a: f32) -> Deriv {
    let a2 = a * a;
    let aa = dot(x, x) - a2;
    let q = max(sqrt(aa * aa + 4.0 * a2 * x.z * x.z), 1e-12);
    let r2 = max(0.5 * (aa + q), 1e-12);
    let r = sqrt(r2);
    let s = r2 * r2 + a2 * x.z * x.z;
    let f = 2.0 * KM * r * r2 / s;
    let d = r2 + a2;
    let ao = -a;
    let l = vec3((r * x.x + ao * x.y) / d, (r * x.y - ao * x.x) / d, x.z / r);
    let gr = vec3(x.x * r2, x.y * r2, x.z * r2 + a2 * x.z) / (r * q);
    let ll = pt + dot(l, p);
    var o: Deriv;
    o.dx = p - f * ll * l;
    var gf = gr * (2.0 * KM * r2 * (3.0 * s - 4.0 * r2 * r2) / (s * s));
    gf.z = gf.z - 4.0 * KM * r2 * r * a2 * x.z / (s * s);
    let c = (x.x * p.x + x.y * p.y) / d - 2.0 * r * (l.x * p.x + l.y * p.y) / d - p.z * x.z / r2;
    let gl = gr * c + vec3((r * p.x - ao * p.y) / d, (r * p.y + ao * p.x) / d, p.z / r);
    o.dp = 0.5 * ll * ll * gf + f * ll * gl;
    return o;
}

// Black-body colour (Tanner Helland fit), linear light, max channel 1.
fn blackbody(t_in: f32) -> vec3<f32> {
    let x = clamp(t_in, 1000.0, 40000.0) / 100.0;
    var r = 1.0;
    var g = 0.0;
    var b = 1.0;
    if (x <= 66.0) { g = 0.39008158 * log(x) - 0.63184144; }
    else { r = 1.29293618 * pow(x - 60.0, -0.1332047592); g = 1.12989086 * pow(x - 60.0, -0.0755148492); }
    if (x < 66.0) { if (x <= 19.0) { b = 0.0; } else { b = 0.54320679 * log(x - 10.0) - 1.19625408; } }
    return pow(clamp(vec3(r, g, b), vec3(0.0), vec3(1.0)), vec3(2.2));
}

// Smooth value noise from hashes (computed once per step, so hashing is cheap enough).
fn vh(c: vec3<f32>) -> f32 { return hash3(c + vec3(0.5)); }
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let w = f * f * (3.0 - 2.0 * f);
    let x00 = mix(vh(i), vh(i + vec3(1.0, 0.0, 0.0)), w.x);
    let x10 = mix(vh(i + vec3(0.0, 1.0, 0.0)), vh(i + vec3(1.0, 1.0, 0.0)), w.x);
    let x01 = mix(vh(i + vec3(0.0, 0.0, 1.0)), vh(i + vec3(1.0, 0.0, 1.0)), w.x);
    let x11 = mix(vh(i + vec3(0.0, 1.0, 1.0)), vh(i + vec3(1.0, 1.0, 1.0)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

fn disk_edge(r: f32, rin: f32, rout: f32) -> f32 {
    return smoothstep(rin * 0.9, rin + 0.25, r) * (1.0 - smoothstep(rout * 0.6, rout, r));
}

// Anti-aliasing of the swirl: detail finer than the patch of gas one pixel covers (foot, in
// Schwarzschild radii, across the ray rd) can't be drawn; left in, it changes from frame to frame
// (shimmer near the inner disk, where the rings wind the swirl into thin stripes and the lens
// squeezes it). Returns how much of each octave of each copy (a: x y, b: z w) is kept; gas_at fades
// the rest to the mean. Only the noise's change across the ray counts (along the ray the gas is
// summed in small pieces). Its change per r_s: the angle pair 3 x grad(turn angle) (round the ring
// 1/r, outward the winding), the third coordinate 9/r outward and 0.8/height upward. Worked out
// once per bent-path step (pieces are at most 4% of the radius apart).
fn swirl_keep(q: vec3<f32>, a: f32, rin: f32, foot: f32, rd: vec3<f32>) -> vec4<f32> {
    let r = ks_r(q, a);
    let rr = max(r, rin);
    let sq = sqrt(KM);
    let den = rr * sqrt(rr) + a * sq;
    // (Inside the inner edge gas_at turns all the gas at the edge's speed: no winding there.)
    let dwdr = select(1.5 * sq * sqrt(rr) / (den * den), 0.0, r < rin);
    let s = u.misc.x / SWIRL_P;
    let fa = fract(s);
    let fb = fract(s + 0.5);
    let rho = max(length(q.xy), 1e-4);
    let er = vec3(q.xy / rho, 0.0);
    let ephi = vec3(-er.y, er.x, 0.0);
    let gz = er * (9.0 / rr) + vec3(0.0, 0.0, 0.8 / (DISK_HR * rr));
    let gz2 = dot(gz, gz) - dot(gz, rd) * dot(gz, rd);
    let ga = 3.0 * (ephi / rho + er * (dwdr * fa * SWIRL_P * 40.0));
    let gb = 3.0 * (ephi / rho + er * (dwdr * fb * SWIRL_P * 40.0));
    let ka = sqrt(max(dot(ga, ga) - dot(ga, rd) * dot(ga, rd) + gz2, 0.0)) * foot;
    let kb = sqrt(max(dot(gb, gb) - dot(gb, rd) * dot(gb, rd) + gz2, 0.0)) * foot;
    return clamp(vec4(1.5 - 2.0 * ka, 1.5 - 4.06 * ka, 1.5 - 2.0 * kb, 1.5 - 4.06 * kb), vec4(0.0), vec4(1.0));
}

struct Gas { col: vec3<f32>, kappa: f32 };

// Gas colour and absorption at q: thin-disk temperature with the inner
// edge at the innermost stable orbit, gas on Kerr circular orbits, redshift from the photon's
// conserved energy and angular momentum, swirl moving with the gas.
fn gas_at(q: vec3<f32>, a: f32, rin: f32, pt: f32, p: vec3<f32>, con: vec4<f32>) -> Gas {
    var o: Gas;
    let r = ks_r(q, a);
    let rr = max(r, rin);
    let f = max(1.0 - sqrt(rin / rr), 0.0);
    let prof = pow(rr / 3.0, -0.75) * pow(f, 0.25) / 0.4878;
    let sq = sqrt(KM);
    let omega = sq / (rr * sqrt(rr) + a * sq);
    let uu = orbit4(q, a, omega);
    let pphi = -q.y * p.x + q.x * p.y;
    let g = 1.0 / max(uu.w * (pt + omega * pphi), 1e-4);
    let hgt = DISK_HR * rr;
    // The swirl turns with the gas (each ring at its own speed). Turned for ever, the rings wind it
    // into ever thinner stripes, and the angle loses precision; so two copies are used, each turned
    // for at most SWIRL_P seconds from a fresh start and faded in and out in turn (a flow-map blend).
    let s = u.misc.x / SWIRL_P;
    let fa = fract(s);
    let fb = fract(s + 0.5);
    let ph = atan2(q.y, q.x);
    let za = log(rr) * 9.0 + q.z / hgt * 0.8;
    let pa = ph - omega * fa * SWIRL_P * 40.0;
    let pb = ph - omega * fb * SWIRL_P * 40.0;
    let oa = vec3(17.0, 31.0, 7.0) * (floor(s) - 64.0 * floor(floor(s) / 64.0));
    let ob = vec3(23.0, 11.0, 29.0) * (floor(s + 0.5) - 64.0 * floor(floor(s + 0.5) / 64.0)) + vec3(5.3, 2.1, 9.7);
    // con: how much of each octave of each copy is kept (swirl_keep, anti-aliasing).
    let ca1 = con.x;
    let ca2 = con.y;
    let cb1 = con.z;
    let cb2 = con.w;
    let qa = vec3(cos(pa) * 3.0, sin(pa) * 3.0, za) + oa;
    let qb = vec3(cos(pb) * 3.0, sin(pb) * 3.0, za) + ob;
    // (An octave faded out entirely is not computed.)
    var na = 0.5;
    var nb = 0.5;
    if (ca1 > 0.0) { na = na + 0.66 * ca1 * (vnoise(qa) - 0.5); }
    if (ca2 > 0.0) { na = na + 0.34 * ca2 * (vnoise(qa * 2.03 + vec3(1.7, 9.2, 4.1)) - 0.5); }
    if (cb1 > 0.0) { nb = nb + 0.66 * cb1 * (vnoise(qb) - 0.5); }
    if (cb2 > 0.0) { nb = nb + 0.34 * cb2 * (vnoise(qb * 2.03 + vec3(1.7, 9.2, 4.1)) - 0.5); }
    let wa = 1.0 - abs(2.0 * fa - 1.0);
    // Blend that keeps the noise's spread (a plain mix of two would be greyer at the halfway point).
    let wn = wa * wa + (1.0 - wa) * (1.0 - wa);
    let n = clamp(0.5 + ((na - 0.5) * wa + (nb - 0.5) * (1.0 - wa)) / sqrt(wn), 0.0, 1.0);
    // Faded noise has a smaller spread, and the streak's mean is 0.4 + 2.4 x its variance (n^3 at
    // 0.5 + e), so the lost part is added back: the gas keeps its average brightness. Value noise
    // variance 0.0342.
    let va = 0.0342 * (0.4356 * ca1 * ca1 + 0.1156 * ca2 * ca2);
    let vb = 0.0342 * (0.4356 * cb1 * cb1 + 0.1156 * cb2 * cb2);
    let vlost = 0.01886 - (wa * wa * va + (1.0 - wa) * (1.0 - wa) * vb) / wn;
    let streak = 0.2 + 1.6 * n * n * n + 2.4 * vlost;
    let i4 = pow(prof * g, 4.0);
    o.col = blackbody(DISK_T * prof * g) * DISK_BRIGHT * (i4 / (1.0 + i4 / 4.0)) * streak;
    let tau_v = (0.3 + 2.2 * sqrt(prof)) * (0.55 + 0.45 * n);
    o.kappa = tau_v / (hgt * 2.5066);
    return o;
}

// Follow the ray through hole `hi` from its entry point (world, camera-relative) `xw` along
// `dw`. Returns a hit (the horizon, or an object the bent path runs into: then ray_o/ray_d are
// set so that ray_o + ray_d * t is the hit point), or NONE with ray_o/ray_d set to where and
// which way the light leaves the sphere. Disk light goes to gr_col / gr_trans.
fn gr_march(hi: u32, xw: vec3<f32>, dw: vec3<f32>) -> Hit {
    let ho = objs[hi];
    let a = ho.mat.x;
    let rin = ho.mat.y;
    let rout = ho.mat.z;
    let rh = ho.ppos.w;
    let rs = 2.0 * ho.size.w;
    let c = obj_centre(ho);
    var x = to_ks((xw - c) / rs);
    let d = normalize(to_ks(dw));
    // Local observer that doesn't turn with the hole (zero angular momentum).
    let xt = vec4(0.0, 0.0, 0.0, 1.0);
    let xp = vec4(-x.y, x.x, 0.0, 0.0);
    let w = -gdot(x, a, xt, xp) / max(gdot(x, a, xp, xp), 1e-12);
    let uo = orbit4(x, a, w);
    var e = vec4(d, 0.0);
    e = e + gdot(x, a, e, uo) * uo;
    e = e / sqrt(max(gdot(x, a, e, e), 1e-12));
    let pc = -uo + e;
    let r0 = ks_r(x, a);
    let l0 = ks_l(x, r0, a);
    let f0 = ks_f(x, r0, a);
    let lp = -pc.w + dot(l0, pc.xyz);
    let pt = -pc.w - f0 * lp;
    var p = pc.xyz + f0 * lp * l0;
    // Objects inside the sphere (up to 8, nearest the hole first is not needed: all are tested
    // on every step). Usually there are none.
    var inside: array<u32, 8>;
    var ni = 0u;
    let n = u32(u.cam_up.w);
    for (var k = 0u; k < n; k = k + 1u) {
        if (k == hi || ni >= 8u) { continue; }
        let b = bounds[k];
        if (length(b.xyz - c) < GR_R * rs + b.w) { inside[ni] = k; ni = ni + 1u; }
    }
    var last = d;
    // For the swirl's anti-aliasing (swirl_keep): the size of the patch one pixel covers, taken
    // as the pixel's angle times the distance gone (as in flat space; the lens's stretching and
    // squeezing near the hole is left out).
    // Which pixel counts: while the view turns or the camera moves by more than a window pixel a
    // frame, the clean-up keeps only a few frames, so the traced pixel (0.6 of it, as for the
    // stars); otherwise the window picture is built from many jittered traces, which already
    // averages detail finer than a window pixel, so half a window pixel (the still picture keeps
    // its fine rings). The bodies moving does not count here: the camera's view is what matters.
    let pwin = 2.0 * u.cam_pos.w / max(u.up.y, 1.0);
    let turn = length(u.cam_fwd.xyz - u.prev_fwd.xyz) + length(u.cam_up.xyz - u.prev_up.xyz)
        + length(u.prev_pos.xyz) / max(length(c), 1e-6);
    let pix = select(0.5 * pwin, 0.6 * 2.0 * u.cam_pos.w / max(u.info.y, 1.0), turn > pwin);
    var gone = length(xw) / rs;
    for (var i = 0; i < GR_STEPS; i = i + 1) {
        let r = ks_r(x, a);
        let k1 = derivs(x, pt, p, a);
        last = k1.dx;
        if (r < rh + 0.01) {
            ray_o = c + rs * from_ks(x);
            ray_d = normalize(from_ks(k1.dx));
            return Hit(0.0, hi);
        }
        let outward = dot(x, k1.dx) > 0.0;
        if (length(x) > GR_R && outward) { break; }
        var ds = STEP_K * r;
        // (Longer steps for outgoing rays far out would start at a different place in each pixel
        // column, and the thin gas above the disk shows that as vertical bars.)
        if (r < 3.0) { ds = ds * 0.5; }
        let hstep = ds / max(length(k1.dx), 1e-12);
        var sx = k1.dx;
        var sp = k1.dp;
        let k2 = derivs(x + 0.5 * hstep * k1.dx, pt, p + 0.5 * hstep * k1.dp, a);
        sx = sx + 2.0 * k2.dx; sp = sp + 2.0 * k2.dp;
        let k3 = derivs(x + 0.5 * hstep * k2.dx, pt, p + 0.5 * hstep * k2.dp, a);
        sx = sx + 2.0 * k3.dx; sp = sp + 2.0 * k3.dp;
        let k4 = derivs(x + hstep * k3.dx, pt, p + hstep * k3.dp, a);
        let xn = x + (hstep / 6.0) * (sx + k4.dx);
        let pn = p + (hstep / 6.0) * (sp + k4.dp);
        let seg = length(xn - x);
        // Objects on this piece of the bent path.
        var tmax = 1.0;
        var hit = Hit(NO_HIT, NONE);
        if (ni > 0u) {
            let sw = c + rs * from_ks(x);
            let dirw = normalize(from_ks(xn - x));
            let lw = rs * seg;
            let tmin = 0.25 * eps_at(sw);
            var bt = lw;
            for (var k = 0u; k < ni; k = k + 1u) {
                let j = inside[k];
                if (!might_hit(j, sw, dirw, tmin, bt)) { continue; }
                let t = hit_obj(j, sw, dirw, tmin);
                if (t < bt) { bt = t; hit = Hit(t, j); }
            }
            if (hit.id != NONE) { tmax = bt / lw; }
        }
        // Thick disk: march through the gas layer in small pieces, in order along the ray.
        // The gas layer is followed out to 5.5 scale heights (density 3e-7 of the middle's); a
        // lower cut shows as a hard top edge that moves with each bent-path step.
        let zmax = 5.5 * DISK_HR * max(max(length(x.xy), length(xn.xy)), 1.0);
        var lo = 0.0;
        var hi2 = 0.0;
        let dz = xn.z - x.z;
        if (abs(dz) < 1e-7) {
            if (abs(x.z) < zmax) { hi2 = 1.0; }
        } else {
            let ta = (-zmax - x.z) / dz;
            let tb = (zmax - x.z) / dz;
            lo = max(0.0, min(ta, tb));
            hi2 = min(1.0, max(ta, tb));
        }
        hi2 = min(hi2, tmax);
        // Nearest the piece comes to the axis (in the disk's plane): a piece can dip inside the
        // disk's edge between its ends.
        let sxy = xn.xy - x.xy;
        let txy = clamp(-dot(x.xy, sxy) / max(dot(sxy, sxy), 1e-12), 0.0, 1.0);
        let dxy = length(x.xy + txy * sxy);
        if (rout > 0.0 && hi2 > lo && dxy < rout + zmax) {
            var tc = lo;
            if (abs(dz) >= 1e-7) { tc = clamp(-x.z / dz, lo, hi2); }
            let qc = mix(x, xn, tc);
            let rc = ks_r(qc, a);
            let hgt = DISK_HR * max(rc, rin);
            let zc = qc.z / hgt;
            // No edge factor here: qc is one point of the piece, and at the disk's outer edge it can
            // lie just outside while the rest of the piece is inside.
            let weight = exp(-0.5 * zc * zc) * (hi2 - lo) * seg / hgt;
            // The gate only skips steps whose gas cannot show.
            if (weight > 1e-5) {
                // Pieces no longer than half the gas layer's thickness and 4% of the radius, and
                // the gas's colour and swirl looked up in each piece (one look-up per bent-path
                // step shows as streaks).
                let len = (hi2 - lo) * seg;
                let n_sub = clamp(i32(ceil(max(len / (0.5 * hgt), len / (0.04 * max(rc, rin))))), 1, 24);
                let dt = (hi2 - lo) / f32(n_sub);
                let foot = pix * (gone + seg * tc);
                let con = swirl_keep(qc, a, rin, foot, normalize(xn - x));
                for (var k = 0; k < n_sub; k = k + 1) {
                    let tk = lo + (f32(k) + 0.5) * dt;
                    let q = mix(x, xn, tk);
                    let rq = ks_r(q, a);
                    let zz = q.z / (DISK_HR * max(rq, rin));
                    let dens = exp(-0.5 * zz * zz) * disk_edge(rq, rin, rout);
                    if (dens < 1e-6) { continue; }
                    let gas = gas_at(q, a, rin, pt, mix(p, pn, tk), con);
                    let tau = gas.kappa * dens * seg * dt;
                    let ab = 1.0 - exp(-tau);
                    gr_col = gr_col + gr_trans * ab * gas.col;
                    gr_trans = gr_trans * (1.0 - ab);
                }
            }
        }
        if (hit.id != NONE) {
            ray_o = c + rs * from_ks(x);
            ray_d = normalize(from_ks(xn - x));
            return hit;
        }
        if (gr_trans < 0.004) {
            // Opaque gas: nothing behind it shows.
            ray_o = c + rs * from_ks(xn);
            ray_d = normalize(from_ks(xn - x));
            return Hit(0.0, hi);
        }
        gone = gone + seg;
        x = xn;
        p = pn;
    }
    ray_o = c + rs * from_ks(x);
    ray_d = normalize(from_ks(last));
    return Hit(NO_HIT, NONE);
}

// Is the straight piece from ray_o (length tend) entering the exact zone of a hole? Returns the
// hole's index and the entry distance (x), or NONE.
struct Entry { id: u32, t: f32 };
fn hole_entry(tend: f32) -> Entry {
    var best = Entry(NONE, tend);
    let nl = u32(u.info2.y);
    for (var k = 0u; k < nl; k = k + 1u) {
        let i = u.lens[k / 4u][k % 4u];
        let o = objs[i];
        if (i32(o.mat.w) != 5) { continue; }
        let c = obj_centre(o);
        let rad = GR_R * 2.0 * o.size.w;
        let oc = ray_o - c;
        let b = dot(oc, ray_d);
        let q = oc - b * ray_d;
        let h = rad * rad - dot(q, q);
        if (h < 0.0) { continue; }
        let s = sqrt(h);
        var t = -b - s;
        if (dot(oc, oc) < rad * rad) { t = 0.0; }
        if (t >= 0.0 && t < best.t) { best = Entry(i, t); }
    }
    return best;
}
