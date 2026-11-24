// Molecules: electrons in many-atom systems, solved with the `hartree` crate (pure Rust
// Hartree-Fock). This file holds the molecule list and the bridge to hartree, at the small STO-3G
// basis.
//
// Units: positions in Bohr, energies in hartree.

use hartree::basis::{BasisSet, ShellData};
use hartree::{Atom, Element, Job, JobOptions, Method, Molecule};

pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529_177_210_903; // CODATA 2018 Bohr radius
const BASIS: &str = "sto-3g";

/// Atoms (symbol, position in Bohr). Every molecule here is neutral with all electrons paired.
#[derive(Clone, Debug)]
pub struct Mol {
    pub atoms: Vec<(String, [f64; 3])>,
}

impl Mol {
    fn hartree(&self) -> Molecule {
        Molecule::new(self.atoms.iter().map(|(s, p)| Atom::new(Element::from_symbol(s).expect("element"), *p)).collect(), 0, 1)
    }
}

#[derive(Clone, Debug)]
pub struct Solved {
    pub mol: Mol,
    pub energy: f64,
    /// Filled orbitals (two electrons each).
    pub n_occ: usize,
    pub n_orbitals: usize,
    pub orbital_energies: Vec<f64>, // hartree
    pub mo_coeff: Vec<f64>,         // n_basis x n_orbitals, row-major (hartree's layout)
    /// The basis functions at this shape (for drawing).
    pub shells: Vec<ShellData>,
}

/// Find the shape of lowest energy from `mol`'s shape (hartree's optimiser with its own tight
/// stopping rule: largest force 3e-6 hartree/Bohr), returning the atoms there, the energy and the
/// number of steps.
pub fn optimise(mol: &Mol) -> Result<(Mol, f64, usize), String> {
    use hartree::opt::{optimize, OptOptions};
    let m = mol.hartree();
    let mut surface = hartree::HfSurface::new(&m, BASIS, hartree::scf::Reference::Rhf)?;
    let r = optimize(&m, &mut surface, &OptOptions::default()).map_err(|e| format!("{e:?}"))?;
    if !r.converged {
        return Err(format!("the shape did not settle in {} steps", r.iterations));
    }
    let mut out = mol.clone();
    for (a, p) in out.atoms.iter_mut().zip(&r.positions) {
        a.1 = *p;
    }
    Ok((out, r.energy, r.iterations))
}

pub fn solve(mol: &Mol) -> Result<Solved, String> {
    let job = Job { molecule: mol.hartree(), basis: BASIS.into(), method: Method::Rhf, options: JobOptions::default() };
    let r = job.run()?;
    let shells = BasisSet::load(BASIS).map_err(|e| e.to_string())?.build(&mol.hartree()).map_err(|e| e.to_string())?.shells().to_vec();
    Ok(Solved {
        mol: mol.clone(),
        energy: r.scf.energy,
        n_occ: r.scf.n_alpha,
        n_orbitals: r.scf.n_orbitals,
        orbital_energies: r.scf.orbital_energies_alpha.clone(),
        mo_coeff: r.scf.mo_coeff_alpha.clone(),
        shells,
    })
}

// The molecules on offer. Each starts from a rough shape: bond lengths are sums of single-bond
// covalent radii (Cordero et al., Dalton Trans. 2008, 2832) and angles are the electron-pair ones.

pub struct MolInfo {
    pub key: &'static str,  // CCCBDB formula
    pub name: &'static str,
}

pub const MOLECULES: [MolInfo; 13] = [
    MolInfo { key: "H2", name: "hydrogen" },
    MolInfo { key: "LiH", name: "lithium hydride" },
    MolInfo { key: "HF", name: "hydrogen fluoride" },
    MolInfo { key: "H2O", name: "water" },
    MolInfo { key: "NH3", name: "ammonia" },
    MolInfo { key: "CH4", name: "methane" },
    MolInfo { key: "N2", name: "nitrogen" },
    MolInfo { key: "CO", name: "carbon monoxide" },
    MolInfo { key: "CO2", name: "carbon dioxide" },
    MolInfo { key: "HCN", name: "hydrogen cyanide" },
    MolInfo { key: "C2H2", name: "acetylene" },
    MolInfo { key: "C2H4", name: "ethylene" },
    MolInfo { key: "CH2O", name: "formaldehyde" },
];

fn radius(s: &str) -> f64 {
    match s {
        "H" => 0.31,
        "Li" => 1.28,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        _ => 1.0,
    }
}

/// The rough starting shape (Bohr) for one of MOLECULES.
pub fn rough_shape(key: &str) -> Option<Mol> {
    let r = |a: &str, b: &str| (radius(a) + radius(b)) * BOHR_PER_ANGSTROM;
    let at = |s: &str, p: [f64; 3]| (s.to_string(), p);
    let (s30, c30) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
    let atoms = match key {
        "H2" => vec![at("H", [0.0, 0.0, 0.0]), at("H", [0.0, 0.0, r("H", "H")])],
        "LiH" => vec![at("Li", [0.0, 0.0, 0.0]), at("H", [0.0, 0.0, r("Li", "H")])],
        "HF" => vec![at("F", [0.0, 0.0, 0.0]), at("H", [0.0, 0.0, r("H", "F")])],
        "H2O" => {
            let d = r("O", "H");
            let h = 109.4712f64.to_radians() / 2.0;
            vec![at("O", [0.0, 0.0, 0.0]), at("H", [0.0, d * h.sin(), -d * h.cos()]), at("H", [0.0, -d * h.sin(), -d * h.cos()])]
        }
        "NH3" | "CH4" => {
            // Tetrahedron directions; ammonia keeps three of the four.
            let c = if key == "NH3" { "N" } else { "C" };
            let d = r(c, "H");
            let k = 1.0 / 3f64.sqrt();
            let dirs = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
            let n = if key == "NH3" { 3 } else { 4 };
            let mut v = vec![at(c, [0.0, 0.0, 0.0])];
            for dd in dirs.iter().take(n) {
                v.push(at("H", [dd[0] * k * d, dd[1] * k * d, dd[2] * k * d]));
            }
            v
        }
        "N2" => vec![at("N", [0.0, 0.0, 0.0]), at("N", [0.0, 0.0, r("N", "N")])],
        "CO" => vec![at("C", [0.0, 0.0, 0.0]), at("O", [0.0, 0.0, r("C", "O")])],
        "CO2" => vec![at("C", [0.0, 0.0, 0.0]), at("O", [0.0, 0.0, r("C", "O")]), at("O", [0.0, 0.0, -r("C", "O")])],
        "HCN" => vec![at("C", [0.0, 0.0, 0.0]), at("H", [0.0, 0.0, -r("H", "C")]), at("N", [0.0, 0.0, r("C", "N")])],
        "C2H2" => {
            let (cc, ch) = (r("C", "C"), r("C", "H"));
            vec![at("C", [0.0, 0.0, cc / 2.0]), at("C", [0.0, 0.0, -cc / 2.0]), at("H", [0.0, 0.0, cc / 2.0 + ch]), at("H", [0.0, 0.0, -cc / 2.0 - ch])]
        }
        "C2H4" => {
            let (cc, ch) = (r("C", "C"), r("C", "H"));
            let z = cc / 2.0 + ch * s30;
            vec![at("C", [0.0, 0.0, cc / 2.0]), at("C", [0.0, 0.0, -cc / 2.0]),
                 at("H", [0.0, ch * c30, z]), at("H", [0.0, -ch * c30, z]), at("H", [0.0, ch * c30, -z]), at("H", [0.0, -ch * c30, -z])]
        }
        "CH2O" => {
            let (co, ch) = (r("C", "O"), r("C", "H"));
            vec![at("O", [0.0, 0.0, co]), at("C", [0.0, 0.0, 0.0]), at("H", [0.0, ch * c30, -ch * s30]), at("H", [0.0, -ch * c30, -ch * s30])]
        }
        _ => return None,
    };
    Some(Mol { atoms })
}

// The electron cloud for drawing (the Atoms tab's dots): density or one orbital at given points.

/// Cube holding the atoms with `pad` Bohr to spare on every side.
pub fn cube_for(m: &Mol, pad: f64) -> ([f64; 3], f64) {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for (_, p) in &m.atoms {
        for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); }
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let half = (0..3).map(|k| (hi[k] - lo[k]) / 2.0).fold(0.0, f64::max) + pad;
    (c, half)
}

/// Values of every orbital asked for at `pts` (rows: points; columns: `cols` of the orbitals).
fn orbitals_at(shells: &[ShellData], c: &[f64], m: usize, cols: &[usize], pts: &[[f64; 3]]) -> Vec<f64> {
    use rayon::prelude::*;
    let nao: usize = shells.iter().map(|s| hartree::dft::ao::n_func(s.l as usize, s.spherical)).sum();
    let k = cols.len();
    let mut out = vec![0.0; pts.len() * k];
    out.par_chunks_mut(2048 * k).zip(pts.par_chunks(2048)).for_each(|(o, p)| {
        let ao = hartree::dft::ao::eval_ao_batch(shells, nao, p, false);
        for (pi, row) in o.chunks_mut(k).enumerate() {
            let phi = &ao.phi[pi * nao..(pi + 1) * nao];
            for (j, &col) in cols.iter().enumerate() {
                let mut v = 0.0;
                for mu in 0..nao { v += c[mu * m + col] * phi[mu]; }
                row[j] = v;
            }
        }
    });
    out
}

/// Electron density (electrons per Bohr^3) at each point (Bohr, molecule frame): the sum over
/// filled orbitals of two times psi^2. Raw numbers, for drawing dots in proportion to it.
pub fn density_values(s: &Solved, pts: &[[f64; 3]]) -> Vec<f64> {
    let nocc = s.n_occ;
    let cols: Vec<usize> = (0..nocc).collect();
    let v = orbitals_at(&s.shells, &s.mo_coeff, s.n_orbitals, &cols, pts);
    let mut rho = vec![0.0; pts.len()];
    for (p, r) in rho.iter_mut().enumerate() {
        for j in 0..nocc { *r += 2.0 * v[p * nocc + j] * v[p * nocc + j]; }
    }
    rho
}

/// Values of one orbital (psi, per Bohr^(3/2)) at each point.
pub fn orbital_values(s: &Solved, i: usize, pts: &[[f64; 3]]) -> Vec<f64> {
    orbitals_at(&s.shells, &s.mo_coeff, s.n_orbitals, &[i], pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEBYE_PER_AU: f64 = 2.541_746_473;

    /// One CCCBDB result: a molecule at its optimised shape.
    struct Ref {
        name: String,
        energy: f64,
        dipole: f64,                    // Debye (CCCBDB gives 3 decimals)
        atoms: Vec<(String, [f64; 3])>,
    }

    fn references() -> Vec<Ref> {
        let mut out = Vec::new();
        for block in include_str!("../data/cccbdb.txt").split("\n\n") {
            let mut r = Ref { name: String::new(), energy: 0.0, dipole: 0.0, atoms: Vec::new() };
            for line in block.lines().filter(|l| !l.starts_with('#')) {
                let w: Vec<&str> = line.split_whitespace().collect();
                match w[0] {
                    "mol" => r.name = w[1].to_string(),
                    "E" => r.energy = w[1].parse().unwrap(),
                    "D" => r.dipole = w[1].parse().unwrap(),
                    s => r.atoms.push((s.to_string(), [w[1].parse().unwrap(), w[2].parse().unwrap(), w[3].parse().unwrap()])),
                }
            }
            if !r.name.is_empty() { out.push(r); }
        }
        out
    }

    fn from_angstrom(atoms: &[(String, [f64; 3])]) -> Mol {
        Mol { atoms: atoms.iter().map(|(s, p)| (s.clone(), p.map(|x| x * BOHR_PER_ANGSTROM))).collect() }
    }

    fn dist(m: &Mol, i: usize, j: usize) -> f64 {
        let (a, b) = (m.atoms[i].1, m.atoms[j].1);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() / BOHR_PER_ANGSTROM
    }

    fn angle(m: &Mol, i: usize, j: usize, k: usize) -> f64 {
        let (a, b, c) = (m.atoms[i].1, m.atoms[j].1, m.atoms[k].1);
        let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (d / (nu * nv)).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn bonded_pairs(m: &Mol) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for i in 0..m.atoms.len() {
            for j in i + 1..m.atoms.len() {
                if dist(m, i, j) < 1.25 * (radius(&m.atoms[i].0) + radius(&m.atoms[j].0)) {
                    v.push((i, j));
                }
            }
        }
        v
    }

    fn shape_key(m: &Mol, like: &Mol) -> (Vec<(String, Vec<f64>)>, Vec<(String, Vec<f64>)>) {
        use std::collections::BTreeMap;
        let mut d: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let n = m.atoms.len();
        for i in 0..n {
            for j in i + 1..n {
                let mut k = [m.atoms[i].0.clone(), m.atoms[j].0.clone()];
                k.sort();
                d.entry(k.join("-")).or_default().push(dist(m, i, j));
            }
        }
        let bonds = bonded_pairs(like);
        let nb = |a: usize| bonds.iter().filter_map(move |&(i, j)| if i == a { Some(j) } else if j == a { Some(i) } else { None });
        let mut a: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for c in 0..n {
            let ns: Vec<usize> = nb(c).collect();
            for x in 0..ns.len() {
                for y in x + 1..ns.len() {
                    let mut ends = [m.atoms[ns[x]].0.clone(), m.atoms[ns[y]].0.clone()];
                    ends.sort();
                    a.entry(format!("{}-{}-{}", ends[0], m.atoms[c].0, ends[1])).or_default().push(angle(m, ns[x], c, ns[y]));
                }
            }
        }
        let sort = |mut v: Vec<f64>| { v.sort_by(|p, q| p.partial_cmp(q).unwrap()); v };
        (d.into_iter().map(|(k, v)| (k, sort(v))).collect(), a.into_iter().map(|(k, v)| (k, sort(v))).collect())
    }

    fn shape_diff(a: &Mol, b: &Mol) -> (f64, f64) {
        let (da, aa) = shape_key(a, b);
        let (db, ab) = shape_key(b, b);
        let mut dr = 0.0f64;
        for ((ka, va), (kb, vb)) in da.iter().zip(&db) {
            assert_eq!(ka, kb);
            for (x, y) in va.iter().zip(vb) { dr = dr.max((x - y).abs()); }
        }
        let mut dt = 0.0f64;
        for ((ka, va), (kb, vb)) in aa.iter().zip(&ab) {
            assert_eq!(ka, kb);
            for (x, y) in va.iter().zip(vb) { dt = dt.max((x - y).abs()); }
        }
        (dr, dt)
    }

    fn dipole(mol: &Mol) -> f64 {
        let mut options = JobOptions::default();
        options.compute_properties = true;
        let r = Job { molecule: mol.hartree(), basis: BASIS.into(), method: Method::Rhf, options }.run().unwrap();
        assert!(r.converged());
        let d = r.properties.as_ref().unwrap().dipole_au;
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * DEBYE_PER_AU
    }

    #[test]
    fn cccbdb_energies_and_dipoles() {
        let refs = references();
        assert_eq!(refs.len(), MOLECULES.len());
        for r in &refs {
            let mol = from_angstrom(&r.atoms);
            let de = solve(&mol).unwrap().energy - r.energy;
            let dd = dipole(&mol) - r.dipole;
            assert!(de.abs() < 1.0e-6, "{}: energy off by {de:e}", r.name);
            assert!(dd.abs() < 1.0e-3, "{}: dipole off by {dd}", r.name);
        }
    }

    #[test]
    fn shapes_from_rough_starts() {
        let refs = references();
        for m in MOLECULES.iter() {
            let r = refs.iter().find(|r| r.name == m.key).unwrap();
            let (got, e, _) = optimise(&rough_shape(m.key).unwrap()).unwrap();
            let (dr, da) = shape_diff(&got, &from_angstrom(&r.atoms));
            // CCCBDB gives coordinates to 1e-4 Angstrom.
            assert!(dr < 4e-4 && da < 0.05, "{}: {dr} A, {da} deg", m.key);
            assert!((e - r.energy).abs() < 1e-6, "{}: {}", m.key, e - r.energy);
        }
    }

    #[test]
    fn grid_holds_the_electrons() {
        let grid = |s: &Solved, n: usize| {
            let (c, half) = cube_for(&s.mol, 7.0);
            let h = 2.0 * half / (n - 1) as f64;
            let mut pts = Vec::with_capacity(n * n * n);
            for z in 0..n { for y in 0..n { for x in 0..n {
                pts.push([c[0] - half + x as f64 * h, c[1] - half + y as f64 * h, c[2] - half + z as f64 * h]);
            } } }
            (pts, h * h * h)
        };
        let refs = references();
        let at = |key: &str| solve(&from_angstrom(&refs.iter().find(|r| r.name == key).unwrap().atoms)).unwrap();
        let s = at("H2");
        let (pts, dv) = grid(&s, 128);
        let ne: f64 = density_values(&s, &pts).iter().sum::<f64>() * dv;
        assert!((ne - 2.0).abs() < 2e-3, "H2 electrons on the grid: {ne}");
        let norm: f64 = orbital_values(&s, 0, &pts).iter().map(|v| v * v).sum::<f64>() * dv;
        assert!((norm - 1.0).abs() < 1e-3, "H2 orbital norm: {norm}");
        let s = at("H2O");
        let (pts, dv) = grid(&s, 128);
        let norm: f64 = orbital_values(&s, 4, &pts).iter().map(|v| v * v).sum::<f64>() * dv;
        assert!((norm - 1.0).abs() < 5e-3, "water HOMO norm: {norm}");
    }
}
