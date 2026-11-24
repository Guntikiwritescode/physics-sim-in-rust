// Solids: crystals solved with density functional theory (LDA) in hartree's periodic code
// (Gaussian basis + plane-wave density, GTH pseudopotentials, k-point sampling). The lattice
// constant is the one with the lowest energy (Birch-Murnaghan fit of E(a)).
//
// Reference: P. Haas, F. Tran, P. Blaha, Phys. Rev. B 79, 085104 (2009), Table I: LDA lattice
// constants from WIEN2k (all-electron FP-LAPW) and experiment corrected for zero-point expansion.

use hartree::periodic::{Cell, KPoint, MonkhorstPack};
use hartree::{run_periodic, Atom, Element, Molecule, PeriodicJob};

pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529_177_210_903;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Structure {
    /// Two atoms on an fcc lattice, the second at (1/4, 1/4, 1/4) a (diamond; zinc blende when
    /// the two differ, as in GaAs).
    Diamond,
    /// Cation at 0, anion at (1/2, 0, 0) a, fcc lattice.
    RockSalt,
}

pub struct SolidInfo {
    pub key: &'static str,
    pub name: &'static str,
    pub atoms: [&'static str; 2],
    pub structure: Structure,
    pub basis: &'static str,
    /// Haas, Tran, Blaha 2009, Table I: the LDA lattice constant (Angstrom).
    pub a_lda: f64,
    /// The lattice constant the picture uses (Angstrom): where this program's energy is lowest
    /// (a_lda for Ge and GaAs).
    pub a0: f64,
    /// Density cutoff (hartree) at which E(a) is smooth: silicon's soft pseudopotential needs 120;
    /// the salts' fluorine and semicore sodium need 500.
    pub ecut: f64,
}

pub const SOLIDS: [SolidInfo; 7] = [
    SolidInfo { key: "Si", name: "silicon", atoms: ["Si", "Si"], structure: Structure::Diamond, basis: "DZVP-GTH-PADE", a_lda: 5.407, a0: 5.4037, ecut: 120.0 },
    SolidInfo { key: "LiF", name: "lithium fluoride", atoms: ["Li", "F"], structure: Structure::RockSalt, basis: "DZVP-GTH", a_lda: 3.919, a0: 3.9236, ecut: 500.0 },
    SolidInfo { key: "LiCl", name: "lithium chloride", atoms: ["Li", "Cl"], structure: Structure::RockSalt, basis: "DZVP-GTH", a_lda: 4.986, a0: 4.9776, ecut: 500.0 },
    SolidInfo { key: "NaF", name: "sodium fluoride", atoms: ["Na", "F"], structure: Structure::RockSalt, basis: "DZVP-GTH", a_lda: 4.507, a0: 4.5165, ecut: 500.0 },
    SolidInfo { key: "NaCl", name: "sodium chloride (salt)", atoms: ["Na", "Cl"], structure: Structure::RockSalt, basis: "DZVP-GTH", a_lda: 5.484, a0: 5.4887, ecut: 500.0 },
    // Diamond and zinc-blende crystals with the MOLOPT basis sets hartree ships for Ga, Ge and As.
    SolidInfo { key: "Ge", name: "germanium", atoms: ["Ge", "Ge"], structure: Structure::Diamond, basis: "DZVP-MOLOPT-PBE-GTH", a_lda: 5.632, a0: 5.632, ecut: 200.0 },
    SolidInfo { key: "GaAs", name: "gallium arsenide", atoms: ["Ga", "As"], structure: Structure::Diamond, basis: "DZVP-MOLOPT-PBE-GTH", a_lda: 5.616, a0: 5.616, ecut: 500.0 },
];

/// fcc primitive cell vectors for cubic lattice constant a (Bohr).
pub fn fcc(a: f64) -> [[f64; 3]; 3] {
    [[0.0, a / 2.0, a / 2.0], [a / 2.0, 0.0, a / 2.0], [a / 2.0, a / 2.0, 0.0]]
}

/// The two atoms of the primitive cell (Bohr).
pub fn basis_atoms(s: &SolidInfo, a: f64) -> [(&'static str, [f64; 3]); 2] {
    match s.structure {
        Structure::Diamond => [(s.atoms[0], [0.0; 3]), (s.atoms[1], [a / 4.0, a / 4.0, a / 4.0])],
        Structure::RockSalt => [(s.atoms[0], [0.0; 3]), (s.atoms[1], [a / 2.0, 0.0, 0.0])],
    }
}

/// Monkhorst-Pack k-points, each k paired with -k (equal energies without magnetism, so one of
/// each pair is kept with double weight; the density from k and -k is the same).
pub fn kpoints(n: usize) -> Vec<KPoint> {
    let all = MonkhorstPack::regular([n, n, n]).unwrap();
    let mut kept: Vec<KPoint> = Vec::new();
    for k in all {
        let neg = k.frac.map(|x| -x);
        let wrap = |f: [f64; 3]| f.map(|x| { let y = x - x.round(); if (y + 0.5).abs() < 1e-9 { 0.5 } else { y } });
        let (kw, nw) = (wrap(k.frac), wrap(neg));
        if let Some(p) = kept.iter_mut().find(|p| wrap(p.frac).iter().zip(&nw).all(|(a, b)| (a - b).abs() < 1e-9)) {
            p.weight += k.weight;
        } else if kw.iter().zip(&nw).all(|(a, b)| (a - b).abs() < 1e-9) {
            kept.push(k); // its own partner (time-reversal invariant point)
        } else {
            kept.push(k);
        }
    }
    kept
}

/// The valence electron density of a crystal on its own grid (the pseudopotential holds the core
/// electrons), with what is needed to look it up anywhere.
pub struct Crystal {
    /// LDA energy per primitive cell (hartree).
    pub energy: f64,
    pub grid: [usize; 3],
    pub density: Vec<f64>,
    pub cell: Cell,
    /// Electrons in the cell counted on the grid (a check against valence()).
    pub n_elec: f64,
}

/// Valence electrons per primitive cell with the GTH-PADE pseudopotentials used here.
pub fn valence(s: &SolidInfo) -> usize {
    let q = |sym: &str| match sym { "Si" => 4, "Li" => 3, "F" => 7, "Na" => 9, "Cl" => 7, "Ge" => 4, "Ga" => 13, "As" => 5, _ => 0 };
    q(s.atoms[0]) + q(s.atoms[1])
}

/// Solve the crystal at lattice constant a (Angstrom) and keep its density.
pub fn crystal(s: &SolidInfo, a_ang: f64, kmesh: usize, ecut: f64) -> Result<Crystal, String> {
    let a = a_ang * BOHR_PER_ANGSTROM;
    let v = fcc(a);
    let cell = Cell::from_vectors(v[0], v[1], v[2]).map_err(|e| e.to_string())?;
    let atoms = basis_atoms(s, a);
    let mol = Molecule::new(atoms.iter().map(|(sym, p)| Atom::new(Element::from_symbol(sym).unwrap(), *p)).collect(), 0, 1);
    let mut job = PeriodicJob::gth_pade(mol, cell, kpoints(kmesh), s.basis)?;
    job.options.e_cut = ecut;
    let r = run_periodic(&job)?;
    if !r.scf.converged { return Err(format!("{} at {a_ang} A: not converged", s.key)); }
    let grid = hartree::periodic::RealSpaceGrid::from_cutoff(cell, ecut);
    let n_elec = r.scf.density.iter().sum::<f64>() * grid.dv();
    Ok(Crystal { energy: r.scf.energy, grid: grid.n(), density: r.scf.density, cell, n_elec })
}

impl Crystal {
    /// Density at a point (Bohr), periodic, trilinear between grid points.
    pub fn density_at(&self, p: [f64; 3]) -> f64 {
        let f = self.cell.cart_to_frac(p);
        let [n1, n2, n3] = self.grid;
        let mut idx = [0usize; 3];
        let mut w = [0.0f64; 3];
        for k in 0..3 {
            let n = [n1, n2, n3][k] as f64;
            let x = (f[k] - f[k].floor()) * n;
            let i = x.floor();
            idx[k] = i as usize % [n1, n2, n3][k];
            w[k] = x - i;
        }
        let at = |i: usize, j: usize, k: usize| self.density[((i % n1) * n2 + (j % n2)) * n3 + (k % n3)];
        let (i, j, k) = (idx[0], idx[1], idx[2]);
        let mut s = 0.0;
        for (di, wi) in [(0, 1.0 - w[0]), (1, w[0])] {
            for (dj, wj) in [(0, 1.0 - w[1]), (1, w[1])] {
                for (dk, wk) in [(0, 1.0 - w[2]), (1, w[2])] {
                    s += wi * wj * wk * at(i + di, j + dj, k + dk);
                }
            }
        }
        s
    }
}

/// Conventional cubic cells along each side of the block shown.
pub const CELLS: f64 = 2.0;
/// k-point mesh and density cutoff (hartree) for the picture.
pub const PIC_K: usize = 2;
pub const PIC_ECUT: f64 = 120.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn birch_murnaghan(v: &[f64], e: &[f64]) -> Option<(f64, f64, f64, f64)> {
        let n = v.len();
        if n < 4 { return None; }
        let xs: Vec<f64> = v.iter().map(|&vi| vi.powf(-2.0 / 3.0)).collect();
        let xm = xs.iter().sum::<f64>() / n as f64;
        let mut m = [[0.0f64; 5]; 4];
        for (x, &ei) in xs.iter().zip(e) {
            let tt = x / xm - 1.0;
            let p = [1.0, tt, tt * tt, tt * tt * tt];
            for i in 0..4 { for j in 0..4 { m[i][j] += p[i] * p[j]; } m[i][4] += p[i] * ei; }
        }
        for c in 0..4 {
            let piv = (c..4).max_by(|&a, &b| m[a][c].abs().partial_cmp(&m[b][c].abs()).unwrap())?;
            m.swap(c, piv);
            if m[c][c].abs() < 1e-300 { return None; }
            for r in 0..4 { if r != c { let f = m[r][c] / m[c][c]; for k in c..5 { m[r][k] -= f * m[c][k]; } } }
        }
        let d: Vec<f64> = (0..4).map(|i| m[i][4] / m[i][i]).collect();
        let (u, w) = (1.0 / xm, -1.0);
        let c = vec![
            d[0] + d[1] * w + d[2] * w * w + d[3] * w * w * w,
            d[1] * u + 2.0 * d[2] * u * w + 3.0 * d[3] * u * w * w,
            d[2] * u * u + 3.0 * d[3] * u * u * w,
            d[3] * u * u * u,
        ];
        let (a, b, cc) = (3.0 * c[3], 2.0 * c[2], c[1]);
        let disc = b * b - 4.0 * a * cc;
        if disc < 0.0 { return None; }
        let (xlo, xhi) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
        let roots = [(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)];
        let x0 = roots.into_iter().filter(|r| *r > 0.9 * xlo && *r < 1.1 * xhi && 2.0 * c[2] + 6.0 * c[3] * r > 0.0).next()?;
        let v0 = x0.powf(-1.5);
        let e0 = c[0] + c[1] * x0 + c[2] * x0 * x0 + c[3] * x0 * x0 * x0;
        let (e1, e2, e3) = (c[1] + 2.0 * c[2] * x0 + 3.0 * c[3] * x0 * x0, 2.0 * c[2] + 6.0 * c[3] * x0, 6.0 * c[3]);
        let (x1, x2) = (-2.0 / 3.0 * v0.powf(-5.0 / 3.0), 10.0 / 9.0 * v0.powf(-8.0 / 3.0));
        let d2 = e2 * x1 * x1 + e1 * x2;
        let b0 = v0 * d2;
        let x3 = -80.0 / 27.0 * v0.powf(-11.0 / 3.0);
        let d3 = e3 * x1 * x1 * x1 + 3.0 * e2 * x1 * x2 + e1 * x3;
        let bp = -(1.0 + v0 * d3 / d2);
        Some((v0, e0, b0, bp))
    }

    fn a_from_v(v: f64) -> f64 { (4.0 * v).cbrt() / BOHR_PER_ANGSTROM }
    fn v_from_a(a_ang: f64) -> f64 { (a_ang * BOHR_PER_ANGSTROM).powi(3) / 4.0 }

    #[test]
    fn birch_murnaghan_fit_is_exact_on_its_own_form() {
        let (v0, e0, b0, bp) = (270.0, -7.9, 0.0035, 4.2);
        let bm = |v: f64| {
            let r = (v0 / v).powf(2.0 / 3.0);
            e0 + 9.0 * v0 * b0 / 16.0 * ((r - 1.0).powi(3) * bp + (r - 1.0).powi(2) * (6.0 - 4.0 * r))
        };
        let vs: Vec<f64> = (0..7).map(|i| v0 * (0.91 + 0.03 * i as f64)).collect();
        let es: Vec<f64> = vs.iter().map(|&v| bm(v)).collect();
        let (v, e, b, p) = birch_murnaghan(&vs, &es).unwrap();
        assert!((v - v0).abs() < 1e-8 && (e - e0).abs() < 1e-12 && (b - b0).abs() < 1e-10 && (p - bp).abs() < 1e-6, "{v} {e} {b} {p}");
    }

    #[test]
    fn silicon_lattice_constant_quick() {
        let s = &SOLIDS[0];
        let (mut vs, mut es) = (Vec::new(), Vec::new());
        for i in 0..5 {
            let a = s.a_lda * (0.98 + 0.01 * i as f64);
            vs.push(v_from_a(a));
            es.push(crystal(s, a, 2, 120.0).unwrap().energy);
        }
        let (v0, _, _, _) = birch_murnaghan(&vs, &es).unwrap();
        let a0 = a_from_v(v0);
        assert!((a0 / s.a_lda - 1.0).abs() < 2e-3, "a0 {a0} vs {}", s.a_lda);
        let c = crystal(s, s.a_lda, 2, 120.0).unwrap();
        assert!((c.n_elec - 8.0).abs() < 1e-3, "{} electrons on the grid", c.n_elec);
    }

    #[test]
    #[ignore]
    fn all_lattice_constants() {
        for s in SOLIDS.iter() {
            let (mut vs, mut es) = (Vec::new(), Vec::new());
            for i in 0..7 {
                let a = s.a_lda * (0.97 + 0.01 * i as f64);
                vs.push(v_from_a(a));
                es.push(crystal(s, a, 2, s.ecut).unwrap().energy);
            }
            let (v0, _, _, _) = birch_murnaghan(&vs, &es).unwrap();
            let a0 = a_from_v(v0);
            assert!((a0 / s.a_lda - 1.0).abs() < 3e-3, "{}: a0 {a0} vs {}", s.key, s.a_lda);
        }
    }

    #[test]
    fn kpoint_pairs() {
        for n in [2, 3, 4] {
            let k = kpoints(n);
            let w: f64 = k.iter().map(|p| p.weight).sum();
            assert!((w - 1.0).abs() < 1e-12);
            if n % 2 == 0 { assert_eq!(k.len(), n * n * n / 2); }
        }
    }
}
