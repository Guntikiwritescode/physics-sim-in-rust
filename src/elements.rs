//! The 118 elements: names, ground-state electron configurations, a solved atom for each (dft.rs),
//! and their light (NIST Atomic Spectra Database lines, data/asd_lines.txt).
//!
//! Configurations: Z = 1-92 as NIST SRD 141 used them (data/nist_configurations.txt). Z = 93-118: PubChem's periodic table of electron
//! configurations (pubchem.ncbi.nlm.nih.gov/periodic-table); from Mt (109) on these are predictions,
//! no measurement exists.

use crate::dft::{self, Grid, Shell};

pub const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

pub const NAMES: [&str; 118] = [
    "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine", "neon",
    "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon", "potassium", "calcium",
    "scandium", "titanium", "vanadium", "chromium", "manganese", "iron", "cobalt", "nickel", "copper", "zinc",
    "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton", "rubidium", "strontium", "yttrium", "zirconium",
    "niobium", "molybdenum", "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin",
    "antimony", "tellurium", "iodine", "xenon", "caesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
    "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium", "thulium", "ytterbium",
    "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium", "iridium", "platinum", "gold", "mercury",
    "thallium", "lead", "bismuth", "polonium", "astatine", "radon", "francium", "radium", "actinium", "thorium",
    "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
    "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium", "hassium", "meitnerium", "darmstadtium",
    "roentgenium", "copernicium", "nihonium", "flerovium", "moscovium", "livermorium", "tennessine", "oganesson",
];

/// Z = 93-118 (PubChem).
const HEAVY: [&str; 26] = [
    "[Rn] 5f^4 6d^1 7s^2", "[Rn] 5f^6 7s^2", "[Rn] 5f^7 7s^2", "[Rn] 5f^7 6d^1 7s^2", "[Rn] 5f^9 7s^2",
    "[Rn] 5f^10 7s^2", "[Rn] 5f^11 7s^2", "[Rn] 5f^12 7s^2", "[Rn] 5f^13 7s^2", "[Rn] 5f^14 7s^2",
    "[Rn] 5f^14 6d^1 7s^2", "[Rn] 5f^14 6d^2 7s^2", "[Rn] 5f^14 6d^3 7s^2", "[Rn] 5f^14 6d^4 7s^2", "[Rn] 5f^14 6d^5 7s^2",
    "[Rn] 5f^14 6d^6 7s^2", "[Rn] 5f^14 6d^7 7s^2", "[Rn] 5f^14 6d^8 7s^2", "[Rn] 5f^14 6d^9 7s^2", "[Rn] 5f^14 6d^10 7s^2",
    "[Rn] 5f^14 6d^10 7s^2 7p^1", "[Rn] 5f^14 6d^10 7s^2 7p^2", "[Rn] 5f^14 6d^10 7s^2 7p^3", "[Rn] 5f^14 6d^10 7s^2 7p^4",
    "[Rn] 5f^14 6d^10 7s^2 7p^5", "[Rn] 5f^14 6d^10 7s^2 7p^6",
];

pub fn config(z: u32) -> Vec<Shell> {
    if z <= 92 { dft::nist_config(z) } else { dft::parse_config(HEAVY[(z - 93) as usize]) }
}

/// Text form, e.g. "[Ar] 3d6 4s2".
pub fn config_text(shells: &[Shell]) -> String {
    let core: [(&str, u32); 6] = [("[Rn]", 86), ("[Xe]", 54), ("[Kr]", 36), ("[Ar]", 18), ("[Ne]", 10), ("[He]", 2)];
    for (name, n) in core {
        let cs = dft::parse_config(name);
        let full = cs.iter().all(|c| shells.iter().any(|s| s.n == c.n && s.l == c.l && (s.occ - c.occ).abs() < 1e-9));
        if full && shells.iter().map(|s| s.occ).sum::<f64>() >= n as f64 {
            let rest: Vec<String> = shells.iter().filter(|s| !cs.iter().any(|c| c.n == s.n && c.l == s.l))
                .map(|s| format!("{}{}", dft::shell_name(s), s.occ)).collect();
            return format!("{name} {}", rest.join(" ")).trim().to_string();
        }
    }
    shells.iter().map(|s| format!("{}{}", dft::shell_name(s), s.occ)).collect::<Vec<_>>().join(" ")
}

/// A solved atom ready to draw: orbital energies and radial tables on a log grid.
#[derive(Clone)]
pub struct Solved {
    pub z: u32,
    pub atom: dft::Atom,
    pub grid: Grid,
}

/// Solve element z on the picture grid.
pub fn solve(z: u32) -> Option<Solved> {
    let g = Grid::display(z as f64);
    let atom = dft::solve(&g, z as f64, &config(z), 1e-8)?;
    Some(Solved { z, atom, grid: g })
}

impl Solved {
    /// Electron density (per Bohr^3) at radius r (Bohr), interpolated in ln r.
    pub fn density_at(&self, r: f64) -> f64 { interp(&self.grid, &self.atom.rho, r) }
}

fn interp(g: &Grid, f: &[f64], r: f64) -> f64 {
    let x0 = g.r[0].ln();
    let t = (r.max(g.r[0]).ln() - x0) / g.h;
    let i = t.floor() as usize;
    if i + 1 >= f.len() { return 0.0; }
    let a = t - i as f64;
    f[i] * (1.0 - a) + f[i + 1] * a
}

pub const ASD: &str = include_str!("../data/asd_lines.txt");

/// One spectral line: air wavelength (nm), transition probability (/s), relative intensity.
#[derive(Clone, Debug)]
pub struct Line { pub nm: f64, pub a: f64, pub intensity: f64 }

/// What the database has for an element.
#[derive(Clone, Debug, PartialEq)]
pub enum LineData { WithRates, IntensitiesOnly, None }

/// The lines of element z (neutral atom, 200-1000 nm).
pub fn lines(z: u32) -> (LineData, Vec<Line>) {
    let head = format!("@ {z} {} ", SYMBOLS[(z - 1) as usize]);
    let Some(start) = ASD.find(&head) else { return (LineData::None, Vec::new()) };
    let mut it = ASD[start..].lines();
    let kind = match it.next().and_then(|h| h.split_whitespace().nth(3)) { Some("A") => LineData::WithRates, Some("I") => LineData::IntensitiesOnly, _ => LineData::None };
    let mut v = Vec::new();
    for l in it {
        if l.starts_with('@') { break; }
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() < 7 { continue; }
        let p = |s: &str| s.parse::<f64>().unwrap_or(0.0);
        v.push(Line { nm: p(f[0]), a: p(f[1]), intensity: p(f[6]) });
    }
    (kind, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_has_its_electrons() {
        for z in 1..=118u32 {
            let n: f64 = config(z).iter().map(|s| s.occ).sum();
            assert_eq!(n, z as f64, "{} has {n} electrons", SYMBOLS[(z - 1) as usize]);
            for s in config(z) { assert!(s.occ <= (2 * (2 * s.l + 1)) as f64, "{} {s:?} over-full", SYMBOLS[(z - 1) as usize]); }
        }
        assert_eq!(config_text(&config(26)), "[Ar] 3d6 4s2");
        assert_eq!(config_text(&config(118)), "[Rn] 5f14 6d10 7s2 7p6");
    }

    #[test]
    fn all_118_elements_solve() {
        use rayon::prelude::*;
        let bad: Vec<u32> = (1..=118u32).into_par_iter().filter(|&z| solve(z).is_none()).collect();
        assert!(bad.is_empty(), "did not converge: {bad:?}");
        let a = solve(118).unwrap().atom;
        let b = dft::solve(&Grid::standard(118.0), 118.0, &config(118), 1e-8).unwrap();
        assert!((a.e_tot - b.e_tot).abs() < 1e-4, "Og: {} vs {}", a.e_tot, b.e_tot);
    }

    #[test]
    fn sodium_d_lines() {
        let (kind, ls) = lines(11);
        assert_eq!(kind, LineData::WithRates);
        for nm in [588.995, 589.592] {
            let l = ls.iter().find(|l| (l.nm - nm).abs() < 0.002).expect("D line");
            assert!((l.a / 6.1e7 - 1.0).abs() < 0.02, "{nm}: A {}", l.a);
        }
        assert_eq!(lines(118).0, LineData::None);
    }
}
