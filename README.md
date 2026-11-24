# rust-black-hole

Rust versions of kavan010's black hole, gravity and atoms sims, in one window with two tabs.

![the black hole tab](screenshot.png)

## Run

You need Rust (https://rustup.rs).

    git clone https://github.com/Guntikiwritescode/rust-black-hole
    cd rust-black-hole
    cargo run --release

F1 shows the controls. `cargo run --release -- --scene atoms` starts on the Atoms tab.

`cargo test --release` runs the tests.

## Credits

- kavan010's sims: https://github.com/kavan010/black_hole, https://github.com/kavan010/gravity_sim, https://github.com/kavan010/Atoms
- Light paths near the black hole: Chan et al. 2018, GRay2, https://arxiv.org/abs/1706.07062
- Picture clean-up: Schied et al. 2017, Spatiotemporal Variance-Guided Filtering
- Molecules and crystals are solved with hartree: https://crates.io/crates/hartree
- Atom data: NIST SRD 141 (https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations) and the NIST Atomic Spectra Database (https://physics.nist.gov/asd)
- Molecule data: NIST CCCBDB, https://cccbdb.nist.gov
- Crystal data: Haas, Tran and Blaha, Phys. Rev. B 79, 085104 (2009)
