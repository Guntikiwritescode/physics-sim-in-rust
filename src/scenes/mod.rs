//! The scenes in the one app (each a `host::Scene`): the Black hole tab and the Atoms tab.

pub mod atoms;
pub mod world;

use crate::host::{Factory, Scene};

fn space() -> Box<dyn Scene> { Box::new(world::WorldScene::new(world::SETUP_ORDER[0])) }
fn atoms() -> Box<dyn Scene> { Box::new(atoms::AtomsScene::new()) }

/// Every scene, in menu order.
pub fn list() -> Vec<Factory> {
    vec![("Black hole", space), ("Atoms", atoms)]
}

/// Opens the window on scene `start` (any case, first letters are enough; --scene picks another).
pub fn open(start: &str) {
    let args: Vec<String> = std::env::args().collect();
    let pick = args.iter().position(|a| a == "--scene").and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or(start);
    let scenes = list();
    let idx = scenes.iter().position(|(n, _)| n.to_lowercase().starts_with(&pick.to_lowercase())).unwrap_or(0);
    // --screenshot out.ppm [seconds]: what the window shows after that long, saved to a file, no window
    if let Some(i) = args.iter().position(|a| a == "--screenshot") {
        let out = args.get(i + 1).cloned().unwrap_or_else(|| "screenshot.ppm".into());
        let secs = args.get(i + 2).and_then(|s| s.parse().ok()).unwrap_or(2.0);
        crate::host::screenshot(scenes, idx, &out, secs, 2880, 1720);
        return;
    }
    crate::host::run(scenes, idx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Input;
    use winit::keyboard::KeyCode;

    #[test]
    fn every_scene_survives_random_use() {
        let mut seed = 7u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f64 / (1u64 << 31) as f64 };
        let keys = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::KeyM, KeyCode::KeyK, KeyCode::KeyG, KeyCode::KeyP,
                    KeyCode::KeyV, KeyCode::KeyB, KeyCode::KeyN, KeyCode::KeyH, KeyCode::KeyC, KeyCode::KeyT, KeyCode::KeyY, KeyCode::KeyU,
                    KeyCode::KeyW, KeyCode::KeyE, KeyCode::KeyZ, KeyCode::KeyX, KeyCode::Space, KeyCode::KeyF, KeyCode::Home, KeyCode::KeyR];
        let all: Vec<Factory> = list();
        for (name, make) in all {
            let mut s = make();
            let mut inp = Input { screen: (1280, 720), mouse: (640.0, 360.0), ..Default::default() };
            for f in 0..600 {
                let r = rnd();
                if r < 0.05 { inp.pressed_left = Some(inp.mouse); inp.left = true; }
                if r > 0.95 { inp.left = false; inp.released_left = true; }
                if inp.left { inp.drag_left = (rnd() * 40.0 - 20.0, rnd() * 40.0 - 20.0); }
                inp.mouse = (rnd() * 1280.0, rnd() * 720.0);
                inp.scroll = if rnd() < 0.1 { rnd() * 4.0 - 2.0 } else { 0.0 };
                inp.right = rnd() < 0.2;
                if f % 25 == 0 {
                    let k = keys[(rnd() * keys.len() as f64) as usize % keys.len()];
                    if name == "Atoms" && matches!(k, KeyCode::KeyW | KeyCode::KeyT) && f > 300 { continue; }
                    s.key(k, true);
                    s.key(k, false);
                }
                s.update(1.0 / 60.0, &inp);
                inp.pressed_left = None;
                inp.released_left = false;
                inp.double = None;
            }
            let st = s.status().join(" | ");
            assert!(!st.contains("NaN") && !st.contains("inf"), "{name}: {st}");
        }
    }
}
