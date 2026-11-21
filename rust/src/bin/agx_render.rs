//! `agx_render`: simulate one render pass (drm_asahi_cmd_render) from a direct-ioctl replay plan.
//!
//! agx_render PLAN.txt [--sim-suffix .sim] [--no-shading] [--max-steps N] [--log]
//!
//! PLAN.txt is what scripts/render_replay.py hands tests/drm_render_replay.c (memory image, command buffer, dump list).
//! Every `dump VA SIZE FILE` line of the plan is written to FILE + suffix (default `.sim`) from the simulated memory, so the
//! hardware and simulated outputs sit side by side. Prints one JSON line: statistics of every stage, plus the predicted
//! vertex / fragment time (`agx_sim::raster::timing`).
use agx_sim::raster::{self, Options, PassInput};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        eprintln!("usage: agx_render PLAN.txt [--sim-suffix S] [--no-shading] [--max-steps N] [--log]");
        std::process::exit(2);
    }
    let die = |m: String| -> ! { println!("{{\"error\":{}}}", serde_json::to_string(&m).unwrap()); std::process::exit(1) };
    let mut opt = Options::default();
    let (mut suffix, mut log) = (".sim".to_string(), false);
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--sim-suffix" => { suffix = a[i + 1].clone(); i += 1; }
            "--no-shading" => opt.no_shading = true,
            "--max-steps" => { opt.max_steps = a[i + 1].parse().unwrap_or_else(|_| die("bad --max-steps".into())); i += 1; }
            "--log" => log = true,
            o => die(format!("unknown option {}", o)),
        }
        i += 1;
    }
    let input = PassInput::from_plan(&a[0]).unwrap_or_else(|e| die(e));
    let r = raster::run_pass(&input, &opt).unwrap_or_else(|e| die(e));
    for (va, size, path) in &input.dumps {
        let b = r.mem.read_bytes(*va, *size as usize).unwrap_or_else(|e| die(e));
        std::fs::write(format!("{}{}", path, suffix), b).unwrap_or_else(|e| die(e.to_string()));
    }
    let t = raster::timing(&r);
    let s = &r.stats;
    let mut j = serde_json::json!({
        "draws": s.draws, "vertices": s.vertices, "vertex_groups": s.vertex_groups, "vertex_instructions": s.vertex_instructions,
        "primitives": s.primitives, "culled": s.culled, "clipped_unsupported": s.clipped_unsupported, "clipped": s.clipped,
        "tiles": s.tiles, "tiles_processed": s.tiles_processed, "bg_groups": s.bg_groups, "eot_groups": s.eot_groups,
        "fragment_groups": s.fragment_groups, "fragment_lanes": s.fragment_lanes, "fragment_samples": s.fragment_samples,
        "fragment_instructions": s.fragment_instructions, "bg_instructions": s.bg_instructions, "eot_instructions": s.eot_instructions,
        "depth_tested": s.depth_tested, "depth_passed": s.depth_passed,
        "predicted_vtx_ns": t.vtx_ns, "predicted_frag_ns": t.frag_ns, "timing": t.parts,
    });
    if log {
        j["log"] = serde_json::json!(r.log);
    }
    println!("{}", j);
}
