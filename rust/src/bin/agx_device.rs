//! `agx_device`: several contexts (GPU VMs) with queues of dispatches on one simulated GPU (`agx_sim::device`).
//!
//! agx_device SPEC.json [--policy round-robin|submission] [--agx-strict GPU]
//!
//! SPEC is a JSON object; file names in it are relative to its directory:
//!
//! ```json
//! {"policy": "round-robin",
//!  "contexts": [
//!    {"vm_slot": 1,
//!     "map": {"0x10000": "in.bin"}, "alloc": {"0x20000": 4096},
//!     "dump": [["0x20000", 4096, "out.bin"]],
//!     "queues": [[{"code": "k.bin", "grid": [64, 1, 1], "wg": [32, 1, 1], "shared": 0, "regs": 128,
//!                  "max_steps": 100000000, "uniforms": {"8": 65536, "9": 0}, "preamble": "pre.bin"}]]}]}
//! ```
//!
//! Each context has its own memory (`map` from a file, `alloc` zero-filled), its VM slot (what SR21 reads) and its
//! queues; commands of a queue run in order. Prints one JSON line per command in the order the device ran them
//! (`context`, `queue`, `point`, `start`, `end`, `instructions`, `groups`, `fault`), then writes the dumps. A faulted
//! command (unmapped access, step limit) has `fault` set and both timestamps 0, and the others still run. Exits 1 if
//! any command faulted.

use agx_sim::device::{Command, Device, Policy};
use agx_sim::func::{Dispatch, Memory};
use serde_json::Value;
use std::path::Path;

fn die(m: &str) -> ! {
    println!("{{\"error\":{}}}", serde_json::to_string(m).unwrap());
    std::process::exit(2)
}

fn num(v: &Value) -> u64 {
    match v {
        Value::Number(n) => n.as_u64().unwrap_or_else(|| die(&format!("bad number {v}"))),
        Value::String(s) => {
            let s = s.trim();
            if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16) } else { s.parse() }
                .unwrap_or_else(|_| die(&format!("bad number {s:?}")))
        }
        _ => die(&format!("expected a number, got {v}")),
    }
}

fn key(k: &str) -> u64 {
    num(&Value::String(k.to_string()))
}

fn dims(v: Option<&Value>) -> [u32; 3] {
    let a = v.and_then(Value::as_array).cloned().unwrap_or_default();
    let g = |i: usize| a.get(i).map(|x| num(x) as u32).unwrap_or(1);
    [g(0), g(1), g(2)]
}

fn read(dir: &Path, f: &Value) -> Vec<u8> {
    let name = f.as_str().unwrap_or_else(|| die("expected a file name"));
    let p = dir.join(name);
    std::fs::read(&p).unwrap_or_else(|e| die(&format!("{}: {}", p.display(), e)))
}

fn command(dir: &Path, c: &Value, vm_slot: Option<u32>) -> Command {
    let mut code = read(dir, c.get("code").unwrap_or_else(|| die("a command needs code")));
    while code.len() > 2 && code[code.len() - 2..] == [0, 0] {
        code.truncate(code.len() - 2); // trailing padding
    }
    let shared = c.get("shared").map(num).unwrap_or(0) as u32;
    let uniforms = c
        .get("uniforms")
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| (key(k) as usize, num(v) as u32)).collect())
        .unwrap_or_default();
    let dispatch = Dispatch {
        code,
        grid: dims(c.get("grid")),
        wg: dims(c.get("wg")),
        shared_bytes: shared,
        uniforms,
        num_registers: c.get("regs").map(num).unwrap_or(128) as usize,
        max_steps: c.get("max_steps").map(num).unwrap_or(100_000_000),
        placement: 0,
        launch: agx_sim::cost::SrLaunch { tg_bytes: shared, context: vm_slot, ..Default::default() },
    };
    Command { dispatch, preamble: c.get("preamble").map(|p| read(dir, p)) }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let Some(spec_path) = a.first().filter(|s| !s.starts_with('-')) else {
        eprintln!("usage: agx_device SPEC.json [--policy round-robin|submission] [--agx-strict GPU]");
        std::process::exit(2);
    };
    let text = std::fs::read(spec_path).unwrap_or_else(|e| die(&format!("{spec_path}: {e}")));
    let spec: Value = serde_json::from_slice(&text).unwrap_or_else(|e| die(&format!("{spec_path}: {e}")));
    let dir = Path::new(spec_path).parent().unwrap_or(Path::new("."));
    let mut policy = spec.get("policy").and_then(Value::as_str).map(|p| Policy::parse(p).unwrap_or_else(|e| die(&e)));
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).unwrap_or_else(|| die(&format!("{} needs a value", a[i])));
        match a[i].as_str() {
            "--policy" => policy = Some(Policy::parse(v).unwrap_or_else(|e| die(&e))),
            "--agx-strict" => agx_sim::isa::mesa::set_mode(Some(agx_sim::isa::mesa::MesaGen::parse(v).unwrap_or_else(|e| die(&e)))),
            o => die(&format!("unknown option {o}")),
        }
        i += 2;
    }
    let mut dev = Device::new(policy.unwrap_or_default());
    let mut dumps = Vec::new();
    for c in spec.get("contexts").and_then(Value::as_array).unwrap_or_else(|| die("the spec needs contexts")) {
        let mut mem = Memory::default();
        for (va, f) in c.get("map").and_then(Value::as_object).into_iter().flatten() {
            mem.map(key(va), read(dir, f));
        }
        for (va, n) in c.get("alloc").and_then(Value::as_object).into_iter().flatten() {
            mem.map(key(va), vec![0; num(n) as usize]);
        }
        let vm_slot = c.get("vm_slot").map(|v| num(v) as u32);
        let ctx = dev.create_context(mem, vm_slot);
        for d in c.get("dump").and_then(Value::as_array).into_iter().flatten() {
            let d = d.as_array().filter(|d| d.len() == 3).unwrap_or_else(|| die("a dump is [VA, BYTES, FILE]"));
            let f = d[2].as_str().unwrap_or_else(|| die("a dump's file is a string"));
            dumps.push((ctx, num(&d[0]), num(&d[1]) as usize, dir.join(f)));
        }
        for (qi, q) in c.get("queues").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let queue = if qi == 0 { 0 } else { dev.create_queue(ctx) };
            for cmd in q.as_array().unwrap_or_else(|| die("a queue is a list of commands")) {
                dev.submit(ctx, queue, command(dir, cmd, vm_slot));
            }
        }
    }
    let done = dev.run();
    for d in &done {
        println!(
            "{{\"context\":{},\"queue\":{},\"point\":{},\"start\":{},\"end\":{},\"instructions\":{},\"groups\":{},\"fault\":{}}}",
            d.context,
            d.queue,
            d.point,
            d.start,
            d.end,
            d.instructions,
            d.groups,
            d.fault.as_ref().map_or("null".to_string(), |f| serde_json::to_string(f).unwrap())
        );
    }
    for (ctx, va, n, f) in dumps {
        let b = dev.contexts[ctx].memory.read_bytes(va, n).unwrap_or_else(|e| die(&e));
        std::fs::write(&f, b).unwrap_or_else(|e| die(&format!("{}: {}", f.display(), e)));
    }
    std::process::exit(if done.iter().any(|d| d.fault.is_some()) { 1 } else { 0 });
}
