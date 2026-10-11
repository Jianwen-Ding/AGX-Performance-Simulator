//! Port of `scripts/verify_cost_model.py`: offline accuracy check of the cost model against every
//! stored hardware measurement. Prints the same lines as the Python verifier; exit status 1 on any
//! failed check.

use agx_sim::cost::{self, read_gz_json, read_json, repo_root, to_ref, Params};
use agx_sim::isa::operand::py_float;
use agx_sim::kernels::{self, counted, e, e2, mem_kernel, store, tg, DEFAULT_CONSTS, MEM_CONSTS};
use serde_json::Value;
use std::io::Write;

struct V {
    p: Params,
    fails: usize,
    /// `--export FILE`: every distinct simulation as one JSON line (decoded instructions, run-length
    /// encoded trace and the cycles), for differential testing against the Python `simulate`.
    export: Option<std::cell::RefCell<(std::fs::File, std::collections::HashSet<(Vec<u8>, usize)>)>>,
}

fn export_line(ins: &[cost::Instr], pcs: &[usize], groups: usize, cycles: f64) -> String {
    let sizes: std::collections::HashMap<usize, usize> = ins.iter().map(|r| (r.pc, r.size)).collect();
    let mut runs: Vec<[usize; 2]> = vec![];
    let mut next = usize::MAX;
    for &pc in pcs {
        match runs.last_mut() {
            // extend the run when pc follows the previous instruction
            Some(last) if pc == next => last[1] += 1,
            _ => runs.push([pc, 1]),
        }
        next = pc + sizes[&pc];
    }
    let instrs: Vec<serde_json::Value> = ins
        .iter()
        .map(|r| serde_json::json!({"pc": r.pc, "size": r.size, "name": r.name, "kind": r.kind.as_str(), "unit": r.unit,
                                     "src": r.src, "dst": r.dst, "slot": r.slot, "reads32": r.reads32}))
        .collect();
    serde_json::json!({"groups": groups, "cycles": cycles, "instrs": instrs, "runs": runs}).to_string()
}

fn stats(errs: &[f64]) -> (f64, f64, f64, f64) {
    let mut a: Vec<f64> = errs.iter().map(|x| x.abs()).collect();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = a.len();
    let mut s: Vec<f64> = errs.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    (a[n / 2], a[(0.9 * n as f64) as usize], a[n - 1], med)
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn rep(b: &[u8], n: usize) -> Vec<u8> {
    b.repeat(n)
}

impl V {
    fn check(&mut self, ok: bool, msg: String) {
        if !ok {
            self.fails += 1;
        }
        println!("{}{}", if ok { "ok   " } else { "FAIL " }, msg);
        std::io::stdout().flush().ok();
    }
    fn sim(&self, code: &[u8], groups: usize) -> f64 {
        let ins = cost::decode(code, &self.p).unwrap();
        let tr = cost::trace(code, 32, 500_000).unwrap();
        let c = cost::simulate(&ins, &tr.pcs, &self.p, groups, false).unwrap().0;
        if let Some(x) = &self.export {
            let mut x = x.borrow_mut();
            if x.1.insert((code.to_vec(), groups)) {
                let line = export_line(&ins, &tr.pcs, groups, c);
                writeln!(x.0, "{}", line).unwrap();
            }
        }
        c
    }
    /// cost_hw.predict: reference units per unit of (phase_b - phase_a).
    fn predict(&self, pa: &[u8], pb: &[u8], units: f64, groups: usize) -> f64 {
        to_ref(self.sim(pb, groups) - self.sim(pa, groups)) / units
    }
    fn pred_loop(&self, body: &[u8], consts: [u32; 4], groups: usize) -> f64 {
        self.predict(&counted(body, 100, true, consts), &counted(body, 500, true, consts), 400.0, groups)
    }
}

fn f3(x: f64) -> String {
    format!("{:.3}", x)
}

fn obj(v: &Value) -> &serde_json::Map<String, Value> {
    v.as_object().unwrap()
}

fn main() {
    let r = repo_root().join("results/2026-10-11-cost-model");
    let load = |f: &str| read_json(&r.join(f)).unwrap();
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    let args: Vec<String> = std::env::args().collect();
    let export = args.iter().position(|a| a == "--export").map(|i| {
        let f = std::fs::File::create(&args[i + 1]).unwrap();
        std::cell::RefCell::new((f, Default::default()))
    });
    let mut v = V { p, fails: 0, export };
    if v.export.is_some() {
        println!("info branch {:?}", v.p.branch.unwrap());
    }

    // ---- bench (FIT)
    let bench = read_gz_json(&cost::default_bench()).unwrap();
    let t_ = 20u32;
    let mut err: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    for (op, bv) in obj(&bench["ops"]) {
        let slope = |ch: u32, groups: usize| {
            let c: Vec<f64> = [16, 64].iter().map(|&s| v.sim(&kernels::bench_loop(op, ch, s, t_), groups)).collect();
            to_ref(c[1] - c[0]) / (t_ * 48) as f64
        };
        let pred = [slope(1, 1), slope(8, 1), slope(8, 4) / 4.0];
        for (i, m) in ["latency", "simd_issue", "gpu"].iter().enumerate() {
            let meas = bv[m]["cycles_per_step"].as_f64().unwrap();
            err[i].push((pred[i] - meas) / meas);
            if std::env::var("AGX_VERBOSE").is_ok() {
                println!("  bench {} {}: measured {:.2} predicted {:.2}", op, m, meas, pred[i]);
            }
        }
    }
    for (i, (m, lim)) in [("latency", 0.03), ("simd_issue", 0.04), ("gpu", 0.06)].iter().enumerate() {
        let (md, p90, mx, _) = stats(&err[i]);
        v.check(md <= *lim, format!("bench (FIT) {}: median |err| {} (limit {}), p90 {}, max {}", m, f3(md), py_float(*lim), f3(p90), f3(mx)));
    }

    // ---- rotation (FIT)
    let rot = load("rotation.json");
    let mut rerr = vec![];
    for (key, meas) in obj(&rot) {
        let (instr, d) = key.split_once('|').unwrap();
        let d: u32 = d.parse().unwrap();
        let body: Vec<u8> = (0..72).flat_map(|k| e(&instr.replace("{r}", &(5 + k % d).to_string()))).collect();
        let pr = (v.pred_loop(&body, DEFAULT_CONSTS, 1) - 35.25) / 72.0;
        let m = meas.as_f64().unwrap();
        rerr.push((pr - m) / m);
    }
    let (md, _, mx, _) = stats(&rerr[..16]);
    v.check(md <= 0.05 && mx <= 0.12, format!("rotation (FIT, 32-bit forms): median |err| {}, max {}", f3(md), f3(mx)));
    let (md, _, mx, _) = stats(&rerr[16..]);
    println!("info rotation (fadd16): median |err| {}, max {} (16-bit writeback not modelled beyond D=9)", f3(md), f3(mx));

    let empty_meas = 35.25;
    let c: Vec<f64> = [200, 1000].iter().map(|&t| v.sim(&counted(&[], t, true, DEFAULT_CONSTS), 1)).collect();
    let empty_pred = to_ref(c[1] - c[0]) / 800.0;
    println!("info empty loop trip: measured {:.2}, predicted {:.2} (reference units)", empty_meas, empty_pred);

    // ---- memory (FIT)
    let mem = load("memory.json");
    let mem_cases: [(&str, Vec<&str>); 2] = [
        ("load line 1 + wait 0", vec!["iadd r13, r1, 32", "device_load 0, i32, x, r6, u8_u9, r13, unsigned", "wait 0"]),
        ("2 loads line 1 + one wait", vec!["iadd r13, r1, 32", "device_load 0, i32, x, r6, u8_u9, r13, unsigned",
                                           "device_load 0, i32, x, r7, u8_u9, r13, unsigned", "wait 0"]),
    ];
    for (name, lines) in &mem_cases {
        let body = rep(&lines.iter().flat_map(|x| e(x)).collect::<Vec<u8>>(), 8);
        let pr = (v.pred_loop(&body, DEFAULT_CONSTS, 1) - empty_pred) / 8.0;
        let m = mem[name].as_f64().unwrap();
        v.check((pr - m).abs() / m < 0.05, format!("memory (FIT) {}: measured {:.1}, predicted {:.1}", name, m, pr));
    }

    // ---- load-after-ALU grid (FIT)
    let grid = load("load-alu-grid.json");
    let lg = "device_load 0, i32, x, r6, u8_u9, r26, unsigned";
    let mut gerr = vec![];
    for (key, m) in obj(&grid) {
        let (a_, b_) = key.split_once(',').unwrap();
        let (a_, b_): (u32, u32) = (a_.parse().unwrap(), b_.parse().unwrap());
        let mut lines: Vec<String> = (0..a_).map(|i| format!("iadd r{}, r{}, 1", 14 + i % 4, 14 + i % 4)).collect();
        lines.push(lg.into());
        lines.extend((0..b_).map(|i| format!("iadd r{}, r{}, 1", 18 + i % 4, 18 + i % 4)));
        lines.push("wait 0".into());
        let body = rep(&lines.iter().flat_map(|x| e(x)).collect::<Vec<u8>>(), 8);
        let pr = (v.pred_loop(&body, MEM_CONSTS, 1) - empty_meas) / 8.0;
        let m = m.as_f64().unwrap();
        gerr.push((pr - m) / m);
    }
    let (md, _, mx, _) = stats(&gerr);
    v.check(md <= 0.03 && mx <= 0.08, format!("load-after-ALU grid (FIT, 30 points): median |err| {}, max {}", f3(md), f3(mx)));

    // ---- memory state probes (FIT)
    let st_ = store(5, 8, 26);
    let l_ = e("device_load 0, i32, x, r6, u8_u9, r26, unsigned");
    let a_ = e("iadd r14, r14, 1");
    let w_ = e("wait 0");
    let ts_ = tg(false, "i32", 5, None, 1, 1);
    let i8_: Vec<u8> = (15..23).flat_map(|r| e(&format!("iadd r{}, r{}, 1", r, r))).collect();
    let ms: Vec<(&str, Vec<u8>)> = vec![
        ("store", cat(&[&st_])), ("store, wait", cat(&[&st_, &w_])), ("iadd, store", cat(&[&a_, &st_])),
        ("iadd, store, wait", cat(&[&a_, &st_, &w_])), ("store, store", cat(&[&st_, &st_])),
        ("store, 8 iadd", cat(&[&st_, &i8_])), ("iadd, wait, store", cat(&[&a_, &w_, &st_])),
        ("iadd, store, store", cat(&[&a_, &st_, &st_])), ("tg store", cat(&[&ts_])), ("iadd, tg store", cat(&[&a_, &ts_])),
        ("load, wait", cat(&[&l_, &w_])), ("load, iadd, wait", cat(&[&l_, &a_, &w_])), ("iadd, load, wait", cat(&[&a_, &l_, &w_])),
        ("iadd, wait, load, wait", cat(&[&a_, &w_, &l_, &w_])), ("iadd, store, load, wait", cat(&[&a_, &st_, &l_, &w_])),
    ];
    let meas_ms = load("memstate.json");
    let r8 = |v: &V, parts: &[u8]| (v.pred_loop(&rep(parts, 8), MEM_CONSTS, 1) - empty_meas) / 8.0;
    let mut merr = vec![];
    for (name, body) in &ms {
        let m = meas_ms[name].as_f64().unwrap();
        merr.push((r8(&v, body) - m) / m);
    }
    let (md, _, mx, _) = stats(&merr);
    v.check(md <= 0.04 && mx <= 0.15, format!("memory state probes (FIT, 15 cases): median |err| {}, max {}", f3(md), f3(mx)));

    // ---- memory-space transitions (FIT)
    let tl_ = tg(true, "i32", 7, None, 0, 1);
    let sp: Vec<(&str, Vec<u8>)> = vec![
        ("dev store, dev store", cat(&[&st_, &st_])), ("tg store, tg store", cat(&[&ts_, &ts_])),
        ("dev store, tg store", cat(&[&st_, &ts_])), ("tg store, dev store", cat(&[&ts_, &st_])),
        ("dev load, wait, tg load, wait", cat(&[&l_, &w_, &tl_, &w_])), ("dev store, tg load, wait", cat(&[&st_, &tl_, &w_])),
        ("tg store, dev load, wait", cat(&[&ts_, &l_, &w_])),
    ];
    let meas_sp = load("spaces.json");
    let mut serr = vec![];
    for (name, body) in &sp {
        let m = meas_sp[name].as_f64().unwrap();
        serr.push((r8(&v, body) - m) / m);
    }
    let (md, _, mx, _) = stats(&serr);
    v.check(md <= 0.03 && mx <= 0.05, format!("memory-space transitions (FIT, 7 cases): median |err| {}, max {}", f3(md), f3(mx)));

    // ---- idle waits (FIT)
    let wt = load("waits.json");
    let mut werr = vec![];
    for (key, m) in obj(&wt) {
        let (slot, k) = key.split_once(',').unwrap();
        let k: usize = k.parse().unwrap();
        if ![1, 8, 16, 30].contains(&k) {
            continue;
        }
        let body = rep(&rep(&e(&format!("wait {}", slot)), k), 8);
        let m = m.as_f64().unwrap();
        werr.push(((v.pred_loop(&body, DEFAULT_CONSTS, 1) - empty_meas) / 8.0 - m) / m);
    }
    let (md, _, mx, _) = stats(&werr);
    v.check(mx <= 0.08, format!("idle wait cost, 1-30 waits per repetition (FIT, 8 cases): median |err| {}, max {}", f3(md), f3(mx)));

    // ---- review2
    let r2 = load("review2.json");
    let w1_ = e("wait 1");
    let tlr = tg(true, "i32", 7, None, 0, 1);
    let i8r: Vec<u8> = (0..8).flat_map(|i| e(&format!("iadd r{}, r{}, 1", 15 + i % 8, 15 + i % 8))).collect();
    let r2v = |n: &str| r2[n].as_f64().unwrap();
    let cases_c: Vec<(&str, Vec<u8>)> = vec![
        ("C tg load, wait", cat(&[&tlr, &w_])), ("C tg load, iadd, wait", cat(&[&tlr, &a_, &w_])),
        ("C tg load, 8 iadd, wait", cat(&[&tlr, &i8r, &w_])),
    ];
    let cerr: Vec<f64> = cases_c.iter().map(|(n, b)| (r8(&v, b) - r2v(n)) / r2v(n)).collect();
    let (md, _, mx, _) = stats(&cerr);
    v.check(mx <= 0.10, format!("threadgroup load deferral (FIT, review2 C, 3 cases): median |err| {}, max {}", f3(md), f3(mx)));
    let werr: Vec<f64> = [0usize, 1, 2, 3, 4, 6, 8]
        .iter()
        .map(|&k| {
            let n = format!("B iadd, {} wait1, load, wait0", k);
            (r8(&v, &cat(&[&a_, &rep(&w1_, k), &l_, &w_])) - r2v(&n)) / r2v(&n)
        })
        .collect();
    let (md, _, mx, _) = stats(&werr);
    v.check(md <= 0.05 && mx <= 0.10, format!("load quiet-time window (iadd, k idle waits, load; 7 cases): median |err| {}, max {}", f3(md), f3(mx)));
    let info: Vec<String> = [
        ("B iadd, 4 wait1, store", cat(&[&a_, &rep(&w1_, 4), &st_])), ("B iadd, 8 wait1, store", cat(&[&a_, &rep(&w1_, 8), &st_])),
        ("E iadd, store, load, wait", cat(&[&a_, &st_, &l_, &w_])), ("A store, 8 iadd", cat(&[&st_, &i8r])),
    ]
    .iter()
    .map(|(n, b)| format!("{}: measured {:.1}, predicted {:.1}", n, r2v(n), r8(&v, b)))
    .collect();
    println!("info known residuals (store window / store->load overlap / first store of a trip): {}", info.join("; "));

    // ---- wait drain (FIT)
    let wd = load("waitdrain.json");
    let mut derr = vec![];
    for (n, t) in [("iadd", "iadd r{r}, r{r}, 1"), ("rcp", "rcp r{r}, r{r}"), ("fadd32", "fadd32 r{r}, r{r}, r24"),
                   ("imadd", "imadd r{r}, r{r}, r25, r24")] {
        let b: Vec<u8> = (0..8).flat_map(|k| cat(&[&e(&t.replace("{r}", &(5 + k).to_string())), &w_])).collect();
        let m = wd[&format!("{}|True", n)].as_f64().unwrap();
        derr.push(((v.pred_loop(&b, DEFAULT_CONSTS, 1) - empty_meas) / 8.0 - m) / m);
    }
    let (md, _, mx, _) = stats(&derr);
    v.check(mx <= 0.10, format!("wait drain after ALU work (FIT, 4 cases): median |err| {}, max {}", f3(md), f3(mx)));

    // ---- exec push/pop and while_icmp chains (FIT)
    let cs = load("ctlspacer.json");
    let pu = e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1");
    let po = e("pop_exec r0l, 1");
    let w0 = e("wait 0");
    let ia = |r: u32| e(&format!("iadd r{}, r{}, 1", r, r));
    let ld = e("device_load 0, i32, x, r6, u8_u9, r26, unsigned");
    let stc = store(5, 8, 26);
    let pair = cat(&[&pu, &po]);
    let mut cm: Vec<(String, Vec<u8>)> = vec![
        ("pair x1".into(), pair.clone()), ("pair x4".into(), rep(&pair, 4)), ("pair x16".into(), rep(&pair, 16)),
        ("push, iadd, pop, iadd".into(), cat(&[&pu, &ia(5), &po, &ia(6)])), ("4 pairs, wait".into(), cat(&[&rep(&pair, 4), &w0])),
        ("iadd, 4 pairs, wait".into(), cat(&[&ia(5), &rep(&pair, 4), &w0])),
    ];
    for k in [1usize, 2, 4, 8] {
        cm.push((format!("load,{}", k), cat(&[&ia(14), &rep(&pair, k), &ld, &w0])));
        cm.push((format!("store,{}", k), cat(&[&ia(14), &rep(&pair, k), &stc])));
    }
    let wh = e("while_icmp r0l, ult, r3, r4, 1");
    cm.extend(vec![
        ("while x4".into(), rep(&wh, 4)), ("while, iadd".into(), cat(&[&wh, &ia(5)])),
        ("while, iadd, iadd same chain".into(), cat(&[&wh, &ia(5), &ia(5)])), ("while, while, iadd".into(), cat(&[&wh, &wh, &ia(5)])),
        ("while, fadd32 chain".into(), cat(&[&wh, &e("fadd32 r5, r5, r24")])), ("4 while, wait".into(), cat(&[&rep(&wh, 4), &w0])),
        ("iadd, 4 while, load, wait".into(), cat(&[&ia(14), &rep(&wh, 4), &ld, &w0])),
    ]);
    let mut cerr = vec![];
    for (n, b) in &cm {
        let mm = if cs[n].is_object() { cs[n]["measured"].as_f64().unwrap() } else { cs[n].as_f64().unwrap() };
        cerr.push(((v.pred_loop(&rep(b, 8), MEM_CONSTS, 1) - empty_meas) / 8.0 - mm) / mm);
    }
    let (md, _, mx, _) = stats(&cerr);
    v.check(mx <= 0.10, format!("exec push/pop and while_icmp chains, before wait / load / store (FIT, {} cases): median |err| {}, max {}", cerr.len(), f3(md), f3(mx)));

    // ---- transcendental pairs and mixes (FIT)
    let spairs = load("sfupairs.json");
    let smx = load("sfumix.json");
    let ops_ = |n: &str| -> &'static str {
        match n {
            "log2" => "log2 r{r}, r{r}",
            "floor" => "floor r{r}, r{r}",
            "rcp" => "rcp r{r}, r{r}",
            "exp2" => "exp2 r{r}, r{r}",
            "rsqrt" => "rsqrt r{r}, r{r}",
            "imadd" => "imadd r{r}, r{r}, r25, r24",
            "bitrev" => "bitrev r{r}, r{r}",
            "convert" => "convert u32_to_f, r{r}, r{r}, rte",
            "fadd32" => "fadd32 r{r}, r{r}, r24",
            _ => panic!("op {}", n),
        }
    };
    let pair_body = |n: &str| -> Vec<u8> {
        let (a, b) = n.split_once('+').unwrap();
        rep(&(0..8).flat_map(|k| e(&ops_(if k % 2 == 0 { a } else { b }).replace("{r}", &(5 + k).to_string()))).collect::<Vec<u8>>(), 4)
    };
    let mut sb: Vec<(String, Vec<u8>)> = obj(&spairs).keys().map(|n| (n.clone(), pair_body(n))).collect();
    let sm: Vec<(&str, Vec<&str>)> = vec![
        ("log2, floor, rcp (independent)", vec!["log2 r6, r6", "floor r7, r7", "rcp r8, r8"]),
        ("4 independent rcp", vec!["rcp r6, r6", "rcp r7, r7", "rcp r8, r8", "rcp r9, r9"]),
        ("rcp + fadd32 + imadd", vec!["rcp r6, r6", "fadd32 r7, r7, r24", "imadd r8, r8, r25, r24"]),
        ("8 independent floor", (5..13).map(|r| Box::leak(format!("floor r{}, r{}", r, r).into_boxed_str()) as &str).collect()),
        ("rcp chain + 3 iadd", vec!["rcp r6, r6", "iadd r7, r7, 1", "iadd r8, r8, 1", "iadd r9, r9, 1"]),
    ];
    for (n, l) in &sm {
        sb.push((n.to_string(), rep(&l.iter().flat_map(|x| e(x)).collect::<Vec<u8>>(), 8)));
    }
    for (groups, key, lim) in [(1usize, "one", 0.08), (4, "four", 0.27)] {
        let mut serr = vec![];
        for (n, b) in &sb {
            let src = if spairs.get(n).is_some() { &spairs[n] } else { &smx[n] };
            let m = src[key].as_f64().unwrap();
            serr.push((v.pred_loop(b, DEFAULT_CONSTS, groups) - m) / m);
        }
        let (md, _, mx, _) = stats(&serr);
        v.check(mx <= lim, format!("transcendental pairs and mixes, {} group(s)/core (FIT, {} cases): median |err| {}, max {} (limit {})", groups, serr.len(), f3(md), f3(mx), py_float(lim)));
    }
    let si = load("sfuimad.json");
    for (groups, key) in [(1usize, "one"), (4, "four")] {
        let mut ierr = vec![];
        let mut worst: Vec<(f64, String, f64, f64)> = vec![];
        for (n, val) in obj(&si) {
            let pr = v.pred_loop(&pair_body(n), DEFAULT_CONSTS, groups);
            let m = val[key].as_f64().unwrap();
            ierr.push((pr - m) / m);
            worst.push((ierr.last().unwrap().abs(), n.clone(), m, pr));
        }
        let (md, _, mx, _) = stats(&ierr);
        let w = worst
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.partial_cmp(&b.2).unwrap()))
            .unwrap();
        println!(
            "info transcendental + integer-multiply/bit-field/convert pairs, {} group(s)/core ({} cases): median |err| {}, max {} ({}: measured {:.1}, predicted {:.1})",
            groups, ierr.len(), f3(md), f3(mx), w.1, w.2, w.3
        );
    }

    // ---- device store after k idle waits (FIT)
    let ssr = load("storeseries.json");
    let mut serr = vec![];
    for (n, val) in obj(&ssr) {
        if n.starts_with("store,") {
            continue;
        }
        let k: usize = n.split_whitespace().nth(1).unwrap().parse().unwrap();
        let wv = e(if n.contains("wait1") { "wait 1" } else { "wait 0" });
        let b = rep(&cat(&[&e("iadd r14, r14, 1"), &rep(&wv, k), &store(5, 8, 26)]), 8);
        let m = val["measured"].as_f64().unwrap();
        serr.push(((v.pred_loop(&b, MEM_CONSTS, 1) - empty_meas) / 8.0 - m) / m);
    }
    let (md, _, mx, _) = stats(&serr);
    v.check(mx <= 0.05, format!("device store after k idle waits (FIT, {} cases, k = 0..16, both slots): median |err| {}, max {}", serr.len(), f3(md), f3(mx)));
    let info: Vec<String> = obj(&ssr)
        .iter()
        .filter(|(n, _)| n.starts_with("store,"))
        .map(|(n, val)| format!("{}: measured {:.1}, predicted {:.1}", n, val["measured"].as_f64().unwrap(), val["predicted"].as_f64().unwrap()))
        .collect();
    println!("info store, k iadd (first store of a trip, open residual): {}", info.join("; "));

    // ---- random memory kernels
    for (fname, role) in [("heldout-mem24.json", "DISCOVERY (revealed the wait drain)"), ("heldout-mem24b.json", "HELD OUT")] {
        let rk = load(fname);
        for (gm, groups, lim) in [("one", 1usize, if fname.contains("b.json") { 0.07 } else { 0.08 }), ("gpu", 4, 0.12)] {
            let mut errs = vec![];
            for x in rk.as_array().unwrap() {
                let (_, body) = mem_kernel(x["seed"].as_u64().unwrap(), 24);
                let pr = v.pred_loop(&body, MEM_CONSTS, groups);
                let m = x[gm]["measured"].as_f64().unwrap();
                errs.push((pr - m) / m);
            }
            let (md, p90, mx, b) = stats(&errs);
            v.check(md <= lim, format!("{} memory kernels ({}), {} group(s)/core: median |err| {} p90 {} max {} bias {:+.3} (limit {})",
                                       fname, role, groups, f3(md), f3(p90), f3(mx), b, py_float(lim)));
        }
    }

    // ---- random ALU kernels
    for (fname, role, lim1, lim4, blim1, blim4) in [
        ("random-kernels.json", "SELECTION", 0.07, 0.12, 0.10, 0.17),
        ("heldout24.json", "HELD OUT (used to choose transcendental blocking)", 0.06, 0.12, 0.09, 0.15),
        ("heldout48.json", "HELD OUT (used to choose transcendental blocking)", 0.045, 0.10, 0.05, 0.12),
        ("heldout32c.json", "HELD OUT (measured after all current rules)", 0.05, 0.05, 0.07, 0.07),
    ] {
        let rk = load(fname);
        for (gm, groups, lim, blim) in [("one", 1usize, lim1, blim1), ("gpu", 4, lim4, blim4)] {
            let (mut errs, mut berrs) = (vec![], vec![]);
            for x in rk.as_array().unwrap() {
                let len = x["ops"].as_array().unwrap().len();
                let (_, body) = kernels::kernel(x["seed"].as_u64().unwrap(), len);
                let pr = v.pred_loop(&body, DEFAULT_CONSTS, groups);
                let m = x[gm]["measured"].as_f64().unwrap();
                errs.push((pr - m) / m);
                berrs.push(((pr - empty_pred) - (m - empty_meas)) / (m - empty_meas));
            }
            let (md, p90, mx, b) = stats(&errs);
            let (bmd, bp90, bmx, _) = stats(&berrs);
            v.check(md <= lim && bmd <= blim, format!(
                "{} ({}), {} group(s)/core: whole trip median {} p90 {} max {} bias {:+.3} | body only median {} p90 {} max {} (limits {}/{})",
                fname, role, groups, f3(md), f3(p90), f3(mx), b, f3(bmd), f3(bp90), f3(bmx), py_float(lim), py_float(blim)));
        }
    }
    // ---- core scaling: 1..32 SIMD groups per core in one workgroup (scripts/scale_probes.py)
    let sc = repo_root().join("results/2026-10-11-core-scaling");
    for (f, lim) in [("grid.json", 0.05), ("chains.json", 0.05)] {
        let d = read_json(&sc.join(f)).unwrap();
        let steps: Vec<u32> = d["steps"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let t_ = 10u32;
        let mut errs = vec![];
        let mut by_g: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
        for r in d["rows"].as_array().unwrap() {
            let Some(meas) = r.get("core_cycles_per_step").and_then(|x| x.as_f64()) else { continue };
            if r.get("idle").and_then(|x| x.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
                continue;
            }
            let (op, ch, g) = (r["op"].as_str().unwrap(), r["chains"].as_u64().unwrap() as u32, r["groups"].as_u64().unwrap());
            let c: Vec<f64> = steps.iter().map(|&s| v.sim(&kernels::bench_loop(op, ch, s, t_), g as usize)).collect();
            let pred = to_ref(c[1] - c[0]) / (t_ * (steps[1] - steps[0])) as f64;
            errs.push((pred - meas) / meas);
            by_g.entry(g).or_default().push(((pred - meas) / meas).abs());
        }
        let (md, p90, mx, b) = stats(&errs);
        let per_g: Vec<String> = by_g.iter().map(|(g, e)| {
            let mut e = e.clone();
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            format!("{}:{:.2}", g, e[e.len() / 2])
        }).collect();
        v.check(md <= lim, format!("core scaling {} (FIT: half units, ports, schedulers; {} rows): median |err| {} p90 {} max {} bias {:+.3} (limit {}); median by groups {}",
            f, errs.len(), f3(md), f3(p90), f3(mx), b, py_float(lim), per_g.join(" ")));
    }
    println!("{} failures", v.fails);
    std::process::exit(if v.fails > 0 { 1 } else { 0 });
}
