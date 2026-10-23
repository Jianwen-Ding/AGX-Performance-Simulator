//! Several contexts on one simulated GPU: each context (a GPU VM) has its own memory and VM slot (what SR21 reads)
//! and one or more queues of commands. Commands run whole, one after another, in an order set by a [`Policy`]; a
//! command that fails (an unmapped access, or a step limit for a kernel that never ends) is recorded as faulted,
//! as the firmware reports it (fence signalled, both timestamps 0: ptx-re-blockers B20), and only its own context's
//! memory is touched.
//!
//! Measured on the M2 Max (ptx-re-blockers B43, B45): commands of different VMs run at the same time, their workgroups
//! interleaved on the same cores, with no time slicing and no implicit order; the only order between queues or VMs is
//! a syncobj wait on another queue's timeline point, which also makes all of that command's writes visible. So a
//! program whose commands in different queues touch the same memory without a wait has no defined result on the
//! hardware, and running commands whole in a [`Policy`]'s order gives one of the hardware's possible outcomes for
//! every program that synchronizes. Waits are [`Command::wait_for`]: a command runs only after the points it waits
//! for. [`Policy::RoundRobin`] (default) takes the next runnable command of each queue in turn,
//! [`Policy::Submission`] the earliest submitted runnable one. Within a queue, commands always run in order and see
//! each other's writes (B32). One buffer object bound in two contexts ([`Device::share`], B36 / B45) is the same
//! memory at each context's own address.
//!
//! Timing ([`Device::timing`], off by default): each command's time alone comes from the dispatch model
//! ([`crate::dispatch`], on the path the functional run takes), and the commands share the GPU as B43 measured: a
//! command starts once its queue's previous command and the points it waits for have ended (plus B45's hand-over for
//! a wait on another queue), runs at full speed while the commands running together need no more than the GPU's
//! workgroup slots, and otherwise all of them slow down alike, by the slots they need over what the GPU has, less
//! the throughput interleaving gains ([`Timing::sharing_gain`]); there is no time slicing and no switch cost.

use crate::cost::Params;
use crate::dispatch;
use crate::func::{self, Dispatch, Memory};

/// One command: a dispatch, optionally after a preamble whose stored uniforms become the dispatch's.
#[derive(Clone, Debug)]
pub struct Command {
    pub dispatch: Dispatch,
    pub preamble: Option<Vec<u8>>,
    /// Timeline points of other queues this command waits for: (context, queue, 1-based point).
    pub wait_for: Vec<(usize, usize, u64)>,
}

impl Command {
    pub fn new(dispatch: Dispatch) -> Command {
        Command { dispatch, preamble: None, wait_for: Vec::new() }
    }
}

/// One buffer object bound in two contexts: `len` bytes at `a.1` in context `a.0` and at `b.1` in context `b.0`.
#[derive(Clone, Copy, Debug)]
pub struct Share {
    pub a: (usize, u64),
    pub b: (usize, u64),
    pub len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Context {
    pub memory: Memory,
    /// The VM slot SR21 reads in this context's commands (unless a command sets its own).
    pub vm_slot: Option<u32>,
    pub queues: Vec<Vec<Command>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Policy {
    /// One command from each queue with work left in turn, contexts in order, then their queues.
    #[default]
    RoundRobin,
    /// In the order `submit` was called.
    Submission,
}

impl Policy {
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "round-robin" => Ok(Policy::RoundRobin),
            "submission" => Ok(Policy::Submission),
            _ => Err(format!("unknown policy {s:?} (round-robin, submission)")),
        }
    }
}

/// The timing model of [`Device::run`].
#[derive(Clone, Debug)]
pub struct Timing {
    pub params: Params,
    /// The GPU: cores, clock, launch costs (the `grid`, `wg`, `regs`, `tg_bytes` and `uniforms` are each command's).
    pub launch: dispatch::Launch,
    /// Combined throughput of commands that together need more workgroup slots than the GPU has, relative to one of
    /// them alone: two full-GPU 0.84 s kernels took 1.49 s together (B43), 2 x 0.84 / 1.49.
    pub sharing_gain: f64,
    /// From the end of a command to the start of a command of another queue that waits for it (B45: ~130 us).
    pub wait_ns: f64,
}

impl Timing {
    pub fn new(params: Params) -> Timing {
        Timing { params, launch: dispatch::Launch::default(), sharing_gain: 2.0 * 0.84 / 1.49, wait_ns: 130_000.0 }
    }
}

/// What happened to one command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub context: usize,
    pub queue: usize,
    /// 1-based position of the command in its queue (its timeline point).
    pub point: u64,
    /// Device order in which it ran (0-based).
    pub order: usize,
    /// Simulated clock: instructions executed on the device before and after it; both 0 for a faulted command, as
    /// the firmware leaves the timestamps of a faulted command.
    pub start: u64,
    pub end: u64,
    pub instructions: u64,
    pub groups: u64,
    /// Why the command failed, if it did.
    pub fault: Option<String>,
    /// With [`Device::timing`]: when it started and ended, in nanoseconds (a faulted command takes no time); None for a
    /// command the dispatch model cannot cost (counted as taking no time).
    pub time_ns: Option<(u64, u64)>,
}

#[derive(Debug, Default)]
pub struct Device {
    pub contexts: Vec<Context>,
    pub policy: Policy,
    /// (context, queue) of each submitted command, in submission order.
    submitted: Vec<(usize, usize)>,
    pub shares: Vec<Share>,
    pub timing: Option<Timing>,
}

impl Device {
    pub fn new(policy: Policy) -> Device {
        Device { policy, ..Default::default() }
    }

    /// Adds a context with one queue and returns its index.
    pub fn create_context(&mut self, memory: Memory, vm_slot: Option<u32>) -> usize {
        self.contexts.push(Context { memory, vm_slot, queues: vec![Vec::new()] });
        self.contexts.len() - 1
    }

    /// Adds a queue to `context` and returns its index.
    pub fn create_queue(&mut self, context: usize) -> usize {
        let c = &mut self.contexts[context];
        c.queues.push(Vec::new());
        c.queues.len() - 1
    }

    /// Binds context `a`'s `len` bytes at `va_a` in context `b` at `va_b` as well (a dma-buf import, or one object
    /// bound in two VMs): they start as `a`'s contents and stay the same memory. `va_b` may already be mapped in `b`
    /// with at least `len` bytes; otherwise it is mapped.
    pub fn share(&mut self, a: usize, va_a: u64, b: usize, va_b: u64, len: usize) -> Result<(), String> {
        let bytes = self.contexts[a].memory.read_bytes(va_a, len)?;
        let mb = &mut self.contexts[b].memory;
        if mb.write_bytes(va_b, &bytes).is_err() {
            mb.map(va_b, bytes);
        }
        self.shares.push(Share { a: (a, va_a), b: (b, va_b), len });
        Ok(())
    }

    /// After a command of context `c`: what it wrote to shared objects is what the other contexts see.
    fn propagate(&mut self, c: usize) -> Result<(), String> {
        for s in self.shares.clone() {
            for (from, to) in [(s.a, s.b), (s.b, s.a)] {
                if from.0 == c {
                    let bytes = self.contexts[c].memory.read_bytes(from.1, s.len)?;
                    self.contexts[to.0].memory.write_bytes(to.1, &bytes)?;
                }
            }
        }
        Ok(())
    }

    /// Queues a command and returns its timeline point in the queue.
    pub fn submit(&mut self, context: usize, queue: usize, command: Command) -> u64 {
        let q = &mut self.contexts[context].queues[queue];
        q.push(command);
        self.submitted.push((context, queue));
        q.len() as u64
    }

    /// The order in which the queued commands run, as (context, queue, index in queue): a command is runnable once the
    /// commands before it in its queue and every point it waits for have run. Commands whose waits can never be met
    /// (a point that is never submitted, or a cycle) are left out and reported as faulted by `run`.
    fn schedule(&self) -> Vec<(usize, usize, usize)> {
        let queues: Vec<(usize, usize)> = self
            .contexts
            .iter()
            .enumerate()
            .flat_map(|(c, ctx)| (0..ctx.queues.len()).map(move |q| (c, q)))
            .collect();
        let mut next: std::collections::HashMap<(usize, usize), usize> = queues.iter().map(|&k| (k, 0)).collect();
        let len = |k: (usize, usize)| self.contexts[k.0].queues[k.1].len();
        let runnable = |next: &std::collections::HashMap<(usize, usize), usize>, k: (usize, usize)| {
            let i = next[&k];
            i < len(k) && self.contexts[k.0].queues[k.1][i].wait_for.iter().all(|&(c, q, p)| {
                next.get(&(c, q)).is_some_and(|&done| done as u64 >= p)
            })
        };
        // submission rank of every command, for Policy::Submission
        let mut rank = std::collections::HashMap::<(usize, usize, usize), usize>::new();
        let mut seen = std::collections::HashMap::<(usize, usize), usize>::new();
        for (n, &k) in self.submitted.iter().enumerate() {
            let i = seen.entry(k).or_default();
            rank.insert((k.0, k.1, *i), n);
            *i += 1;
        }
        let mut order = Vec::new();
        let mut cursor = 0;
        loop {
            let pick = match self.policy {
                Policy::Submission => queues
                    .iter()
                    .filter(|&&k| runnable(&next, k))
                    .min_by_key(|&&k| rank.get(&(k.0, k.1, next[&k])).copied().unwrap_or(usize::MAX))
                    .copied(),
                Policy::RoundRobin => (0..queues.len())
                    .map(|j| queues[(cursor + j) % queues.len()])
                    .find(|&k| runnable(&next, k))
                    .inspect(|k| cursor = queues.iter().position(|q| q == k).unwrap() + 1),
            };
            let Some(k) = pick else { break };
            order.push((k.0, k.1, next[&k]));
            *next.get_mut(&k).unwrap() += 1;
        }
        order
    }

    /// The time of `d` alone on the GPU in nanoseconds, and the fraction of the GPU's workgroup slots it fills.
    fn time_alone(t: &Timing, d: &Dispatch, mem: &Memory) -> Result<(f64, f64), String> {
        let mut l = t.launch.clone();
        (l.grid, l.wg, l.tg_bytes, l.uniforms) = (d.grid, d.wg, d.shared_bytes, d.uniforms.clone());
        if d.launch.regs != 0 {
            l.regs = d.launch.regs;
        }
        let prepared = func::PreparedDispatch::new(d)?;
        let tracer = |wg: [u32; 3]| prepared.trace_workgroup(mem, wg);
        let r = dispatch::simulate_with(&d.code, &l, &t.params, Some(&tracer))?;
        let slots = l.cores as f64 * r.wgs_per_core.max(1) as f64;
        Ok((r.time_ns, (r.workgroups as f64 / slots).min(1.0)))
    }

    /// Start and end of each command of `order` (B43's sharing, see the module documentation), given its time alone
    /// and slot fraction.
    fn overlap(&self, t: &Timing, order: &[(usize, usize, usize)], alone: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let index: std::collections::HashMap<(usize, usize, usize), usize> =
            order.iter().enumerate().map(|(n, &k)| (k, n)).collect();
        // what each command waits for: the previous command of its queue (no hand-over) and its wait points
        let deps: Vec<Vec<(usize, f64)>> = order
            .iter()
            .map(|&(c, q, i)| {
                let mut v = Vec::new();
                if i > 0 {
                    v.extend(index.get(&(c, q, i - 1)).map(|&n| (n, 0.0)));
                }
                for &(wc, wq, p) in &self.contexts[c].queues[q][i].wait_for {
                    if p >= 1 {
                        let hand_over = if (wc, wq) == (c, q) { 0.0 } else { t.wait_ns };
                        v.extend(index.get(&(wc, wq, p as usize - 1)).map(|&n| (n, hand_over)));
                    }
                }
                v
            })
            .collect();
        let n = order.len();
        let (mut start, mut end) = (vec![f64::NAN; n], vec![f64::NAN; n]);
        let mut left: Vec<f64> = alone.iter().map(|a| a.0).collect();
        let mut running: Vec<usize> = Vec::new();
        let mut now = 0.0f64;
        loop {
            // commands whose dependencies have all ended: their release times
            let release = |k: usize| -> Option<f64> {
                deps[k].iter().try_fold(0.0f64, |r, &(d, h)| (!end[d].is_nan()).then(|| r.max(end[d] + h)))
            };
            for k in 0..n {
                if start[k].is_nan() && release(k).is_some_and(|r| r <= now) {
                    start[k] = now;
                    running.push(k);
                }
            }
            let need: f64 = running.iter().map(|&k| alone[k].1).sum();
            let speed = if need <= 1.0 { 1.0 } else { (t.sharing_gain / need).min(1.0) };
            let finish = running.iter().map(|&k| now + left[k] / speed).fold(f64::INFINITY, f64::min);
            let next_release = (0..n)
                .filter(|&k| start[k].is_nan())
                .filter_map(release)
                .filter(|&r| r > now)
                .fold(f64::INFINITY, f64::min);
            let next = finish.min(next_release);
            if !next.is_finite() {
                break;
            }
            for &k in &running {
                left[k] -= (next - now) * speed;
            }
            now = next;
            running.retain(|&k| {
                let done = left[k] <= 1e-9 * alone[k].0.max(1.0);
                if done {
                    end[k] = now;
                }
                !done
            });
        }
        start.into_iter().zip(end).map(|(s, e)| (if s.is_nan() { 0.0 } else { s }, if e.is_nan() { 0.0 } else { e })).collect()
    }

    /// Runs every queued command and empties the queues.
    pub fn run(&mut self) -> Vec<Completion> {
        let order = self.schedule();
        let mut clock = 0u64;
        let mut out = Vec::with_capacity(order.len());
        // per command run: (time alone, slot fraction), for the timing model
        let mut alone = Vec::with_capacity(order.len());
        for (n, &(c, q, i)) in order.iter().enumerate() {
            let ctx = &mut self.contexts[c];
            let cmd = &ctx.queues[q][i];
            let mut d = cmd.dispatch.clone();
            if d.launch.context.is_none() {
                d.launch.context = ctx.vm_slot;
            }
            let mut cost = None;
            let r = (|| {
                if let Some(p) = &cmd.preamble {
                    d.uniforms = func::run_preamble(p, &d.uniforms, &mut ctx.memory, d.num_registers)
                        .map_err(|e| format!("preamble: {e}"))?;
                }
                // the timing does not change what runs: a command the dispatch model cannot cost has no time
                if let Some(t) = &self.timing {
                    cost = Some(Self::time_alone(t, &d, &ctx.memory));
                }
                func::run(&d, &mut ctx.memory)
            })();
            let mut done = Completion {
                context: c,
                queue: q,
                point: i as u64 + 1,
                order: n,
                start: 0,
                end: 0,
                instructions: 0,
                groups: 0,
                fault: None,
                time_ns: None,
            };
            match r {
                Ok(st) => {
                    done.start = clock;
                    clock += st.instructions;
                    done.end = clock;
                    done.instructions = st.instructions;
                    done.groups = st.groups;
                }
                Err(e) => done.fault = Some(e),
            }
            if let Err(e) = self.propagate(c) {
                done.fault.get_or_insert(format!("shared object: {e}"));
            }
            let costed = done.fault.is_some() || matches!(cost, Some(Ok(_)));
            alone.push((if done.fault.is_some() { None } else { cost.and_then(Result::ok) }, costed));
            out.push(done);
        }
        if let Some(t) = &self.timing {
            let times = self.overlap(t, &order, &alone.iter().map(|a| a.0.unwrap_or_default()).collect::<Vec<_>>());
            for ((done, time), &(_, costed)) in out.iter_mut().zip(times).zip(&alone) {
                done.time_ns = costed.then(|| (time.0.round() as u64, time.1.round() as u64));
            }
        }
        // commands whose waits were never met did not run: report them as faults (fence never signalled)
        for (c, ctx) in self.contexts.iter().enumerate() {
            for (q, cmds) in ctx.queues.iter().enumerate() {
                for i in 0..cmds.len() {
                    if !order.contains(&(c, q, i)) {
                        out.push(Completion {
                            context: c,
                            queue: q,
                            point: i as u64 + 1,
                            order: out.len(),
                            start: 0,
                            end: 0,
                            instructions: 0,
                            groups: 0,
                            fault: Some("not run: waits for a point that is never signalled".into()),
                            time_ns: None,
                        });
                    }
                }
            }
        }
        for ctx in &mut self.contexts {
            ctx.queues.iter_mut().for_each(Vec::clear);
        }
        self.submitted.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{e, get_sr, mov, store};

    const BUF: u64 = 0x10000;

    fn dispatch(code: Vec<u8>, buf: u64) -> Dispatch {
        Dispatch {
            code,
            grid: [32, 1, 1],
            wg: [32, 1, 1],
            shared_bytes: 0,
            uniforms: vec![(8, buf as u32), (9, (buf >> 32) as u32)],
            num_registers: 8,
            max_steps: 1000,
            placement: 0,
            launch: Default::default(),
        }
    }

    /// out[tid] = value, out at u8_u9
    fn store_value(value: u32, buf: u64) -> Command {
        let code = [get_sr(80, 1), mov(2, value), store(2, 8, 1), e("wait 0"), e("stop")].concat();
        Command::new(dispatch(code, buf))
    }

    fn memory() -> Memory {
        let mut m = Memory::default();
        m.map(BUF, vec![0; 128]);
        m
    }

    fn word(m: &Memory, va: u64) -> u32 {
        m.read(va, 4).unwrap() as u32
    }

    #[test]
    fn contexts_have_their_own_memory_at_the_same_address() {
        let mut dev = Device::new(Policy::RoundRobin);
        let a = dev.create_context(memory(), Some(1));
        let b = dev.create_context(memory(), Some(2));
        dev.submit(a, 0, store_value(7, BUF));
        dev.submit(b, 0, store_value(9, BUF));
        let done = dev.run();
        assert!(done.iter().all(|d| d.fault.is_none()), "{done:?}");
        assert_eq!(word(&dev.contexts[a].memory, BUF + 4 * 31), 7);
        assert_eq!(word(&dev.contexts[b].memory, BUF + 4 * 31), 9);
    }

    #[test]
    fn a_faulting_context_leaves_the_others_alone() {
        let mut dev = Device::new(Policy::RoundRobin);
        let a = dev.create_context(memory(), None);
        let b = dev.create_context(memory(), None);
        // a stores outside its memory; b stores twice, before and after a's fault
        dev.submit(a, 0, store_value(1, 0x20000));
        dev.submit(b, 0, store_value(5, BUF));
        dev.submit(b, 0, store_value(6, BUF));
        let done = dev.run();
        let fa = done.iter().find(|d| d.context == a).unwrap();
        assert!(fa.fault.as_deref().is_some_and(|f| f.contains("unmapped")), "{fa:?}");
        assert_eq!((fa.start, fa.end), (0, 0), "a faulted command has no timestamps");
        let bs: Vec<&Completion> = done.iter().filter(|d| d.context == b).collect();
        assert!(bs.iter().all(|d| d.fault.is_none() && d.end > d.start), "{bs:?}");
        assert_eq!(bs.iter().map(|d| d.point).collect::<Vec<_>>(), [1, 2], "a queue runs in order");
        assert_eq!(word(&dev.contexts[b].memory, BUF), 6);
        assert_eq!(word(&dev.contexts[a].memory, BUF), 0, "a's own buffer is untouched");
    }

    #[test]
    fn a_kernel_that_never_ends_is_a_fault_of_its_context_only() {
        let mut dev = Device::new(Policy::RoundRobin);
        let a = dev.create_context(memory(), None);
        let b = dev.create_context(memory(), None);
        let spin = Command::new(dispatch([e("jmp_exec_any pc+0"), e("stop")].concat(), BUF));
        dev.submit(a, 0, spin);
        dev.submit(b, 0, store_value(4, BUF));
        let done = dev.run();
        assert!(done[0].fault.as_deref().is_some_and(|f| f.contains("step limit")), "{done:?}");
        assert!(done[1].fault.is_none());
        assert_eq!(word(&dev.contexts[b].memory, BUF), 4);
    }

    #[test]
    fn sr21_reads_the_vm_slot_of_the_context() {
        let mut dev = Device::new(Policy::Submission);
        let a = dev.create_context(memory(), Some(3));
        let b = dev.create_context(memory(), Some(11));
        for c in [a, b] {
            let code = [get_sr(80, 1), get_sr(21, 2), store(2, 8, 1), e("wait 0"), e("stop")].concat();
            dev.submit(c, 0, Command::new(dispatch(code, BUF)));
        }
        let done = dev.run();
        assert!(done.iter().all(|d| d.fault.is_none()), "{done:?}");
        assert_eq!(word(&dev.contexts[a].memory, BUF), 3);
        assert_eq!(word(&dev.contexts[b].memory, BUF), 11);
    }

    #[test]
    fn a_shared_object_is_the_same_memory_in_both_contexts() {
        // B45: one object bound in VM a at BUF and in VM b at 0x40000; b reads what a wrote once a's command has run
        let mut dev = Device::new(Policy::Submission);
        let a = dev.create_context(memory(), Some(1));
        let b = dev.create_context(Memory::default(), Some(2));
        dev.share(a, BUF, b, 0x40000, 128).unwrap();
        dev.submit(a, 0, store_value(7, BUF));
        let mut copy = Command::new(dispatch([get_sr(80, 1), e("device_load 0, i32, x, r2, u8_u9, r1, unsigned"), e("wait 0"), mov(3, 1), e("iadd r2, r2, r3"),
                                              store(2, 8, 1), e("wait 0"), e("stop")].concat(), 0x40000));
        copy.wait_for = vec![(a, 0, 1)];
        dev.submit(b, 0, copy);
        let done = dev.run();
        assert!(done.iter().all(|d| d.fault.is_none()), "{done:?}");
        assert_eq!(word(&dev.contexts[b].memory, 0x40000 + 4 * 5), 8, "b read a's 7 and stored 8");
        assert_eq!(word(&dev.contexts[a].memory, BUF + 4 * 5), 8, "and a sees b's store");
    }

    #[test]
    fn a_wait_orders_commands_of_different_contexts() {
        let mut dev = Device::new(Policy::Submission);
        let a = dev.create_context(memory(), None);
        let b = dev.create_context(memory(), None);
        let mut first = store_value(1, BUF);
        first.wait_for = vec![(b, 0, 1)];
        dev.submit(a, 0, first);
        dev.submit(b, 0, store_value(2, BUF));
        let ran: Vec<usize> = dev.run().iter().map(|d| d.context).collect();
        assert_eq!(ran, [b, a], "a's command waits for b's although it was submitted first");
    }

    #[test]
    fn a_wait_that_is_never_signalled_is_reported() {
        let mut dev = Device::new(Policy::RoundRobin);
        let a = dev.create_context(memory(), None);
        let mut cmd = store_value(1, BUF);
        cmd.wait_for = vec![(a, 0, 5)];
        dev.submit(a, 0, cmd);
        let done = dev.run();
        assert_eq!(done.len(), 1);
        assert!(done[0].fault.as_deref().is_some_and(|f| f.contains("never signalled")), "{done:?}");
        assert_eq!(word(&dev.contexts[a].memory, BUF), 0);
    }

    #[test]
    fn policies_order_the_queues() {
        let mut dev = Device::new(Policy::RoundRobin);
        let a = dev.create_context(memory(), None);
        let b = dev.create_context(memory(), None);
        let q = dev.create_queue(a);
        for (c, qu) in [(a, 0), (a, 0), (a, q), (b, 0)] {
            dev.submit(c, qu, store_value(1, BUF));
        }
        let ran: Vec<(usize, usize, u64)> = dev.run().iter().map(|d| (d.context, d.queue, d.point)).collect();
        assert_eq!(ran, [(a, 0, 1), (a, q, 1), (b, 0, 1), (a, 0, 2)]);
        dev.policy = Policy::Submission;
        for (c, qu) in [(b, 0), (a, 0), (a, q), (a, 0)] {
            dev.submit(c, qu, store_value(1, BUF));
        }
        let ran: Vec<(usize, usize, u64)> = dev.run().iter().map(|d| (d.context, d.queue, d.point)).collect();
        assert_eq!(ran, [(b, 0, 1), (a, 0, 1), (a, q, 1), (a, 0, 2)]);
    }

    fn timing() -> Timing {
        Timing::new(Params::new().unwrap())
    }

    // the overlap model on given times alone (ns) and slot fractions, for commands submitted as (context, queue)
    fn overlap(dev: &mut Device, cmds: &[(usize, usize, Vec<(usize, usize, u64)>)], alone: &[(f64, f64)]) -> Vec<(f64, f64)> {
        for (c, q, w) in cmds {
            let mut cmd = store_value(1, BUF);
            cmd.wait_for = w.clone();
            dev.submit(*c, *q, cmd);
        }
        let order = dev.schedule();
        let alone: Vec<(f64, f64)> = order
            .iter()
            .map(|k| alone[cmds.iter().enumerate().filter(|(_, x)| (x.0, x.1) == (k.0, k.1)).nth(k.2).unwrap().0])
            .collect();
        let times = dev.overlap(&timing(), &order, &alone);
        let mut by_cmd = vec![(0.0, 0.0); cmds.len()];
        for (k, t) in order.iter().zip(times) {
            by_cmd[cmds.iter().enumerate().filter(|(_, x)| (x.0, x.1) == (k.0, k.1)).nth(k.2).unwrap().0] = t;
        }
        by_cmd
    }

    #[test]
    fn full_gpu_commands_of_two_contexts_share_it_as_measured() {
        // B43: two full-GPU 0.84 s kernels started together both take 1.49 s and end together
        let mut dev = Device::new(Policy::RoundRobin);
        let (a, b) = (dev.create_context(memory(), None), dev.create_context(memory(), None));
        let t = overlap(&mut dev, &[(a, 0, vec![]), (b, 0, vec![])], &[(0.84e9, 1.0), (0.84e9, 1.0)]);
        for (s, e) in t {
            assert_eq!(s, 0.0);
            assert!((e - 1.49e9).abs() < 1e3, "{e}");
        }
    }

    #[test]
    fn a_small_command_beside_another_is_not_slowed_down() {
        let mut dev = Device::new(Policy::RoundRobin);
        let (a, b) = (dev.create_context(memory(), None), dev.create_context(memory(), None));
        let t = overlap(&mut dev, &[(a, 0, vec![]), (b, 0, vec![])], &[(1e6, 0.5), (2e5, 0.25)]);
        assert_eq!(t, [(0.0, 1e6), (0.0, 2e5)]);
    }

    #[test]
    fn a_queue_runs_its_commands_one_after_another_and_a_wait_costs_the_hand_over() {
        let mut dev = Device::new(Policy::RoundRobin);
        let (a, b) = (dev.create_context(memory(), None), dev.create_context(memory(), None));
        let t = overlap(&mut dev, &[(a, 0, vec![]), (a, 0, vec![]), (b, 0, vec![(a, 0, 1)])], &[(1e6, 1.0), (1e6, 1.0), (1e6, 0.5)]);
        assert_eq!(t[0], (0.0, 1e6));
        assert_eq!(t[1].0, 1e6, "the second command of a's queue starts when the first ends");
        assert_eq!(t[2].0, 1e6 + 130_000.0, "b starts a hand-over after the point it waits for");
        assert!(t[1].1 > 2e6, "the two overlap and share the GPU");
    }

    #[test]
    fn timed_commands_get_the_dispatch_model_time() {
        let mut dev = Device::new(Policy::RoundRobin);
        dev.timing = Some(timing());
        let a = dev.create_context(memory(), None);
        let b = dev.create_context(memory(), None);
        dev.submit(a, 0, store_value(1, BUF));
        dev.submit(b, 0, store_value(2, BUF));
        dev.submit(a, 0, store_value(3, 0x20000));
        let done = dev.run();
        let (s0, e0) = done[0].time_ns.unwrap();
        assert!(e0 > s0 && s0 == 0, "{done:?}");
        assert_eq!(done[1].time_ns.unwrap().0, 0, "the other context's command runs at the same time");
        assert!(done[2].fault.is_some());
        let (s2, e2) = done[2].time_ns.unwrap();
        assert_eq!((s2, e2), (e0, e0), "a faulted command takes no time");
    }
}
