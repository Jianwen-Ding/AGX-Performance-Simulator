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
}

#[derive(Debug, Default)]
pub struct Device {
    pub contexts: Vec<Context>,
    pub policy: Policy,
    /// (context, queue) of each submitted command, in submission order.
    submitted: Vec<(usize, usize)>,
    pub shares: Vec<Share>,
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

    /// Runs every queued command and empties the queues.
    pub fn run(&mut self) -> Vec<Completion> {
        let order = self.schedule();
        let mut clock = 0u64;
        let mut out = Vec::with_capacity(order.len());
        for (n, &(c, q, i)) in order.iter().enumerate() {
            let ctx = &mut self.contexts[c];
            let cmd = &ctx.queues[q][i];
            let mut d = cmd.dispatch.clone();
            if d.launch.context.is_none() {
                d.launch.context = ctx.vm_slot;
            }
            let r = (|| {
                if let Some(p) = &cmd.preamble {
                    d.uniforms = func::run_preamble(p, &d.uniforms, &mut ctx.memory, d.num_registers)
                        .map_err(|e| format!("preamble: {e}"))?;
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
            out.push(done);
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
}
