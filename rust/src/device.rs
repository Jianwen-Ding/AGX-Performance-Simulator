//! Several contexts on one simulated GPU: each context (a GPU VM) has its own memory and VM slot (what SR21 reads)
//! and one or more queues of commands. Commands run whole, one after another, in an order set by a [`Policy`]; a
//! command that fails (an unmapped access, or a step limit for a kernel that never ends) is recorded as faulted,
//! as the firmware reports it (fence signalled, both timestamps 0: ptx-re-blockers B20), and only its own context's
//! memory is touched.
//!
//! What the hardware does with commands of different VMs (overlap, time slicing, priorities, slot reuse) is not
//! measured yet (ptx-re-blockers B43-B45), so the interleaving is a choice: [`Policy::RoundRobin`] (default) takes
//! one command from each non-empty queue in turn, [`Policy::Submission`] runs them in the order they were submitted.
//! Within a queue, commands always run in order and see each other's writes (B32). Memory shared between VMs is not
//! modelled (B45).

use crate::func::{self, Dispatch, Memory};

/// One command: a dispatch, optionally after a preamble whose stored uniforms become the dispatch's.
#[derive(Clone, Debug)]
pub struct Command {
    pub dispatch: Dispatch,
    pub preamble: Option<Vec<u8>>,
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

    /// Queues a command and returns its timeline point in the queue.
    pub fn submit(&mut self, context: usize, queue: usize, command: Command) -> u64 {
        let q = &mut self.contexts[context].queues[queue];
        q.push(command);
        self.submitted.push((context, queue));
        q.len() as u64
    }

    /// The order in which the queued commands run, as (context, queue, index in queue).
    fn schedule(&self) -> Vec<(usize, usize, usize)> {
        match self.policy {
            Policy::Submission => {
                let mut next = std::collections::HashMap::<(usize, usize), usize>::new();
                self.submitted
                    .iter()
                    .map(|&k| {
                        let n = next.entry(k).or_default();
                        *n += 1;
                        (k.0, k.1, *n - 1)
                    })
                    .collect()
            }
            Policy::RoundRobin => {
                let queues: Vec<(usize, usize, usize)> = self
                    .contexts
                    .iter()
                    .enumerate()
                    .flat_map(|(c, ctx)| ctx.queues.iter().enumerate().map(move |(q, cmds)| (c, q, cmds.len())))
                    .collect();
                let rounds = queues.iter().map(|q| q.2).max().unwrap_or(0);
                (0..rounds).flat_map(|i| queues.iter().filter(move |q| i < q.2).map(move |q| (q.0, q.1, i))).collect()
            }
        }
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
            out.push(done);
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
        Command { dispatch: dispatch(code, buf), preamble: None }
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
        let spin = Command { dispatch: dispatch([e("jmp_exec_any pc+0"), e("stop")].concat(), BUF), preamble: None };
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
            dev.submit(c, 0, Command { dispatch: dispatch(code, BUF), preamble: None });
        }
        let done = dev.run();
        assert!(done.iter().all(|d| d.fault.is_none()), "{done:?}");
        assert_eq!(word(&dev.contexts[a].memory, BUF), 3);
        assert_eq!(word(&dev.contexts[b].memory, BUF), 11);
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
