//! Order-preserving parallel map for the offline checkers (independent simulations on every core).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

thread_local! {
    static LOCAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Limit the maps this thread starts to `n` threads (0: the default). For callers that already run one
/// simulation per core, so that the simulations' own maps do not multiply the threads.
pub fn set_local_threads(n: usize) {
    LOCAL.with(|c| c.set(n));
}

/// Worker threads: this thread's limit if set, else `AGX_THREADS` if set, else the available parallelism.
pub fn threads() -> usize {
    let l = LOCAL.with(|c| c.get());
    if l > 0 {
        return l;
    }
    std::env::var("AGX_THREADS").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1))
}

/// `f` over `items`, results in item order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    map_by(items, |_| 0, f)
}

/// `map`, starting the items with the highest `cost` first (balance when a few items dominate).
pub fn map_by<T: Sync, R: Send>(items: &[T], cost: impl Fn(&T) -> u64, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = threads().min(items.len());
    if n <= 1 {
        return items.iter().map(f).collect();
    }
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(cost(&items[i])));
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = (0..items.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..n {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= items.len() {
                    break;
                }
                let i = idx[k];
                let r = f(&items[i]);
                *out[i].lock().unwrap() = Some(r);
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}
