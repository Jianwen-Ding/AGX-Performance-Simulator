//! Shared immutable instruction cells, indexed by byte PC.
//! Keep common short programs dense; allocate tail pages only when visited.
use std::sync::OnceLock;

const PREFIX_BYTES: usize = 8192;
const PAGE_BYTES: usize = 512;
type Cell<T> = OnceLock<Box<T>>;

pub(crate) struct DecodeCache<T, const STRIDE: usize> {
    first: Vec<Cell<T>>,
    pages: Vec<OnceLock<Box<[Cell<T>]>>>,
}

impl<T, const STRIDE: usize> DecodeCache<T, STRIDE> {
    pub(crate) fn new(bytes: usize) -> Self {
        let slots = bytes.div_ceil(STRIDE);
        Self {
            first: (0..slots.min(PREFIX_BYTES / STRIDE)).map(|_| OnceLock::new()).collect(),
            pages: (0..slots.saturating_sub(PREFIX_BYTES / STRIDE).div_ceil(PAGE_BYTES / STRIDE)).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Caller checks that PC is inside the immutable program. STRIDE=1 also
    /// keeps odd addresses distinct so the functional decoder can reject them.
    #[inline]
    pub(crate) fn at(&self, pc: usize) -> &Cell<T> {
        debug_assert_eq!(pc % STRIDE, 0);
        let slot = pc / STRIDE;
        if slot < self.first.len() {
            &self.first[slot]
        } else {
            self.tail(slot - PREFIX_BYTES / STRIDE)
        }
    }

    #[cold]
    fn tail(&self, tail: usize) -> &Cell<T> {
        let page = self.pages[tail / (PAGE_BYTES / STRIDE)].get_or_init(||
            (0..PAGE_BYTES / STRIDE).map(|_| OnceLock::new()).collect());
        &page[tail % (PAGE_BYTES / STRIDE)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_stays_sparse_and_boundary_cells_are_distinct() {
        let cache = DecodeCache::<usize, 2>::new(1_048_577); // odd final byte is still a slot
        assert_eq!(cache.first.len(), PREFIX_BYTES / 2);
        assert!(cache.pages.iter().all(|p| p.get().is_none()));
        for pc in [0, 8190, 8192, 8702, 8704, 1_048_576] {
            cache.at(pc).set(Box::new(pc)).unwrap();
        }
        for pc in [0, 8190, 8192, 8702, 8704, 1_048_576] {
            assert_eq!(**cache.at(pc).get().unwrap(), pc);
        }
        assert_eq!(cache.pages.iter().filter(|p| p.get().is_some()).count(), 3);
        assert!(DecodeCache::<usize, 2>::new(0).first.is_empty());
        assert!(DecodeCache::<usize, 2>::new(PREFIX_BYTES).pages.is_empty());
        let bytes = DecodeCache::<usize, 1>::new(PREFIX_BYTES + 3);
        bytes.at(PREFIX_BYTES).set(Box::new(1)).unwrap();
        bytes.at(PREFIX_BYTES + 1).set(Box::new(2)).unwrap();
        assert_eq!(**bytes.at(PREFIX_BYTES).get().unwrap(), 1);
        assert_eq!(**bytes.at(PREFIX_BYTES + 1).get().unwrap(), 2);
    }

    #[test]
    fn concurrent_page_misses_share_the_same_cells() {
        let cache = DecodeCache::<usize, 2>::new(16384);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for pc in (PREFIX_BYTES..16384).step_by(2) {
                        let cell = cache.at(pc);
                        let _ = cell.set(Box::new(pc));
                        assert_eq!(**cell.get().unwrap(), pc);
                        assert!(std::ptr::eq(cell, cache.at(pc)));
                    }
                });
            }
        });
        assert!(cache.pages.iter().all(|p| p.get().is_some()));
    }
}
