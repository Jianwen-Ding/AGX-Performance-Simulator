//! Immutable memory-trace operands. Register values and lane masks are never cached.
use crate::{cost::tg_base, isa::{self, exec::{CoreState, SIMD_WIDTH}, fget, Fields}};

#[derive(Clone, Copy)]
struct DeviceAddress {
    at: Option<u64>, a: Option<u64>, ot: Option<u64>, o: Option<u64>, ou: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost;

    #[test]
    fn cached_addresses_match_legacy_helpers_across_masks_and_fields() {
        let mut st = CoreState::new(128, None);
        for pattern in 0..4 {
            for t in 0..32 {
                st.exec[t] = match pattern { 0 => true, 1 => t % 4 != 1, 2 => t == 0 || t == 31, _ => false };
                st.set_reg64(12, t, 0x1003 + 64 * (t % 3) as i128).unwrap();
                st.set_reg32(5, t, if t % 2 == 0 { t as i128 } else { -1 - t as i128 }).unwrap();
                st.set_reg16(4, t, (t % 8 * 4) as i128).unwrap();
            }
            st.uniforms[24] = 0x2007;
            st.uniforms[25] = 0;
            st.uniforms[26] = 0;
            st.uniforms[27] = 0;
            for fmt in 0..16 { for shift in 0..4 { for mask in [0, 1, 5, 15] {
                for at in 0..2 { for ot in 0..2 { for ou in 0..2 {
                    let f = vec![("F", fmt), ("s", shift), ("mask", mask), ("At", at), ("A", 24),
                        ("Ot", ot), ("O", if ot == 0 { 10 } else { 0xfffd }), ("Ou", ou)];
                    assert_eq!(Access::device(&f).and_then(|a| a.capture(&st)), cost::mem_segments(&f, &st), "{f:?}");
                    assert_eq!(Access::atomic(&f).capture(&st), cost::atomic_words(&f, &st));
                    for key in ["At", "A", "Ot", "O", "Ou", "F", "s", "mask"] {
                        let partial: Fields = f.iter().copied().filter(|(k, _)| *k != key).collect();
                        assert_eq!(Access::device(&partial).and_then(|a| a.capture(&st)), cost::mem_segments(&partial, &st));
                        assert_eq!(Access::atomic(&partial).capture(&st), cost::atomic_words(&partial, &st));
                    }
                } } }
            } } }
            for at in 0..4 { for fmt in 0..8 { for mask in [0, 1, 5, 15] { for ot in 0..2 {
                let f = vec![("At", at), ("A", 4), ("Ot", ot), ("O", if ot == 0 { 10 } else { 7 }), ("F", fmt), ("mask", mask)];
                assert_eq!(Access::threadgroup(&f).capture(&st), cost::tg_words(&f, &st));
                for key in ["At", "A", "Ot", "O", "F", "mask"] {
                    let partial: Fields = f.iter().copied().filter(|(k, _)| *k != key).collect();
                    assert_eq!(Access::threadgroup(&partial).capture(&st), cost::tg_words(&partial, &st));
                }
            } } } }
        }
    }

    #[test]
    fn atomic_device_segments_match_independent_byte_oracles_across_masks_and_alignment() {
        for (op, variant, bytes) in [(0, 2, 4), (3, 1, 8), (3, 3, 8)] {
            let fields = vec![("At", 0), ("A", 24), ("Ot", 0), ("O", 10), ("Ou", 0), ("op", op), ("u5", variant)];
            let access = Access::atomic(&fields);
            for low in 0..64 { for mask in [0u32, 1, u32::MAX, 0x80808080, 0x55555555] {
                let mut st = CoreState::new(128, None);
                let mut expected = std::collections::BTreeSet::new();
                for lane in 0..32 {
                    st.exec[lane] = mask & (1 << lane) != 0;
                    let base = 0x4000 + 128 * lane as i128 + low;
                    let offset = lane as i128 % 3 - 1;
                    st.set_reg64(12, lane, base).unwrap();st.set_reg32(5, lane, offset).unwrap();
                    if st.exec[lane] { for byte in 0..bytes { expected.insert(((base + 4 * offset + byte) / 64) as u64); } }
                }
                let legacy = access.capture(&st).unwrap();
                assert_eq!(access.device_segments(&st, &legacy), Some(expected.into_iter().collect()));
            } }
        }
    }

    #[test]
    fn only_device_segments_drop_duplicate_lanes_and_components() {
        let mut st = CoreState::new(128, None);
        for t in 0..32 { st.set_reg64(12, t, 0x1000).unwrap(); }
        let f = vec![("At", 0), ("A", 24), ("Ot", 1), ("O", 0), ("F", 2), ("s", 0), ("mask", 15)];
        assert_eq!(Access::device(&f).unwrap().capture(&st), Some(vec![64]));
        assert_eq!(Access::atomic(&f).capture(&st), Some(vec![1024; 32]));
        let tg = vec![("At", 2), ("A", 0), ("Ot", 1), ("O", 0), ("F", 2), ("mask", 5)];
        assert_eq!(Access::threadgroup(&tg).capture(&st), Some([0, 2].repeat(32)));
        assert!(Access::decode("device_load.TODO", &f, false).is_none());
        assert!(Access::decode("device_load.TODO", &f, true).is_some());
    }
}

impl DeviceAddress {
    fn new(f: &Fields) -> Self {
        Self { at: fget(f, "At"), a: fget(f, "A"), ot: fget(f, "Ot"), o: fget(f, "O"), ou: fget(f, "Ou") }
    }

    #[inline(always)]
    fn eval(&self, st: &CoreState, t: usize) -> Option<(i128, i128)> {
        let base = if self.at? != 0 { st.ureg64((self.a? >> 1) as usize).ok()? }
            else { st.get_reg64((self.a? >> 1) as usize, t).ok()? };
        let off = if self.ot? != 0 { isa::desc::sign_extend(self.o? as i128, 16) } else {
            let o = st.get_reg32((self.o? >> 1) as usize, t).ok()?;
            if self.ou? == 0 { isa::desc::sign_extend(o, 32) } else { o }
        };
        Some((base, off))
    }
}

#[derive(Clone, Copy)]
struct ThreadgroupAddress {
    at: Option<u64>, a: Option<u64>, ot: Option<u64>, o: Option<u64>,
}

impl ThreadgroupAddress {
    fn new(f: &Fields) -> Self {
        Self { at: fget(f, "At"), a: fget(f, "A"), ot: fget(f, "Ot"), o: fget(f, "O") }
    }

    #[inline(always)]
    fn eval(&self, st: &CoreState, t: usize) -> Option<(i128, i128)> {
        let base = tg_base(st, self.at?, self.a?, t).ok()? as i128;
        let idx = if self.ot? != 0 { self.o? as i128 } else { st.get_reg16(self.o? as usize, t).ok()? };
        Some((base, idx))
    }
}

// Boxed only for memory instructions, so ordinary decoded records remain compact.
pub(crate) struct Access(AccessKind);

enum AccessKind {
    Segments { address: DeviceAddress, item: i128, shift: u64, mask: u64 },
    Atomic { address: DeviceAddress, bytes: u64 },
    Threadgroup { address: ThreadgroupAddress, item: i128, mask: u64 },
}

impl Access {
    pub(crate) fn decode(name: &str, f: &Fields, device_todo: bool) -> Option<Box<Self>> {
        let access = match name {
            "device_load" | "device_store" => Self::device(f)?,
            "device_load.TODO" | "device_store.TODO" if device_todo => Self::device(f)?,
            "atomic" => Self::atomic(f),
            n if n.starts_with("threadgroup_") && n != "threadgroup_barrier" => Self::threadgroup(f),
            _ => return None,
        };
        Some(Box::new(access))
    }

    pub(crate) fn device(f: &Fields) -> Option<Self> {
        // Preserve the tracing helpers' field/default semantics, including when every lane is inactive.
        let fmt = fget(f, "F")?;
        let shift = fget(f, "s")?;
        let packed = matches!(fmt, 8 | 12 | 13);
        let mask = if packed { 15 } else { fget(f, "mask")? };
        let item = match fmt { 1 | 3 | 6 | 7 => 2, 2 => 4, _ => 1 };
        Some(Self(AccessKind::Segments { address: DeviceAddress::new(f), item,
            shift: if packed || shift == 3 { 2 } else { shift }, mask }))
    }

    pub(crate) fn atomic(f: &Fields) -> Self {
        let op = fget(f, "op").unwrap_or(0);
        let variant = fget(f, "u5").unwrap_or(0);
        let bytes = if op >= 11 && variant == 2 { 0 } else if op == 3 && variant & 1 != 0 { 8 } else { 4 };
        Self(AccessKind::Atomic { address: DeviceAddress::new(f), bytes })
    }

    pub(crate) fn threadgroup(f: &Fields) -> Self {
        let item = match fget(f, "F") { Some(1 | 3 | 6 | 7) => 2, Some(2) | None => 4, _ => 1 };
        Self(AccessKind::Threadgroup { address: ThreadgroupAddress::new(f), item, mask: fget(f, "mask").unwrap_or(1) })
    }

    /// Preserve full byte ranges separately from legacy cost word addresses.
    /// `captured` is the already calculated legacy access vector.
    pub(crate) fn device_segments(&self, st: &CoreState, captured: &[u64]) -> Option<Vec<u64>> {
        match &self.0 {
            AccessKind::Segments { .. } => Some(captured.to_vec()),
            AccessKind::Threadgroup { .. } => None,
            AccessKind::Atomic { address, bytes } => {
                let mut segments = vec![];
                if *bytes == 0 { return Some(segments); }
                for t in 0..SIMD_WIDTH {
                    if !st.exec[t] { continue; }
                    let (base, off) = address.eval(st, t)?;
                    let first = (base + 4 * off) as u64;
                    let last = first.checked_add(bytes - 1)?;
                    segments.extend((first >> 6)..=(last >> 6));
                }
                segments.sort_unstable();segments.dedup();Some(segments)
            }
        }
    }

    pub(crate) fn capture(&self, st: &CoreState) -> Option<Vec<u64>> {
        match &self.0 {
            AccessKind::Segments { address, item, shift, mask } => {
                // Reserve once instead of repeatedly growing the lane/component vector.
                // Sorting still handles arbitrary scatter; atomic/TG vectors are not sets.
                let mut segs = Vec::with_capacity(SIMD_WIDTH * (mask & 15).count_ones() as usize);
                for t in 0..SIMD_WIDTH {
                    if !st.exec[t] { continue; }
                    let (base, off) = address.eval(st, t)?;
                    let first = (base & !(item - 1)) + (off << shift) * item;
                    for c in 0..4 {
                        if mask & (1 << c) != 0 {
                            segs.push(((first + c * item) >> 6) as u64);
                        }
                    }
                }
                segs.sort_unstable();
                segs.dedup();
                Some(segs)
            }
            AccessKind::Atomic { address, .. } => {
                let mut words = Vec::with_capacity(SIMD_WIDTH);
                for t in 0..SIMD_WIDTH {
                    if st.exec[t] {
                        let (base, off) = address.eval(st, t)?;
                        words.push(((base + 4 * off) >> 2) as u64);
                    }
                }
                Some(words)
            }
            AccessKind::Threadgroup { address, item, mask } => {
                let mut words = Vec::with_capacity(4 * SIMD_WIDTH);
                for t in 0..SIMD_WIDTH {
                    if !st.exec[t] { continue; }
                    let (base, idx) = address.eval(st, t)?;
                    for c in 0..4 {
                        if mask & (1 << c) != 0 { words.push(((base + 4 * item * idx + item * c) / 4) as u64); }
                    }
                }
                Some(words)
            }
        }
    }
}
