//! CPython's `random.Random` (MT19937 with `init_by_array` seeding) for integer seeds, with the
//! exact `random()`, `getrandbits`, `_randbelow`, `randint` and `choice` algorithms, so kernels
//! generated from a seed match `scripts/cost_hw.py` bit for bit.

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    idx: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut r = PyRandom { mt: [0; N], idx: N + 1 };
        let mut key: Vec<u32> = vec![];
        let mut s = seed;
        while s > 0 {
            key.push(s as u32);
            s >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.idx = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let kl = key.len();
        let mut k = N.max(kl);
        while k > 0 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= kl {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x80000000;
        self.idx = N;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        const MAG01: [u32; 2] = [0, 0x9908b0df];
        if self.idx >= N {
            let mut kk = 0;
            while kk < N - M {
                let y = (self.mt[kk] & 0x80000000) | (self.mt[kk + 1] & 0x7fffffff);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            while kk < N - 1 {
                let y = (self.mt[kk] & 0x80000000) | (self.mt[kk + 1] & 0x7fffffff);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            let y = (self.mt[N - 1] & 0x80000000) | (self.mt[0] & 0x7fffffff);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `getrandbits(k)` for 0 < k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!(k <= 32);
        if k == 0 {
            return 0;
        }
        (self.genrand_u32() >> (32 - k)) as u64
    }

    /// `_randbelow_with_getrandbits(n)`
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        a + self.randbelow((b - a + 1) as u64) as i64
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len() as u64) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_cpython() {
        // python3 -c "import random; r=random.Random(1000); print(repr(r.random()), r.randint(5,12),
        //             r.getrandbits(32), r.choice(list(range(57))))"
        let mut r = PyRandom::new(1000);
        assert_eq!(r.random(), 0.7773566427005639);
        assert_eq!(r.randint(5, 12), 6);
        assert_eq!(r.getrandbits(32), 1690537025);
        let v: Vec<i32> = (0..57).collect();
        assert_eq!(*r.choice(&v), 22);
        assert_eq!(PyRandom::new(0).random(), 0.8444218515250481);
        assert_eq!(PyRandom::new((1 << 40) + 5).random(), 0.5043802970418443);
    }
}
