pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut r = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        r.step();
        r.state = r.state.wrapping_add(seed);
        r.step();
        r
    }

    fn step(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.step()
    }

    pub fn below(&mut self, bound: u32) -> u32 {
        let t = bound.wrapping_neg() % bound;
        loop {
            let r = self.step();
            if r >= t {
                return r % bound;
            }
        }
    }

    pub fn unit(&mut self) -> f64 {
        (self.step() as f64) / 4294967296.0
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = (self.step() as f64 + 1.0) / 4294967297.0;
        let u2 = (self.step() as f64) / 4294967296.0;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
