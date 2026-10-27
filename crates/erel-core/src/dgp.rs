use crate::pcg::Pcg32;

pub struct Synth {
    pub n: usize,
    pub t: usize,
    pub d: usize,
    pub alpha: f64,
    pub sig_lo: f64,
    pub sig_hi: f64,
    pub innov: f64,
}

pub struct Arms {
    pub n: usize,
    pub t: usize,
    pub d: usize,
    pub truth: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

impl Synth {
    pub fn sigma(&self, c: usize) -> f64 {
        if self.d <= 1 {
            self.sig_lo
        } else {
            self.sig_lo + (self.sig_hi - self.sig_lo) * (c as f64) / ((self.d - 1) as f64)
        }
    }

    pub fn generate(&self, seed: u64) -> Arms {
        let (n, t, d) = (self.n, self.t, self.d);
        let mut r = Pcg32::new(seed, 0xE5E1);
        let mut truth = vec![0.0f64; n * t * d];
        let mut a = vec![0.0f64; n * t * d];
        let mut b = vec![0.0f64; n * t * d];
        let keep = (1.0 - self.alpha * self.alpha).max(0.0).sqrt();
        for i in 0..n {
            for c in 0..d {
                let sg = self.sigma(c);
                let mut vel = self.innov * r.normal();
                let mut pos = r.normal();
                for tt in 0..t {
                    let idx = (i * t + tt) * d + c;
                    truth[idx] = pos;
                    a[idx] = pos + sg * r.normal();
                    b[idx] = pos + sg * r.normal();
                    let z = r.normal();
                    vel = self.alpha * vel + keep * self.innov * z;
                    pos += vel;
                }
            }
        }
        Arms { n, t, d, truth, a, b }
    }
}

pub struct ArmsK {
    pub n: usize,
    pub t: usize,
    pub d: usize,
    pub truth: Vec<f64>,
    pub arms: Vec<Vec<f64>>,
}

impl Synth {

    pub fn generate_k(&self, seed: u64, k: usize) -> ArmsK {
        let (n, t, d) = (self.n, self.t, self.d);
        let mut r = Pcg32::new(seed, 0xE5E1);
        let mut truth = vec![0.0f64; n * t * d];
        let mut arms = vec![vec![0.0f64; n * t * d]; k];
        let keep = (1.0 - self.alpha * self.alpha).max(0.0).sqrt();
        for i in 0..n {
            for c in 0..d {
                let sg = self.sigma(c);
                let mut vel = self.innov * r.normal();
                let mut pos = r.normal();
                for tt in 0..t {
                    let idx = (i * t + tt) * d + c;
                    truth[idx] = pos;
                    for a in arms.iter_mut() {
                        a[idx] = pos + sg * r.normal();
                    }
                    let z = r.normal();
                    vel = self.alpha * vel + keep * self.innov * z;
                    pos += vel;
                }
            }
        }
        ArmsK { n, t, d, truth, arms }
    }
}

impl Arms {
    pub fn disp(src: &[f64], n: usize, t: usize, d: usize, _lag: usize) -> Vec<f64> {
        let mut u = vec![0.0f64; n * (t - 1) * d];
        for i in 0..n {
            for tt in 0..(t - 1) {
                for c in 0..d {
                    let hi = src[(i * t + tt + 1) * d + c];
                    let lo = src[(i * t + tt) * d + c];
                    u[(i * (t - 1) + tt) * d + c] = hi - lo;
                }
            }
        }
        u
    }
}
