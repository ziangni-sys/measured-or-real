use crate::pcg::Pcg32;

pub struct Corr {
    pub n: usize,
    pub k: usize,
    pub r_true: f64,
    pub rho_x: f64,
    pub rho_y: f64,
    pub heavy: bool,
}

impl Corr {
    pub fn sym(n: usize, k: usize, r_true: f64, rho: f64) -> Self {
        Corr { n, k, r_true, rho_x: rho, rho_y: rho, heavy: false }
    }
}

pub struct Est {
    pub corr_obs: f64,
    pub rho_hat_x: f64,
    pub rho_hat_y: f64,
    pub joint: f64,
    pub joint_bad: bool,
    pub joint_psd: f64,
}

fn psd_corr(a: f64, b: f64, c: f64) -> f64 {
    let tr = a + b;
    let r = (((a - b) * 0.5).powi(2) + c * c).sqrt();
    let (l1, l2) = (0.5 * tr + r, 0.5 * tr - r);
    if l1 <= 0.0 {
        return 0.0;
    }
    let (mut ex, mut ey) = (c, l1 - a);
    let nrm = (ex * ex + ey * ey).sqrt();
    if nrm < 1e-12 {
        ex = 1.0;
        ey = 0.0;
    } else {
        ex /= nrm;
        ey /= nrm;
    }
    let (l1p, l2p) = (l1.max(0.0), l2.max(0.0));
    let m00 = l1p * ex * ex + l2p * ey * ey;
    let m11 = l1p * ey * ey + l2p * ex * ex;
    let m01 = (l1p - l2p) * ex * ey;
    if m00 > 0.0 && m11 > 0.0 {
        (m01 / (m00 * m11).sqrt()).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn icc(rep: &[Vec<f64>], n: usize, k: usize) -> (f64, Vec<f64>) {

    let mut mean = vec![0.0f64; n];
    for j in 0..k {
        for i in 0..n {
            mean[i] += rep[j][i];
        }
    }
    for m in mean.iter_mut() {
        *m /= k as f64;
    }
    let grand = mean.iter().sum::<f64>() / n as f64;
    let mut ssb = 0.0;
    for i in 0..n {
        ssb += (mean[i] - grand) * (mean[i] - grand);
    }
    let msb = ssb * k as f64 / (n as f64 - 1.0);
    let mut ssw = 0.0;
    for j in 0..k {
        for i in 0..n {
            ssw += (rep[j][i] - mean[i]) * (rep[j][i] - mean[i]);
        }
    }
    let msw = ssw / (n as f64 * (k as f64 - 1.0));
    let rho = if msb <= 0.0 { 0.0 } else { 1.0 - msw / msb };
    (rho, mean)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut dx, mut dy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    let den = (dx * dy).sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

impl Corr {
    pub fn estimate(&self, seed: u64) -> Est {
        let (n, k) = (self.n, self.k);
        let mut r = Pcg32::new(seed, 0x51D);

        let sgx = (k as f64 * (1.0 - self.rho_x) / self.rho_x).sqrt();
        let sgy = (k as f64 * (1.0 - self.rho_y) / self.rho_y).sqrt();
        let heavy = self.heavy;
        let noise = |r: &mut Pcg32| -> f64 {
            if heavy {

                let z = r.normal();
                let c = r.normal().powi(2) + r.normal().powi(2) + r.normal().powi(2);
                (z / (c / 3.0).sqrt()) / 3.0_f64.sqrt()
            } else {
                r.normal()
            }
        };
        let mut xr = vec![vec![0.0f64; n]; k];
        let mut yr = vec![vec![0.0f64; n]; k];
        let a = self.r_true;
        let b = (1.0 - a * a).max(0.0).sqrt();
        for i in 0..n {
            let z1 = r.normal();
            let z2 = r.normal();
            let (xt, yt) = (z1, a * z1 + b * z2);
            for j in 0..k {
                xr[j][i] = xt + sgx * noise(&mut r);
                yr[j][i] = yt + sgy * noise(&mut r);
            }
        }
        let (rx, xbar) = icc(&xr, n, k);
        let (ry, ybar) = icc(&yr, n, k);

        let cov = |a: &[f64], b: &[f64]| -> f64 {
            let ma = a.iter().sum::<f64>() / n as f64;
            let mb = b.iter().sum::<f64>() / n as f64;
            a.iter().zip(b).map(|(&u, &v)| (u - ma) * (v - mb)).sum::<f64>() / (n as f64 - 1.0)
        };
        let (mut vx, mut vy, mut cxy, mut np) = (0.0f64, 0.0, 0.0, 0.0);
        for p in 0..k {
            for q in 0..k {
                if p != q {
                    vx += cov(&xr[p], &xr[q]);
                    vy += cov(&yr[p], &yr[q]);
                    cxy += cov(&xr[p], &yr[q]);
                    np += 1.0;
                }
            }
        }
        let (vx, vy, cxy) = (vx / np, vy / np, cxy / np);
        let (joint, joint_bad) = if vx > 0.0 && vy > 0.0 {
            let raw = cxy / (vx * vy).sqrt();
            (clip(raw), raw.abs() > 1.0)
        } else {
            (0.0, true)
        };
        // Project the 2x2 latent moment covariance onto the PSD cone by eigenvalue
        // truncation, then read off its correlation. This nearest-PSD projection is
        // not generally the constrained REML/ML estimator.
        let joint_psd = psd_corr(vx, vy, cxy);
        Est { corr_obs: pearson(&xbar, &ybar), rho_hat_x: rx, rho_hat_y: ry, joint, joint_bad, joint_psd }
    }
}

fn clip(v: f64) -> f64 {
    v.clamp(-1.0, 1.0)
}

pub fn raw(e: &Est) -> f64 {
    clip(e.corr_obs)
}

pub fn naive(e: &Est) -> f64 {
    let d = (e.rho_hat_x * e.rho_hat_y).max(1e-9).sqrt();
    e.corr_obs / d
}

pub fn clipped(e: &Est) -> f64 {
    clip(naive(e))
}

pub fn floored(e: &Est, c: f64) -> f64 {
    let d = (e.rho_hat_x.max(c) * e.rho_hat_y.max(c)).sqrt();
    clip(e.corr_obs / d)
}

pub fn bootstrap_floor(e: &Est, n: usize, k: usize, seed: u64) -> f64 {
    let b = 120u64;
    let rx0 = e.rho_hat_x.clamp(0.02, 0.99);
    let ry0 = e.rho_hat_y.clamp(0.02, 0.99);
    let r0 = clip(e.corr_obs / (rx0 * ry0).sqrt());
    let sim = Corr { n, k, r_true: r0, rho_x: rx0, rho_y: ry0, heavy: false };
    let cs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];
    let mut best_c = 0.1;
    let mut best = f64::INFINITY;
    for &c in &cs {
        let mut m = 0.0;
        for s in 0..b {
            let eb = sim.estimate(seed.wrapping_mul(1_000_003).wrapping_add(s));
            m += (floored(&eb, c) - r0).powi(2);
        }
        if m < best {
            best = m;
            best_c = c;
        }
    }
    floored(e, best_c)
}

pub fn pospart(e: &Est) -> f64 {
    let (px, py) = (e.rho_hat_x.max(0.0), e.rho_hat_y.max(0.0));
    if px <= 0.0 || py <= 0.0 {
        return clip(e.corr_obs);
    }
    clip(e.corr_obs / (px * py).sqrt())
}

fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * ln_gamma(1.0 - x).exp())
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        let t = x + g + 0.5;
        for (i, &ci) in c.iter().enumerate().skip(1) {
            a += ci / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (mut c, mut d) = (1.0, 1.0 - (a + b) * x / (a + 1.0));
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + aa2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn f_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    betai(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))
}

fn f_quantile(p: f64, d1: f64, d2: f64) -> f64 {
    let (mut lo, mut hi) = (1e-6, 1e6);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if f_cdf(mid, d1, d2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub fn ctdf_ucb(e: &Est, n: usize, k: usize, alpha: f64) -> f64 {
    let d1 = (n * (k - 1)) as f64;
    let d2 = (n - 1) as f64;

    let q_gate = f_quantile(1.0 - alpha, d2, d1);
    let gate = |rh: f64| 1.0 / (1.0 - rh).max(1e-9) > q_gate;
    if !gate(e.rho_hat_x) || !gate(e.rho_hat_y) {
        return clip(e.corr_obs);
    }

    let q_up = f_quantile(1.0 - alpha, d1, d2);

    let ux = (1.0 - (1.0 - e.rho_hat_x) / q_up).max(1e-3);
    let uy = (1.0 - (1.0 - e.rho_hat_y) / q_up).max(1e-3);
    clip(e.corr_obs / (ux * uy).sqrt())
}

pub fn gate_refuses(e: &Est, n: usize, k: usize, alpha: f64) -> bool {
    let d1 = (n * (k - 1)) as f64;
    let d2 = (n - 1) as f64;
    let q_gate = f_quantile(1.0 - alpha, d2, d1);
    let g = |rh: f64| 1.0 / (1.0 - rh).max(1e-9) > q_gate;
    !g(e.rho_hat_x) || !g(e.rho_hat_y)
}

pub fn ctdf_ucb_nogate(e: &Est, n: usize, k: usize, alpha: f64) -> f64 {
    let d1 = (n * (k - 1)) as f64;
    let d2 = (n - 1) as f64;
    let q_up = f_quantile(1.0 - alpha, d1, d2);
    let ux = (1.0 - (1.0 - e.rho_hat_x) / q_up).max(1e-3);
    let uy = (1.0 - (1.0 - e.rho_hat_y) / q_up).max(1e-3);
    clip(e.corr_obs / (ux * uy).sqrt())
}

pub fn gate_plugin(e: &Est, n: usize, k: usize, alpha: f64) -> f64 {
    let d1 = (n * (k - 1)) as f64;
    let d2 = (n - 1) as f64;
    let q_gate = f_quantile(1.0 - alpha, d2, d1);
    let gate = |rh: f64| 1.0 / (1.0 - rh).max(1e-9) > q_gate;
    if !gate(e.rho_hat_x) || !gate(e.rho_hat_y) {
        return clip(e.corr_obs);
    }
    clipped(e)
}

pub fn bayes_shrink(e: &Est, n: usize, k: usize) -> f64 {
    let d1 = (n * (k - 1)) as f64;
    let d2 = (n - 1) as f64;
    let m = 24usize;
    let fq: Vec<f64> = (0..m)
        .map(|i| f_quantile((i as f64 + 0.5) / m as f64, d1, d2))
        .collect();
    let rx: Vec<f64> = fq.iter().map(|&f| (1.0 - (1.0 - e.rho_hat_x) / f).clamp(1e-3, 1.0)).collect();
    let ry: Vec<f64> = fq.iter().map(|&f| (1.0 - (1.0 - e.rho_hat_y) / f).clamp(1e-3, 1.0)).collect();
    let mut s = 0.0;
    for &a in &rx {
        for &b in &ry {
            s += clip(e.corr_obs / (a * b).sqrt());
        }
    }
    s / (m * m) as f64
}
