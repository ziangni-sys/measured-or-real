use crate::dgp::{Arms, ArmsK};

fn pooled(x: &[f64], y: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for k in 0..x.len() {
        num += x[k] * y[k];
        dx += x[k] * x[k];
        dy += y[k] * y[k];
    }
    let den = (dx * dy).sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

fn gather(u: &[f64], n: usize, m: usize, d: usize, s0: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * (m - 1) * d];
    for i in 0..n {
        for s in 0..(m - 1) {
            for c in 0..d {
                out[(i * (m - 1) + s) * d + c] = u[(i * m + s + s0) * d + c];
            }
        }
    }
    out
}

fn same(ua: &[f64], ub: &[f64], n: usize, m: usize, d: usize) -> f64 {
    pooled(&ua[..n * m * d], &ub[..n * m * d])
}

pub struct Est {
    pub truth: f64,
    pub naive: f64,
    pub cross: f64,
    pub rel: f64,
    pub disatt: f64,
}

pub fn estimate(arms: &Arms) -> Est {
    let (n, t, d) = (arms.n, arms.t, arms.d);
    let ua = Arms::disp(&arms.a, n, t, d, 1);
    let ub = Arms::disp(&arms.b, n, t, d, 1);
    let ut = Arms::disp(&arms.truth, n, t, d, 1);
    let m = t - 1;

    let ut0 = gather(&ut, n, m, d, 0);
    let ut1 = gather(&ut, n, m, d, 1);
    let truth = same(&ut0, &ut1, n, m - 1, d);

    let ua0 = gather(&ua, n, m, d, 0);
    let ua1 = gather(&ua, n, m, d, 1);
    let naive = same(&ua0, &ua1, n, m - 1, d);

    let ub1 = gather(&ub, n, m, d, 1);
    let cross = same(&ua0, &ub1, n, m - 1, d);

    let rel = pooled(&ua, &ub);

    let disatt = if rel.abs() < 1e-9 { 0.0 } else { cross / rel };
    Est { truth, naive, cross, rel, disatt }
}

pub fn per_dim_reliability(arms: &Arms) -> Vec<f64> {
    let (n, t, d) = (arms.n, arms.t, arms.d);
    let mut num = vec![0.0f64; d];
    let mut da = vec![0.0f64; d];
    let mut db = vec![0.0f64; d];
    for i in 0..n {
        for tt in 0..t {
            for c in 0..d {
                let idx = (i * t + tt) * d + c;
                num[c] += arms.a[idx] * arms.b[idx];
                da[c] += arms.a[idx] * arms.a[idx];
                db[c] += arms.b[idx] * arms.b[idx];
            }
        }
    }
    (0..d)
        .map(|c| {
            let den = (da[c] * db[c]).sqrt();
            if den == 0.0 { 0.0 } else { (num[c] / den).clamp(0.0, 1.0) }
        })
        .collect()
}

pub struct Recon {
    pub naive: f64,
    pub ensemble: f64,
    pub shrunk: f64,
    pub shrunk_ens: f64,
}

fn nrmse(x: &[f64], truth: &[f64]) -> f64 {
    let mut se = 0.0;
    let mut st = 0.0;
    for k in 0..x.len() {
        se += (x[k] - truth[k]) * (x[k] - truth[k]);
        st += truth[k] * truth[k];
    }
    (se / st).sqrt()
}

fn mean_of(arms: &[Vec<f64>], sel: &[usize], len: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; len];
    for &j in sel {
        for idx in 0..len {
            m[idx] += arms[j][idx];
        }
    }
    let k = sel.len() as f64;
    for v in m.iter_mut() {
        *v /= k;
    }
    m
}

fn per_dim_corr(x: &[f64], y: &[f64], n: usize, t: usize, d: usize) -> Vec<f64> {
    let mut num = vec![0.0f64; d];
    let mut dx = vec![0.0f64; d];
    let mut dy = vec![0.0f64; d];
    for i in 0..n {
        for tt in 0..t {
            for c in 0..d {
                let idx = (i * t + tt) * d + c;
                num[c] += x[idx] * y[idx];
                dx[c] += x[idx] * x[idx];
                dy[c] += y[idx] * y[idx];
            }
        }
    }
    (0..d)
        .map(|c| {
            let den = (dx[c] * dy[c]).sqrt();
            if den == 0.0 { 0.0 } else { (num[c] / den).clamp(0.0, 1.0) }
        })
        .collect()
}

pub struct Pareto {
    pub k: usize,
    pub plain: f64,
    pub weighted: f64,
}

pub fn pareto(a: &ArmsK, k: usize) -> Pareto {
    let (n, t, d) = (a.n, a.t, a.d);
    let len = n * t * d;
    let all: Vec<usize> = (0..k).collect();
    let mean = mean_of(&a.arms, &all, len);

    let h = k / 2;
    let rho_k = if h == 0 {
        vec![1.0f64; d]
    } else {
        let m1 = mean_of(&a.arms, &all[..h], len);
        let m2 = mean_of(&a.arms, &all[h..2 * h], len);

        let rho_h = per_dim_corr(&m1, &m2, n, t, d);

        rho_h
            .iter()
            .map(|&r| (2.0 * r / (1.0 + r)).clamp(0.0, 1.0))
            .collect()
    };

    let mut wt = vec![0.0f64; len];
    for i in 0..n {
        for tt in 0..t {
            for c in 0..d {
                let idx = (i * t + tt) * d + c;
                wt[idx] = rho_k[c] * mean[idx];
            }
        }
    }
    Pareto { k, plain: nrmse(&mean, &a.truth), weighted: nrmse(&wt, &a.truth) }
}

pub fn recon(arms: &Arms) -> Recon {
    let (n, t, d) = (arms.n, arms.t, arms.d);
    let rel = per_dim_reliability(arms);
    let len = n * t * d;
    let mut ens = vec![0.0f64; len];
    let mut sa = vec![0.0f64; len];
    let mut se = vec![0.0f64; len];
    for i in 0..n {
        for tt in 0..t {
            for c in 0..d {
                let idx = (i * t + tt) * d + c;
                let m = 0.5 * (arms.a[idx] + arms.b[idx]);
                ens[idx] = m;
                sa[idx] = rel[c] * arms.a[idx];
                se[idx] = rel[c] * m;
            }
        }
    }
    Recon {
        naive: nrmse(&arms.a, &arms.truth),
        ensemble: nrmse(&ens, &arms.truth),
        shrunk: nrmse(&sa, &arms.truth),
        shrunk_ens: nrmse(&se, &arms.truth),
    }
}
