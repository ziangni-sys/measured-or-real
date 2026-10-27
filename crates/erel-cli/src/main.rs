use erel_core::dgp::Synth;
use erel_core::reliability::{estimate, pareto, recon};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * q).round() as usize;
    v[idx]
}

fn synth(out: &str) {
    let reps = 40u64;
    let alphas = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let sigmas = [0.5, 1.0, 2.0];
    let f = File::create(out).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "alpha,sigma,truth,naive,naive_lo,naive_hi,disatt,disatt_lo,disatt_hi,rel"
    )
    .unwrap();
    let mut mae_naive = 0.0;
    let mut mae_disatt = 0.0;
    let mut cover_naive = 0usize;
    let mut cover_disatt = 0usize;
    let mut cells = 0usize;
    for &alpha in &alphas {
        for &sigma in &sigmas {
            let s = Synth { n: 400, t: 20, d: 16, alpha, sig_lo: sigma, sig_hi: sigma, innov: 1.0 };
            let mut nv = Vec::with_capacity(reps as usize);
            let mut dv = Vec::with_capacity(reps as usize);
            let mut truth = 0.0;
            let mut rel = 0.0;
            for r in 0..reps {
                let e = estimate(&s.generate(1000 + r));
                nv.push(e.naive);
                dv.push(e.disatt);
                truth += e.truth;
                rel += e.rel;
            }
            let k = reps as f64;
            truth /= k;
            rel /= k;
            let nmean = nv.iter().sum::<f64>() / k;
            let dmean = dv.iter().sum::<f64>() / k;
            let (nlo, nhi) = (quantile(&mut nv.clone(), 0.025), quantile(&mut nv.clone(), 0.975));
            let (dlo, dhi) = (quantile(&mut dv.clone(), 0.025), quantile(&mut dv.clone(), 0.975));
            writeln!(
                w,
                "{alpha},{sigma},{truth:.4},{nmean:.4},{nlo:.4},{nhi:.4},{dmean:.4},{dlo:.4},{dhi:.4},{rel:.4}"
            )
            .unwrap();
            mae_naive += (nmean - truth).abs();
            mae_disatt += (dmean - truth).abs();
            cover_naive += (nlo <= truth && truth <= nhi) as usize;
            cover_disatt += (dlo <= truth && truth <= dhi) as usize;
            cells += 1;
        }
    }
    let c = cells as f64;
    println!("wrote {out}");
    println!("MAE   naive={:.4}  disatt={:.4}  ({:.1}x lower)", mae_naive / c, mae_disatt / c, (mae_naive / mae_disatt));
    println!("CI cover truth  naive={}/{cells}  disatt={}/{cells}", cover_naive, cover_disatt);
}

fn recon_bench() {
    let reps = 40u64;
    let his = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0];
    println!("noisy-regime sweep: fixed clean floor sig_lo=0.1, growing heterogeneity sig_hi");
    println!("sig_hi  naive   ensemble  shrunk+ens   gain_vs_ensemble");
    for &hi in &his {
        let s = Synth { n: 600, t: 20, d: 32, alpha: 0.3, sig_lo: 0.1, sig_hi: hi, innov: 1.0 };
        let (mut na, mut en, mut se) = (0.0, 0.0, 0.0);
        for r in 0..reps {
            let rc = recon(&s.generate(2000 + r));
            na += rc.naive;
            en += rc.ensemble;
            se += rc.shrunk_ens;
        }
        let k = reps as f64;
        let (na, en, se) = (na / k, en / k, se / k);
        println!("{hi:>5.1}  {na:.4}  {en:.4}   {se:.4}     {:+.1}%", 100.0 * (en - se) / en);
    }
}

fn pareto_bench(out: &str) {
    let reps = 40u64;
    let ks = [2usize, 3, 4, 6, 8, 12, 16, 24];
    let s = Synth { n: 600, t: 20, d: 32, alpha: 0.3, sig_lo: 0.3, sig_hi: 4.0, innov: 1.0 };
    let f = File::create(out).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "k,plain,weighted").unwrap();
    let mut plain = Vec::new();
    let mut weighted = Vec::new();
    println!("seed-budget Pareto: shared latent, heterogeneous noise (sig 0.3..4.0)");
    println!("  k    plain-NRMSE  weighted-NRMSE   error-cut");
    for &k in &ks {
        let (mut p, mut q) = (0.0, 0.0);
        for r in 0..reps {
            let pr = pareto(&s.generate_k(3000 + r, k), k);
            p += pr.plain;
            q += pr.weighted;
        }
        let (p, q) = (p / reps as f64, q / reps as f64);
        writeln!(w, "{k},{p:.4},{q:.4}").unwrap();
        println!("{k:>4}   {p:.4}       {q:.4}        {:+.1}%", 100.0 * (q - p) / p);
        plain.push((k, p));
        weighted.push((k, q));
    }
    println!("wrote {out}");

    let target = weighted.iter().find(|(k, _)| *k == 4).unwrap().1;
    let need = plain.iter().find(|(_, v)| *v <= target).map(|(k, _)| *k);
    match need {
        Some(kp) => println!(
            "HEADLINE: weighted @ 4 seeds (NRMSE {target:.4}) = plain @ {kp} seeds  -> {:.1}x seed budget",
            kp as f64 / 4.0
        ),
        None => println!("HEADLINE: plain never matches weighted @ 4 seeds within k<=24"),
    }
}

fn disatt2_bench() {
    use erel_core::disatt::{
        bayes_shrink, bootstrap_floor, clipped, ctdf_ucb, floored, gate_refuses, naive, pospart,
        raw, Corr,
    };
    let reps = 2000u64;
    let alpha = 0.1f64;
    let floors = [0.02, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];

    let scen: [(&str, usize, usize, f64, f64, f64, bool); 8] = [
        ("symmetric rho=.2", 50, 3, 0.2, 0.2, 0.4, false),
        ("asymmetric .3/.8", 50, 3, 0.3, 0.8, 0.4, false),
        ("heavy-tail t3", 50, 3, 0.2, 0.2, 0.4, true),
        ("r*=0.0", 50, 3, 0.2, 0.2, 0.0, false),
        ("r*=0.8", 50, 3, 0.2, 0.2, 0.8, false),
        ("N=20", 20, 3, 0.2, 0.2, 0.4, false),
        ("N=100", 100, 3, 0.2, 0.2, 0.4, false),
        ("N=500", 500, 3, 0.2, 0.2, 0.4, false),
    ];
    println!("complete disattenuation table, MSE vs truth over {reps} reps (alpha={alpha})");
    println!("  scenario           naive     clip    raw     joint(bad%)  jPSD    pp      bayes    boot     UCB      oracle   refuse%");
    let mut max_se = 0.0f64;
    for &(lab, n, k, rx, ry, rt, hv) in &scen {
        let c = Corr { n, k, r_true: rt, rho_x: rx, rho_y: ry, heavy: hv };
        // methods: 0 naive, 1 clip, 2 raw, 3 joint(MoM), 4 jPSD, 5 pp, 6 bayes, 7 boot, 8 UCB
        let mut sum = [0.0f64; 9];
        let mut sq = [0.0f64; 9];
        let (mut jbad, mut refuse) = (0usize, 0usize);
        let mut mfl = vec![0.0f64; floors.len()];
        for s in 0..reps {
            let e = c.estimate(1 + s);
            let vals = [
                naive(&e),
                clipped(&e),
                raw(&e),
                e.joint,
                e.joint_psd,
                pospart(&e),
                bayes_shrink(&e, n, k),
                bootstrap_floor(&e, n, k, 7 + s),
                ctdf_ucb(&e, n, k, alpha),
            ];
            for i in 0..9 {
                let d = (vals[i] - rt).powi(2);
                sum[i] += d;
                sq[i] += d * d;
            }
            if e.joint_bad {
                jbad += 1;
            }
            if gate_refuses(&e, n, k, alpha) {
                refuse += 1;
            }
            for (fi, &fc) in floors.iter().enumerate() {
                mfl[fi] += (floored(&e, fc) - rt).powi(2);
            }
        }
        let rf = reps as f64;
        let mse = sum.map(|v| v / rf);
        let oracle = mfl.iter().cloned().fold(f64::INFINITY, f64::min) / rf;
        for i in 1..9 {
            let se = ((sq[i] / rf - mse[i] * mse[i]) / rf).max(0.0).sqrt();
            max_se = max_se.max(se);
        }
        let ns = if mse[0] > 1e5 { format!("{:>8.1e}", mse[0]) } else { format!("{:>8.4}", mse[0]) };
        println!(
            "  {lab:<18} {ns}  {:.4}  {:.4}  {:.4}({:>3.0}%)  {:.4}  {:.4}  {:.4}   {:.4}   {:.4}   {:.4}   {:>4.0}%",
            mse[1], mse[2], mse[3], 100.0 * jbad as f64 / rf, mse[4], mse[5], mse[6], mse[7], mse[8],
            oracle, 100.0 * refuse as f64 / rf
        );
    }
    println!("  max Monte-Carlo SE over all deployable methods and cells: {max_se:.4} ({reps} reps)");
}

fn ablate_bench() {
    use erel_core::disatt::{ctdf_ucb, ctdf_ucb_nogate, floored, gate_plugin, raw, Corr};
    let reps = 4000u64;
    let alpha = 0.1f64;
    let r_true = 0.4f64;
    let cells = [(20usize, 0.1f64), (20, 0.2), (50, 0.1), (50, 0.2), (50, 0.5), (500, 0.2)];
    println!("gate ablation: r*={r_true}, K=3, alpha={alpha}, MSE over {reps} reps");
    println!("  N    rho    raw      UCB(full)  UCB-nogate  gate+plugin  fixed.1");
    for &(n, rho) in &cells {
        let c = Corr::sym(n, 3, r_true, rho);
        let (mut mr, mut mu, mut mng, mut mgp, mut mf) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for s in 0..reps {
            let e = c.estimate(1 + s);
            mr += (raw(&e) - r_true).powi(2);
            mu += (ctdf_ucb(&e, n, 3, alpha) - r_true).powi(2);
            mng += (ctdf_ucb_nogate(&e, n, 3, alpha) - r_true).powi(2);
            mgp += (gate_plugin(&e, n, 3, alpha) - r_true).powi(2);
            mf += (floored(&e, 0.1) - r_true).powi(2);
        }
        let rf = reps as f64;
        println!(
            "{n:>4} {rho:>5.1}   {:.4}   {:.4}     {:.4}      {:.4}       {:.4}",
            mr / rf, mu / rf, mng / rf, mgp / rf, mf / rf
        );
    }
}

fn disatt_bench() {
    use erel_core::disatt::{clipped, ctdf_ucb, naive, raw, Corr};
    let reps = 4000u64;
    let n = 50usize;
    let r_true = 0.4f64;
    let alpha = 0.1f64;
    let rhos = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
    let ks = [2usize, 3, 5];
    println!("correlation disattenuation: N={n}, r_true={r_true}, alpha={alpha}, MSE vs truth over {reps} reps");
    println!("  rho   K   MSE_naive   MSE_raw(donothing)  MSE_UCB(rule)  UCB-raw  refuse  P(|nv|>1)");
    for &rho in &rhos {
        for &k in &ks {
            let c = Corr::sym(n, k, r_true, rho);
            let (mut mn, mut mraw, mut mucb) = (0.0f64, 0.0, 0.0);
            let mut over = 0usize;
            let mut refuse = 0usize;
            for s in 0..reps {
                let e = c.estimate(1 + s);
                let nv = naive(&e);
                mn += (nv - r_true).powi(2);
                mraw += (raw(&e) - r_true).powi(2);
                let rk = ctdf_ucb(&e, n, k, alpha);
                mucb += (rk - r_true).powi(2);
                if (rk - clipped(&e)).abs() < 1e-12 {
                    refuse += 1;
                }
                if nv.abs() > 1.0 {
                    over += 1;
                }
            }
            let rf = reps as f64;
            let (mn, mraw, mucb) = (mn / rf, mraw / rf, mucb / rf);
            let mn_s = if mn > 1e6 { format!("{:>9.1e}", mn) } else { format!("{:>9.4}", mn) };
            let flag = if mucb < mraw { "UCB<raw" } else { "raw<=UCB" };
            println!(
                "{rho:>5.1} {k:>3}  {mn_s}       {mraw:.4}          {mucb:.4}    {:+.4} {flag}   {:.2}   {:>5.3}",
                mucb - mraw,
                refuse as f64 / rf,
                over as f64 / rf
            );
        }
    }

    println!("\nalpha sensitivity at rho=0.2, K=3, N={n}, r_true={r_true}:");
    println!("  alpha   MSE_UCB   refuse   (smaller alpha = more conservative)");
    let c = Corr::sym(n, 3, r_true, 0.2);
    for &al in &[0.05f64, 0.1, 0.25, 0.4] {
        let (mut m, mut rf_) = (0.0f64, 0usize);
        for s in 0..reps {
            let e = c.estimate(1 + s);
            let rk = erel_core::disatt::ctdf_ucb(&e, n, 3, al);
            m += (rk - r_true).powi(2);
            if (rk - clipped(&e)).abs() < 1e-12 {
                rf_ += 1;
            }
        }
        println!("  {al:>5.2}   {:.4}    {:.2}", m / reps as f64, rf_ as f64 / reps as f64);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()).unwrap_or("") {
        "version" => println!("erel-core {}", erel_core::version()),
        "synth" => synth(args.get(2).map(|s| s.as_str()).unwrap_or("data/synth.csv")),
        "recon" => recon_bench(),
        "pareto" => pareto_bench(args.get(2).map(|s| s.as_str()).unwrap_or("data/pareto.csv")),
        "disatt" => disatt_bench(),
        "disatt2" => disatt2_bench(),
        "ablate" => ablate_bench(),
        _ => {
            eprintln!("usage: erel <version|synth [out.csv]|recon|pareto [out.csv]|disatt|disatt2|ablate>");
            std::process::exit(2);
        }
    }
}
