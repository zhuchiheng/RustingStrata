//! Does an "expert-contiguous" weight layout help `moe_gemv`? Measured at the Inkling 2 Mini decode shape: the 8
//! experts one routed token picks, over both of a layer's expert matrices (`xg = 8`, so every slot reads the same
//! input row). A decode step makes 48 of each per token.
//!
//! Three layouts of the same bytes:
//!   scattered ids  the router's pick: 8 regions one stride apart inside the whole expert tensor (today)
//!   adjacent ids   ids 0..7: those same 8 regions are one contiguous run (what a reordered pool would look like)
//!   staged copy    the 8 slices gathered into one contiguous buffer by dtod, then read with ids 0..7 (the full
//!                  "gather once, then stream it" idea, copies included)
//!
//! Run with `cargo test --release --test moe_layout -- --ignored --nocapture`, `STRATA_MODEL` pointing at an
//! Inkling 2 file.
mod common;
use rusting_strata::backend::{Buf, GpuTensor};
use rusting_strata::gguf::Gguf;
use rusting_strata::quant::GgmlType;

const HBM_PEAK_GBS: f64 = 898.0;
/// Layer visits per decode token: 24 layers, 12 of which sit in the loop that runs 3 times.
const VISITS: f64 = 48.0;

#[test]
#[ignore]
fn expert_layout_bandwidth() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    if !g.tensors.contains_key("blk.0.ffn_gate_up_exps.weight") {
        eprintln!("SKIP: no ffn_gate_up_exps.weight (not an Inkling 2 file)");
        return;
    }
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        for name in ["blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"] {
            let info = &g.tensors[name];
            let (n_expert, cols, rows) = (info.dims[2] as usize, info.dims[0] as usize, info.dims[1] as usize);
            let ty = GgmlType::from_u32(info.ty).unwrap();
            let data = g.bytes(info);
            let stride = data.len() / n_expert;
            eprintln!("{bk}: {name} {n_expert} experts of {rows} x {cols} {}, {:.2} MB per expert",
                ty.name(), stride as f64 / 1e6);

            let db = gpu.alloc(data.len()).unwrap();
            gpu.htod(data, &db, 0).unwrap();
            let dt = GpuTensor::new(ty, rows, cols, Buf::new(db.addr(), data.len(), ()));
            let scattered: Vec<i32> = vec![3, 17, 40, 77, 101, 120, 125, 127];
            let adjacent: Vec<i32> = (0..8).collect();
            let ds = common::up(gpu, &scattered);
            let da = common::up(gpu, &adjacent);
            let dsi = common::up(gpu, &adjacent);
            let x = common::up(gpu, &vec![0.01f32; cols]);
            let y = gpu.zeros::<f32>(8 * rows).unwrap();

            let time = |f: &dyn Fn()| {
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_millis() < 300 { f(); gpu.sync().unwrap(); }
                let t0 = std::time::Instant::now();
                for _ in 0..100 { f(); }
                gpu.sync().unwrap();
                t0.elapsed().as_secs_f64() / 100.0
            };
            // Median of three: the first two measurements of a kernel here differ by 20 %.
            let med = |f: &dyn Fn()| {
                let mut r: Vec<f64> = (0..3).map(|_| time(f)).collect();
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                r[1]
            };

            let staged = gpu.alloc(8 * stride).unwrap();
            let gather = || for (i, &e) in scattered.iter().enumerate() {
                gpu.dtod(&db, e as usize * stride, &staged, i * stride, stride).unwrap();
            };
            let st = GpuTensor::new(ty, rows, cols, Buf::new(staged.addr(), 8 * stride, ()));
            let moe = |w: &GpuTensor, stride: usize, ids: &Buf| gpu.moe_gemv(ty, rows, cols, w.ptr, stride, ids.addr(), 8,
                x.addr(), 0, 8, y.addr(), None, 0).unwrap();

            let s_sc = med(&|| moe(&dt, stride, &ds));
            let s_ad = med(&|| moe(&dt, stride, &da));
            let s_g = med(&gather);
            let s_st = med(&|| moe(&st, stride, &dsi));
            let s_gs = med(&|| { gather(); moe(&st, stride, &dsi); });

            let bytes = 8 * stride;
            let row = |k: &str, s: f64| {
                let gbs = bytes as f64 / s / 1e9;
                eprintln!("  {k:26} {:7.1} us  {:6.1} GB/s  {:5.1}% of peak   {:5.2} ms per token's {VISITS} visits",
                    s * 1e6, gbs, 100.0 * gbs / HBM_PEAK_GBS, VISITS * s * 1e3);
                s
            };
            row("scattered ids (today)", s_sc);
            row("adjacent ids 0..7", s_ad);
            row("dtod gather only", s_g);
            row("kernel over staged copy", s_st);
            row("gather + kernel", s_gs);
            eprintln!("  {:26} adjacent {:+.1} % on the kernel; gather+stream {:+.1} % (negative = worse)",
                "", 100.0 * (s_sc / s_ad - 1.0), 100.0 * (s_sc / s_gs - 1.0));
        }
    }
}