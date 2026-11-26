//! How `moe_gemv` scales with the number of rows it verifies at once, at the Inkling 2 Mini decode shape. A
//! speculative step hands it `nt` token rows (`k = nt * 8` slots, `xg = 8`); the weight bytes it must read grow with
//! the union of the routed experts, so with disjoint experts the honest lower bound is `nt` times one row's bytes.
//!
//! Measured for the two expert matrices a decode step visits, and against the same experts for every row (which the
//! kernel's duplicate-slot path can share), so the gap between the two is that path's worth.
//!
//! Run with `cargo test --release --test moe_batch -- --ignored --nocapture`, `STRATA_MODEL` pointing at an
//! Inkling 2 file.
mod common;
use rusting_strata::backend::{Buf, GpuTensor};
use rusting_strata::gguf::Gguf;
use rusting_strata::quant::GgmlType;

const HBM_PEAK_GBS: f64 = 898.0;

#[test]
#[ignore]
fn moe_gemv_row_scaling() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    if !g.tensors.contains_key("blk.0.ffn_gate_up_exps.weight") {
        eprintln!("SKIP: no ffn_gate_up_exps.weight (not an Inkling 2 file)");
        return;
    }
    // Three rows of disjoint experts: what a verify batch of 3 drafted tokens routes to, roughly (each token picks
    // its own 8 of 128). `same` repeats one set, the case the kernel's duplicate slots can share.
    let rows_ids: [&[i32]; 3] = [
        &[3, 17, 40, 77, 101, 120, 125, 127],
        &[4, 18, 41, 78, 102, 121, 126, 6],
        &[5, 19, 42, 79, 103, 122, 124, 7],
    ];
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        for name in ["blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"] {
            let info = &g.tensors[name];
            let (n_expert, cols, rows) = (info.dims[2] as usize, info.dims[0] as usize, info.dims[1] as usize);
            let ty = GgmlType::from_u32(info.ty).unwrap();
            let data = g.bytes(info);
            let stride = data.len() / n_expert;
            eprintln!("{bk}: {name} {rows} x {cols} {}, {:.2} MB per expert", ty.name(), stride as f64 / 1e6);

            let db = gpu.alloc(data.len()).unwrap();
            gpu.htod(data, &db, 0).unwrap();
            let dt = GpuTensor::new(ty, rows, cols, Buf::new(db.addr(), data.len(), ()));
            let y = gpu.zeros::<f32>(3 * 8 * rows).unwrap();

            let time = |f: &dyn Fn()| {
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_millis() < 300 { f(); gpu.sync().unwrap(); }
                let t0 = std::time::Instant::now();
                for _ in 0..100 { f(); }
                gpu.sync().unwrap();
                t0.elapsed().as_secs_f64() / 100.0
            };
            let med = |f: &dyn Fn()| {
                let mut r: Vec<f64> = (0..3).map(|_| time(f)).collect();
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                r[1]
            };

            let mut base = 0.0;
            for nt in 1..=3 {
                let ids: Vec<i32> = (0..nt).flat_map(|t| rows_ids[t].iter().copied()).collect();
                let di = common::up(gpu, &ids);
                let x = common::up(gpu, &vec![0.01f32; nt * cols]);
                let s = med(&|| gpu.moe_gemv(ty, rows, cols, dt.ptr, stride, di.addr(), nt * 8, x.addr(), cols, 8,
                    y.addr(), None, 0).unwrap());
                if nt == 1 { base = s; }
                let bytes = nt * 8 * stride;
                let gbs = bytes as f64 / s / 1e9;
                eprintln!("  {nt} row(s) disjoint: {:7.1} us  {:6.1} GB/s  {:5.1}% of peak   x{:.2} of one row",
                    s * 1e6, gbs, 100.0 * gbs / HBM_PEAK_GBS, s / base);
            }
            // Same 8 experts for every row: the kernel can share one weight read across the rows.
            let ids: Vec<i32> = (0..3).flat_map(|_| rows_ids[0].iter().copied()).collect();
            let di = common::up(gpu, &ids);
            let x = common::up(gpu, &vec![0.01f32; 3 * cols]);
            let s = med(&|| gpu.moe_gemv(ty, rows, cols, dt.ptr, stride, di.addr(), 24, x.addr(), cols, 8,
                y.addr(), None, 0).unwrap());
            eprintln!("  3 rows, one expert set: {:7.1} us   x{:.2} of one row (the duplicate-slot path)", s * 1e6, s / base);
        }
    }
}