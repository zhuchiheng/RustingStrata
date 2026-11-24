//! `moe_gemv` at the Inkling 2 Mini decode shape, measured on its own so a change to the kernel can be judged
//! without the rest of the model. The real call is 8 experts of one layer's `ffn_gate_up_exps` (2304 x 2304, Q4_K)
//! with `xg = 8`: every slot reads the same input row, so the kernel streams 8 * 2.99 MB per layer visit, 48 times a
//! token.
//!
//! Run with `cargo test --release --test inkling2_moe_bw -- --ignored --nocapture` and `STRATA_MODEL` pointing at an
//! Inkling 2 file (the tensor type comes from the file, so a Q4_K model measures Q4_K).
mod common;
use rusting_strata::backend::{Backend, Buf, GpuTensor};
use rusting_strata::gguf::Gguf;
use rusting_strata::quant::GgmlType;

/// Peak of the reference card in this repo's roofline (tests/kernels.rs).
const HBM_PEAK_GBS: f64 = 898.0;

#[test]
#[ignore]
fn inkling2_moe_gemv_bandwidth() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    // The Inkling 2 file is read through Gguf directly: Weights::new refuses `inkling2_loop` (its loader is the
    // inkling2 module's own), but the tensor bytes and the ggml type are all this needs.
    let Some(info) = g.tensors.get("blk.0.ffn_gate_up_exps.weight") else {
        eprintln!("SKIP: no ffn_gate_up_exps.weight (not an Inkling 2 file)");
        return;
    };
    let n_expert = info.dims[2] as usize;
    let cols = info.dims[0] as usize;
    let rows = info.dims[1] as usize;
    let ty = GgmlType::from_u32(info.ty).unwrap();
    let data = g.bytes(info);
    eprintln!("blk.0.ffn_gate_up_exps: {n_expert} experts of {rows} x {cols}, {}", ty.name());

    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        let db = gpu.alloc(data.len()).unwrap();
        gpu.htod(data, &db, 0).unwrap();
        let stride = data.len() / n_expert;
        let dt = GpuTensor::new(ty, rows, cols, Buf::new(db.addr(), data.len(), ()));
        // The 8 experts a decode token routes to: arbitrary but fixed, as the router would pick.
        let ids: Vec<i32> = vec![3, 17, 40, 77, 101, 120, 125, 127];
        let di = common::up(gpu, &ids);
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
        let call = || gpu.moe_gemv(ty, rows, cols, dt.ptr, stride, di.addr(), 8, x.addr(), 0, 8, y.addr(), None, 0).unwrap();
        // Three separate timing runs: the first two measurements of this kernel differed by 20%, so one sample is
        // not enough to judge a change.
        let runs: Vec<f64> = (0..3).map(|_| time(&call)).collect();
        let s = {
            let mut v = runs.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[1]
        };
        eprintln!("     runs: {}", runs.iter().map(|x| format!("{:.1}us", x * 1e6)).collect::<Vec<_>>().join(" / "));
        let bytes = 8 * stride;
        let gbs = bytes as f64 / s / 1e9;
        eprintln!("  {bk}: {:.1} us, {:.1} GB/s ({:.0}% of {HBM_PEAK_GBS}), {:.1} MB per call",
            s * 1e6, gbs, 100.0 * gbs / HBM_PEAK_GBS, bytes as f64 / 1e6);
        // What a token costs: 48 layer visits of this.
        eprintln!("     -> one token's experts: {:.2} ms at this rate ({:.0} tok/s if it were all)",
            48.0 * s * 1e3, 1.0 / (48.0 * s));

        // Correctness: the moe path must equal 8 dense GEMVs over the same experts. This is the check the change to
        // the row mapping (NWARPS instead of a hard-wired 8) has to keep passing.
        let s8v = || {
            for (i, &e) in ids.iter().enumerate() {
                let off = e as usize * stride;
                let wt = GpuTensor::new(ty, rows, cols, Buf::new(dt.ptr + off as u64, stride, ()));
                gpu.gemv(&wt, x.addr(), y.addr() + 4 * i as u64 * rows as u64).unwrap();
            }
        };
        let ref_y = gpu.zeros::<f32>(8 * rows).unwrap();
        for (i, &e) in ids.iter().enumerate() {
            let off = e as usize * stride;
            let wt = GpuTensor::new(ty, rows, cols, Buf::new(dt.ptr + off as u64, stride, ()));
            gpu.gemv(&wt, x.addr(), ref_y.addr() + 4 * i as u64 * rows as u64).unwrap();
        }
        gpu.sync().unwrap();
        let moe_out: Vec<f32> = gpu.get(&y, 0, 8 * rows).unwrap();
        let ref_out: Vec<f32> = gpu.get(&ref_y, 0, 8 * rows).unwrap();
        let num: f32 = moe_out.iter().zip(&ref_out).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = ref_out.iter().map(|b| b * b).sum::<f32>().max(1e-30);
        let rel = (num / den).sqrt();
        eprintln!("     correctness vs 8 dense gemv: rel err {rel:.3e}");
        assert!(rel < 1e-3, "{bk}: moe_gemv differs from 8 dense gemv (rel {rel:.3e})");
        let _ = s8v;

        // The same data as one dense GEMV per expert, for reference: 8 launches instead of one.
        // Two experts per slot group (xg = 4): each slot reads its own input row, so this exercises the other
        // branch of the kernel's slot handling.
        let x4 = common::up(gpu, &vec![0.01f32; 4 * cols]);
        let s_xg4 = time(&|| gpu.moe_gemv(ty, rows, cols, dt.ptr, stride, di.addr(), 8, x4.addr(), cols, 4, y.addr(), None, 0).unwrap());
        eprintln!("     xg=4 (2 rows per slot): {:.1} us ({:.0}% of xg=8)", s_xg4 * 1e6, 100.0 * s_xg4 / s);

        let s8 = time(&|| {
            for (i, &e) in ids.iter().enumerate() {
                let off = e as usize * stride;
                let wt = GpuTensor::new(ty, rows, cols, Buf::new(dt.ptr + off as u64, stride, ()));
                gpu.gemv(&wt, x.addr(), y.addr() + 4 * i as u64 * rows as u64).unwrap();
            }
        });
        eprintln!("     dense gemv x8: {:.1} us ({:.0}% of the moe path)", s8 * 1e6, 100.0 * s / s8);
    }
}
