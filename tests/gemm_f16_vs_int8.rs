//! Expert GEMM throughput against positions per tile, at a fixed number of workgroups (one per expert), and the same
//! shapes with the weights expanded to F16 (`gemm_fs`) instead of decoded per K step (`gemmi_16`).
//!
//! The F16 copy is 3.56 times the bytes of the NVFP4 one, so it starts at a disadvantage: if it is still faster at the
//! same shape, the decode is the cost. Vary only the positions per tile, never the grid, or the curve measures
//! parallelism instead of the tile.
//!
//! Run with `cargo test --release --test gemm_f16_vs_int8 -- --ignored --nocapture`, `STRATA_MODEL` pointing at an
//! Inkling 2 file.
mod common;
use rusting_strata::backend::{Backend, Buf, GpuTensor};
use rusting_strata::gguf::Gguf;
use rusting_strata::quant::{GgmlType, dequant_row};

#[test]
#[ignore]
fn expert_gemm_tile_sizes() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let Some(info) = g.tensors.get("blk.0.ffn_gate_up_exps.weight") else {
        eprintln!("SKIP: no ffn_gate_up_exps.weight (not an Inkling 2 file)");
        return;
    };
    let (ne, cols, rows) = (info.dims[2] as usize, info.dims[0] as usize, info.dims[1] as usize);
    let ty = GgmlType::from_u32(info.ty).unwrap();
    let data = g.bytes(info);
    let stride = data.len() / ne;
    eprintln!("blk.0.ffn_gate_up_exps: {ne} experts of {rows} x {cols} {}, {:.2} MB per expert",
        ty.name(), stride as f64 / 1e6);

    // Expert 0 as F16: the values `gemmi_16` multiplies, already expanded.
    let mut f32v = vec![0f32; rows * cols];
    {
        let rowb = ty.row_bytes(cols as u64) as usize;
        for r in 0..rows {
            dequant_row(ty, &data[r * rowb..r * rowb + rowb], &mut f32v[r * cols..(r + 1) * cols]);
        }
    }
    let cbytes: Vec<u8> = f32v.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    eprintln!("  F16 copy: {:.2} MB ({:.2}x the {})", cbytes.len() as f64 / 1e6, cbytes.len() as f64 / stride as f64, ty.name());

    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        let db = gpu.alloc(data.len()).unwrap();
        gpu.htod(data, &db, 0).unwrap();
        let w8 = GpuTensor::new(ty, rows, cols, Buf::new(db.addr(), data.len(), ()));
        let ctr = gpu.alloc(cbytes.len()).unwrap();
        gpu.htod(&cbytes, &ctr, 0).unwrap();
        let wf = GpuTensor::new(GgmlType::F16, rows, cols, Buf::new(ctr.addr(), cbytes.len(), ()));
        let _ = (&w8, &wf);

        let time = |f: &dyn Fn()| {
            for _ in 0..2 { f(); }
            gpu.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..5 { f(); }
            gpu.sync().unwrap();
            t0.elapsed().as_secs_f64() / 5.0
        };
        // Every tile reads expert 0: the shape is what is measured, not the data. The grid is one workgroup per
        // expert at every position count, so the curve is the tile's, not the GPU's occupancy.
        for c in [8usize, 16, 32, 64, 96, 128, 256] {
            let m = ne * c;
            let tiles: Vec<u64> = (0..ne).flat_map(|e| [db.addr(), (c * e) as u64 | (c as u64) << 32]).collect();
            let dtl = common::up(gpu, &tiles);
            let dx = common::up(gpu, &vec![0.01f32; m * cols]);
            let dy = gpu.zeros::<f32>(m * rows).unwrap();
            let s8 = time(&|| gpu.gemm(ty, rows, cols, dtl.addr(), ne, dx.addr(), cols, m, 0, dy.addr(), rows, 0).unwrap());
            let sf = time(&|| gpu.gemm(GgmlType::F16, rows, cols, dtl.addr(), ne, dx.addr(), cols, m, 0, dy.addr(), rows, 0).unwrap());
            let flops = 2.0 * (m * rows * cols) as f64;
            eprintln!("  {bk} {c:3} pos x {ne} tiles ({m:5} rows): {} {s8*1e3:8.3} ms {:6.2} TFLOPS | F16 {sf*1e3:8.3} ms {:6.2} TFLOPS | F16/int8 {:5.2}x",
                ty.name(), flops / s8 / 1e12, flops / sf / 1e12, s8 / sf);
        }
    }
}
