//! Prefill GEMM throughput against the positions per tile, at a grid that always keeps the card busy. This is the
//! curve that says what a prefill kernel can reach and where the real one sits on it, so a change to the tiling can be
//! judged without a per-op profiler.
//!
//! Two grids are measured, because they answer different questions:
//!   fixed grid   one workgroup per expert (128): the tile's own efficiency, no parallelism to hide latency with
//!   busy grid    enough workgroups to fill every CU (304 on an RX 7900 XTX): what a real prefill looks like
//!
//! Run with `cargo test --release --test prefill_tile_curve -- --ignored --nocapture`, `STRATA_MODEL` pointing at an
//! Inkling 2 file.
mod common;
use rusting_strata::backend::{Buf, GpuTensor};
use rusting_strata::gguf::Gguf;
use rusting_strata::quant::GgmlType;

/// FP16 peak of the reference card: 96 CUs x 128 lanes x 2 (FMA) x 2 (FLOP) x 2.5 GHz, roughly 61 TFLOPS.
const FP16_PEAK_TFLOPS: f64 = 61.0;

#[test]
#[ignore]
fn prefill_tile_curve() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let Some(info) = g.tensors.get("blk.0.ffn_gate_up_exps.weight") else {
        eprintln!("SKIP: no ffn_gate_up_exps.weight (not an Inkling 2 file)");
        return;
    };
    let (ne, cols, rows) = (info.dims[2] as usize, info.dims[0] as usize, info.dims[1] as usize);
    let ty = GgmlType::from_u32(info.ty).unwrap();
    let data = g.bytes(info);
    eprintln!("blk.0.ffn_gate_up_exps: {ne} experts of {rows} x {cols} {}", ty.name());

    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        let db = gpu.alloc(data.len()).unwrap();
        gpu.htod(data, &db, 0).unwrap();
        let time = |f: &dyn Fn()| {
            for _ in 0..2 { f(); }
            gpu.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..5 { f(); }
            gpu.sync().unwrap();
            t0.elapsed().as_secs_f64() / 5.0
        };
        // A busy grid needs many tiles. Every tile reads expert 0, so the weight bytes stay one expert's and the
        // curve is the tile's, not the memory system's.
        for &(label, tiles_wanted, sizes) in &[
            ("fixed grid (128 tiles)", ne, &[8usize, 16, 32, 48, 64, 96, 128][..]),
            ("busy grid (256 tiles)", 256, &[8usize, 16, 32][..]),
        ] {
            eprintln!("  {bk} {label}");
            for &c in sizes {
                let n_tiles = tiles_wanted.max(ne);
                let per = (n_tiles + ne - 1) / ne;  // tiles per expert
                let m = n_tiles * c;
                let mut tiles: Vec<u64> = Vec::with_capacity(2 * ne * per);
                for e in 0..ne {
                    for j in 0..per {
                        tiles.push(db.addr());
                        tiles.push(((c * (e * per + j)) as u64) | ((c as u64) << 32));
                    }
                }
                let n_tiles = tiles.len() / 2;
                let dtl = common::up(gpu, &tiles);
                let dx = common::up(gpu, &vec![0.01f32; m * cols]);
                let dy = gpu.zeros::<f32>(m * rows).unwrap();
                let s = time(&|| gpu.gemm(ty, rows, cols, dtl.addr(), n_tiles, dx.addr(), cols, m, 0, dy.addr(), rows, 0).unwrap());
                let flops = 2.0 * (m * rows * cols) as f64;
                let tf = flops / s / 1e12;
                eprintln!("    {c:3} pos x {n_tiles:3} tiles ({m:6} rows): {s:8.3} ms  {tf:6.2} TFLOPS  {:4.0}% of {FP16_PEAK_TFLOPS:.0}",
                    100.0 * tf / FP16_PEAK_TFLOPS);
            }
        }
    }
}
