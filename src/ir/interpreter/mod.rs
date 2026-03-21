//! IR interpreter – walks SSA programs using a register-file architecture.
//!
//! All public entry-points return `Result<(), String>`.  The fuzz target
//! decides what to do: in `PanicOnFirstError` mode it panics (so libFuzzer
//! saves the crash artifact), in `Continuous` it logs to stderr and moves on.

pub(crate) mod shape;
mod tensor_program;
mod autograd;

pub use tensor_program::run_tensor_program;
pub use autograd::run_autograd_program;

use burn::backend::NdArray;
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::Backend;

// use burn::tensor::backend::AutodiffBackend;
use crate::ir::program::FuzzConfig;

use super::ops::TensorInstr;
use shape::{Shape2, resolve_broadcast_compatible, resolve_matmul_compatible, resolve_concat_compatible};

type PlainB = NdArray;

// ─── shared utilities ────────────────────────────────────────────────────────

/// Cycle raw bytes and map to f32 values in [-1, 1].
fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

/// Wrap a closure that may panic, converting the panic into `Err(String)`.
fn catch_as_result<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Result<(), String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    result.map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "<unknown panic payload>".to_string()
        }
    })
}

/// Compare two output vectors element-wise with relative tolerance.
#[cfg(feature = "oracle-tch")]
fn compare_outputs(ndarray: &[f32], libtorch: &[f32], label: &str) {
    assert_eq!(ndarray.len(), libtorch.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0_usize;
    for (_i, (&a, &b)) in ndarray.iter().zip(libtorch).enumerate() {
        if a.is_nan() && b.is_nan() { continue; }
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0_f32);
        if abs_diff > 1e-4_f32 * scale {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("oracle detected {} mismatches in {}", mismatches, label);
    }
}

// ─── shared instruction evaluator ────────────────────────────────────────────

/// Evaluate one [`TensorInstr`] against the register file, using `shapes`
/// to ensure binary operands are shape-compatible.
/// Evaluate one [`TensorInstr`] against the register file, using `shapes`
/// to ensure binary operands are shape-compatible.
fn eval_tensor_instr<B: Backend>(
    regs: &[Tensor<B, 2>],
    shapes: &[Shape2],
    instr: &TensorInstr,
    config: &FuzzConfig,
) -> Tensor<B, 2> {
    let n = regs.len();

    match instr {
        TensorInstr::Add(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() + regs[bi].clone()
        }
        TensorInstr::Sub(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() - regs[bi].clone()
        }
        TensorInstr::Mul(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() * regs[bi].clone()
        }
        TensorInstr::Matmul(a, b) => {
            let ai = a.resolve(n);
            match resolve_matmul_compatible(shapes, ai, b) {
                Some(bi) => regs[ai].clone().matmul(regs[bi].clone()),
                None => regs[ai].clone(),
            }
        }

        TensorInstr::Neg(r) => regs[r.resolve(n)].clone().neg(),
        TensorInstr::Abs(r) => regs[r.resolve(n)].clone().abs(),
        TensorInstr::Exp(r) => regs[r.resolve(n)].clone().exp(),

        TensorInstr::Log(r) => {
            let x = regs[r.resolve(n)].clone();
            if config.safe_math {
                x.clamp(1e-6_f32, 1e6_f32).log()
            } else {
                x.log()
            }
        }

        TensorInstr::Sqrt(r) => {
            let x = regs[r.resolve(n)].clone();
            if config.safe_math {
                x.clamp(0.0_f32, 1e6_f32).sqrt()
            } else {
                x.sqrt()
            }
        }

        TensorInstr::Relu(r) => activation::relu(regs[r.resolve(n)].clone()),
        TensorInstr::Sigmoid(r) => activation::sigmoid(regs[r.resolve(n)].clone()),
        TensorInstr::Tanh(r) => activation::tanh(regs[r.resolve(n)].clone()),
        TensorInstr::SumAll(r) => regs[r.resolve(n)].clone().sum().unsqueeze::<2>(),
        TensorInstr::MeanAll(r) => regs[r.resolve(n)].clone().mean().unsqueeze::<2>(),

        TensorInstr::SumDim(r, d) => {
            let dim = *d as usize % 2;
            regs[r.resolve(n)].clone().sum_dim(dim)
        }
        TensorInstr::MeanDim(r, d) => {
            let dim = *d as usize % 2;
            regs[r.resolve(n)].clone().mean_dim(dim)
        }

        TensorInstr::Transpose(r) => regs[r.resolve(n)].clone().transpose(),

        TensorInstr::Concat(a, b, d) => {
            let ai = a.resolve(n);
            let dim = *d as usize % 2;
            match resolve_concat_compatible(shapes, ai, b, dim) {
                Some(bi) => Tensor::cat(vec![regs[ai].clone(), regs[bi].clone()], dim),
                None => regs[ai].clone(),
            }
        }

        TensorInstr::Repeat(r, d, c) => {
            let dim = *d as usize % 2;
            let count = (*c as usize).clamp(1, 4);
            regs[r.resolve(n)].clone().repeat_dim(dim, count)
        }

        TensorInstr::Clamp(r) => regs[r.resolve(n)].clone().clamp(-1e6_f32, 1e6_f32),
    }
}
