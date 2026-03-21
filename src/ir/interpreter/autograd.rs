//! Autograd program SSA interpreter with gradient collection.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use super::shape::Shape2;
use super::shape::after_diff_op;
use super::{bytes_to_floats, catch_as_result, eval_tensor_instr};
use crate::ir::ops::DiffOp;
use crate::ir::program::{AutogradProgram, FuzzConfig};

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

type DiffB = Autodiff<NdArray>;

fn make_leaf<AB: AutodiffBackend>(raw: &[u8], rows: usize, cols: usize, device: &AB::Device) -> Tensor<AB, 2> {
    Tensor::<AB, 1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// Evaluate one non-Leaf [`DiffOp`] against the register file.
/// Returns `None` for `Leaf` (handled by the main loop).
fn eval_diff_op<AB: AutodiffBackend>(
    regs: &[Tensor<AB, 2>],
    shapes: &[Shape2],
    op: &DiffOp,
    config: &FuzzConfig,
) -> Option<Tensor<AB, 2>> {
    match op {
        DiffOp::Leaf { .. } => None,
        DiffOp::Instr(instr) => Some(eval_tensor_instr(regs, shapes, instr, config)),
    }
}

/// Run `prog` on any autodiff backend, returning gradient data for every leaf
/// in introduction order.
fn collect_grads<AB: AutodiffBackend>(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    device: &AB::Device,
) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(config.min_dim, config.max_dim);
    let cols = (prog.cols as usize).clamp(config.min_dim, config.max_dim);

    // r0 = seed leaf (always present)
    let leaf_0 = make_leaf::<AB>(
        prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]),
        rows,
        cols,
        device,
    );
    let mut regs: Vec<Tensor<AB, 2>> = vec![leaf_0];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];
    let mut leaf_indices: Vec<usize> = vec![0];
    let mut leaf_shapes: Vec<(usize, usize)> = vec![(rows, cols)];
    let mut leaf_count: usize = 1;

    for op in &prog.ops {
        let (val, out_shape) = match op {
            DiffOp::Leaf { seed, rows: lr, cols: lc } => {
                if leaf_count < config.max_leaves {
                    let pool_idx = if prog.leaf_seeds.is_empty() {
                        0
                    } else {
                        *seed as usize % prog.leaf_seeds.len()
                    };
                    let raw = prog
                        .leaf_seeds
                        .get(pool_idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let leaf_rows = (*lr as usize).clamp(config.min_dim, config.max_dim);
                    let leaf_cols = (*lc as usize).clamp(config.min_dim, config.max_dim);
                    let leaf = make_leaf::<AB>(raw, leaf_rows, leaf_cols, device);
                    leaf_indices.push(regs.len());
                    leaf_shapes.push((leaf_rows, leaf_cols));
                    leaf_count += 1;
                    (leaf, Shape2(leaf_rows, leaf_cols))
                } else {
                    let alias_idx = (*seed as usize) % regs.len();
                    (regs[alias_idx].clone(), shapes[alias_idx])
                }
            }
            _ => {
                let out_shape = after_diff_op(&shapes, op)
                    .expect("non-Leaf op returned None shape");
                let val = eval_diff_op(&regs, &shapes, op, config)
                    .expect("non-Leaf op returned None");
                (val, out_shape)
            }
        };
        regs.push(val);
        shapes.push(out_shape);
    }

    // backward from the last register
    let last = regs.last().expect("register file is empty").clone();
    let grads = last.backward();

    leaf_indices
        .iter()
        .zip(leaf_shapes.iter())
        .map(|(&ri, &(lr, lc))| {
            match regs[ri].grad(&grads) {
                Some(g) => g
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_else(|e| panic!("into_data for r{ri} grad failed: {e}")),
                None => vec![0.0_f32; lr * lc],
            }
        })
        .collect()
}

/// Run AutogradProgram against Autodiff<NdArray> backend.
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = collect_grads::<DiffB>(prog, config, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_autograd_oracle(prog, config, nd);
    }))
}

#[cfg(feature = "oracle-tch")]
type DiffTch = Autodiff<LibTorch>;

#[cfg(feature = "oracle-tch")]
fn run_autograd_oracle(prog: &AutogradProgram, config: &FuzzConfig, nd: Vec<Vec<f32>>) {
    let lt = collect_grads::<DiffTch>(prog, config, &LibTorchDevice::Cpu);
    if nd.len() != lt.len() {
        panic!("oracle leaf count mismatch: NdArray={}, LibTorch={}", nd.len(), lt.len());
    }
    for (i, (nd_grad, lt_grad)) in nd.iter().zip(lt.iter()).enumerate() {
        compare_outputs(nd_grad, lt_grad, &format!("grad r{i}"));
    }
}
