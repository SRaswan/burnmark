//! Autograd program SSA interpreter with gradient collection.

use burn::tensor::{Device, Gradients, Tensor};

use super::shape::Shape2;
use super::shape::after_diff_op;
use super::{assert_agreement, bytes_to_floats, catch_as_result, device_for, eval_tensor_instr};
use crate::ir::ops::DiffOp;
use crate::ir::program::{AutogradProgram, FuzzConfig};

fn make_leaf(raw: &[u8], rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::<1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// Evaluate one non-Leaf [`DiffOp`] against the register file.
/// Returns `None` for `Leaf` (handled by the main loop).
fn eval_diff_op(
    regs: &[Tensor<2>],
    shapes: &[Shape2],
    op: &DiffOp,
) -> Option<Tensor<2>> {
    match op {
        DiffOp::Leaf { .. } => None,
        DiffOp::Instr(instr) => Some(eval_tensor_instr(regs, shapes, instr)),
    }
}

/// Run `prog` on whichever backend `device` selects (must be an autodiff
/// device, i.e. built with `.autodiff()`), returning gradient data for every
/// leaf in introduction order.
fn collect_grads(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    device: &Device,
) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let leaf_0 = make_leaf(
        prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]),
        rows,
        cols,
        device,
    );
    let mut regs: Vec<Tensor<2>> = vec![leaf_0];
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
                    let leaf_rows = (*lr as usize).clamp(1, 16);
                    let leaf_cols = (*lc as usize).clamp(1, 16);
                    let leaf = make_leaf(raw, leaf_rows, leaf_cols, device);
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
                // Defensive backstop matching the plain TensorProgram path
                // (see `Shape2::exceeds_cap`'s doc comment) — the generator
                // already keeps shapes far under this via MAX_DIM, so this
                // should never actually trigger here, but stopping early is
                // cheap insurance against a multi-gigabyte allocation rather
                // than an OOM abort.
                if out_shape.exceeds_cap() {
                    break;
                }
                let val = eval_diff_op(&regs, &shapes, op)
                    .expect("non-Leaf op returned None");
                (val, out_shape)
            }
        };
        regs.push(val);
        shapes.push(out_shape);
    }

    // backward from the last register
    let last = regs.last().expect("register file is empty").clone();
    let grads: Gradients = last.backward();

    leaf_indices
        .iter()
        .zip(leaf_shapes.iter())
        .map(|(&ri, &(lr, lc))| {
            match regs[ri].grad(&grads) {
                Some(g) => g
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap_or_else(|e| panic!("into_data for r{ri} grad failed: {e}")),
                None => vec![0.0_f32; lr * lc],
            }
        })
        .collect()
}

/// Run `prog` on every backend in `config.backends` and require their per-leaf
/// gradients to agree.  The first entry is the reference side; a single entry
/// simply executes the program with no comparison.
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        // `.autodiff()` wraps whichever backend the device selects, so every
        // entry gets gradient tracking without the interpreter having to know
        // which backend it is running on.
        let grads: Vec<(&'static str, Vec<Vec<f32>>)> = config
            .backends
            .iter()
            .map(|&backend| {
                let device = device_for(backend).autodiff();
                (backend.name(), collect_grads(prog, config, &device))
            })
            .collect();

        let Some(((ref_name, reference), others)) = grads.split_first() else {
            return;
        };
        for (name, leaves) in others {
            if leaves.len() != reference.len() {
                panic!(
                    "leaf count mismatch: {ref_name}={}, {name}={}",
                    reference.len(),
                    leaves.len()
                );
            }
        }

        for leaf in 0..reference.len() {
            let view: Vec<(&'static str, &[f32])> = grads
                .iter()
                .map(|(name, per_leaf)| (*name, per_leaf[leaf].as_slice()))
                .collect();
            assert_agreement(&view, &format!("grad r{leaf}"));
        }
    }))
}
