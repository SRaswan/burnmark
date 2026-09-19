//! SSA driver shared by every target.
//!
//! [`Framework`] is the crate's only trait. It abstracts the scaffolding *around*
//! each target's 22-arm instruction match — register file, leaf/alias protocol,
//! `exceeds_cap` break, grad collection — never the arms themselves. Adding an op
//! does not touch this file.
//!
//! Every target runs the *same program* by construction because the walk lives here
//! rather than being copied per target. A leaf at the wrong index, or a break that
//! fires on one target but not another, produces false divergences — bugs in the
//! oracle that look exactly like bugs in the framework under test.
//!
//! To add a target: write a 22-arm free function, implement [`Framework`], add one
//! arm each to `eval_on_target` / `grads_on_target`. Nothing here changes.

use super::shape::{Shape2, after_diff_op, after_tensor_instr};
use crate::ir::ops::{DiffOp, TensorInstr};
use crate::ir::program::{AutogradProgram, FuzzConfig, TensorProgram};

/// One framework's answers to the seven questions the SSA driver has to ask.
/// Every method is one to three lines; anything longer means driver logic has
/// leaked into a target.
pub(crate) trait Framework {
    /// This framework's tensor type.
    type Tensor;
    /// Whatever `backward()` hands back.  burn returns a `Gradients` store,
    /// candle a `GradStore`, and libtorch nothing at all — it writes gradients
    /// onto the leaves themselves, so `tch_raw` uses `()` here.
    type Grads;

    /// Build `r0` (or any non-differentiated input) from seed bytes.
    /// Must go through `bytes_to_floats` so all targets see identical data.
    fn input(&self, raw: &[u8], rows: usize, cols: usize) -> Self::Tensor;

    /// Build a `requires_grad` leaf from seed bytes.
    fn leaf(&self, raw: &[u8], rows: usize, cols: usize) -> Self::Tensor;

    /// Reference an existing register as an alias.
    ///
    /// **Must share the autograd node, not copy it.** A copy silently drops the
    /// aliased leaf's gradient — only visible in programs that hit the leaf cap.
    /// burn's `Tensor::clone` and candle's `Arc` clone are correct; libtorch
    /// needs `shallow_clone` specifically.
    fn alias(&self, tensor: &Self::Tensor) -> Self::Tensor;

    /// Evaluate one instruction. The 22-arm match stays a free function in each
    /// target's module; `shapes` must be passed to the shared operand resolvers
    /// so every target picks the same register for every operand.
    fn eval(
        &self,
        regs: &[Self::Tensor],
        shapes: &[Shape2],
        instr: &TensorInstr,
    ) -> Self::Tensor;

    /// Differentiate the root. Seed must be **ones of the root's shape** — burn
    /// does this, candle does too; libtorch refuses a non-scalar root so `tch_raw`
    /// sums first (`d(Σy)/dx == Σᵢ ∂yᵢ/∂x · 1`). Any other reduction silently
    /// rescales every gradient.
    ///
    /// Don't guard a root with no gradient tracking — burn panics there and that
    /// panic is the signal (it was bug #3).
    fn backward(&self, root: &Self::Tensor) -> Self::Grads;

    /// Gradient for one leaf, or `None` if it never reached the root.
    /// The driver reports `None` as zeros.
    fn grad(&self, grads: &Self::Grads, leaf: &Self::Tensor) -> Option<Self::Tensor>;

    /// Read a tensor out as row-major `f32`. Must call `contiguous()` first —
    /// `Transpose` leaves a strided view and blitting it reorders elements.
    fn to_vec(&self, tensor: &Self::Tensor) -> Vec<f32>;
}

// ─── the drivers ─────────────────────────────────────────────────────────────

/// Clamp a raw dimension byte to `[1, 16]` — same bound the generator uses.
/// Lives here so all targets clamp identically; a different clamp means a
/// different program and every comparison against it is meaningless.
#[inline]
fn clamp_dim(raw: u8) -> usize {
    (raw as usize).clamp(1, 16)
}

/// Run a plain SSA [`TensorProgram`], returning the final register's data.
pub(crate) fn eval_program<F: Framework>(f: &F, prog: &TensorProgram) -> Vec<f32> {
    let rows = clamp_dim(prog.rows);
    let cols = clamp_dim(prog.cols);

    let mut regs: Vec<F::Tensor> = vec![f.input(&prog.values, rows, cols)];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];

    for instr in &prog.ops {
        let out_shape = after_tensor_instr(&shapes, instr);
        // TensorProgram has no shape-aware generator, so chained Repeat/Concat/
        // Matmul can balloon. All targets break here, so they all break together.
        if out_shape.exceeds_cap() {
            break;
        }
        let val = f.eval(&regs, &shapes, instr);
        regs.push(val);
        shapes.push(out_shape);
    }

    f.to_vec(regs.last().expect("register file is empty"))
}

/// Run an [`AutogradProgram`], returning gradient data for every leaf in
/// introduction order. `leaf_shapes` is kept separately because an unreachable
/// leaf is reported as zeros of the leaf's shape, not the root's.
pub(crate) fn collect_grads<F: Framework>(
    f: &F,
    prog: &AutogradProgram,
    config: &FuzzConfig,
) -> Vec<Vec<f32>> {
    let rows = clamp_dim(prog.rows);
    let cols = clamp_dim(prog.cols);

    let leaf_0 = f.leaf(
        prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]),
        rows,
        cols,
    );
    let mut regs: Vec<F::Tensor> = vec![leaf_0];
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
                    let leaf_rows = clamp_dim(*lr);
                    let leaf_cols = clamp_dim(*lc);
                    let leaf = f.leaf(raw, leaf_rows, leaf_cols);
                    leaf_indices.push(regs.len());
                    leaf_shapes.push((leaf_rows, leaf_cols));
                    leaf_count += 1;
                    (leaf, Shape2(leaf_rows, leaf_cols))
                } else {
                    // Leaf cap: alias an existing register. `alias` must share
                    // the autograd node — a copy silently drops its gradient.
                    let alias_idx = (*seed as usize) % regs.len();
                    (f.alias(&regs[alias_idx]), shapes[alias_idx])
                }
            }
            _ => {
                let out_shape =
                    after_diff_op(&shapes, op).expect("non-Leaf op returned None shape");
                if out_shape.exceeds_cap() {
                    break;
                }
                let DiffOp::Instr(instr) = op else {
                    unreachable!("Leaf handled above")
                };
                (f.eval(&regs, &shapes, instr), out_shape)
            }
        };
        regs.push(val);
        shapes.push(out_shape);
    }

    let grads = f.backward(regs.last().expect("register file is empty"));

    leaf_indices
        .iter()
        .zip(leaf_shapes.iter())
        .map(|(&ri, &(lr, lc))| match f.grad(&grads, &regs[ri]) {
            Some(g) => f.to_vec(&g),
            // The leaf never reached the root.  burn reports this as `None`,
            // candle as an absent store entry, libtorch as an undefined tensor;
            // all three mean zeros of the leaf's shape.
            None => vec![0.0_f32; lr * lc],
        })
        .collect()
}
