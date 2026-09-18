//! The one SSA driver, shared by every target.
//!
//! # What this is, and — more importantly — what it is not
//!
//! This module defines the crate's only trait, and the boundary it draws is
//! narrow on purpose.  [`Framework`] is **not** an abstraction over the
//! instruction set: it has no arm per [`TensorInstr`], and adding an op to the
//! IR does not touch it.  `TensorInstr` remains a vocabulary that each target
//! consumes with its own `match`, exactly as before — see `ir/ops.rs`.
//!
//! What it abstracts is the *scaffolding around* that match, which is a
//! genuinely different thing and was genuinely duplicated.  Running an SSA
//! program is the same algorithm no matter who executes the instructions:
//!
//! * build `r0` from seed bytes, clamped to `[1,16]`
//! * for each op, ask `after_tensor_instr` for the output shape, stop if it
//!   `exceeds_cap`, evaluate, push the value *and the predicted shape*
//! * for autograd: track which registers are leaves and what shape each had,
//!   introduce a new leaf until `max_leaves`, then alias an existing register
//!   instead, and after `backward()` report one gradient per leaf in
//!   introduction order, zeros where a leaf never reached the root
//!
//! Every line of that was written three times — once per target — and it is
//! the half that is easy to get subtly wrong: a leaf recorded at the wrong
//! index, an `exceeds_cap` break in one target and not another, an alias that
//! copies instead of sharing.  Those are not bugs in burn; they are bugs in the
//! oracle, and they manufacture exactly the false divergences this fuzzer
//! exists to avoid.  A divergence is only meaningful if both sides ran *the
//! same program*, and that is now true by construction rather than by three
//! files agreeing.
//!
//! # Why a trait here and not over the ops
//!
//! A trait over the execution surface — 22 ops plus data in/out plus autodiff,
//! ~27 methods — would not remove a single `match` arm.  It would relocate each
//! target's 22 arms into an `impl` block and add a generic parameter to
//! everything that touches a tensor.  The arms are irreducibly per-framework
//! (`broadcast_add` vs `+`, `sum_keepdim` vs `sum_dim_intlist`, rank-0 vs
//! `[1,1]` reductions); there is nothing shared there to factor out.
//!
//! The driver is the opposite case: nothing in it is per-framework except the
//! seven operations below, and all seven are one to three lines in every
//! implementation.  So the trait is placed where the sharing actually is.
//!
//! # Adding a target
//!
//! Write the 22-arm `match` as a free function, implement [`Framework`] over
//! it, and add one arm to `eval_on_target` / `grads_on_target`.  Nothing in
//! this file changes, and neither does the IR, the generator, shape
//! propagation, the operand resolvers or `values_diverge`.

use super::shape::{Shape2, after_diff_op, after_tensor_instr};
use crate::ir::ops::{DiffOp, TensorInstr};
use crate::ir::program::{AutogradProgram, FuzzConfig, TensorProgram};

/// One framework's answers to the seven questions the SSA driver has to ask.
///
/// Implementations are expected to be thin: every method here is one to three
/// lines over the framework's own API, and anything longer is a sign that logic
/// belonging in the driver has leaked into a target.
pub(crate) trait Framework {
    /// This framework's tensor type.
    type Tensor;
    /// Whatever `backward()` hands back.  burn returns a `Gradients` store,
    /// candle a `GradStore`, and libtorch nothing at all — it writes gradients
    /// onto the leaves themselves, so `tch_raw` uses `()` here.
    type Grads;

    /// Build `r0` (or any non-differentiated input) from fuzzer seed bytes.
    ///
    /// Must go through `bytes_to_floats`: every target has to see bit-identical
    /// inputs or the comparison means nothing.
    fn input(&self, raw: &[u8], rows: usize, cols: usize) -> Self::Tensor;

    /// Build a `requires_grad` leaf from fuzzer seed bytes.
    ///
    /// Same data as [`Framework::input`]; what differs is only that gradients
    /// are tracked back to it.
    fn leaf(&self, raw: &[u8], rows: usize, cols: usize) -> Self::Tensor;

    /// Reference an existing register as a second name for the same value.
    ///
    /// **Must share the autograd node, not copy it.**  This is how the driver
    /// expresses "the leaf cap is reached, alias an existing register instead",
    /// and a copying implementation makes the aliased leaf's gradient come back
    /// short — silently, and only in programs that hit the cap.  burn's
    /// `Tensor::clone` and candle's `Arc` clone already do the right thing;
    /// libtorch needs `shallow_clone` specifically.
    fn alias(&self, tensor: &Self::Tensor) -> Self::Tensor;

    /// Evaluate one instruction — the 22-arm `match`, which stays a free
    /// function in each target's own module.
    ///
    /// `shapes` carries the *predicted* shape of every register so the shared
    /// operand resolvers (`resolve_broadcast_compatible` and friends) can be
    /// called from here.  Calling them, rather than reimplementing operand
    /// choice, is what guarantees every target picks the same registers.
    fn eval(
        &self,
        regs: &[Self::Tensor],
        shapes: &[Shape2],
        instr: &TensorInstr,
    ) -> Self::Tensor;

    /// Differentiate the root.
    ///
    /// The seed matters and is not free to choose: burn seeds the root gradient
    /// with **ones of the root's shape**, so an implementation must do the same
    /// or every gradient it reports is rescaled by a constant nobody notices.
    /// candle's `backward()` already seeds that way; libtorch refuses a
    /// non-scalar root, so `tch_raw` sums first — which is exactly the ones-seed
    /// (`d(Σy)/dx == Σᵢ ∂yᵢ/∂x · 1`), and no other reduction is.
    ///
    /// A root that does not track gradients is **not** to be guarded against.
    /// burn panics there ("Tensor::backward requires a tracked autodiff
    /// tensor") and that panic was a real bug — `powf_scalar(0)` detaching from
    /// the graph, bug #3.  A guard hides exactly that class.
    fn backward(&self, root: &Self::Tensor) -> Self::Grads;

    /// The gradient accumulated for one leaf, or `None` if it never reached the
    /// root.  The driver reports `None` as zeros, which is what burn does.
    fn grad(&self, grads: &Self::Grads, leaf: &Self::Tensor) -> Option<Self::Tensor>;

    /// Read a tensor out as row-major `f32`.
    ///
    /// Must materialise a non-contiguous tensor rather than blitting it:
    /// `Transpose` leaves a strided view in both non-burn targets, and reading
    /// one without a copy reports its elements in the wrong order — a fake
    /// divergence on every program containing a transpose.
    fn to_vec(&self, tensor: &Self::Tensor) -> Vec<f32>;
}

// ─── the drivers ─────────────────────────────────────────────────────────────

/// Shapes are clamped into `[1, 16]` on the way in.
///
/// The bound is the generator's, not a framework's, so it lives here: a target
/// that clamped differently would run a different program and every comparison
/// against it would be meaningless.
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
        // Chained Repeat/Concat/Matmul on this unconstrained op stream can
        // otherwise compound into a multi-gigabyte single allocation (see
        // `Shape2::exceeds_cap`'s doc comment) — stop here rather than let it
        // OOM the whole fuzz process; not a bug, just a generation-space gap.
        //
        // Every target breaks at the same instruction because they all break
        // *here*, which is the point of the shape prediction being shared.
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
/// introduction order.
///
/// Leaves and intermediates share one flat register file, so a leaf is just an
/// index into it; `leaf_indices` records which. `leaf_shapes` records what shape
/// each had, because an unreachable leaf has to be reported as zeros of the
/// right length, and by then its tensor tells us nothing about what the
/// gradient *would* have looked like.
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
                    // Leaf cap reached: alias an existing register rather than
                    // introduce another.  `alias` must share the autograd node
                    // — see its doc comment.
                    let alias_idx = (*seed as usize) % regs.len();
                    (f.alias(&regs[alias_idx]), shapes[alias_idx])
                }
            }
            _ => {
                let out_shape =
                    after_diff_op(&shapes, op).expect("non-Leaf op returned None shape");
                // Defensive backstop matching the plain TensorProgram path —
                // the generator already keeps shapes far under this via
                // MAX_DIM, so this should never trigger here.
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
