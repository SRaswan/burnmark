//! Raw tch-rs interpreter — the same SSA IR, executed against libtorch directly.
//!
//! This is the crate's second *tensor-producing* consumer of [`TensorInstr`],
//! and its first non-burn one.  `eval_tensor_instr_tch` is a **sibling** of
//! `eval_tensor_instr`, not a generic version of it: a new consumer of the IR
//! is a new `match`, not a new trait.  At two targets, two concrete
//! interpreters read better than one generic one, and the IR, the generator,
//! shape propagation, `values_diverge` and target selection all carry over
//! untouched.
//!
//! # Why this target exists
//!
//! `Backend::LibTorch` already runs libtorch — but it reaches it through
//! burn-tch, so burn's FFI/translation layer sits inside the reference side.
//! Comparing `libtorch` against `tch-raw` puts that layer, and only that layer,
//! under test on the forward pass: identical kernels on both sides, burn's
//! bridge on one of them.  That is the class the original 0.20.1 `swap_dims`
//! bug belonged to — a shallow clone across the FFI boundary corrupting
//! gradients, not a math bug — and no pairing of burn-internal backends can
//! isolate it.
//!
//! **On the backward pass it buys considerably more than that**, for a reason
//! worth stating plainly: `Device::libtorch().autodiff()` differentiates with
//! **burn-autodiff**.  libtorch supplies the tensors; burn supplies the
//! derivative formulas and the graph.  Every burn backend therefore shares one
//! autograd implementation, so no burn-vs-burn pairing — `cpu` vs `flex` vs
//! `ndarray` vs `libtorch` — can ever disagree about a *derivative*, only about
//! the forward kernels those derivatives call.  Raw tch-rs brings libtorch's
//! own autograd, which makes this the first and only pairing in this fuzzer
//! where the backward pass itself is implemented twice, independently.  Given
//! that `fuzz_autograd` is where every bug so far has come from, that is the
//! larger half of the case for this target.
//!
//! It is also the oracle burn cannot deprecate: `Device::libtorch()` is going
//! away on burn `main`, but `tch` is an independent project and what burn is
//! dropping is the bridge, not the library.
//!
//! # Fidelity notes
//!
//! Every arm below is written to match what burn's own libtorch backend emits
//! for the same instruction, so that a divergence means a real disagreement
//! rather than a harness mismatch.  The three places that needed care:
//!
//! * `SumDim`/`MeanDim` keep the reduced dimension (`keepdim = true`), which is
//!   what `after_tensor_instr` predicts and what burn-tch passes.
//! * `Repeat` is *tile* semantics (`[a,b] ×2 → [a,b,a,b]`), which is what both
//!   burn's `repeat_dim` and libtorch's `repeat` do — not `repeat_interleave`.
//! * `backward()` is seeded with ones (see [`collect_grads`]).

use tch::{Kind, Tensor};

use super::bytes_to_floats;
use super::shape::{
    Shape2, after_diff_op, after_tensor_instr, resolve_broadcast_compatible,
    resolve_concat_compatible, resolve_matmul_compatible,
};
use crate::ir::ops::{DiffOp, POWF_EXPONENTS, TensorInstr};
use crate::ir::program::{AutogradProgram, FuzzConfig, TensorProgram};

/// Every tensor here is `f32`, matching burn's `Tensor<2>`.
const KIND: Kind = Kind::Float;

// ─── data in / out ───────────────────────────────────────────────────────────

/// Build one 2-D input tensor from fuzzer seed bytes, via the *same*
/// `bytes_to_floats` the burn side uses — both targets must see bit-identical
/// inputs or every comparison is meaningless.
fn make_tensor(raw: &[u8], rows: usize, cols: usize, requires_grad: bool) -> Tensor {
    let t = Tensor::from_slice(bytes_to_floats(raw, rows * cols).as_slice())
        .reshape([rows as i64, cols as i64]);
    // Set after the reshape so the *reshaped* tensor is the autograd leaf,
    // matching burn's `.reshape(..).require_grad()` ordering.
    if requires_grad { t.set_requires_grad(true) } else { t }
}

/// Flatten to `Vec<f32>` — the counterpart of burn's
/// `.into_data().try_to_vec::<f32>()`.
///
/// `contiguous()` is not optional: `Transpose` leaves a strided view, and the
/// underlying `at_copy_data` blits raw memory, so a non-contiguous tensor would
/// hand the oracle its elements in the wrong order — a fake divergence on every
/// program containing a transpose.
fn to_vec(t: &Tensor) -> Vec<f32> {
    let flat = t.contiguous().reshape([-1_i64]);
    Vec::<f32>::try_from(&flat).unwrap_or_else(|e| panic!("tch-raw into_data failed: {e}"))
}

// ─── instruction evaluator ───────────────────────────────────────────────────

/// Evaluate one [`TensorInstr`] against the register file, using `shapes` to
/// legalise binary operands.
///
/// The operand-resolution calls are shared with the burn interpreter rather
/// than reimplemented, so both targets are guaranteed to pick the *same*
/// registers for every instruction.
fn eval_tensor_instr_tch(regs: &[Tensor], shapes: &[Shape2], instr: &TensorInstr) -> Tensor {
    let n = regs.len();
    match instr {
        TensorInstr::Add(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            &regs[ai] + &regs[bi]
        }
        TensorInstr::Sub(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            &regs[ai] - &regs[bi]
        }
        TensorInstr::Mul(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            &regs[ai] * &regs[bi]
        }
        TensorInstr::Div(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            &regs[ai] / &regs[bi]
        }
        TensorInstr::Matmul(a, b) => {
            let ai = a.resolve(n);
            match resolve_matmul_compatible(shapes, ai, b) {
                Some(bi) => regs[ai].matmul(&regs[bi]),
                None => regs[ai].shallow_clone(),
            }
        }
        TensorInstr::Neg(r)     => regs[r.resolve(n)].neg(),
        TensorInstr::Abs(r)     => regs[r.resolve(n)].abs(),
        TensorInstr::Exp(r)     => regs[r.resolve(n)].exp(),
        TensorInstr::Log(r)     => regs[r.resolve(n)].log(),
        TensorInstr::Sqrt(r)    => regs[r.resolve(n)].sqrt(),
        TensorInstr::PowfScalar(r, e) => {
            let exp = POWF_EXPONENTS[*e as usize % POWF_EXPONENTS.len()];
            regs[r.resolve(n)].pow_tensor_scalar(exp as f64)
        }
        TensorInstr::Relu(r)    => regs[r.resolve(n)].relu(),
        TensorInstr::Sigmoid(r) => regs[r.resolve(n)].sigmoid(),
        TensorInstr::Tanh(r)    => regs[r.resolve(n)].tanh(),
        // burn does `.sum().unsqueeze::<2>()`; libtorch's `sum` is 0-dim, so
        // reshape rather than unsqueeze to land on the same [1,1].
        TensorInstr::SumAll(r)  => regs[r.resolve(n)].sum(KIND).reshape([1_i64, 1]),
        TensorInstr::MeanAll(r) => regs[r.resolve(n)].mean(KIND).reshape([1_i64, 1]),
        TensorInstr::SumDim(r, d) => {
            let dim = *d as i64 % 2;
            regs[r.resolve(n)].sum_dim_intlist(Some([dim].as_slice()), true, KIND)
        }
        TensorInstr::MeanDim(r, d) => {
            let dim = *d as i64 % 2;
            regs[r.resolve(n)].mean_dim(Some([dim].as_slice()), true, KIND)
        }
        TensorInstr::Transpose(r) => regs[r.resolve(n)].transpose(0, 1),
        TensorInstr::Concat(a, b, d) => {
            let ai = a.resolve(n);
            let dim = *d as usize % 2;
            match resolve_concat_compatible(shapes, ai, b, dim) {
                Some(bi) => Tensor::cat(&[&regs[ai], &regs[bi]], dim as i64),
                None => regs[ai].shallow_clone(),
            }
        }
        TensorInstr::Repeat(r, d, c) => {
            let dim = *d as usize % 2;
            let count = (*c as usize).clamp(1, 4) as i64;
            let mut factors = [1_i64, 1];
            factors[dim] = count;
            regs[r.resolve(n)].repeat(factors)
        }
        TensorInstr::Clamp(r) => regs[r.resolve(n)].clamp(-1e6_f64, 1e6_f64),
    }
}

// ─── program runners ─────────────────────────────────────────────────────────

/// Run a plain SSA [`TensorProgram`] against libtorch directly.
///
/// Mirrors the burn path in `tensor_program.rs` step for step, including the
/// `exceeds_cap` break — the two must stop at the same instruction or they
/// would compare different programs.
pub(super) fn eval_tensor_program(prog: &TensorProgram) -> Vec<f32> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let mut regs: Vec<Tensor> = vec![make_tensor(&prog.values, rows, cols, false)];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];

    for instr in &prog.ops {
        let out_shape = after_tensor_instr(&shapes, instr);
        if out_shape.exceeds_cap() {
            break;
        }
        let val = eval_tensor_instr_tch(&regs, &shapes, instr);
        regs.push(val);
        shapes.push(out_shape);
    }

    to_vec(&regs.pop().expect("register file is empty"))
}

/// Run an [`AutogradProgram`] against libtorch directly, returning gradient data
/// for every leaf in introduction order.
///
/// # The backward seed
///
/// burn's `backward()` seeds the root gradient with **ones of the root's
/// shape** (`Gradients::new_with_hook` registers `float_ones`), while
/// libtorch's `backward()` refuses a non-scalar root outright.  Summing the
/// root first is exactly that ones-seed — `d(Σy)/dx == Σᵢ ∂yᵢ/∂x · 1` — so the
/// two halves stay comparable.  Reducing any other way (mean, first element)
/// would silently rescale every gradient and make every comparison wrong.
///
/// A root that does not track gradients is left to fail here rather than being
/// guarded: burn panics in that situation too ("Tensor::backward requires a
/// tracked autodiff tensor"), and that panic *was* a real bug — `powf_scalar(0)`
/// detaching from the graph. A guard would hide exactly that class.
pub(super) fn collect_grads(prog: &AutogradProgram, config: &FuzzConfig) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let leaf_0 = make_tensor(
        prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]),
        rows,
        cols,
        true,
    );
    let mut regs: Vec<Tensor> = vec![leaf_0];
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
                    let leaf = make_tensor(raw, leaf_rows, leaf_cols, true);
                    leaf_indices.push(regs.len());
                    leaf_shapes.push((leaf_rows, leaf_cols));
                    leaf_count += 1;
                    (leaf, Shape2(leaf_rows, leaf_cols))
                } else {
                    // Leaf cap reached: alias an existing register.  A shallow
                    // clone shares the autograd node, exactly as burn's
                    // `Tensor::clone` does, so gradients still reach the
                    // original leaf.
                    let alias_idx = (*seed as usize) % regs.len();
                    (regs[alias_idx].shallow_clone(), shapes[alias_idx])
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
                (eval_tensor_instr_tch(&regs, &shapes, instr), out_shape)
            }
        };
        regs.push(val);
        shapes.push(out_shape);
    }

    let last = regs.last().expect("register file is empty").shallow_clone();
    last.sum(KIND).backward();

    leaf_indices
        .iter()
        .zip(leaf_shapes.iter())
        .map(|(&ri, &(lr, lc))| {
            let grad = regs[ri].grad();
            // An undefined grad is libtorch's `None`: the leaf never reached
            // the root.  burn reports the same case as zeros.
            if grad.defined() {
                to_vec(&grad)
            } else {
                vec![0.0_f32; lr * lc]
            }
        })
        .collect()
}

// ─── fidelity tests ──────────────────────────────────────────────────────────

/// Does this interpreter mean the same thing by each instruction as burn's own
/// libtorch backend does?
///
/// # What is shared, and what is not
///
/// The two sides share the **forward** kernels: both end up in the same C++
/// library, so for the forward pass there is no legitimate numerical difference
/// and any divergence is a mistranslation in one of the 22 arms above — a wrong
/// `keepdim`, tile-vs-interleave, a missing `contiguous()`. That is asserted
/// exhaustively below, and it is what keeps a real `libtorch` vs `tch-raw`
/// report meaningful.
///
/// The **backward** pass is a different matter, and it is the more interesting
/// half.  `Device::libtorch().autodiff()` differentiates with **burn-autodiff**;
/// libtorch's tensors are only the thing burn's own derivative formulas are
/// evaluated on.  So the gradients compared here come from two genuinely
/// independent autograd implementations — which is why the backward pass is
/// deliberately *not* asserted per-op here: disagreement is the signal this
/// target exists to produce, not a defect in the translation.  As of
/// 0.22.0-pre.3 two instructions already disagree:
///
/// * `x.powf(0)` — burn drops the node from the graph entirely and panics
///   ("Node should have a step registered"); libtorch keeps it.  Known bug,
///   patched in `fuzz/Cargo.toml` but not in this crate's dependency graph.
/// * `x.powf(-2)` at `x == 0` — burn yields `NaN`, libtorch `-inf`, for a
///   derivative that is singular there.
///
/// What *is* asserted about the backward pass is the plumbing — leaf
/// introduction, aliasing, unreachable leaves, and the ones-seeded root — since
/// an error there would corrupt every gradient comparison rather than report a
/// real one.
#[cfg(all(test, feature = "oracle-tch"))]
mod matches_burns_libtorch_backend {
    use crate::ir::interpreter::{run_autograd_program, run_tensor_program};
    use crate::ir::ops::{DiffOp, POWF_EXPONENTS, Reg, TensorInstr};
    use crate::ir::program::{AutogradProgram, Backend, FuzzConfig, Target, TensorProgram};

    /// Spans negative, zero and positive: `log`/`sqrt`/`powf` need negatives to
    /// reach their NaN paths, and `div` a zero to reach infinity.
    const SEED: [u8; 9] = [0, 32, 64, 128, 160, 192, 224, 255, 96];

    fn config() -> FuzzConfig {
        FuzzConfig {
            targets: vec![Target::Burn(Backend::LibTorch), Target::TchRaw],
            ..FuzzConfig::default()
        }
    }

    /// Every [`TensorInstr`] variant, on a square `r0` so that matmul, concat
    /// and transpose are all legal without operand fixup.
    fn every_instruction() -> Vec<TensorInstr> {
        let r = Reg(0);
        let mut all = vec![
            TensorInstr::Add(r, r),
            TensorInstr::Sub(r, r),
            TensorInstr::Mul(r, r),
            TensorInstr::Div(r, r),
            TensorInstr::Neg(r),
            TensorInstr::Abs(r),
            TensorInstr::Exp(r),
            TensorInstr::Log(r),
            TensorInstr::Sqrt(r),
            TensorInstr::Relu(r),
            TensorInstr::Sigmoid(r),
            TensorInstr::Tanh(r),
            TensorInstr::SumAll(r),
            TensorInstr::MeanAll(r),
            TensorInstr::Transpose(r),
            TensorInstr::Matmul(r, r),
            TensorInstr::Clamp(r),
        ];
        // Dimensional ops on both dims, every repeat count, every exponent.
        for dim in 0..2_u8 {
            all.push(TensorInstr::SumDim(r, dim));
            all.push(TensorInstr::MeanDim(r, dim));
            all.push(TensorInstr::Concat(r, r, dim));
            for count in 1..=4_u8 {
                all.push(TensorInstr::Repeat(r, dim, count));
            }
        }
        for e in 0..POWF_EXPONENTS.len() as u8 {
            all.push(TensorInstr::PowfScalar(r, e));
        }
        all
    }

    fn assert_forward_agrees(ops: Vec<TensorInstr>, what: &str) {
        let prog = TensorProgram { rows: 3, cols: 3, values: SEED.to_vec(), ops };
        if let Err(msg) = run_tensor_program(&prog, &config()) {
            panic!("forward translation mismatch for {what}:\n{prog}\n{msg}");
        }
    }

    #[test]
    fn forward_matches_for_every_instruction() {
        for instr in every_instruction() {
            let line = instr.ssa_line("r1", 1);
            assert_forward_agrees(vec![instr], &line);
        }
    }

    /// Chained, so each op must also agree about what it *received*.  A
    /// transpose leaves a strided view in libtorch, which is the case a missing
    /// `contiguous()` on the way out would silently reorder — invisible to the
    /// isolated tests above.
    #[test]
    fn forward_matches_chained_onto_a_transpose() {
        for instr in every_instruction() {
            let line = instr.ssa_line("r2", 2);
            assert_forward_agrees(
                vec![TensorInstr::Transpose(Reg(0)), instr],
                &format!("r1 = r0.T; {line}"),
            );
        }
    }

    /// The gradient *plumbing*, on ops whose derivatives are smooth and finite
    /// everywhere here — so the two autograd implementations have nothing
    /// legitimate to disagree about, and a failure means this interpreter
    /// mishandled a leaf, an alias, or the backward seed.
    fn assert_backward_plumbing_agrees(ops: Vec<DiffOp>, what: &str) {
        let config = config();
        let prog = AutogradProgram {
            rows: 3,
            cols: 3,
            leaf_seeds: vec![SEED.to_vec(), SEED.iter().rev().copied().collect()],
            ops,
        };
        if let Err(msg) = run_autograd_program(&prog, &config) {
            panic!("gradient plumbing mismatch for {what}:\n{}\n{msg}", prog.ssa(config.max_leaves));
        }
    }

    /// burn seeds the root gradient with ones of the root's shape; libtorch
    /// refuses a non-scalar root, so this interpreter sums first.  If that were
    /// the wrong reduction every gradient would be rescaled — and a root whose
    /// shape differs from the leaf's is where it would show.
    #[test]
    fn backward_seed_matches_for_shape_changing_roots() {
        for root in [
            TensorInstr::SumAll(Reg(1)),
            TensorInstr::MeanAll(Reg(1)),
            TensorInstr::SumDim(Reg(1), 0),
            TensorInstr::MeanDim(Reg(1), 1),
            TensorInstr::Transpose(Reg(1)),
            TensorInstr::Concat(Reg(1), Reg(1), 0),
            TensorInstr::Repeat(Reg(1), 1, 3),
            TensorInstr::Matmul(Reg(1), Reg(1)),
        ] {
            let line = root.ssa_line("r2", 2);
            assert_backward_plumbing_agrees(
                vec![
                    DiffOp::Instr(TensorInstr::Tanh(Reg(0))),
                    DiffOp::Instr(root),
                ],
                &format!("r1 = tanh(r0); {line}"),
            );
        }
    }

    /// Multiple leaves, a leaf that never reaches the root, and an aliasing
    /// leaf past the cap — the three places gradient plumbing could differ.
    #[test]
    fn backward_plumbing_matches_across_leaves() {
        let leaf = |seed: u8, rows: u8, cols: u8| DiffOp::Leaf { seed, rows, cols };
        assert_backward_plumbing_agrees(
            vec![
                leaf(1, 3, 3),
                DiffOp::Instr(TensorInstr::Mul(Reg(0), Reg(1))),
                leaf(0, 3, 3),
                DiffOp::Instr(TensorInstr::Matmul(Reg(2), Reg(3))),
                DiffOp::Instr(TensorInstr::SumAll(Reg(4))),
            ],
            "two extra leaves, both reachable",
        );
        assert_backward_plumbing_agrees(
            vec![
                // Introduced but never used: zeros on both sides — burn's
                // `None`, libtorch's undefined tensor.
                leaf(1, 2, 5),
                DiffOp::Instr(TensorInstr::Tanh(Reg(0))),
            ],
            "unreachable leaf",
        );
        assert_backward_plumbing_agrees(
            vec![
                leaf(0, 3, 3),
                leaf(1, 3, 3),
                leaf(0, 3, 3),
                // Four leaves exist now (r0 + 3), so this one aliases instead.
                // The alias must share the autograd node, not copy it, or the
                // original leaf's gradient comes back short.
                leaf(2, 3, 3),
                DiffOp::Instr(TensorInstr::Add(Reg(4), Reg(1))),
                DiffOp::Instr(TensorInstr::Mul(Reg(5), Reg(2))),
            ],
            "leaf cap reached, aliasing",
        );
    }
}
