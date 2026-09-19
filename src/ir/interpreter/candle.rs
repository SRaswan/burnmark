//! candle interpreter — the same SSA IR, executed against candle.
//!
//! This is the crate's third *tensor-producing* consumer of [`TensorInstr`],
//! and its second non-burn one.  `eval_tensor_instr_candle` is a **sibling** of
//! `eval_tensor_instr` and of `eval_tensor_instr_tch`, not a generic version of
//! either: a new consumer of the IR is a new `match`, not a new trait.  The IR,
//! the generator, shape propagation, the operand resolvers, `values_diverge`
//! and target selection all carry over untouched; only this file is new.
//!
//! # Why this target exists
//!
//! Every other target in this fuzzer shares an implementation with some other
//! target, which bounds what a disagreement between them can mean:
//!
//! * The four burn backends share **burn-autodiff**.  `Device::libtorch()
//!   .autodiff()` differentiates with burn's formulas, not libtorch's, so `cpu`
//!   vs `flex` vs `ndarray` vs `libtorch` can disagree about a forward kernel
//!   but never about a *derivative*.
//! * [`Target::TchRaw`](crate::ir::program::Target::TchRaw) closes that gap on
//!   the backward pass, but on the **forward** pass it is the same libtorch
//!   kernels `Backend::LibTorch` already runs — what differs there is burn's
//!   FFI bridge, which is valuable but is not a second opinion about the math.
//!
//! candle shares neither half with anything.  Its kernels are its own and its
//! autograd is its own, so it is the first target here that is an independent
//! implementation on **both** passes at once.  Two consequences follow, and the
//! second is the one that matters:
//!
//! 1. A `candle` vs `libtorch` forward agreement is *evidence* — two unrelated
//!    implementations landing on the same number — where `tch-raw` vs
//!    `libtorch` agreement is close to a tautology.
//! 2. With `tch-raw` also compiled in, a divergence is **triageable by
//!    majority**.  `BACKENDS=tch-raw,candle,flex` runs two independent oracles
//!    against burn: where both oracles agree and burn does not, burn is wrong
//!    and the report is a finding; where the two oracles split, the question is
//!    about candle rather than about burn.  That is the thing no pairing in
//!    this fuzzer could do before, and it is why candle is worth a second
//!    non-burn interpreter rather than a third burn device.
//!
//! # Fidelity notes
//!
//! Every arm below is written to mean what burn means by the same instruction,
//! so that a divergence is a real disagreement rather than a harness mismatch.
//! The places that needed care:
//!
//! * **Broadcasting is opt-in.**  candle's `add`/`sub`/`mul`/`div` require
//!   identical shapes and error otherwise; the `broadcast_*` variants are the
//!   ones with burn's semantics.  Using the plain forms would turn every
//!   legally-broadcast operand pair — which `resolve_broadcast_compatible`
//!   deliberately produces — into a harness error.
//! * **`SumDim`/`MeanDim` keep the reduced dimension** (`*_keepdim`), which is
//!   what `after_tensor_instr` predicts and what burn's `sum_dim`/`mean_dim` do.
//!   candle's plain `sum`/`mean` squeeze it instead.
//! * **`SumAll`/`MeanAll` reduce to rank 0** in candle, not to `[1,1]`; burn's
//!   `.sum().unsqueeze::<2>()` lands on `[1,1]`, so these reshape.
//! * **`Repeat` is tile semantics.**  candle's `repeat` is built on `cat`, so
//!   `[a,b] ×2 → [a,b,a,b]`, matching burn's `repeat_dim` and libtorch's
//!   `repeat` — not `repeat_interleave`.
//! * **Data out must be contiguous**, for the reason it must be on the tch side:
//!   `Transpose` leaves a strided view, and flattening one without a copy would
//!   hand the oracle its elements in the wrong order — a fake divergence on
//!   every program containing a transpose.
//! * **`sigmoid` lives in `candle-nn`, not `candle-core`.**  It is a real op
//!   there (a `CustomOp1` whose backward is `s·(1-s)`, which is burn's formula),
//!   so this target is pulled in rather than the op being open-coded as
//!   `(1 + (-x).exp()).recip()`.  An open-coded version would differ from burn
//!   in *composition* — three ops where burn has one, with three chances to
//!   round differently and a backward assembled from the chain rather than the
//!   closed form — and those differences would be indistinguishable from the
//!   backend bugs this fuzzer is looking for.
//!
//! One thing that needed no care at all, unlike tch: **the backward seed
//! already matches.**  candle's `backward()` seeds the root gradient with
//! `ones_like()` of the root, which is exactly what burn's
//! `Gradients::new_with_hook` does with `float_ones`.  libtorch, by contrast,
//! refuses a non-scalar root and forced `tch_raw` to sum first.  Nothing here
//! rescales anything.

use candle_core::backprop::GradStore;
use candle_core::{Device, Result, Tensor, Var};

use super::bytes_to_floats;
use super::driver::Framework;
use super::shape::{
    Shape2, resolve_broadcast_compatible, resolve_concat_compatible,
    resolve_matmul_compatible,
};
use crate::ir::ops::{POWF_EXPONENTS, TensorInstr};

/// candle's CPU device.  Cheap to construct (a unit enum variant), so unlike a
/// real GPU backend there is nothing to hoist out of the per-iteration path.
const DEVICE: Device = Device::Cpu;

// ─── data in / out ───────────────────────────────────────────────────────────

/// Build one 2-D input tensor from fuzzer seed bytes, via the *same*
/// `bytes_to_floats` every other target uses — all targets must see
/// bit-identical inputs or every comparison is meaningless.
fn make_tensor(raw: &[u8], rows: usize, cols: usize) -> Tensor {
    Tensor::from_slice(
        bytes_to_floats(raw, rows * cols).as_slice(),
        (rows, cols),
        &DEVICE,
    )
    .unwrap_or_else(|e| panic!("candle input construction failed: {e}"))
}

/// Build one 2-D `requires_grad` leaf.
///
/// candle tracks gradients for `Var`s specifically — there is no
/// `require_grad()` flag on an ordinary tensor — so a leaf is a `Var` unwrapped
/// back into the `Tensor` the register file holds.  The unwrapped tensor keeps
/// `is_variable()`, which is what `backward()` stops at, and `Tensor::clone`
/// shares its `TensorId`, so an aliased leaf still resolves to the same
/// gradient entry (burn's `Tensor::clone` and tch's `shallow_clone` both behave
/// this way too).
fn make_leaf(raw: &[u8], rows: usize, cols: usize) -> Tensor {
    Var::from_slice(
        bytes_to_floats(raw, rows * cols).as_slice(),
        (rows, cols),
        &DEVICE,
    )
    .unwrap_or_else(|e| panic!("candle leaf construction failed: {e}"))
    .into_inner()
}

/// Flatten to `Vec<f32>` — the counterpart of burn's
/// `.into_data().try_to_vec::<f32>()`.
///
/// `contiguous()` is not optional: `Transpose` leaves a strided view, and
/// flattening one without materialising it would report the elements in the
/// wrong order — a fake divergence on every program containing a transpose.
fn to_vec(t: &Tensor) -> Vec<f32> {
    t.contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap_or_else(|e| panic!("candle into_data failed: {e}"))
}

// ─── instruction evaluator ───────────────────────────────────────────────────

/// Evaluate one [`TensorInstr`] against the register file, using `shapes` to
/// legalise binary operands.
///
/// The operand-resolution calls are shared with the other interpreters rather
/// than reimplemented, so every target is guaranteed to pick the *same*
/// registers for every instruction — which is what makes a divergence mean
/// disagreement about math rather than about which program was run.
///
/// Returns candle's own `Result` so the 22 arms stay one line each; the caller
/// attaches the failing SSA line.
fn eval_tensor_instr_candle(
    regs: &[Tensor],
    shapes: &[Shape2],
    instr: &TensorInstr,
) -> Result<Tensor> {
    let n = regs.len();
    match instr {
        TensorInstr::Add(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].broadcast_add(&regs[bi])
        }
        TensorInstr::Sub(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].broadcast_sub(&regs[bi])
        }
        TensorInstr::Mul(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].broadcast_mul(&regs[bi])
        }
        TensorInstr::Div(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].broadcast_div(&regs[bi])
        }
        TensorInstr::Matmul(a, b) => {
            let ai = a.resolve(n);
            match resolve_matmul_compatible(shapes, ai, b) {
                // candle's CPU matmul rejects some strided layouts outright
                // (`MatMulUnexpectedStriding`), and `Transpose` — one of the 22
                // — leaves exactly such a view.  Free when the operand is
                // already contiguous: candle returns the same tensor, adding no
                // graph node, so this costs nothing on the common path and
                // nothing to the gradient on either.
                Some(bi) => regs[ai].contiguous()?.matmul(&regs[bi].contiguous()?),
                None => Ok(regs[ai].clone()),
            }
        }
        TensorInstr::Neg(r)     => regs[r.resolve(n)].neg(),
        TensorInstr::Abs(r)     => regs[r.resolve(n)].abs(),
        TensorInstr::Exp(r)     => regs[r.resolve(n)].exp(),
        TensorInstr::Log(r)     => regs[r.resolve(n)].log(),
        TensorInstr::Sqrt(r)    => regs[r.resolve(n)].sqrt(),
        TensorInstr::PowfScalar(r, e) => {
            let exp = POWF_EXPONENTS[*e as usize % POWF_EXPONENTS.len()];
            regs[r.resolve(n)].powf(exp as f64)
        }
        TensorInstr::Relu(r)    => regs[r.resolve(n)].relu(),
        TensorInstr::Sigmoid(r) => candle_nn::ops::sigmoid(&regs[r.resolve(n)]),
        TensorInstr::Tanh(r)    => regs[r.resolve(n)].tanh(),
        // burn does `.sum().unsqueeze::<2>()`; candle's whole-tensor reductions
        // are rank 0, so reshape rather than unsqueeze to land on the same
        // `[1,1]`.
        TensorInstr::SumAll(r)  => regs[r.resolve(n)].sum_all()?.reshape((1, 1)),
        TensorInstr::MeanAll(r) => regs[r.resolve(n)].mean_all()?.reshape((1, 1)),
        TensorInstr::SumDim(r, d) => {
            let dim = *d as usize % 2;
            regs[r.resolve(n)].sum_keepdim(dim)
        }
        TensorInstr::MeanDim(r, d) => {
            let dim = *d as usize % 2;
            regs[r.resolve(n)].mean_keepdim(dim)
        }
        TensorInstr::Transpose(r) => regs[r.resolve(n)].transpose(0, 1),
        TensorInstr::Concat(a, b, d) => {
            let ai = a.resolve(n);
            let dim = *d as usize % 2;
            match resolve_concat_compatible(shapes, ai, b, dim) {
                Some(bi) => Tensor::cat(&[&regs[ai], &regs[bi]], dim),
                None => Ok(regs[ai].clone()),
            }
        }
        TensorInstr::Repeat(r, d, c) => {
            let dim = *d as usize % 2;
            let count = (*c as usize).clamp(1, 4);
            let mut factors = [1_usize, 1];
            factors[dim] = count;
            regs[r.resolve(n)].repeat(factors.to_vec())
        }
        TensorInstr::Clamp(r) => regs[r.resolve(n)].clamp(-1e6_f32, 1e6_f32),
    }
}

/// [`eval_tensor_instr_candle`] with the failing SSA line attached.
///
/// candle reports shape and dtype problems as `Err` where burn and libtorch
/// panic, so without this an operand-legalisation mistake in this file would
/// surface as a bare `ShapeMismatchBinaryOp` with nothing naming the
/// instruction that produced it.
fn eval_or_panic(regs: &[Tensor], shapes: &[Shape2], instr: &TensorInstr) -> Tensor {
    eval_tensor_instr_candle(regs, shapes, instr).unwrap_or_else(|e| {
        panic!(
            "candle failed on `{}`: {e}",
            instr.ssa_line(&format!("r{}", regs.len()), regs.len())
        )
    })
}

// ─── the `Framework` impl ────────────────────────────────────────────────────

/// candle as a target.
///
/// Stateless: [`DEVICE`] is a unit enum variant, so there is nothing to carry.
/// The SSA walk — register file, leaf bookkeeping, `exceeds_cap` break,
/// aliasing — lives in [`driver`](super::driver) and is shared with every other
/// target, which is what guarantees this target and burn run the *same*
/// program.
pub(super) struct Candle;

impl Framework for Candle {
    type Tensor = Tensor;
    type Grads = GradStore;

    fn input(&self, raw: &[u8], rows: usize, cols: usize) -> Tensor {
        make_tensor(raw, rows, cols)
    }

    fn leaf(&self, raw: &[u8], rows: usize, cols: usize) -> Tensor {
        make_leaf(raw, rows, cols)
    }

    fn alias(&self, tensor: &Tensor) -> Tensor {
        // candle's `Tensor` is an `Arc` over shared state, so a clone keeps the
        // same `TensorId` and therefore resolves to the same `GradStore` entry
        // — exactly what `alias` requires, and what burn's `Tensor::clone` and
        // tch's `shallow_clone` do.  A `copy()` would be wrong.
        tensor.clone()
    }

    fn eval(&self, regs: &[Tensor], shapes: &[Shape2], instr: &TensorInstr) -> Tensor {
        eval_or_panic(regs, shapes, instr)
    }

    /// # The backward seed
    ///
    /// Nothing to reconcile here, unlike the tch side: candle seeds the root
    /// gradient with `ones_like()` of the root and burn seeds it with
    /// `float_ones` of the root, so both already mean the same thing by
    /// `backward()`.  Nothing rescales.
    ///
    /// A root that does not track gradients is not guarded.  burn panics there
    /// ("Tensor::backward requires a tracked autodiff tensor") and that panic
    /// *was* bug #3 — `powf_scalar(0)` detaching from the graph.  candle simply
    /// returns the ones-seed and no gradient for the leaf, which is a divergence
    /// this target should *report*; a guard on either side would hide the class.
    fn backward(&self, root: &Tensor) -> GradStore {
        root.backward()
            .unwrap_or_else(|e| panic!("candle backward failed: {e}"))
    }

    fn grad(&self, grads: &GradStore, leaf: &Tensor) -> Option<Tensor> {
        // An absent entry is candle's way of saying the leaf never reached the
        // root.  The driver reports that as zeros, which is what burn does.
        grads.get(leaf).cloned()
    }

    fn to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        to_vec(tensor)
    }
}

// ─── fidelity tests ──────────────────────────────────────────────────────────

/// Does this interpreter mean the same thing by each instruction as burn does?
///
/// # Why this file's tests are weaker than `tch_raw.rs`'s, and have to be
///
/// `tch_raw` can assert forward agreement with burn's LibTorch backend
/// *exhaustively*, on negatives and zeros and every NaN path, because both
/// sides end in the same C++ library: there is no legitimate numerical
/// difference between them, so every disagreement is a mistranslation in one of
/// its 22 arms.  candle shares no kernel with anything here, which is the whole
/// point of adding it — and it means an assertion of that shape would be
/// asserting that two independent implementations agree about `sign(NaN)` and
/// `0^-2`, i.e. asserting away the signal this target exists to produce.
///
/// So the translation is pinned in two narrower ways instead:
///
/// * **Shapes, exhaustively and unconditionally** — the tests directly below.
///   Every instruction's actual output shape must equal what
///   `after_tensor_instr` predicted, isolated and chained.  This is pure
///   translation, no arithmetic involved, and it is where a wrong `keepdim`, a
///   squeezed whole-tensor reduction, a transposed `repeat` factor or a rank-0
///   root would show up.  It needs no other target compiled in.
/// * **Values, on inputs where the implementations have nothing to disagree
///   about** — the [`against_libtorch`](matches_burn::against_libtorch)
///   submodule.  [`WELL_BEHAVED`](matches_burn::WELL_BEHAVED) is strictly
///   positive and bounded away from 0 and 1, so every one of the 22
///   instructions — `log`, `sqrt`, every exponent in `POWF_EXPONENTS`, the
///   `Div` that would otherwise divide by zero — stays finite and smooth.
///
/// Anything outside that is deliberately left to the fuzzer to report.
#[cfg(test)]
mod matches_burn {
    use super::*;
    use crate::ir::interpreter::shape::after_tensor_instr;
    use crate::ir::ops::Reg;

    /// Strictly positive, bounded away from 0 and 1, all distinct.
    ///
    /// `bytes_to_floats` maps a byte to `b/128 - 1`, so bytes above 128 give
    /// `(0, 1)`.  Distinctness matters for `Repeat`: a tile and an interleave of
    /// a constant are the same tensor, so a uniform seed would not tell them
    /// apart.
    pub(super) const WELL_BEHAVED: [u8; 9] = [160, 176, 192, 208, 224, 240, 255, 144, 200];

    /// Every [`TensorInstr`] variant, on a square `r0` so that matmul, concat
    /// and transpose are all legal without operand fixup.
    pub(super) fn every_instruction() -> Vec<TensorInstr> {
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

    /// Run `ops` through candle, asserting at every step that the tensor candle
    /// actually produced has the shape `after_tensor_instr` predicted.
    ///
    /// The predicted shape — not the observed one — is what gets pushed into
    /// `shapes`, exactly as the driver does, so a silent drift would compound
    /// rather than self-correct.
    fn assert_shapes_match_prediction(ops: &[TensorInstr], rows: usize, cols: usize) {
        let mut regs = vec![make_tensor(&WELL_BEHAVED, rows, cols)];
        let mut shapes = vec![Shape2(rows, cols)];

        for instr in ops {
            let predicted = after_tensor_instr(&shapes, instr);
            let line = instr.ssa_line(&format!("r{}", regs.len()), regs.len());
            let val = eval_or_panic(&regs, &shapes, instr);
            assert_eq!(
                val.dims(),
                [predicted.rows(), predicted.cols()],
                "candle's shape for `{line}` is not what after_tensor_instr predicted"
            );
            regs.push(val);
            shapes.push(predicted);
        }
    }

    /// The assertion that needs no second target: a rank-0 root, a squeezed
    /// reduction, a `keepdim` flipped the wrong way or a transposed `repeat`
    /// factor all land here.
    #[test]
    fn shapes_match_the_ir_prediction() {
        for instr in every_instruction() {
            assert_shapes_match_prediction(&[instr], 3, 3);
        }
        // Chained onto a transpose, and on a non-square input, so a factor pair
        // or a reduced dim applied to the wrong axis cannot hide behind
        // rows == cols.
        for instr in every_instruction() {
            assert_shapes_match_prediction(&[TensorInstr::Transpose(Reg(0)), instr], 2, 5);
        }
    }

    /// The `[1,1]` reshape on the whole-tensor reductions is the one place this
    /// interpreter changes a rank rather than translating an op, so it gets its
    /// own assertion rather than relying on the sweep above to cover it.
    #[test]
    fn whole_tensor_reductions_are_rank_2() {
        for instr in [TensorInstr::SumAll(Reg(0)), TensorInstr::MeanAll(Reg(0))] {
            let regs = vec![make_tensor(&WELL_BEHAVED, 2, 5)];
            let out = eval_or_panic(&regs, &[Shape2(2, 5)], &instr);
            assert_eq!(out.dims(), [1, 1], "candle reductions squeeze by default");
        }
    }

    /// `Repeat` must tile, not interleave — the distinction bug #5 turns on.
    ///
    /// Asserted directly against the expected element order rather than against
    /// another target, because *both* semantics are shape-identical and burn's
    /// own backward disagrees with its forward here.
    #[test]
    fn repeat_tiles_rather_than_interleaves() {
        let regs = vec![make_tensor(&[160, 192], 1, 2)]; // [0.25, 0.5]
        let out = eval_or_panic(&regs, &[Shape2(1, 2)], &TensorInstr::Repeat(Reg(0), 1, 2));
        assert_eq!(
            to_vec(&out),
            vec![0.25, 0.5, 0.25, 0.5],
            "tile is [a,b,a,b]; interleave would be [a,a,b,b]"
        );
    }

    /// Value agreement, against libtorch.
    ///
    /// Gated on `oracle-tch` because libtorch is the strongest reference
    /// available and the one whose special-value conventions the rest are judged
    /// against.  burn's CPU backends are deliberately **not** used here: two of
    /// them are known wrong on `sign(NaN)` (bug #2) and `recip` on aarch64 is
    /// ~0.2% off in the published macerator (bug #1), so a failure against them
    /// would be ambiguous between this file and a known burn bug.
    ///
    /// On [`WELL_BEHAVED`] input a divergence from libtorch is a translation
    /// error with high probability, not a special-value opinion — which is what
    /// makes these assertable at all between two independent implementations.
    #[cfg(feature = "oracle-tch")]
    mod against_libtorch {
        use super::*;
        use crate::ir::interpreter::{run_autograd_program, run_tensor_program};
        use crate::ir::ops::DiffOp;
        use crate::ir::program::{AutogradProgram, Backend, FuzzConfig, Target, TensorProgram};

        fn config() -> FuzzConfig {
            FuzzConfig {
                targets: vec![Target::Burn(Backend::LibTorch), Target::Candle],
                ..FuzzConfig::default()
            }
        }

        fn assert_forward_agrees(ops: Vec<TensorInstr>, what: &str) {
            let prog = TensorProgram {
                rows: 3,
                cols: 3,
                values: WELL_BEHAVED.to_vec(),
                ops,
            };
            if let Err(msg) = run_tensor_program(&prog, &config()) {
                panic!("forward divergence from libtorch for {what}:\n{prog}\n{msg}");
            }
        }

        #[test]
        fn forward_matches_libtorch_for_every_instruction() {
            for instr in every_instruction() {
                let line = instr.ssa_line("r1", 1);
                assert_forward_agrees(vec![instr], &line);
            }
        }

        /// Chained, so each op must also agree about what it *received*.  A
        /// transpose leaves a strided view in candle as it does in libtorch,
        /// which is the case a missing `contiguous()` on the way out — or into
        /// `matmul` — would silently reorder, invisible to the isolated tests
        /// above.
        #[test]
        fn forward_matches_libtorch_chained_onto_a_transpose() {
            for instr in every_instruction() {
                let line = instr.ssa_line("r2", 2);
                assert_forward_agrees(
                    vec![TensorInstr::Transpose(Reg(0)), instr],
                    &format!("r1 = r0.T; {line}"),
                );
            }
        }

        /// The gradient *plumbing* — leaf introduction, aliasing, unreachable
        /// leaves, and the seed at the root.  An error in any of those would
        /// corrupt every gradient this target ever reports rather than produce
        /// one real finding, so unlike per-op derivative agreement it is
        /// asserted.
        ///
        /// Most of that plumbing now lives in the shared driver, so these also
        /// serve as the driver's own regression tests — via a target whose
        /// `alias` and `grad` are implemented completely differently from
        /// burn's.
        fn assert_backward_plumbing_agrees(ops: Vec<DiffOp>, what: &str) {
            let config = config();
            let prog = AutogradProgram {
                rows: 3,
                cols: 3,
                leaf_seeds: vec![
                    WELL_BEHAVED.to_vec(),
                    WELL_BEHAVED.iter().rev().copied().collect(),
                ],
                ops,
            };
            if let Err(msg) = run_autograd_program(&prog, &config) {
                panic!(
                    "gradient plumbing mismatch for {what}:\n{}\n{msg}",
                    prog.ssa(config.max_leaves)
                );
            }
        }

        /// burn and candle both seed the root gradient with ones of the root's
        /// shape, so unlike `tch_raw` there is no reduction to get wrong here —
        /// which is exactly why it is worth an assertion: "nothing to do" is a
        /// claim, and a root whose shape differs from the leaf's is where a
        /// wrong one would show.
        ///
        /// `Repeat` is safe to include despite bug #5 — burn's `repeat_dim`
        /// backward regroups the incoming gradient as though the forward had
        /// interleaved, but the incoming gradient *is* the ones-seed here, and
        /// every regrouping of ones sums to the same thing.  The bug needs a
        /// non-uniform gradient to become visible, which is the fuzzer's job,
        /// not this test's.
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
        ///
        /// The aliasing case is the one that matters most: candle identifies a
        /// gradient by `TensorId`, so an `alias` that *copied* rather than
        /// shared the `Arc` would silently lose the original leaf's
        /// contribution.
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
                    // `None`, candle's absent GradStore entry.
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
                    // Four leaves exist now (r0 + 3), so this one aliases.
                    leaf(2, 3, 3),
                    DiffOp::Instr(TensorInstr::Add(Reg(4), Reg(1))),
                    DiffOp::Instr(TensorInstr::Mul(Reg(5), Reg(2))),
                ],
                "leaf cap reached, aliasing",
            );
        }
    }
}
