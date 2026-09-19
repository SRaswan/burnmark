//! Raw tch-rs interpreter — libtorch called directly, no burn in the path.
//!
//! [`Backend::LibTorch`] also reaches libtorch, but through burn-tch. Comparing
//! `libtorch` vs `tch-raw` puts burn's FFI bridge under test on its own — the
//! class the original 0.20.1 `swap_dims` bug belonged to.
//!
//! On the backward pass: `Device::libtorch().autodiff()` uses burn-autodiff, so
//! every burn backend shares one set of derivative formulas. This target brings
//! libtorch's own autograd, making it the only pairing where the backward pass is
//! implemented twice independently.
//!
//! Fidelity notes (places that needed care):
//! - `SumDim`/`MeanDim` use `keepdim = true`, matching burn and the shape predictor.
//! - `Repeat` is tile semantics (`[a,b] ×2 → [a,b,a,b]`), not `repeat_interleave`.
//! - `backward()` sums the root before calling it — see [`TchRaw::backward`].

use tch::{Kind, Tensor};

use super::bytes_to_floats;
use super::driver::Framework;
use super::shape::{
    Shape2, resolve_broadcast_compatible, resolve_concat_compatible,
    resolve_matmul_compatible,
};
use crate::ir::ops::{POWF_EXPONENTS, TensorInstr};

/// Every tensor here is `f32`, matching burn's `Tensor<2>`.
const KIND: Kind = Kind::Float;

// ─── data in / out ───────────────────────────────────────────────────────────

/// Build one 2-D input tensor from seed bytes via `bytes_to_floats`.
fn make_tensor(raw: &[u8], rows: usize, cols: usize, requires_grad: bool) -> Tensor {
    let t = Tensor::from_slice(bytes_to_floats(raw, rows * cols).as_slice())
        .reshape([rows as i64, cols as i64]);
    // Set after the reshape so the *reshaped* tensor is the autograd leaf,
    // matching burn's `.reshape(..).require_grad()` ordering.
    if requires_grad { t.set_requires_grad(true) } else { t }
}

/// Flatten to `Vec<f32>`. `contiguous()` is required — `Transpose` leaves a
/// strided view and blitting it reorders elements.
fn to_vec(t: &Tensor) -> Vec<f32> {
    let flat = t.contiguous().reshape([-1_i64]);
    Vec::<f32>::try_from(&flat).unwrap_or_else(|e| panic!("tch-raw into_data failed: {e}"))
}

// ─── instruction evaluator ───────────────────────────────────────────────────

/// Evaluate one [`TensorInstr`]. Uses the shared operand resolvers so this
/// target picks the same registers as every other target.
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

// ─── the `Framework` impl ────────────────────────────────────────────────────

/// Raw tch-rs as a target. Stateless — libtorch's CPU device needs no handle.
pub(super) struct TchRaw;

impl Framework for TchRaw {
    type Tensor = Tensor;
    /// libtorch has no gradient store: `backward()` writes gradients onto the
    /// leaf tensors themselves, and [`Framework::grad`] reads them back off.
    type Grads = ();

    fn input(&self, raw: &[u8], rows: usize, cols: usize) -> Tensor {
        make_tensor(raw, rows, cols, false)
    }

    fn leaf(&self, raw: &[u8], rows: usize, cols: usize) -> Tensor {
        make_tensor(raw, rows, cols, true)
    }

    fn alias(&self, tensor: &Tensor) -> Tensor {
        // `shallow_clone` shares the autograd node (matching burn's `Tensor::clone`).
        // `copy()` would silently drop the aliased leaf's gradient.
        tensor.shallow_clone()
    }

    fn eval(&self, regs: &[Tensor], shapes: &[Shape2], instr: &TensorInstr) -> Tensor {
        eval_tensor_instr_tch(regs, shapes, instr)
    }

    /// libtorch refuses a non-scalar root, so we sum first. That is exactly the
    /// ones-seed burn uses (`d(Σy)/dx == Σᵢ ∂yᵢ/∂x · 1`). Any other reduction
    /// silently rescales every gradient. Don't guard a root with no grad — that
    /// panic is the signal (it was bug #3).
    fn backward(&self, root: &Tensor) {
        root.sum(KIND).backward()
    }

    fn grad(&self, _grads: &(), leaf: &Tensor) -> Option<Tensor> {
        let grad = leaf.grad();
        // An undefined grad is libtorch's `None`: the leaf never reached the
        // root.  The driver reports that as zeros, which is what burn does.
        if grad.defined() { Some(grad) } else { None }
    }

    fn to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        to_vec(tensor)
    }
}

// ─── fidelity tests ──────────────────────────────────────────────────────────

/// Does this interpreter mean the same thing by each instruction as burn's own
/// libtorch backend?
///
/// Forward pass: both sides end in the same C++ library, so any divergence is a
/// mistranslation (wrong `keepdim`, tile-vs-interleave, missing `contiguous()`).
/// Asserted exhaustively.
///
/// Backward pass: *not* asserted per-op — disagreement is the signal. The two
/// implementations are genuinely independent (burn-autodiff vs libtorch autograd).
/// What is asserted is the plumbing: leaves, aliasing, unreachable leaves, the
/// backward seed.
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
