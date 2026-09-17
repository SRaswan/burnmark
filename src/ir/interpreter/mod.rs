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

use burn::tensor::{activation, Device, Tensor};

use super::ops::{TensorInstr, POWF_EXPONENTS};
use super::program::Backend;
use shape::{Shape2, resolve_broadcast_compatible, resolve_matmul_compatible, resolve_concat_compatible};

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

// ─── differential oracle ─────────────────────────────────────────────────────

/// Relative tolerance for comparing finite values across backends.
///
/// Named rather than inlined because it is a bug-hiding knob: `macerator`'s own
/// `recip` test used a `2^-8` tolerance, loose enough to hide a real ~0.2%
/// precision bug for as long as that bug existed.
pub const TOLERANCE: f32 = 1e-4;

/// The device for one [`Backend`].
///
/// Burn 0.22 dropped the `Backend` type parameter from `Tensor`, so which
/// backend runs an op is a property of the device, not of the tensor's type.
/// That is what lets a single non-generic interpreter serve every backend.
fn device_for(backend: Backend) -> Device {
    match backend {
        Backend::NdArray => {
            #[allow(deprecated)]
            Device::ndarray()
        }
        #[cfg(feature = "oracle-flex")]
        Backend::Flex => Device::flex(),
        #[cfg(feature = "oracle-tch")]
        Backend::LibTorch => Device::libtorch(),
        #[cfg(feature = "oracle-cpu")]
        Backend::Cpu => Device::cpu(),
        #[allow(unreachable_patterns)]
        unavailable => panic!(
            "backend {} is not compiled into this build",
            unavailable.name()
        ),
    }
}

/// Whether two backends' values for the same element disagree.
///
/// Special values are branched on explicitly rather than left to the tolerance
/// check, because `NaN`/`inf` arithmetic silently defeats it: `(NaN - x).abs()`
/// is `NaN` and `NaN > t` is `false`, while an infinite operand makes `scale` —
/// and therefore the threshold itself — infinite, so `inf > inf` is `false` too.
/// A comparison written only as `abs_diff > TOLERANCE * scale` therefore reports
/// **agreement** for every pair involving a `NaN` or an infinity, in either
/// direction, which is what this harness used to do.
///
/// That mattered: three of the four backend bugs found so far are special-value
/// bugs, so the old comparison was blind to its own subject matter. The
/// burn-flex `sign(NaN)` divergence (`NaN` where LibTorch returns `-0.0`) is
/// precisely the shape it ran straight past.
fn values_diverge(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        // Both-NaN is agreement: NaN payloads carry no meaning here, and
        // backends are not expected to produce matching bit patterns.
        return a.is_nan() != b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        // Infinities must match exactly, sign included.
        return a != b;
    }
    let abs_diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0_f32);
    abs_diff > TOLERANCE * scale
}

/// Compare every non-reference backend against the reference (the first entry),
/// returning one line per diverging backend.
///
/// Reports *all* diverging backends rather than stopping at the first, because
/// one root cause can make two backends wrong in different ways: the `sign(NaN)`
/// bug makes NdArray return `1/x` and burn-flex return `NaN` for the same input,
/// and a first-mismatch-wins report would name only one of them.
fn divergences(results: &[(&'static str, &[f32])], label: &str) -> Vec<String> {
    let Some((&(ref_name, reference), others)) = results.split_first() else {
        return Vec::new();
    };
    let mut report = Vec::new();
    for &(name, values) in others {
        if values.len() != reference.len() {
            report.push(format!(
                "{label}: {name} produced {} elements, {ref_name} produced {}",
                values.len(),
                reference.len()
            ));
            continue;
        }
        let diverging: Vec<usize> = (0..reference.len())
            .filter(|&i| values_diverge(reference[i], values[i]))
            .collect();
        if let Some(&first) = diverging.first() {
            report.push(format!(
                "{label}: {name} diverges from {ref_name} at {}/{} elements \
                 (first at [{first}]: {name}={:?}, {ref_name}={:?})",
                diverging.len(),
                reference.len(),
                values[first],
                reference[first],
            ));
        }
    }
    report
}

/// Panic if any backend diverges from the reference — the fuzz target turns that
/// panic into a saved crash artifact.
fn assert_agreement(results: &[(&'static str, &[f32])], label: &str) {
    let report = divergences(results, label);
    if !report.is_empty() {
        panic!("differential divergence:\n{}", report.join("\n"));
    }
}

// ─── shared instruction evaluator ────────────────────────────────────────────

/// Evaluate one [`TensorInstr`] against the register file, using `shapes`
/// to ensure binary operands are shape-compatible.
///
/// Burn 0.22 dropped the `Backend` type parameter from `Tensor` — which
/// backend actually runs an op is now a property of the `Device` a tensor
/// was created on, not of the tensor's type.  So this function (and the
/// register file it operates on) no longer needs to be generic at all; the
/// same code path handles NdArray and LibTorch registers alike.
fn eval_tensor_instr(
    regs: &[Tensor<2>],
    shapes: &[Shape2],
    instr: &TensorInstr,
) -> Tensor<2> {
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
        TensorInstr::Div(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() / regs[bi].clone()
        }
        TensorInstr::Matmul(a, b) => {
            let ai = a.resolve(n);
            match resolve_matmul_compatible(shapes, ai, b) {
                Some(bi) => regs[ai].clone().matmul(regs[bi].clone()),
                None => regs[ai].clone(),
            }
        }
        TensorInstr::Neg(r)       => regs[r.resolve(n)].clone().neg(),
        TensorInstr::Abs(r)       => regs[r.resolve(n)].clone().abs(),
        TensorInstr::Exp(r)       => regs[r.resolve(n)].clone().exp(),
        TensorInstr::Log(r)       => regs[r.resolve(n)].clone().log(),
        TensorInstr::Sqrt(r)      => regs[r.resolve(n)].clone().sqrt(),
        TensorInstr::PowfScalar(r, e) => {
            let exp = POWF_EXPONENTS[*e as usize % POWF_EXPONENTS.len()];
            regs[r.resolve(n)].clone().powf_scalar(exp)
        }
        TensorInstr::Relu(r)      => activation::relu(regs[r.resolve(n)].clone()),
        TensorInstr::Sigmoid(r)   => activation::sigmoid(regs[r.resolve(n)].clone()),
        TensorInstr::Tanh(r)      => activation::tanh(regs[r.resolve(n)].clone()),
        TensorInstr::SumAll(r)    => regs[r.resolve(n)].clone().sum().unsqueeze::<2>(),
        TensorInstr::MeanAll(r)   => regs[r.resolve(n)].clone().mean().unsqueeze::<2>(),
        TensorInstr::SumDim(r, d)  => {
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
        TensorInstr::Clamp(r)     => regs[r.resolve(n)].clone().clamp(-1e6_f32, 1e6_f32),
    }
}

#[cfg(test)]
mod tests {
    use super::{divergences, values_diverge};

    /// Every case here was silently reported as *agreement* by the previous
    /// comparison, which only evaluated `abs_diff > TOLERANCE * scale`.
    #[test]
    fn special_value_divergence_is_detected() {
        // NaN vs -0.0 is the exact signature of the sign(NaN) bug.
        assert!(values_diverge(f32::NAN, -0.0));
        assert!(values_diverge(-0.0, f32::NAN), "and in the other direction");
        assert!(values_diverge(f32::NAN, 42.0));
        assert!(values_diverge(f32::INFINITY, f32::NEG_INFINITY));
        assert!(values_diverge(f32::INFINITY, 1.0));
        assert!(values_diverge(1.0, f32::NEG_INFINITY));
    }

    #[test]
    fn matching_special_values_agree() {
        // NaN payloads are not meaningful across backends, so both-NaN agrees.
        assert!(!values_diverge(f32::NAN, f32::NAN));
        assert!(!values_diverge(f32::INFINITY, f32::INFINITY));
        assert!(!values_diverge(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn finite_comparison_still_honours_tolerance() {
        assert!(!values_diverge(1.0, 1.0));
        assert!(!values_diverge(1.0, 1.000_01), "inside 1e-4 relative");
        assert!(values_diverge(1.0, 1.01), "outside 1e-4 relative");
        // Signed zero alone is not a divergence; only NaN-vs-(-0.0) is.
        assert!(!values_diverge(0.0, -0.0));
    }

    /// The real measured three-way result for `abs(log(x))`'s gradient on
    /// published 0.22.0-pre.3: NdArray and burn-flex are each wrong, in
    /// *different* ways. A first-mismatch-wins report would name only one.
    #[test]
    fn every_diverging_backend_is_reported() {
        let libtorch = [-0.0_f32, -4.0, 0.5, -0.0];
        let ndarray = [-2.0_f32, -4.0, 0.5, -0.333_333_34];
        let flex = [f32::NAN, -4.0, 0.5, f32::NAN];

        let report = divergences(
            &[("libtorch", &libtorch), ("ndarray", &ndarray), ("flex", &flex)],
            "grad r0",
        );
        assert_eq!(report.len(), 2, "both wrong backends must be named: {report:?}");
        assert!(report[0].contains("ndarray") && report[0].contains("2/4"));
        assert!(report[1].contains("flex") && report[1].contains("2/4"));
        assert!(report[1].contains("NaN"), "report should show the offending value");
    }

    /// The reference side is whichever backend is listed first, so the same
    /// three results reported against a different reference name a different
    /// set of culprits.
    #[test]
    fn reference_side_is_the_first_entry() {
        let ndarray = [-2.0_f32, -4.0];
        let flex = [f32::NAN, -4.0];
        let report = divergences(&[("flex", &flex), ("ndarray", &ndarray)], "g");
        assert_eq!(report.len(), 1);
        assert!(report[0].contains("ndarray diverges from flex"), "{report:?}");
    }

    #[test]
    fn agreement_and_single_backend_produce_no_report() {
        let a = [1.0_f32, 2.0];
        assert!(divergences(&[("ref", &a), ("other", &a)], "x").is_empty());
        // One backend selected: nothing to compare against.
        assert!(divergences(&[("ref", &a)], "x").is_empty());
        assert!(divergences(&[], "x").is_empty());
    }

    #[test]
    fn shape_mismatch_is_reported_not_panicked() {
        let short = [1.0_f32];
        let long = [1.0_f32, 2.0];
        let report = divergences(&[("ref", &long), ("other", &short)], "x");
        assert_eq!(report.len(), 1);
        assert!(report[0].contains("1 elements"));
    }
}
