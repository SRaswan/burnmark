/// Burn 0.22.0-pre.3 autodiff bug, found by fuzz_autograd after adding
/// `powf_scalar` to its op set: every one of ~640 divergent runs in a
/// 25-minute continuous-mode session reduced to a program with a `powf(0)`
/// upstream of the final `.backward()` call.
///
/// `x.powf_scalar(0.0)` computes the mathematically-correct forward value
/// (`1` everywhere) but returns a tensor completely disconnected from the
/// autodiff graph, because `Backend::float_powi_scalar`'s default
/// implementation special-cases the exponent-0 case as
/// `Self::float_ones(lhs.shape(), ...)` — a tensor *constructor* with no
/// relation to `lhs` at all
/// (burn-backend-0.22.0-pre.3/src/backend/ops/tensor.rs:1062), unlike every
/// other arm (`1 => lhs`, `2 => B::float_mul(lhs.clone(), lhs)`, etc.), which
/// all derive their result from `lhs` through a real, autodiff-tracked
/// backend op. `burn-autodiff` doesn't override `float_powi_scalar` at all,
/// so this is inherited by every backend, not just NdArray.
///
/// The result is the same on every backend (the defect is in the shared
/// default, not any one backend's override) but *how loudly* it shows up
/// depends on which burn commit you're on:
///
/// - Against published crates.io `0.22.0-pre.3`
///   (`Tensor::backward`/`Tensor::grad` in that version have no
///   tracked-tensor precondition check at all — confirmed directly against
///   the registry source): `.backward()` silently succeeds and
///   `x.grad(&grads)` quietly comes back `None`, on *both* NdArray and
///   LibTorch — indistinguishable from "you forgot `.require_grad()`" even
///   though you didn't. This is what running this example as-is shows you
///   today.
/// - Against current `origin/main` (where a fix would actually land):
///   main has since added a helpful `Tensor::backward()` precondition that
///   panics with "requires a tracked autodiff tensor" instead of silently
///   returning nothing useful — a legitimate improvement in general, but it
///   means this pre-existing, unrelated bug now trips a *false positive* of
///   that check, blaming a `.require_grad()` call that was actually made
///   correctly. This is the form the fuzzer actually caught (640 divergent
///   `fuzz_autograd` runs, all this exact panic), and the form the
///   regression test in this fix targets.
///
/// Either way, it's the same root cause: `float_powi_scalar`'s exponent-0
/// arm builds its result via `float_ones`, a tensor constructor with no
/// relation to the input, instead of composing tracked ops the way every
/// other arm does.
///
/// cargo run --example powi_scalar_zero_grad_bug --features oracle-tch --release
use burn::tensor::{Device, Tensor};

#[allow(deprecated)] // Device::ndarray() — see note in src/ir/interpreter/tensor_program.rs
fn main() {
    let nd = Device::ndarray().autodiff();
    let lt = Device::libtorch().autodiff();

    println!("x = [2.0, -3.0, 0.0], require_grad; y = x.powf_scalar(0.0); y.sum().backward()\n");

    run_on("NdArray", &nd);
    run_on("LibTorch", &lt);
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| e.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<unknown panic payload>".to_string())
}

fn run_on(name: &str, device: &Device) {
    let x = Tensor::<1>::from_floats([2.0, -3.0, 0.0], device).require_grad();
    let y = x.clone().powf_scalar(0.0);
    let y_vals = y.clone().into_data().try_to_vec::<f32>().unwrap();
    println!("{name}: forward = {y_vals:?}  (mathematically correct: x^0 == 1 everywhere)");

    // Both the .backward() call and the .grad() lookup are wrapped, rather
    // than asserting one specific failure shape up front: the underlying
    // defect (float_powi_scalar's exponent-0 arm detaching from the graph
    // via a bare `float_ones` constructor) is backend-agnostic — it lives in
    // a shared burn-backend default no backend overrides — but *how* that
    // surfaces downstream (a hard panic in .backward() itself, vs a later
    // `.grad()` quietly coming back `None` because backward() never found a
    // path to this leaf) can depend on backend-specific checkpointing
    // details. Report whichever actually happens instead of assuming.
    let backward_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        y.sum().backward()
    }));
    match backward_result {
        Err(e) => {
            println!("{name}: y.sum().backward() panicked: {}", panic_msg(e));
        }
        Ok(grads) => match x.grad(&grads) {
            Some(g) => {
                let g_vals = g.into_data().try_to_vec::<f32>().unwrap();
                println!("{name}: grad = {g_vals:?}");
            }
            None => {
                println!(
                    "{name}: backward() ran, but x.grad() is None — x was never reached, \
                     meaning y was disconnected from x's node despite require_grad()"
                );
            }
        },
    }
    println!();
}
