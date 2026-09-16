# `x.powf_scalar(0)` silently detaches from the autodiff graph

Found by `burnmark`'s `fuzz_autograd` target after adding `powf_scalar` (and
`Div`) to its op set, immediately after the SIMD `recip()` precision bug and
the `sign(NaN)` bug were both patched in locally. Not filed anywhere yet —
this is the write-up to file from, not a filed report.

## TL;DR

`x.powf_scalar(0.0)` (equivalently `x.powi_scalar(0)`, and reached from
`powf_scalar` for *any* float value that rounds to integer `0`) computes the
mathematically-correct forward value (`1` everywhere) but returns a tensor
that is completely disconnected from the autodiff graph — not "zero
gradient", but structurally indistinguishable from a tensor that never
called `.require_grad()` at all. This is a bug in `burn-backend`'s shared
default implementation, not any one backend — every backend inherits it
identically, since `burn-autodiff` doesn't override the function where the
bug lives (confirmed directly against both NdArray and LibTorch).

How loudly this shows up depends on which commit you're on:

- **Published crates.io `0.22.0-pre.3`** (confirmed directly against the
  registry source, git sha `13f0a12b71ad83c1f9edeac22dea325dcb612397`):
  `Tensor::backward`/`Tensor::grad` have no tracked-tensor precondition
  check at all. `.backward()` silently succeeds and `x.grad(&grads)` quietly
  comes back `None` — indistinguishable from "you forgot `.require_grad()`"
  even though you didn't. Silently wrong, not a crash.
- **Current `origin/main`** (where a fix would actually land): main has
  since added a helpful precondition to `Tensor::backward()` that panics
  instead of quietly returning nothing useful — a legitimate improvement in
  general, but this pre-existing, unrelated bug now trips a *false
  positive* of that check:

  ```
  Tensor::backward requires a tracked autodiff tensor; call
  Tensor::autodiff().require_grad() on the source leaf before computing the
  output
  ```

  ...blaming a `.require_grad()` call that was actually made correctly.
  This is the form burnmark's fuzzer actually caught, and the form the
  regression test added with the fix targets.

Either way, same root cause, same fix.

## Reproduction

Minimal, no fuzzer needed — `examples/powi_scalar_zero_grad_bug.rs` in this
repo, runnable as-is against the currently-published crate:

```rust
let x = Tensor::<1>::from_floats([2.0, -3.0, 0.0], &device).require_grad();

let y = x.clone().powf_scalar(0.0);
println!("{:?}", y.to_data()); // [1.0, 1.0, 1.0] -- correct

let grads = y.sum().backward();
println!("{:?}", x.grad(&grads)); // None, on every backend -- x was never reached
```

Run it: `cargo run --example powi_scalar_zero_grad_bug --features oracle-tch
--release` (needs `LIBTORCH`/`DYLD_LIBRARY_PATH` set for the LibTorch oracle
side). Against published crates.io `0.22.0-pre.3` (what the example actually
builds against — no local patches applied there, deliberately, so it
reflects what anyone hitting this today sees) this prints `None` for both
NdArray and LibTorch; against current `origin/main` the same scenario panics
instead (see TL;DR above for why).

### Via the fuzzer

Every one of ~640 divergent runs in a 25-minute `MODE=continuous` session
(`cargo +nightly fuzz run fuzz_autograd --features oracle-tch -- \
-max_total_time=1500`, right after `PowfScalar` was added to the generator)
reduced to the exact same error message, e.g.:

```
=== AutogradProgram [1×16] (max_leaves=8) ===
r0 [1×16] = leaf(1×16, 1 seed bytes)  [requires_grad, seed]
r1 [1×16] = r0.powf(0)
r2 [1×16] = r1.powf(0)
r3 [1×16] = r1.powf(0)
r4 [1×16] = -r1
r5 [1×16] = sum(r2, dim=0)
grads = backward(r5)
grad r0 [1×16] = r0.grad(grads)  # None → zeros if unreachable

error: Tensor::backward requires a tracked autodiff tensor; call Tensor::autodiff().require_grad() on the source leaf before computing the output
```

Every single reduction across all ~640 occurrences had a `powf(0)` upstream
of the final `.backward()` call — no other program shape produced this
error, which is what pointed straight at the exponent-0 case specifically
(confirmed directly with a standalone check: `powf(0.0)` panics, `powf(2.0)`
on the same leaf does not).

This one root cause was, in turn, drowning out anything else the fuzzer
might find with `powf_scalar` in the mix — same "one bug dominates the
crash population" pattern as the `recip()` precision bug before it.

## Root cause

`Tensor::powf_scalar` dispatches through `Backend::float_powf_scalar`
(`burn-backend-0.22.0-pre.3/crates/burn-backend/src/backend/ops/tensor.rs:1110`),
whose default implementation special-cases integer-valued exponents:

```rust
fn float_powf_scalar(tensor: FloatTensor<B>, value: Scalar) -> FloatTensor<B> {
    if let Some(exp) = value.try_as_integer() {
        Self::float_powi_scalar(tensor, exp)
    } else {
        Self::float_powf_scalar_impl(tensor, value)
    }
}
```

`0.0` is representable as the integer `0`, so it goes to
`float_powi_scalar` (same file, line 1060), whose default implementation
special-cases several small integer exponents as cheaper ops than generic
exponentiation:

```rust
fn float_powi_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B> {
    match rhs.elem::<i64>() {
        0 => Self::float_ones(lhs.shape(), &lhs.device(), lhs.dtype().into()),
        1 => lhs,
        2 => B::float_mul(lhs.clone(), lhs),
        -1 => Self::float_recip(lhs),
        -2 => Self::float_recip(B::float_mul(lhs.clone(), lhs)),
        _ => Self::float_powi_scalar_impl(lhs, rhs),
    }
}
```

Every arm except `0` derives its result from `lhs` through a real backend
op — `1` returns `lhs` itself, `2`/`-1`/`-2` compose `float_mul`/
`float_recip`, both of which `burn-autodiff` overrides with a proper
`Backward` node wired to the input. The `0` arm is different in kind:
`float_ones` is a tensor *constructor*, with no relationship to `lhs`
whatsoever. Mathematically `x^0 == 1` for every `x`, so the forward value is
fine — but on `Autodiff<B>`, `AutodiffTensor::new(B::float_ones(...))`
produces a tensor with no parent node at all. There is no way for the
autodiff wrapper (which works by intercepting individual ops, each
independently deciding `Tracked`/`UnTracked` from whether *its own* input
requires grad) to attach `lhs`'s gradient requirement to a tensor that was
never actually computed from `lhs`.

`burn-autodiff` doesn't override `float_powi_scalar` (or `float_powf_scalar`)
at all — confirmed by grep, no match in
`burn-autodiff-0.22.0-pre.3/src/ops/tensor.rs` — so every backend, not just
NdArray, inherits this exact default and this exact bug. It isn't
backend-specific and it isn't limited to the ergonomic `powf_scalar` entry
point: anything that reaches `float_powi_scalar(_, 0)` hits it.

## Suggested fix

Compute the same constant through ops that *do* preserve the backward edge,
the same way the `2`/`-1`/`-2` arms already do:

```rust
0 => {
    let dtype = lhs.dtype();
    B::float_add_scalar(
        B::float_mul_scalar(lhs, Scalar::new(0.0_f32, &dtype)),
        Scalar::new(1.0_f32, &dtype),
    )
}
```

`0 * lhs + 1` is the same forward value as `float_ones`, computed through
`float_mul_scalar` and `float_add_scalar` — both fundamental enough that
`burn-autodiff` already overrides them with real `Backward` nodes, so the
result stays properly attached to `lhs`'s node and the gradient comes out
correctly as `0` (a tracked, differentiable constant) instead of the
tensor being untracked outright.

Validated: full `burn-backend-tests` suite against NdArray after the fix —
`--test tensor` 1670 passed, `--test autodiff` 626 passed (including a new
`should_diff_powf_scalar_zero_exponent` test asserting both the forward
value and the gradient), 0 failures. Confirmed the new test fails with the
exact panic above against the unmodified code, passes after.

## Where this should be filed

Squarely `tracel-ai/burn` — the bug is in `burn-backend`'s own shared
default implementation, not a dependency. Not filed anywhere yet; this doc
is the draft to file from, not a filed report.
