# `relu(relu(NaN))`'s gradient is wrong at the SIMD tail — mechanism not fully pinned down

Found by `burnmark`'s `fuzz_autograd` target immediately after the
`sign(NaN)`, macerator `recip()` precision, and `powf_scalar(0)` bugs were
all patched in locally — a 25-minute continuous-mode run still produced 647
crashes, all reducing to the same shape below. **Investigation stopped here
at the user's request** (mid deep-dive into checkpointing internals) — this
is the precise, reproducible signature and the strongest lead so far, not a
pinned-down single line the way the other three bugs' docs are. Treat this
as "ready to open an issue with a great repro", not "ready to open a PR with
a fix".

## TL;DR

`x.log().relu().relu()` (`log` producing `NaN` because `x < 0`, then two
chained `relu` calls) gives the **correct** gradient (`d/dx log(x) = 1/x`,
passed through unmasked since NaN fails every numeric comparison) on every
element **except the last `n mod 4` elements**, where NdArray gives `-0.0`
instead of the correct value. LibTorch is correct everywhere. No panic, no
crash — silently wrong numbers, and only at a tensor-size-dependent tail.

Precise conditions, each individually confirmed necessary:
- **Needs `NaN`.** The identical shape with a plain (non-NaN) negative input
  (`x = -1.0`, no `log()`) shows zero divergence at any size tested.
- **Needs *two* chained `relu` calls.** `x.log().relu()` (one relu) shows
  zero divergence at any size tested, including the exact sizes that fail
  with a second `relu` stacked on top.
- **Needs `n mod 4 != 0`.** Exhaustively confirmed over `n` from 16 to 257:
  0 divergent elements whenever `n` is a multiple of 4; exactly `n mod 4`
  divergent elements otherwise, and they are always the *last* `n mod 4`
  elements of the tensor. 4 matches the `f32` NEON lane width on aarch64
  (this was run on Apple Silicon) — strongly suggestive of a SIMD
  full-vector-vs-scalar-remainder split, the same general shape as the
  macerator `recip()` bug, but a different code path and a different
  failure mode (a hard wrong zero, not a precision loss).

## Reproduction

Minimal (pseudocode; see below for exact values used during investigation):

```rust
let device = Device::ndarray().autodiff();
let n = 121; // 121 mod 4 == 1
let x = Tensor::<1>::from_floats(vec![-1.0_f32; n].as_slice(), &device).require_grad();

let y = x.clone().log().relu().relu(); // NaN throughout
let grads = y.sum().backward();
let grad = x.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();

// grad[0..120] == -1.0 (correct: d/dx log(x) = 1/x, unmasked since NaN
//                        fails every comparison in relu_backward's mask)
// grad[120]    == -0.0 on NdArray (WRONG), -1.0 on LibTorch (correct)
```

Swept `n` over `{16, 31, 32, 33, 63, 64, 65, 120, 121, 127, 128, 129, 200,
255, 256, 257}`: 0 diffs at every multiple of 4, exactly `n mod 4` diffs
(always the trailing elements) at every other size.

## What's ruled out, and where the remaining suspicion sits

`relu`'s backward pass (`burn-backend`'s default `relu_backward`, inherited
by NdArray — confirmed no override) is:

```rust
fn relu_backward(output: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
    let mask = B::float_lower_equal_elem(output, 0f32.into(), bool_dtype);
    B::float_mask_fill(grad, mask, 0.into())
}
```

`float_lower_equal_elem` → NdArray's `lower_equal_elem` → a
macerator-SIMD-accelerated comparison (`dispatch_cmp_scalar_simd!` /
`try_cmp_scalar_simd`, `burn-ndarray/src/ops/simd/cmp.rs`) with an explicit
three-tier loop: 8-wide vector chunks, then single-vector chunks, then a
final **scalar** loop (`Op::apply(*input, rhs)`, plain `<=`) over whatever's
left after both vector phases — which is exactly where a NaN tail element
would be evaluated if this function is the culprit. On its face this scalar
fallback looks correct (`NaN <= 0.0` is `false` in plain Rust, same as
every other backend), and it's *identical* regardless of how many `relu`s
precede it — yet the bug only appears with **two**, not one. That's the
part investigation didn't resolve: whatever differs between the one-relu
and two-relu cases isn't in this comparison function's own logic, since it
runs the same way either time.

The leading hypothesis, not confirmed: `relu`'s autodiff wrapper
(`burn-autodiff/src/ops/activation.rs`) is a `.memory_bound()` /
`RetroForward`-checkpointed op — it doesn't unconditionally save its forward
output, it can be asked to *recompute* it later via `RetroRelu::<B>::new(...)`
calling `B::relu(...)` again. Stacking two such ops changes the
checkpointing graph shape (the outer `relu`'s backward needs the inner
`relu`'s *output* as `relu_backward`'s `output` argument, which may now come
from a fresh recomputation rather than the tensor already sitting in memory
from the forward pass) in a way a single `relu` never exercises. If that
recomputed tensor's SIMD `clamp_min` forward pass (`NdArrayMathOps::clamp_min`,
also macerator-backed) handles a NaN tail element differently than the
originally-computed one, `relu_backward`'s mask on the *recomputed* copy
could disagree with what actually flowed through the graph the first time —
but this wasn't traced through the checkpointer far enough to confirm it
before stopping. Forward-only spot checks during investigation didn't catch
`clamp_min` returning a wrong tail value on its own, but weren't exhaustive
across the same `n`-sweep the backward divergence was confirmed over.

## Next steps for whoever picks this up

- Trace `Checkpointer::retrieve_node_output` for the outer `relu`'s state in
  exactly this two-relu case to confirm whether it's a fresh recomputation
  or the originally-saved tensor, and diff that tensor's last `n mod 4`
  elements against what the forward pass actually produced.
- If it is a recomputation, check `NdArrayMathOps::clamp_min`'s own SIMD
  tail handling (`burn-ndarray/src/ops/base.rs` — same macerator-dispatch
  shape as the comparison op above) for a NaN-specific remainder bug
  independent of the one already found and ruled out in `recip()`.
- Confirm the SIMD-width correlation on x86 too (this was only run on
  aarch64 / 4-wide `f32` NEON lanes); if it's genuinely
  `lane-width`-shaped rather than aarch64-specific, that's further
  evidence for the SIMD-remainder theory over something checkpointing-only.

## Where this should be filed

`tracel-ai/burn` (`burn-ndarray` and/or `burn-autodiff`, pending which side
the trace above lands on) — worth opening as an issue with this repro and
investigation notes even without a pinned-down single line; the repro alone
(exact `n mod 4` tail divergence, NaN-and-two-relus-specific) is precise
enough for someone with checkpointing-internals context to likely spot it
quickly. Not filed anywhere yet.
