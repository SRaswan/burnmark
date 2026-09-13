# SIMD `recip()` silently loses precision on every accelerated backend

Found by `burnmark`'s `fuzz_autograd` target within minutes of pointing it at
Burn `0.22.0-pre.3`. Not filed anywhere yet — this is the write-up to file
from, not a filed report.

## TL;DR

`burn-ndarray`'s SIMD-accelerated `recip()` (reciprocal) returns results
accurate to only ~9-14 bits instead of full `f32` precision, on **any**
tensor with >= 32 elements, on **every** major SIMD backend it ships
(x86 SSE2/AVX, AVX-512, and ARM NEON). No panic, no NaN — it just silently
returns numbers that are wrong by up to ~0.2%. It corrupts `log()`'s gradient
and `sigmoid()`'s forward pass too, since both are built on `recip`.

Root cause is one line each in **[macerator](https://github.com/wingertge/macerator)**
(a burn-ndarray dependency, maintained by a Burn core dev), not in burn
itself. Fix belongs there; burn-ndarray/burn just need to pick up the fix
(or work around it) once it lands.

## Reproduction

Minimal, no autodiff required — `examples/simd_recip_precision_bug.rs` in
this repo:

```rust
let n = 33; // 32 (one full SIMD chunk) + 1 scalar remainder
let data = vec![-1.0_f32; n];
let nd_recip = Tensor::<1>::from_floats(data.as_slice(), &Device::ndarray()).recip();
let lt_recip = Tensor::<1>::from_floats(data.as_slice(), &Device::libtorch()).recip();
```

```
recip(-1.0) over 33 elements:
  NdArray  [0..4]  = [-0.9980469, -0.9980469, -0.9980469, -0.9980469] ... [32] = -1
  LibTorch [0..4]  = [-1.0, -1.0, -1.0, -1.0] ... [32] = -1
```

Elements `0..32` (one full SIMD chunk) come back as `-0.9980469` instead of
the exact `-1.0`; element `32` (the scalar remainder, below the SIMD
threshold) is exact. `-0.9980469 == -511/512 == -(1 - 2^-9)` — the textbook
accuracy bound of ARM's unrefined `vrecpe` estimate instruction.

Run it: `cargo run --example simd_recip_precision_bug --features oracle-tch --release`
(needs `LIBTORCH`/`DYLD_LIBRARY_PATH` set for the LibTorch oracle side).

### Via the fuzzer

The very first `fuzz_autograd` crash on 0.22.0-pre.3 was this same bug,
reached through `log()`'s gradient:

```
=== AutogradProgram [7×7] (max_leaves=4) ===
r0 [7×7] = leaf(7×7, 1 seed bytes)  [requires_grad, seed]
r1 [7×7] = log(r0)
grads = backward(r1)
grad r0 [7×7] = r0.grad(grads)
error: oracle detected 48 mismatches in grad r0
```

r0 is all `-1.0` (49 elements: 48 through the SIMD path, 1 scalar remainder
— same 48-imprecise/1-exact split as above). `d/dx log(x) = 1/x`, so the
gradient computation is `recip(x)`, and NdArray's SIMD path returns
`-0.9980469` per element instead of `-1.0`. Crash artifact saved at
`fuzz/artifacts/fuzz_autograd/crash-8c836557c70d855befc1fc350c33a1a55062fff0`.

A 5-minute continuous-mode run (`MODE=continuous cargo +nightly fuzz run
fuzz_autograd --features oracle-tch -- -max_total_time=300`) found 261
crashes, **all this same root cause** — it's common enough that it currently
drowns out any other distinct bug the fuzzer might otherwise find.

## Root cause

`burn-ndarray` gates its SIMD unary ops behind a size threshold
(`burn_ndarray::ops::simd::base::should_use_simd`, `burn-ndarray-0.22.0-pre.3/src/ops/simd/base.rs:17-19`):

```rust
pub fn should_use_simd(len: usize) -> bool {
    len >= 32
}
```

Below 32 elements everything goes through the exact scalar `.map()`
fallback (`burn-ndarray-0.22.0-pre.3/src/ops/base.rs:932`,
`NdArrayMathOps::recip`) — which is why a casual manual check on small
tensors won't show this at all. At or above 32 elements, full SIMD-width
chunks go through `RecipVec` (`burn-ndarray-0.22.0-pre.3/src/ops/simd/unary.rs:20-33`),
which just calls the SIMD vector type's `.recip()` — i.e. macerator's
`VRecip` impl for the target arch. None of macerator's vectorized
backends apply a refinement step after the hardware estimate:

- **aarch64** (`macerator-0.3.4/src/backend/aarch64.rs:173`):
  `impl_unop!(recip, vrecpeq, f32, f64);` — raw NEON `vrecpeq_f32`.
  ARM's own docs guarantee only ~2^-8..2^-9 relative accuracy for this
  instruction; it's meant to be paired with 1-2 `vrecpsq_f32`
  Newton-Raphson refinement steps to reach full `f32` precision. Verified
  empirically above (Apple Silicon).
- **x86 SSE2/AVX** (`macerator-0.3.4/src/backend/x86/v2.rs:87`,
  `v3.rs:99`): `impl_unop!(recip, _mm_rcp, f32)` /
  `_mm256_rcp` — Intel's `RCPPS`/`VRCPPS`, documented to ~12-bit
  (2^-12) relative accuracy, again with no refinement. **Not yet verified
  empirically** (no x86 machine on hand for this investigation) — flagged
  by source inspection only.
- **x86 AVX-512** (`macerator-0.3.4/src/backend/x86/v4.rs:366`):
  `_mm512_rcp14` — 14-bit precision, same story. Also not empirically
  verified here.
- The plain scalar and wasm32 backends (`scalar.rs:270`, `wasm32.rs:260`)
  use `impl_unop_scalar!(recip, recip, ...)`, i.e. real `f32::recip()` — no
  bug there, which matches every "small tensor" and "scalar remainder"
  case being exact in the reproduction above.

So: this is a general "fast-reciprocal-without-refinement" bug across
essentially every accelerated SIMD backend macerator ships, not an
aarch64/Apple-Silicon-specific quirk — aarch64 is just the one it was
actually run and confirmed on.

`log()`'s backward pass computes `grad * B::float_recip(input)`
(`burn-autodiff-0.22.0-pre.3/src/ops/tensor.rs:2648-2652`), and
`sigmoid()`'s forward pass is `1/(1+exp(-x))`
(`burn-tensor-0.22.0-pre.3/src/tensor/activation/base.rs`) — both go
through the same `float_recip`, so both silently inherit this on any
tensor with >= 32 elements. That's essentially every real layer in a real
model (any hidden dim, batch, or sequence length worth using a framework
for), which makes this a genuine training-correctness bug, not just an
edge case a fuzzer happened to poke at.

## Suggested fix (for macerator)

Add a Newton-Raphson refinement iteration after the hardware estimate on
every vectorized backend, e.g. for aarch64:

```rust
// y0 = vrecpeq_f32(x)              (~2^-8 relative error)
// y1 = y0 * vrecpsq_f32(x, y0)     (one NR step -> ~2^-16, full f32 precision)
let y0 = vrecpeq_f32(x);
let y1 = vmulq_f32(vrecpsq_f32(x, y0), y0);
```

and the SSE2/AVX/AVX-512 equivalent using `_mm_mul_ps`/`_mm_sub_ps`
around `_mm_rcp_ps` (the classic one-Newton-step reciprocal refinement —
well documented, e.g. in Intel's own optimization manuals). Alternatively,
burn-ndarray could stop routing `recip` through the approximate SIMD path
at all (use the exact `_mm_div_ps`/`vdivq_f32` per-lane instead) if the
perf win of the unrefined estimate isn't worth the accuracy loss.

## Where this should be filed

The bug is in `macerator`, not `burn` — that's where the actual fix has to
land. `tracel-ai/burn` is arguably still worth a linked issue since burn
users hit this without ever knowing macerator exists, and they'd want to
pin a fixed macerator version once one exists. Neither has been filed yet;
this doc is the draft to file from, not a filed report.
