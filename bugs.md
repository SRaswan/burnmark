# Bugs

Bugs found by this fuzzer, with root causes and where the fix landed.

*Updated 2026-09-26.*

## Status

The fuzzer runs against burn `main`; the root crate and `examples/` pin published
`0.22.0-pre.4`. **All five bugs below are fixed upstream — the queue is empty.**
Three were fixed by our PRs; two were filed and fixed by other contributors while
they sat here triaged but unfiled. The original 0.20.1 `swap_dims` bug is also fixed.

| # | Bug | Repo | Fix | Filed by |
|---|---|---|---|---|
| 1 | aarch64 SIMD `recip()` skips Newton–Raphson refinement → ~0.2% precision loss on ≥32 elements; corrupts `log()` gradient and `sigmoid()` forward | macerator | [#47](https://github.com/wingertge/macerator/pull/47) merged, unreleased | us |
| 2 | `sign(NaN)` reads the raw sign bit → `abs(log(x))`'s gradient returns `±1/x` for `x < 0` instead of `0` | burn-ndarray, burn-flex | [#5665](https://github.com/tracel-ai/burn/pull/5665) merged, in `pre.4` | us |
| 3 | `powf_scalar(0.0)` detaches from the autodiff graph → zero gradient for all inputs | burn-backend (autodiff) | [#5692](https://github.com/tracel-ai/burn/pull/5692) merged, in `pre.4` | us |
| 4 | `x.log().relu().relu()` with `x < 0`: gradient wrong in exactly the last `n mod 4` elements | burn-ndarray | [#5733](https://github.com/tracel-ai/burn/pull/5733) merged, in `pre.4` | **someone else** |
| 5 | `repeat_dim` backward groups gradients as if the forward had interleaved, but the forward tiles → wrong gradient whenever the repeated dim has size > 1 | burn-autodiff (all backends) | [#5837](https://github.com/tracel-ai/burn/pull/5837) merged, **`main` only** | **someone else** |

Repros in `examples/`, writeups in `docs/`.

---

## Bug #1 — merged into macerator, but burn was fixed separately

Our Newton–Raphson fix is merged on macerator `main` and not yet released (the
release PR is still open). It no longer matters to burn either way: burn stopped
routing `recip` through macerator's `VRecip` in
[#5553](https://github.com/tracel-ai/burn/pull/5553) (in `pre.4`), replacing the
estimate with exact SIMD division — `burn-flex` does the same. burn still pins
macerator `0.3.4`, which the merged fix cannot semver-patch anyway.

`examples/simd_recip_precision_bug.rs` therefore no longer reproduces on `pre.4`;
it now asserts the fixed behaviour instead.

## Bugs #4 and #5 — lost the race

Both had complete writeups here and were never filed. Both were then filed and
fixed by other people:

- **#4** — issue [#5716](https://github.com/tracel-ai/burn/issues/5716) →
  PR [#5733](https://github.com/tracel-ai/burn/pull/5733), merged the same day,
  5 days after our writeup was committed here. The fix routes f32/f64 scalar-bound
  clamps away from the SIMD min/max path so NaN survives consistently across the
  vector body and the scalar tail. Our writeup's checkpointing hypothesis was
  wrong; its *second* lead — `clamp_min`'s own NaN handling, not the
  `RetroForward` recomputation — was the right one.
- **#5** — issue [#5833](https://github.com/tracel-ai/burn/issues/5833) →
  PR [#5837](https://github.com/tracel-ai/burn/pull/5837), **13 hours** from
  issue to merged fix. The issue carried a repro and a root-cause line but no
  patch, and that was enough.

The takeaway is process, not analysis: both writeups were more complete than the
issues that beat them. **File the issue at triage, before root-causing** — a good
repro is sufficient to get a fix landed, and it puts a timestamp on the find.

Bug #5 remains live in the latest *published* burn: `pre.4` was cut before #5837
landed. That is why `fuzz/Cargo.toml` points at `main` rather than a release.

---

## Next leads

Nothing triaged is left, so the work is finding new bugs. Verified starting points:

- **Fresh campaign against `main`.** `fuzz/Cargo.toml` now tracks
  `tracel-ai/burn` `main` with no local patches, so a run surfaces genuinely new
  divergences rather than any of the five above.
- **[#5284](https://github.com/tracel-ai/burn/issues/5284) — backends disagree on
  `inf` vs `NaN` when matmul overflows** (open, unassigned, labelled
  `bug`+`design`). Reproduces on `main` with no libtorch needed: `ndarray` gives
  `[inf]`, `flex` gives `[NaN]` for `[[1e30, -1e30]] @ [[1e30], [1e30]]`. Exactly
  this fuzzer's class, and burn has not settled the policy — which also decides
  what `values_diverge` should do on overflow.
- **[#4596](https://github.com/tracel-ai/burn/issues/4596) — autodiff training
  regression in 0.21.0-pre, "needs a repro"** (open, unassigned). Producing a
  minimal repro from a loss-curve report is what this harness is for.
- **[#4688](https://github.com/tracel-ai/burn/issues/4688) — `repeat_dim` on an
  outer dim of size 1 leaves zeros.** Does *not* reproduce on `ndarray` or
  `flex`, so it is CubeCL-`cpu`-specific; needs `oracle-cpu`.
