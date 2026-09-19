# Bugs

Bugs found by this fuzzer, with root causes, PRs, and filing status.

## Status

Targeting Burn `0.22.0-pre.3`. The original 0.20.1 `swap_dims` bug is confirmed fixed.

| # | Bug | Repo | PR | Status |
|---|---|---|---|---|
| 1 | aarch64 SIMD `recip()` skips Newton–Raphson refinement → ~0.2% precision loss on ≥32 elements; corrupts `log()` gradient and `sigmoid()` forward | macerator | [#46](https://github.com/wingertge/macerator/pull/46) | pending |
| 2 | `sign(NaN)` reads the raw sign bit → `abs(log(x))`'s gradient returns `±1/x` for `x < 0` instead of `0` | burn-ndarray, burn-flex | [#5665](https://github.com/tracel-ai/burn/pull/5665) | **merged** |
| 3 | `powf_scalar(0.0)` detaches from the autodiff graph → zero gradient for all inputs | burn-backend (autodiff) | [#5692](https://github.com/tracel-ai/burn/pull/5692) | **merged** |
| 4 | `x.log().relu().relu()` with `x < 0`: gradient wrong in exactly the last `n mod 4` elements | burn-ndarray (suspected) | — | open, not root-caused |
| 5 | `repeat_dim` backward groups gradients as if the forward had interleaved, but the forward tiles → wrong gradient whenever the repeated dim has size > 1 | burn autodiff (all backends) | — | open |

Repros for bugs 1–3 are in `examples/`; writeups in `docs/`.

---

## Open: bug #4 — relu-chain NaN tail

`x.log().relu().relu()` where `x < 0`, so `log(x)` is NaN throughout. NdArray's gradient is correct everywhere *except the last `n mod 4` elements*, where it gives `-0.0` instead of the correct passthrough. LibTorch is correct everywhere. No panic — silently wrong, size-dependent.

Each condition confirmed necessary by direct experiment:
- Needs NaN input (a plain negative input diverges nowhere).
- Needs **two** chained `relu`s — one alone is clean at every size tested, including sizes that fail with two.
- Needs `n mod 4 != 0` — swept n = 16..257, zero divergence at every multiple of 4, exactly `n mod 4` trailing divergent elements otherwise. 4 is the `f32` NEON lane width.

Traced as far as: `relu`'s backward masks through `float_lower_equal_elem`, which is macerator-SIMD-accelerated with its own scalar-tail fallback. That fallback runs identically regardless of how many `relu`s precede it, yet the bug needs two — so the fallback alone isn't the cause.

Leading hypothesis: `relu`'s `.memory_bound()` / `RetroForward` checkpointing causes the second `relu` to recompute the first's output, and that recomputation's SIMD `clamp_min` handles the NaN tail differently. Investigation stopped at burn's checkpointing internals.

**Cheapest next experiment:** build `burn-ndarray` without the `simd` feature (scalar path, same backend) and run the repro. If divergence vanishes, macerator's tail handling is confirmed. If it persists, macerator is cleared and the checkpointing hypothesis is the lead.

---

## Open: bug #5 — `repeat_dim` backward

Found by the `tch-raw` target within a minute of it being wired in. No burn-vs-burn pairing could find this — every burn backend shares burn-autodiff, so they all compute the same wrong gradient.

Minimal repro — `x` is `[2×2]`, `y = sum(log(x.repeat_dim(0, 2)))`, expected `dy/dx == 2/x`:

```
x = [0.225, 0.35, 0.475, 0.6]

dy/dx  burn    = [6.549708, 4.5238094, 6.549708, 4.5238094]
       libtorch = [8.888889, 5.714286, 4.2105265, 3.3333333]   ← 2/x, correct
```

Burn's numbers are exactly `[1/x₀+1/x₂, 1/x₁+1/x₃, ...]` — the grouping from `repeat_interleave`. The forward tiles, the backward un-tiles as though it had interleaved. The gradient is reshaped `[orig_dim, k, …]` and reduced over the wrong axis; it should be `[k, orig_dim, …]` reduced over axis 0.

Reproduces identically on `ndarray` and `libtorch` burn backends — places it in burn-autodiff. Correct only when the repeated dimension has size 1 (the two groupings coincide). Swept `[1×4] [2×2] [2×3] [3×2] [4×1] [2×5] [3×3]` × `dim ∈ {0,1}` × `k ∈ {2,3}` — every case with `size(dim) > 1` is wrong.

**Why burn's own tests miss it:** with `sum(repeat_dim(x, d, k))`, every upstream gradient is `1`, so any mis-grouping sums to the same `k`. It takes a non-uniform upstream gradient (the `log` here) to expose the permutation.

Not yet root-caused to a line in burn's source. No `examples/` repro written yet.
