/// Burn 0.22.0-pre.3 numerical-correctness bug, found by fuzz_autograd within
/// minutes of pointing burnmark at 0.22 (crash artifact:
/// fuzz/artifacts/fuzz_autograd/crash-8c836557c70d855befc1fc350c33a1a55062fff0).
///
/// On aarch64 (Apple Silicon and other ARM64 hosts), burn-ndarray's
/// SIMD-accelerated `recip()` silently returns results accurate to only
/// ~9 bits instead of full f32 precision, once a tensor has >= 32 elements
/// (burn_ndarray::ops::simd::base::should_use_simd threshold). Root cause:
/// macerator's aarch64 backend maps the `recip` SIMD op straight onto the
/// NEON `vrecpeq_f32` *reciprocal-estimate* instruction with no
/// Newton-Raphson refinement step —
/// https://github.com/wingertge/macerator/blob/main/src/backend/aarch64.rs
/// (`impl_unop!(recip, vrecpeq, f32, f64);`). ARM's own docs guarantee only
/// ~2^-8..2^-9 relative accuracy for that instruction; it's meant to be
/// paired with 1-2 `vrecpsq_f32` refinement steps, which never happen here.
///
/// This isn't just `Tensor::recip()`: `log()`'s gradient is `grad * recip(x)`
/// (burn-autodiff), and `sigmoid()`'s forward pass is `1/(1+exp(-x))`, so
/// both silently inherit the same ~0.2% error on any tensor with >= 32
/// elements — no panic, no NaN, just quietly wrong numbers on every
/// Apple-Silicon Mac.
///
/// cargo run --example simd_recip_precision_bug --features oracle-tch --release

use burn::tensor::{Device, Tensor};

#[allow(deprecated)] // Device::ndarray() — see note in src/ir/interpreter/tensor_program.rs
fn main() {
    let nd = Device::ndarray();
    let lt = Device::libtorch();

    // 33 = 32 (one SIMD-eligible chunk) + 1 scalar remainder, so the split
    // is visible in a single tensor: elements 0..32 go through the
    // imprecise SIMD path, element 32 through the exact scalar fallback.
    let n = 33;
    let data = vec![-1.0_f32; n];

    let nd_recip: Tensor<1> = Tensor::from_floats(data.as_slice(), &nd).recip();
    let lt_recip: Tensor<1> = Tensor::from_floats(data.as_slice(), &lt).recip();

    let nd_vals = nd_recip.into_data().try_to_vec::<f32>().unwrap();
    let lt_vals = lt_recip.into_data().try_to_vec::<f32>().unwrap();

    println!("recip(-1.0) over {n} elements:");
    println!("  NdArray  [0..4]  = {:?} ... [{}] = {}", &nd_vals[0..4], n - 1, nd_vals[n - 1]);
    println!("  LibTorch [0..4]  = {:?} ... [{}] = {}", &lt_vals[0..4], n - 1, lt_vals[n - 1]);
    println!(
        "  expected exact value: -1.0 everywhere; NdArray's SIMD lanes give {} instead (rel. error {:.4}%)",
        nd_vals[0],
        (nd_vals[0] - lt_vals[0]).abs() * 100.0
    );
    assert_eq!(lt_vals, vec![-1.0_f32; n], "LibTorch is exact, as expected");
    assert_ne!(nd_vals, lt_vals, "reproduces the NdArray SIMD recip precision bug");
}
