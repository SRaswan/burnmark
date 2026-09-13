/// Burn autograd example — reproduces the 0.20.1 LibTorch swap_dims /
/// memory-sharing bug (fixed upstream as of 0.22).
/// cargo run --example transpose_bug --features oracle-tch

use burn::tensor::{Device, Tensor};

fn run(device: &Device, label: &str) {
    let x_0: Tensor<2> = Tensor::full([3, 3], 0.5_f32, device).require_grad();

    let t0 = x_0.clone();
    let t1 = t0.clone().transpose();
    let t2 = t1.clone() + t0.clone();
    let grads = t2.backward();

    let x_grad = x_0.grad(&grads).unwrap();
    // d/dx log(x) = 1/x, so we expect x_grad to be full of 2.0
    println!("[{label}] x_0.grad = {}", x_grad.into_data());
}

#[allow(deprecated)] // Device::ndarray() — see note in src/ir/interpreter/tensor_program.rs
fn main() {
    run(&Device::ndarray().autodiff(), "NdArray");
    run(&Device::libtorch().autodiff(), "LibTorch");
}
