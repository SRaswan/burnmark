/// Burn autograd example
/// cargo run --example simple_autograd --features oracle-tch

use burn::tensor::{Device, Tensor};

fn run(device: &Device, label: &str) {
    let x_0: Tensor<2> = Tensor::full([3, 3], 0.5_f32, device).require_grad();

    let t0 = x_0.clone();
    let t1 = t0.clone();
    let t2 = t1.log();
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
