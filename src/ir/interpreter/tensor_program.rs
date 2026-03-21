//! Plain tensor-program SSA interpreter.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use super::shape::Shape2;
use super::shape::after_tensor_instr;
use super::{PlainB, bytes_to_floats, catch_as_result, eval_tensor_instr};
use crate::ir::program::TensorProgram;

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

/// Run a plain SSA TensorProgram on any backend, returning the final
/// register's data as `Vec<f32>`.
fn eval_tensor_program<B: Backend>(prog: &TensorProgram, device: &B::Device) -> Vec<f32> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let r0: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(
            bytes_to_floats(&prog.values, rows * cols).as_slice(),
            device,
        )
        .reshape([rows, cols]);

    let mut regs: Vec<Tensor<B, 2>> = vec![r0];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];

    for instr in &prog.ops {
        let out_shape = after_tensor_instr(&shapes, instr);
        let val = eval_tensor_instr(&regs, &shapes, instr);
        regs.push(val);
        shapes.push(out_shape);
    }

    regs.pop()
        .expect("register file is empty")
        .into_data()
        .to_vec::<f32>()
        .expect("into_data failed")
}

/// Run a plain SSA TensorProgram against NdArray (+ LibTorch oracle when enabled).
pub fn run_tensor_program(prog: &TensorProgram) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = eval_tensor_program::<PlainB>(prog, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_tensor_program_oracle(prog, nd);
    }))
}

#[cfg(feature = "oracle-tch")]
fn run_tensor_program_oracle(prog: &TensorProgram, nd: Vec<f32>) {
    let lt = eval_tensor_program::<LibTorch>(prog, &LibTorchDevice::Cpu);
    compare_outputs(&nd, &lt, "tensor_program");
}
