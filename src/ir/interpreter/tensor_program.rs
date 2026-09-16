//! Plain tensor-program SSA interpreter.

use burn::tensor::{Device, Tensor};

use super::shape::Shape2;
use super::shape::after_tensor_instr;
use super::{bytes_to_floats, catch_as_result, eval_tensor_instr};
use crate::ir::program::TensorProgram;

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

/// Run a plain SSA TensorProgram on whichever backend `device` selects.
fn eval_tensor_program(prog: &TensorProgram, device: &Device) -> Vec<f32> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let r0: Tensor<2> =
        Tensor::<1>::from_floats(
            bytes_to_floats(&prog.values, rows * cols).as_slice(),
            device,
        )
        .reshape([rows, cols]);

    let mut regs: Vec<Tensor<2>> = vec![r0];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];

    for instr in &prog.ops {
        let out_shape = after_tensor_instr(&shapes, instr);
        // Chained Repeat/Concat/Matmul on this unconstrained op stream can
        // otherwise compound into a multi-gigabyte single allocation (see
        // `Shape2::exceeds_cap`'s doc comment) — stop here rather than let it
        // OOM the whole fuzz process; not a bug, just a generation-space gap.
        if out_shape.exceeds_cap() {
            break;
        }
        let val = eval_tensor_instr(&regs, &shapes, instr);
        regs.push(val);
        shapes.push(out_shape);
    }

    regs.pop()
        .expect("register file is empty")
        .into_data()
        .try_to_vec::<f32>()
        .expect("into_data failed")
}

/// Run a plain SSA TensorProgram against NdArray.
///
/// `Device::ndarray()` is deprecated in favor of `Device::flex()` as of 0.22,
/// but burn-flex is a distinct pure-Rust implementation with its own
/// semantics — swapping backends here would change what we're differentially
/// testing, not just how we spell it. Keep NdArray as the "left" oracle side
/// until burn-flex has had a comparable amount of scrutiny.
#[allow(deprecated)]
pub fn run_tensor_program(prog: &TensorProgram) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = eval_tensor_program(prog, &Device::ndarray());
        #[cfg(feature = "oracle-tch")]
        run_tensor_program_oracle(prog, nd);
    }))
}

#[cfg(feature = "oracle-tch")]
fn run_tensor_program_oracle(prog: &TensorProgram, nd: Vec<f32>) {
    let lt = eval_tensor_program(prog, &Device::libtorch());
    compare_outputs(&nd, &lt, "tensor_program");
}
