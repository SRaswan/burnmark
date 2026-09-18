//! Plain tensor-program SSA interpreter.

use burn::tensor::{Device, Tensor};

use super::shape::Shape2;
use super::shape::after_tensor_instr;
use super::{assert_agreement, bytes_to_floats, catch_as_result, device_for, eval_tensor_instr};
use crate::ir::program::{FuzzConfig, Target, TensorProgram};

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

/// Run `prog` on one target, dispatching to that target's interpreter.
///
/// The burn arm covers every burn *device* at once — 0.22 made the backend a
/// property of the device, so one interpreter serves them all.  A non-burn
/// target gets its own arm, because it has its own tensor type and therefore
/// its own interpreter over the same IR.
fn eval_on_target(prog: &TensorProgram, target: Target) -> Vec<f32> {
    match target {
        Target::Burn(backend) => eval_tensor_program(prog, &device_for(backend)),
        #[cfg(feature = "oracle-tch-raw")]
        Target::TchRaw => super::tch_raw::eval_tensor_program(prog),
        #[cfg(feature = "oracle-candle")]
        Target::Candle => super::candle::eval_tensor_program(prog),
        #[allow(unreachable_patterns)]
        unavailable => panic!(
            "target {} is not compiled into this build",
            unavailable.name()
        ),
    }
}

/// Run a plain SSA TensorProgram on every target in `config.targets` and
/// require them to agree.  The first entry is the reference side; a single entry
/// simply executes the program with no comparison.
pub fn run_tensor_program(prog: &TensorProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let outputs: Vec<(&'static str, Vec<f32>)> = config
            .targets
            .iter()
            .map(|&target| (target.name(), eval_on_target(prog, target)))
            .collect();

        let view: Vec<(&'static str, &[f32])> = outputs
            .iter()
            .map(|(name, out)| (*name, out.as_slice()))
            .collect();
        assert_agreement(&view, "tensor_program");
    }))
}
