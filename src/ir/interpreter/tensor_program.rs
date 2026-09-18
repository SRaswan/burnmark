//! Plain tensor-program entry point: run one [`TensorProgram`] on every
//! selected target and require them to agree.
//!
//! The SSA walk itself lives in [`driver`](super::driver) and is shared by every
//! target; this file is only target dispatch plus the comparison.

use super::driver;
use super::{BurnTarget, assert_agreement, catch_as_result};
use crate::ir::program::{FuzzConfig, Target, TensorProgram};

/// Run `prog` on one target, dispatching to that target's [`Framework`] impl.
///
/// [`Framework`]: super::driver::Framework
///
/// The burn arm covers every burn *device* at once — 0.22 made the backend a
/// property of the device, so one implementation serves them all.  A non-burn
/// target gets its own arm, because it has its own tensor type and therefore
/// its own 22-arm `match`.
fn eval_on_target(prog: &TensorProgram, target: Target) -> Vec<f32> {
    match target {
        Target::Burn(backend) => driver::eval_program(&BurnTarget::forward(backend), prog),
        #[cfg(feature = "oracle-tch-raw")]
        Target::TchRaw => driver::eval_program(&super::tch_raw::TchRaw, prog),
        #[cfg(feature = "oracle-candle")]
        Target::Candle => driver::eval_program(&super::candle::Candle, prog),
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
