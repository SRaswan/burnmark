//! Autograd entry point: run one [`AutogradProgram`] on every selected target
//! and require their per-leaf gradients to agree.
//!
//! The SSA walk, the leaf bookkeeping and the backward call all live in
//! [`driver`](super::driver) and are shared by every target; this file is only
//! target dispatch plus the comparison.

use super::driver;
use super::{BurnTarget, assert_agreement, catch_as_result};
use crate::ir::program::{AutogradProgram, FuzzConfig, Target};

/// Collect per-leaf gradients from one target, dispatching to its [`Framework`]
/// impl.
///
/// [`Framework`]: super::driver::Framework
///
/// On the burn side, `.autodiff()` wraps whichever backend the device selects,
/// so every burn entry gets gradient tracking without the driver having to know
/// which backend it is running on — and, more to the point, every burn entry
/// gets the *same* derivative formulas, which is why a non-burn target is the
/// only thing that can disagree with burn about a derivative at all.
fn grads_on_target(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    target: Target,
) -> Vec<Vec<f32>> {
    match target {
        Target::Burn(backend) => {
            driver::collect_grads(&BurnTarget::autodiff(backend), prog, config)
        }
        #[cfg(feature = "oracle-tch-raw")]
        Target::TchRaw => driver::collect_grads(&super::tch_raw::TchRaw, prog, config),
        #[cfg(feature = "oracle-candle")]
        Target::Candle => driver::collect_grads(&super::candle::Candle, prog, config),
        #[allow(unreachable_patterns)]
        unavailable => panic!(
            "target {} is not compiled into this build",
            unavailable.name()
        ),
    }
}

/// Run `prog` on every target in `config.targets` and require their per-leaf
/// gradients to agree.  The first entry is the reference side; a single entry
/// simply executes the program with no comparison.
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let grads: Vec<(&'static str, Vec<Vec<f32>>)> = config
            .targets
            .iter()
            .map(|&target| (target.name(), grads_on_target(prog, config, target)))
            .collect();

        let Some(((ref_name, reference), others)) = grads.split_first() else {
            return;
        };
        for (name, leaves) in others {
            if leaves.len() != reference.len() {
                panic!(
                    "leaf count mismatch: {ref_name}={}, {name}={}",
                    reference.len(),
                    leaves.len()
                );
            }
        }

        for leaf in 0..reference.len() {
            let view: Vec<(&'static str, &[f32])> = grads
                .iter()
                .map(|(name, per_leaf)| (*name, per_leaf[leaf].as_slice()))
                .collect();
            assert_agreement(&view, &format!("grad r{leaf}"));
        }
    }))
}
