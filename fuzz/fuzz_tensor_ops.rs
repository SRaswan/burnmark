//! Fuzz frontend tensor API (plain, no autograd)
//! env config: MODE (default panic), BACKENDS (default: all compiled in)

#![no_main]

use fuzzburn::ir::{interpreter, FuzzConfig, HarnessMode};
use fuzzburn::ir::program::TensorProgram;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|prog: TensorProgram| {
    let config = FuzzConfig::from_env();
    if prog.ops.len() < config.min_ops { return; }
    if let Err(msg) = interpreter::run_tensor_program(&prog, &config) {
        match config.mode {
            HarnessMode::PanicOnFirstError => {
                panic!("fuzz_tensor_ops CRASH:\n{prog}\nerror: {msg}");
            }
            HarnessMode::Continuous => {
                eprintln!("\n=== CRASH (fuzz_tensor_ops, continuing) ===\n{prog}\nerror: {msg}\n");
            }
        }
    }
});
