//! Root AST nodes and runtime configuration for fuzz programs.

use std::fmt;
use arbitrary::Arbitrary;
use super::ops::{DiffOp, TensorInstr};

// ─── harness mode ─────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarnessMode {
    PanicOnFirstError,
    Continuous,
}

// ─── fuzz config ──────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Upper bound on the number of distinct leaf tensors that may be introduced
    pub max_leaves: usize,
    /// Minimum number of ops a program must have; smaller inputs are skipped.
    pub min_ops: usize,
    pub mode: HarnessMode,

    pub min_dim: usize,
    pub max_dim: usize,

    pub safe_math: bool,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            max_leaves: 4,
            min_ops: 0,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
            safe_math: true,
        }
    }
}

impl FuzzConfig {
    pub fn from_env() -> Self {
        let max_leaves = std::env::var("MAX_LEAVES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 8);

        let min_ops = std::env::var("MIN_OPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let mode = match std::env::var("MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "continuous" => HarnessMode::Continuous,
            _ => HarnessMode::PanicOnFirstError,
        };

        let min_dim = std::env::var("FUZZ_MIN_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 4096);

        let max_dim = std::env::var("FUZZ_MAX_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(min_dim, 4096);
        
        let safe_math = std::env::var("FUZZ_SAFE_MATH")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        FuzzConfig {
            max_leaves,
            min_ops,
            mode,
            min_dim,
            max_dim,
            safe_math,
        }
    }
}
// ─── plain tensor program (SSA) ──────────────────────────────────────────────

/// SSA tensor program. `r0` is seeded from `values`; every [`TensorInstr`]
/// appends a new register to the file.
#[derive(Arbitrary, Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorInstr>,
}

impl TensorProgram {
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;

        let rows = (self.rows as usize).clamp(config.min_dim, config.max_dim);
        let cols = (self.cols as usize).clamp(config.min_dim, config.max_dim);

        let mut s = String::new();
        let _ = writeln!(s, "=== TensorProgram [{}×{}] ===", rows, cols);
        let _ = writeln!(
            s,
            "r0 = input({}×{}, {} seed bytes)",
            rows,
            cols,
            self.values.len()
        );

        let mut num_regs: usize = 1;
        for instr in &self.ops {
            let out = format!("r{}", num_regs);
            let _ = writeln!(s, "{}", instr.ssa_line(&out, num_regs));
            num_regs += 1;
        }
        let _ = write!(s, "result = r{}.into_data()", num_regs - 1);
        s
    }
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cfg = FuzzConfig::default();
        write!(f, "{}", self.ssa(&cfg))
    }
}

// ─── autograd program (SSA) ──────────────────────────────────────────────────

/// SSA autograd program.
///
/// `r0` is always the seed leaf (`requires_grad`). [`DiffOp::Leaf`]
/// instructions introduce additional leaves (up to `max_leaves`). All other
/// [`DiffOp`] variants reference registers by [`Reg`] and push new values.
///
/// The register file is a flat `Vec<Tensor>` — leaves and intermediates share
/// the same index space, so the fuzzer can freely compose any DAG.
#[derive(Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    /// Pool of seed byte-vectors for leaf tensors.
    pub leaf_seeds: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
}

impl AutogradProgram {
    /// Pretty-print the program in SSA form, simulating register resolution
    /// and annotating every line with the output shape.
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;
        use super::shape::Shape2;
        use super::interpreter::shape::after_diff_op;

        let rows = (self.rows as usize).clamp(config.min_dim, config.max_dim);
        let cols = (self.cols as usize).clamp(config.min_dim, config.max_dim);

        let mut s = String::new();

        let _ = writeln!(
            s,
            "=== AutogradProgram [{}×{}] (max_leaves={}) ===",
            rows, cols, config.max_leaves
        );

        // r0 = seed leaf (always present)
        let seed0_len = self.leaf_seeds.first().map(|v| v.len()).unwrap_or(0);
        let r0_shape = Shape2(rows, cols);
        let _ = writeln!(
            s,
            "r0 {r0_shape} = leaf({}×{}, {} seed bytes)  [requires_grad, seed]",
            rows, cols, seed0_len
        );

        let mut num_regs: usize = 1;
        let mut shapes: Vec<Shape2> = vec![r0_shape];
        let mut leaf_count: usize = 1;
        let mut leaf_reg_indices: Vec<usize> = vec![0];

        for op in &self.ops {
            let out = format!("r{}", num_regs);
            let out_shape = match op {
                DiffOp::Leaf { seed, rows: lr, cols: lc } => {
                    if leaf_count < config.max_leaves {
                        let pool_idx = if self.leaf_seeds.is_empty() {
                            0
                        } else {
                            *seed as usize % self.leaf_seeds.len()
                        };
                        let seed_len =
                            self.leaf_seeds.get(pool_idx).map(|v| v.len()).unwrap_or(0);

                        let leaf_rows = (*lr as usize).clamp(config.min_dim, config.max_dim);
                        let leaf_cols = (*lc as usize).clamp(config.min_dim, config.max_dim);
                        let sh = Shape2(leaf_rows, leaf_cols);
                        let _ = writeln!(
                            s,
                            "{out} {sh} = leaf({leaf_rows}×{leaf_cols}, {seed_len} seed bytes)  \
                             [requires_grad, leaf #{leaf_count}]",
                        );
                        leaf_reg_indices.push(num_regs);
                        leaf_count += 1;
                        sh
                    } else {
                        let src = (*seed as usize) % num_regs;
                        let sh = shapes[src];
                        let _ = writeln!(s, "{out} {sh} = r{src}  # leaf cap reached, alias");
                        sh
                    }
                }
                _ => {
                    let sh = after_diff_op(&shapes, op).expect("non-Leaf op shape");
                    // print SSA line without the dummy "_ = " prefix
                    let rhs = op
                        .ssa_line("_", num_regs)
                        .trim_start_matches("_ = ")
                        .to_string();
                    let _ = writeln!(s, "{out} {sh} = {rhs}");
                    sh
                }
            };
            shapes.push(out_shape);
            num_regs += 1;
        }

        let last = num_regs - 1;
        let _ = writeln!(s, "grads = backward(r{last})");
        for &ri in &leaf_reg_indices {
            let sh = shapes[ri];
            let _ = writeln!(
                s,
                "grad r{ri} {sh} = r{ri}.grad(grads)  # None → zeros if unreachable"
            );
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cfg = FuzzConfig::default();
        write!(f, "{}", self.ssa(&cfg))
    }
}