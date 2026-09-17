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

// ─── backend selection ────────────────────────────────────────────────────────

/// A backend the interpreter can run a program on.
///
/// Which backends a build *can* run is decided at compile time by cargo
/// features; which it *does* run is chosen at runtime by `BACKENDS`.  Keeping
/// those separate means one binary can be pointed at any pairing without a
/// rebuild — `libtorch,ndarray`, `libtorch,flex`, `flex,ndarray`, or `all`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Deprecated as of burn 0.22 (slated for removal in favour of `Flex`), but
    /// still where most known bugs live, so still worth testing.
    NdArray,
    /// burn-flex: the pure-Rust CPU backend replacing NdArray.  A distinct
    /// implementation, not a rename — the two disagree in practice.
    Flex,
    /// LibTorch via `tch`.  A separate project following PyTorch's documented
    /// special-value conventions, which makes it the best available reference.
    LibTorch,
}

impl Backend {
    /// Name used in `BACKENDS` and in divergence reports.
    pub const fn name(self) -> &'static str {
        match self {
            Backend::NdArray => "ndarray",
            Backend::Flex => "flex",
            Backend::LibTorch => "libtorch",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "ndarray" | "nd" => Some(Backend::NdArray),
            "flex" => Some(Backend::Flex),
            "libtorch" | "tch" | "torch" => Some(Backend::LibTorch),
            _ => None,
        }
    }

    /// Whether this build was compiled with support for this backend.
    pub const fn is_compiled_in(self) -> bool {
        match self {
            Backend::NdArray => true,
            Backend::Flex => cfg!(feature = "oracle-flex"),
            Backend::LibTorch => cfg!(feature = "oracle-tch"),
        }
    }

    /// Every backend this build can run, most-trustworthy first.
    pub fn available() -> Vec<Self> {
        [Backend::LibTorch, Backend::NdArray, Backend::Flex]
            .into_iter()
            .filter(|b| b.is_compiled_in())
            .collect()
    }
}

/// Backends to run, **reference side first** — every other backend is compared
/// against the first, and divergences are reported as `<backend> vs <reference>`.
///
/// Parsed from `BACKENDS`: a comma-separated list of names, or `all`.  Unset
/// keeps the historical pairing (LibTorch as reference against NdArray when
/// `oracle-tch` is compiled in, NdArray alone otherwise) so existing corpora and
/// artifacts stay comparable.  A single entry runs the program without any
/// comparison, which is useful for smoke and throughput runs.
///
/// An unparseable or unavailable selection panics rather than silently dropping
/// a requested side: quietly comparing fewer backends than asked for would
/// manufacture exactly the false confidence this fuzzer exists to prevent.
pub fn backends_from_env() -> Vec<Backend> {
    let requested = match std::env::var("BACKENDS") {
        Err(_) => return Backend::available(),
        Ok(raw) if raw.trim().is_empty() => return Backend::available(),
        Ok(raw) if raw.trim().eq_ignore_ascii_case("all") => return Backend::available(),
        Ok(raw) => raw,
    };

    let mut selected: Vec<Backend> = Vec::new();
    for token in requested.split(',') {
        let name = token.trim().to_lowercase();
        if name.is_empty() {
            continue;
        }
        let backend = Backend::from_name(&name).unwrap_or_else(|| {
            panic!(
                "invalid BACKENDS entry {name:?}: expected one of \
                 ndarray, flex, libtorch (or `all`)"
            )
        });
        if !backend.is_compiled_in() {
            let feature = match backend {
                Backend::Flex => "oracle-flex",
                Backend::LibTorch => "oracle-tch",
                Backend::NdArray => unreachable!("NdArray is always compiled in"),
            };
            panic!(
                "BACKENDS requested {name:?}, which this build does not support \
                 — rebuild with --features {feature}. Available: {:?}",
                Backend::available().iter().map(|b| b.name()).collect::<Vec<_>>()
            );
        }
        if !selected.contains(&backend) {
            selected.push(backend);
        }
    }

    if selected.is_empty() {
        panic!("BACKENDS named no usable backend");
    }
    selected
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

    /// Backends to run and cross-check, reference side first.
    pub backends: Vec<Backend>,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            max_leaves: 4,
            min_ops: 0,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
            backends: Backend::available(),
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

        FuzzConfig {
            max_leaves,
            min_ops,
            mode,
            min_dim,
            max_dim,
            backends: backends_from_env(),
        }
    }
}
// ─── plain tensor program (SSA) ──────────────────────────────────────────────

/// SSA tensor program.  `r0` is seeded from `values`; every [`TensorInstr`]
/// appends a new register to the file.
#[derive(Arbitrary, Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorInstr>,
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        writeln!(f, "=== TensorProgram [{}×{}] ===", rows, cols)?;
        writeln!(f, "r0 = input({}×{}, {} seed bytes)", rows, cols, self.values.len())?;
        let mut num_regs: usize = 1;
        for instr in &self.ops {
            let out = format!("r{}", num_regs);
            writeln!(f, "{}", instr.ssa_line(&out, num_regs))?;
            num_regs += 1;
        }
        write!(f, "result = r{}.into_data()", num_regs - 1)
    }
}

// ─── autograd program (SSA) ──────────────────────────────────────────────────

/// SSA autograd program.
///
/// `r0` is always the seed leaf (`requires_grad`).  [`DiffOp::Leaf`]
/// instructions introduce additional leaves (up to `max_leaves`).  All other
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
    /// simulating register resolution
    /// and annotating every line with the output shape.
    pub fn ssa(&self, max_leaves: usize) -> String {
        use std::fmt::Write;
        use super::shape::Shape2;
        use super::interpreter::shape::after_diff_op;

        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        let mut s = String::new();

        let _ = writeln!(
            s,
            "=== AutogradProgram [{}×{}] (max_leaves={}) ===",
            rows, cols, max_leaves
        );

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
                    if leaf_count < max_leaves {
                        let pool_idx = if self.leaf_seeds.is_empty() {
                            0
                        } else {
                            *seed as usize % self.leaf_seeds.len()
                        };
                        let seed_len =
                            self.leaf_seeds.get(pool_idx).map(|v| v.len()).unwrap_or(0);
                        let leaf_rows = (*lr as usize).clamp(1, 16);
                        let leaf_cols = (*lc as usize).clamp(1, 16);
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
                        let _ = writeln!(
                            s,
                            "{out} {sh} = r{src}  # leaf cap reached, alias"
                        );
                        sh
                    }
                }
                _ => {
                    let sh = after_diff_op(&shapes, op)
                        .expect("non-Leaf op shape");
                    let _ = writeln!(s, "{out} {sh} = {}", op.ssa_line("_", num_regs).trim_start_matches("_ = "));
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
            let _ = writeln!(s, "grad r{ri} {sh} = r{ri}.grad(grads)  # None → zeros if unreachable");
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ssa(4))
    }
}
#[cfg(test)]
mod backend_selection_tests {
    use super::Backend;

    #[test]
    fn ndarray_is_always_available() {
        assert!(Backend::NdArray.is_compiled_in());
        assert!(Backend::available().contains(&Backend::NdArray));
    }

    #[test]
    fn available_lists_reference_side_first() {
        // LibTorch is the most independently-trustworthy side, so when it is
        // compiled in it must sort ahead of the burn backends.
        let available = Backend::available();
        if available.contains(&Backend::LibTorch) {
            assert_eq!(available[0], Backend::LibTorch);
        }
        assert!(available.iter().all(|b| b.is_compiled_in()));
    }

    #[test]
    fn names_round_trip() {
        for backend in [Backend::NdArray, Backend::Flex, Backend::LibTorch] {
            assert_eq!(
                Backend::from_name(backend.name()),
                Some(backend),
                "{} must parse back to itself",
                backend.name()
            );
        }
        assert_eq!(Backend::from_name("tch"), Some(Backend::LibTorch));
        assert_eq!(Backend::from_name("nonsense"), None);
    }
}
