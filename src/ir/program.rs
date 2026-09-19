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

/// A burn backend. Compile-time features control availability; `BACKENDS` env
/// var controls which runs at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Deprecated in 0.22; still where most known bugs live.
    NdArray,
    /// Pure-Rust CPU backend replacing NdArray. A distinct implementation — the
    /// two disagree in practice.
    Flex,
    /// LibTorch via `tch`. Deprecated on burn `main`; days as reference are numbered.
    LibTorch,
    /// CubeCL CPU. Runs the same kernels as burn-cuda/wgpu but on CPU. Not
    /// deprecated, and correct on `sign(NaN)` where ndarray/flex are wrong —
    /// hence the reference slot.
    Cpu,
}

impl Backend {
    /// All backends, most-trustworthy first. The one place this order lives —
    /// `available()` filters without reordering, so head = default reference.
    pub const ALL: [Backend; 4] = [
        Backend::Cpu,
        Backend::LibTorch,
        Backend::NdArray,
        Backend::Flex,
    ];

    /// Name used in `BACKENDS` and in divergence reports.
    pub const fn name(self) -> &'static str {
        match self {
            Backend::NdArray => "ndarray",
            Backend::Flex => "flex",
            Backend::LibTorch => "libtorch",
            Backend::Cpu => "cpu",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "ndarray" | "nd" => Some(Backend::NdArray),
            "flex" => Some(Backend::Flex),
            "libtorch" | "tch" | "torch" => Some(Backend::LibTorch),
            // Not aliased to bare "cubecl": metal/cuda/rocm/wgpu are CubeCL
            // backends too, so that name will be ambiguous the moment a second
            // one is wired in.
            "cpu" | "cubecl-cpu" => Some(Backend::Cpu),
            _ => None,
        }
    }

    /// Whether this build was compiled with support for this backend.
    pub const fn is_compiled_in(self) -> bool {
        match self {
            Backend::NdArray => true,
            Backend::Flex => cfg!(feature = "oracle-flex"),
            Backend::LibTorch => cfg!(feature = "oracle-tch"),
            Backend::Cpu => cfg!(feature = "oracle-cpu"),
        }
    }

    /// Backends compiled into this build, most-trustworthy first.
    pub fn available() -> Vec<Self> {
        Backend::ALL
            .into_iter()
            .filter(|b| b.is_compiled_in())
            .collect()
    }
}

// ─── target selection ─────────────────────────────────────────────────────────

/// Which framework executes a program. `Backend` is the axis *within* burn;
/// `Target` is the axis above — including non-burn frameworks over the same IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// burn, on one of its devices.
    Burn(Backend),
    /// libtorch via raw tch-rs, no burn in the path.
    ///
    /// Forward: same kernels as `Backend::LibTorch`, different only by burn's FFI
    /// bridge — puts that bridge under test on its own.
    ///
    /// Backward: brings libtorch's own autograd. Every burn backend shares
    /// burn-autodiff, so no burn-vs-burn pairing can disagree about a derivative.
    /// This is the only pairing where the backward pass is independently implemented.
    TchRaw,
    /// candle: no burn and no libtorch anywhere in the path.
    ///
    /// Independent on both passes. With `tch-raw` also running, enables majority-vote
    /// triage (`BACKENDS=tch-raw,candle,flex`): both oracles agreeing against burn
    /// is a confirmed finding.
    Candle,
}

impl Target {
    /// All targets, most-trustworthy first. The one place this order lives —
    /// `available()` filters without reordering, so head = default reference.
    /// Non-burn targets lead: no burn bug can reach them. Within burn, `cpu` leads
    /// (not deprecated, correct on `sign(NaN)`).
    pub const ALL: [Target; 6] = [
        Target::TchRaw,
        Target::Candle,
        Target::Burn(Backend::Cpu),
        Target::Burn(Backend::LibTorch),
        Target::Burn(Backend::NdArray),
        Target::Burn(Backend::Flex),
    ];

    /// Name used in `BACKENDS` and in divergence reports.
    pub const fn name(self) -> &'static str {
        match self {
            Target::Burn(backend) => backend.name(),
            Target::TchRaw => "tch-raw",
            Target::Candle => "candle",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            // Deliberately disjoint from "libtorch"/"tch"/"torch" (burn's backend).
            // The point is to run them against each other; a shared name defeats that.
            "tch-raw" | "raw-tch" | "tch_raw" | "rawtch" => Some(Target::TchRaw),
            "candle" | "candle-core" => Some(Target::Candle),
            other => Backend::from_name(other).map(Target::Burn),
        }
    }

    /// Whether this build was compiled with support for this target.
    pub const fn is_compiled_in(self) -> bool {
        match self {
            Target::Burn(backend) => backend.is_compiled_in(),
            Target::TchRaw => cfg!(feature = "oracle-tch-raw"),
            Target::Candle => cfg!(feature = "oracle-candle"),
        }
    }

    /// The cargo feature that enables this target. NdArray has no feature (always
    /// compiled in) — that arm is unreachable from the only caller.
    const fn feature(self) -> &'static str {
        match self {
            Target::Burn(Backend::NdArray) => "ndarray (always compiled in)",
            Target::Burn(Backend::Flex) => "oracle-flex",
            Target::Burn(Backend::LibTorch) => "oracle-tch",
            Target::Burn(Backend::Cpu) => "oracle-cpu",
            Target::TchRaw => "oracle-tch-raw",
            Target::Candle => "oracle-candle",
        }
    }

    /// Targets compiled into this build, most-trustworthy first.
    pub fn available() -> Vec<Self> {
        Target::ALL
            .into_iter()
            .filter(|t| t.is_compiled_in())
            .collect()
    }
}

/// Parse `BACKENDS` into an ordered target list, reference first. Unset or `all`
/// returns every compiled-in target. Panics on an unknown or uncompiled name —
/// silently dropping a requested target would manufacture false confidence.
pub fn targets_from_env() -> Vec<Target> {
    let requested = match std::env::var("BACKENDS") {
        Err(_) => return Target::available(),
        Ok(raw) if raw.trim().is_empty() => return Target::available(),
        Ok(raw) if raw.trim().eq_ignore_ascii_case("all") => return Target::available(),
        Ok(raw) => raw,
    };

    let mut selected: Vec<Target> = Vec::new();
    for token in requested.split(',') {
        let name = token.trim().to_lowercase();
        if name.is_empty() {
            continue;
        }
        let target = Target::from_name(&name).unwrap_or_else(|| {
            panic!(
                "invalid BACKENDS entry {name:?}: expected one of \
                 ndarray, flex, libtorch, cpu, tch-raw, candle (or `all`)"
            )
        });
        if !target.is_compiled_in() {
            panic!(
                "BACKENDS requested {name:?}, which this build does not support \
                 — rebuild with --features {}. Available: {:?}",
                target.feature(),
                Target::available().iter().map(|t| t.name()).collect::<Vec<_>>()
            );
        }
        if !selected.contains(&target) {
            selected.push(target);
        }
    }

    if selected.is_empty() {
        panic!("BACKENDS named no usable target");
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

    /// Targets to run and cross-check, reference side first.
    pub targets: Vec<Target>,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            max_leaves: 4,
            min_ops: 0,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
            targets: Target::available(),
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
            targets: targets_from_env(),
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
        use super::interpreter::shape::{
            Shape2, after_tensor_instr,
            resolve_broadcast_compatible, resolve_concat_compatible, resolve_matmul_compatible,
        };
        use super::ops::POWF_EXPONENTS;

        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        writeln!(f, "=== TensorProgram [{}×{}] ===", rows, cols)?;
        writeln!(f, "r0 {} = input({} seed bytes)", Shape2(rows, cols), self.values.len())?;

        let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];
        let total = self.ops.len();

        for (i, instr) in self.ops.iter().enumerate() {
            let n = shapes.len();
            let out_shape = after_tensor_instr(&shapes, instr);
            if out_shape.exceeds_cap() {
                writeln!(
                    f,
                    "... ({} remaining op(s) skipped: r{n} would be {out_shape})",
                    total - i,
                )?;
                break;
            }
            // Show the same resolved operands the driver uses so the display
            // matches what was actually computed.
            let rhs = match instr {
                TensorInstr::Add(a, b) => {
                    let ai = a.resolve(n);
                    let bi = resolve_broadcast_compatible(&shapes, ai, b);
                    format!("r{ai} + r{bi}")
                }
                TensorInstr::Sub(a, b) => {
                    let ai = a.resolve(n);
                    let bi = resolve_broadcast_compatible(&shapes, ai, b);
                    format!("r{ai} - r{bi}")
                }
                TensorInstr::Mul(a, b) => {
                    let ai = a.resolve(n);
                    let bi = resolve_broadcast_compatible(&shapes, ai, b);
                    format!("r{ai} * r{bi}")
                }
                TensorInstr::Div(a, b) => {
                    let ai = a.resolve(n);
                    let bi = resolve_broadcast_compatible(&shapes, ai, b);
                    format!("r{ai} / r{bi}")
                }
                TensorInstr::Matmul(a, b) => {
                    let ai = a.resolve(n);
                    match resolve_matmul_compatible(&shapes, ai, b) {
                        Some(bi) => format!("r{ai} @ r{bi}"),
                        None => format!("r{ai}  # matmul: no compatible b, passthrough"),
                    }
                }
                TensorInstr::Neg(r)    => format!("-r{}", r.resolve(n)),
                TensorInstr::Abs(r)    => format!("abs(r{})", r.resolve(n)),
                TensorInstr::Exp(r)    => format!("exp(r{})", r.resolve(n)),
                TensorInstr::Log(r)    => format!("log(r{})", r.resolve(n)),
                TensorInstr::Sqrt(r)   => format!("sqrt(r{})", r.resolve(n)),
                TensorInstr::PowfScalar(r, e) => {
                    let exp = POWF_EXPONENTS[*e as usize % POWF_EXPONENTS.len()];
                    format!("r{}.powf({exp})", r.resolve(n))
                }
                TensorInstr::Relu(r)   => format!("relu(r{})", r.resolve(n)),
                TensorInstr::Sigmoid(r)=> format!("sigmoid(r{})", r.resolve(n)),
                TensorInstr::Tanh(r)   => format!("tanh(r{})", r.resolve(n)),
                TensorInstr::SumAll(r) => format!("sum(r{})  # → [1×1]", r.resolve(n)),
                TensorInstr::MeanAll(r)=> format!("mean(r{})  # → [1×1]", r.resolve(n)),
                TensorInstr::SumDim(r, d) => {
                    format!("sum(r{}, dim={})", r.resolve(n), *d as usize % 2)
                }
                TensorInstr::MeanDim(r, d) => {
                    format!("mean(r{}, dim={})", r.resolve(n), *d as usize % 2)
                }
                TensorInstr::Transpose(r) => format!("r{}.T", r.resolve(n)),
                TensorInstr::Concat(a, b, d) => {
                    let ai = a.resolve(n);
                    let dim = *d as usize % 2;
                    match resolve_concat_compatible(&shapes, ai, b, dim) {
                        Some(bi) => format!("cat([r{ai}, r{bi}], dim={dim})"),
                        None => format!("r{ai}  # concat: no compatible b, passthrough"),
                    }
                }
                TensorInstr::Repeat(r, d, c) => {
                    let dim = *d as usize % 2;
                    let count = (*c as usize).clamp(1, 4);
                    format!("r{}.repeat(dim={dim}, ×{count})", r.resolve(n))
                }
                TensorInstr::Clamp(r) => format!("clamp(r{}, -1e6, 1e6)", r.resolve(n)),
            };
            writeln!(f, "r{n} {out_shape} = {rhs}")?;
            shapes.push(out_shape);
        }

        write!(f, "result = r{}", shapes.len() - 1)
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
        // The head of this list is the default reference side, so it must be a
        // backend believed correct: CubeCL CPU when compiled in, else LibTorch.
        // Both are measured correct on `sign(NaN)`; NdArray and Flex are not.
        let available = Backend::available();
        if available.contains(&Backend::Cpu) {
            assert_eq!(available[0], Backend::Cpu);
        } else if available.contains(&Backend::LibTorch) {
            assert_eq!(available[0], Backend::LibTorch);
        }
        assert!(available.iter().all(|b| b.is_compiled_in()));
    }

    #[test]
    fn cpu_outranks_libtorch_as_reference() {
        // burn-tch is deprecated on burn `main`; CubeCL CPU is not.  When both
        // are compiled in, CPU must take the reference slot — this is the
        // ordering that survives burn-tch's removal.
        let order = |b| Backend::ALL.iter().position(|x| *x == b);
        assert!(order(Backend::Cpu) < order(Backend::LibTorch));
        assert!(order(Backend::LibTorch) < order(Backend::NdArray));
    }

    #[test]
    fn names_round_trip() {
        for backend in Backend::ALL {
            assert_eq!(
                Backend::from_name(backend.name()),
                Some(backend),
                "{} must parse back to itself",
                backend.name()
            );
        }
        assert_eq!(Backend::from_name("tch"), Some(Backend::LibTorch));
        assert_eq!(Backend::from_name("cubecl-cpu"), Some(Backend::Cpu));
        assert_eq!(Backend::from_name("nonsense"), None);
    }
}

#[cfg(test)]
mod target_selection_tests {
    use super::{Backend, Target};

    #[test]
    fn target_names_round_trip() {
        for target in Target::ALL {
            assert_eq!(
                Target::from_name(target.name()),
                Some(target),
                "{} must parse back to itself",
                target.name()
            );
        }
        assert_eq!(Target::from_name("nonsense"), None);
    }

    /// The two libtorch routes must never share a name.  They reach the same
    /// C++ library and differ only by burn's bridge, so the whole point is to
    /// run them *against each other* — a name that could resolve to either
    /// would silently turn that comparison into a target compared with itself.
    #[test]
    fn raw_tch_and_burns_libtorch_backend_are_distinct() {
        for burn_name in ["libtorch", "tch", "torch"] {
            assert_eq!(
                Target::from_name(burn_name),
                Some(Target::Burn(Backend::LibTorch)),
                "{burn_name:?} must stay burn's tch backend"
            );
        }
        for raw_name in ["tch-raw", "raw-tch", "tch_raw", "rawtch"] {
            assert_eq!(
                Target::from_name(raw_name),
                Some(Target::TchRaw),
                "{raw_name:?} must be the non-burn target"
            );
        }
        assert_ne!(Target::TchRaw.name(), Target::Burn(Backend::LibTorch).name());
    }

    /// Raw tch-rs is libtorch minus burn's bridge, so it cannot be wrong
    /// *because of* a burn bug — which no other entry can claim.  It therefore
    /// takes the reference slot ahead of CubeCL CPU, which in turn stays ahead
    /// of burn's own (deprecated) LibTorch backend.
    #[test]
    fn raw_tch_leads_the_trust_order() {
        let order = |t| Target::ALL.iter().position(|x| *x == t);
        assert!(order(Target::TchRaw) < order(Target::Candle));
        assert!(order(Target::Candle) < order(Target::Burn(Backend::Cpu)));
        assert!(order(Target::Burn(Backend::Cpu)) < order(Target::Burn(Backend::LibTorch)));
        assert!(order(Target::Burn(Backend::LibTorch)) < order(Target::Burn(Backend::NdArray)));
    }

    /// Both non-burn targets must outrank every burn one.  A burn backend in
    /// the reference slot means the oracle can be wrong *because of* the bug it
    /// is looking for, which is the failure mode `Target` exists to avoid;
    /// neither `tch-raw` nor `candle` has burn anywhere in its path.
    #[test]
    fn non_burn_targets_outrank_every_burn_target() {
        let last_non_burn = Target::ALL
            .iter()
            .rposition(|t| !matches!(t, Target::Burn(_)))
            .expect("at least one non-burn target");
        let first_burn = Target::ALL
            .iter()
            .position(|t| matches!(t, Target::Burn(_)))
            .expect("at least one burn target");
        assert!(last_non_burn < first_burn, "{:?}", Target::ALL);
    }

    /// `Target::ALL` must cover `Backend::ALL`: a burn backend with no target
    /// slot would be selectable by name and then never actually run.
    #[test]
    fn every_burn_backend_has_a_target_slot() {
        for backend in Backend::ALL {
            assert!(
                Target::ALL.contains(&Target::Burn(backend)),
                "{} has no Target slot",
                backend.name()
            );
        }
        assert_eq!(
            Target::ALL.len(),
            Backend::ALL.len() + 2,
            "burn backends + the two non-burn targets (tch-raw, candle)"
        );
    }

    /// Burn's ordering must survive being lifted into `Target::ALL` — the two
    /// lists disagreeing would make the default reference depend on which one
    /// a reader happened to consult.
    #[test]
    fn burn_ordering_is_preserved_inside_target_ordering() {
        let lifted: Vec<Backend> = Target::ALL
            .iter()
            .filter_map(|t| match t {
                Target::Burn(b) => Some(*b),
                Target::TchRaw | Target::Candle => None,
            })
            .collect();
        assert_eq!(lifted, Backend::ALL.to_vec());
    }

    #[test]
    fn available_targets_are_compiled_in() {
        let available = Target::available();
        assert!(available.iter().all(|t| t.is_compiled_in()));
        // NdArray is unconditional, so the list is never empty.
        assert!(available.contains(&Target::Burn(Backend::NdArray)));
    }
}

#[cfg(test)]
mod tensor_program_display_tests {
    use super::*;
    use crate::ir::ops::{Reg, TensorInstr};

    /// TensorProgram::Display must show the same register indices that the driver
    /// uses — i.e. the resolver-chosen ones, not the raw Reg.0 % n values. These
    /// diverge whenever the preferred `b` operand is not broadcast-compatible and
    /// the resolver falls back to an earlier register.
    #[test]
    fn display_shows_resolved_operands_not_raw_reg() {
        // r0: [2×3]. r1 = Neg(r0) → [2×3].
        // r2 = Add(r1, Reg(255)) — Reg(255) resolves to 255 % 2 = 1, which is
        // broadcast-compatible ([2×3] vs [2×3]), so resolved b = 1.
        let prog = TensorProgram {
            rows: 2,
            cols: 3,
            values: vec![1, 2, 3],
            ops: vec![
                TensorInstr::Neg(Reg(0)),
                TensorInstr::Add(Reg(1), Reg(255)),
            ],
        };
        let s = format!("{prog}");
        // Reg(255) resolves to 255 % 2 = 1 → display must say "r1 + r1".
        assert!(
            s.contains("r1 + r1"),
            "expected 'r1 + r1' (Reg(255) % 2 = 1) in:\n{s}"
        );
        // All instructions must be shown (neither cap nor passthrough).
        assert!(!s.contains("skipped"), "no instructions should be skipped: {s}");
    }

    /// When the resolved `b` is incompatible and the resolver falls back, the
    /// display must show the actual fallback register, NOT the raw Reg.0 % n.
    /// Here we contrive a situation where the preferred b is shape-incompatible.
    #[test]
    fn display_shows_fallback_register_when_preferred_b_is_incompatible() {
        // r0: [2×3]. r1 = Transpose(r0) → [3×2].
        // r2 = Add(r0, r1): r0 is [2×3], r1 is [3×2] → NOT broadcast-compatible.
        // The resolver scans backwards from r1 and finds r0 (broadcast-compatible
        // with itself), so resolved b = 0. Display must say "r0 + r0", not "r0 + r1".
        let prog = TensorProgram {
            rows: 2,
            cols: 3,
            values: vec![1, 2, 3],
            ops: vec![
                TensorInstr::Transpose(Reg(0)),          // r1 = r0.T → [3×2]
                TensorInstr::Add(Reg(0), Reg(1)),         // r2 = r0 + r1?
            ],
        };
        let s = format!("{prog}");
        // r0 [2×3] is not broadcast-compatible with r1 [3×2], so the resolver
        // falls back: scans (1, 0) in reverse and takes r0 (self-compatible).
        // Displayed b must be r0, not r1.
        assert!(
            s.contains("r0 + r0"),
            "expected fallback 'r0 + r0' ([2×3] incompatible with [3×2]) in:\n{s}"
        );
    }

    /// exceeds_cap path: the display must note the truncation and not panic.
    #[test]
    fn display_notes_truncation_on_cap_overflow() {
        // Chain Repeat(Reg(i), dim=0, count=4) using the previous register each
        // time: r0=[1×1], r1=[4×1], r2=[16×1], ... r11=[4194304×1] > 2^20.
        // Reg(i).resolve(i+1) = i because i < i+1, so each step repeats the
        // previous output.
        let mut ops = vec![];
        for i in 0..12u8 {
            ops.push(TensorInstr::Repeat(Reg(i), 0, 4));
        }
        let prog = TensorProgram { rows: 1, cols: 1, values: vec![42], ops };
        let s = format!("{prog}");
        assert!(
            s.contains("skipped"),
            "expected truncation note in:\n{s}"
        );
    }

    /// Shapes annotated on every line must match after_tensor_instr's prediction.
    #[test]
    fn display_shape_annotations_match_predictor() {
        use crate::ir::interpreter::shape::{after_tensor_instr, Shape2};
        let prog = TensorProgram {
            rows: 3,
            cols: 4,
            values: vec![0u8; 12],
            ops: vec![
                TensorInstr::Transpose(Reg(0)),     // [3×4] → [4×3]
                TensorInstr::SumDim(Reg(1), 0),     // [4×3] → [1×3]
                TensorInstr::Abs(Reg(2)),            // [1×3] → [1×3]
            ],
        };
        let s = format!("{prog}");
        // Verify the annotated shapes appear in the output.
        assert!(s.contains("[4×3]"), "transpose output shape missing in:\n{s}");
        assert!(s.contains("[1×3]"), "sum_dim output shape missing in:\n{s}");

        // Cross-check with the predictor directly.
        let mut shapes = vec![Shape2(3, 4)];
        for instr in &prog.ops {
            let predicted = after_tensor_instr(&shapes, instr);
            shapes.push(predicted);
        }
        assert_eq!(shapes[1], Shape2(4, 3));
        assert_eq!(shapes[2], Shape2(1, 3));
        assert_eq!(shapes[3], Shape2(1, 3));
    }
}

#[cfg(test)]
mod autograd_generator_invariants {
    use arbitrary::{Arbitrary, Unstructured};
    use crate::ir::interpreter::shape::{
        resolve_broadcast_compatible, resolve_matmul_compatible, resolve_concat_compatible,
        Shape2, after_diff_op,
    };
    use crate::ir::ops::{DiffOp, TensorInstr};
    use crate::ir::program::AutogradProgram;

    /// The shape-aware generator stores exact register indices in every instruction
    /// (not fuzzy Reg values that might require resolver fallback). Verify this by
    /// replaying the program through the resolver and checking it never uses a
    /// different register than the one stored.
    ///
    /// If this fails the generator is producing programs where the driver silently
    /// executes different ops than the generator intended.
    fn check_exact_refs(prog: &AutogradProgram, max_leaves: usize) {
        let rows = (prog.rows as usize).clamp(1, 16);
        let cols = (prog.cols as usize).clamp(1, 16);
        let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];
        let mut leaf_count = 1usize;

        for (step, op) in prog.ops.iter().enumerate() {
            let n = shapes.len();
            match op {
                DiffOp::Leaf { seed, rows: lr, cols: lc } => {
                    if leaf_count < max_leaves {
                        let r = (*lr as usize).clamp(1, 16);
                        let c = (*lc as usize).clamp(1, 16);
                        shapes.push(Shape2(r, c));
                        leaf_count += 1;
                    } else {
                        // Alias: seed % n must be in-bounds, which it is by construction.
                        let alias_idx = (*seed as usize) % n;
                        shapes.push(shapes[alias_idx]);
                    }
                }
                DiffOp::Instr(instr) => {
                    let stored_matches = match instr {
                        TensorInstr::Add(a, b) | TensorInstr::Sub(a, b)
                        | TensorInstr::Mul(a, b) | TensorInstr::Div(a, b) => {
                            let ai = a.resolve(n);
                            let resolved_bi = resolve_broadcast_compatible(&shapes, ai, b);
                            resolved_bi == b.resolve(n)
                        }
                        TensorInstr::Matmul(a, b) => {
                            let ai = a.resolve(n);
                            // Generator falls back to unary when there are no matmul-
                            // compatible registers, so the matmul arm is only emitted
                            // when one exists — the stored b must be it.
                            match resolve_matmul_compatible(&shapes, ai, b) {
                                Some(resolved_bi) => resolved_bi == b.resolve(n),
                                None => false,
                            }
                        }
                        TensorInstr::Concat(a, b, d) => {
                            let ai = a.resolve(n);
                            let dim = *d as usize % 2;
                            match resolve_concat_compatible(&shapes, ai, b, dim) {
                                Some(resolved_bi) => resolved_bi == b.resolve(n),
                                None => false,
                            }
                        }
                        _ => true, // Unary ops: single Reg, no fallback possible.
                    };
                    assert!(
                        stored_matches,
                        "step {step}: generator stored a Reg that the driver would not \
                         pick directly — program would execute different ops than generated.\n\
                         instr: {:?}\nshapes: {shapes:?}",
                        instr
                    );
                    let out_shape = after_diff_op(&shapes, op).unwrap();
                    shapes.push(out_shape);
                }
            }
        }
    }

    #[test]
    fn generated_programs_use_exact_register_refs() {
        // Deterministic seed corpus: exercise the generator across many inputs.
        let seeds: &[&[u8]] = &[
            &[0u8; 200],
            &[255u8; 200],
            // Alternating bytes to exercise both halves of every branch.
            &{
                let mut v = [0u8; 200];
                for (i, b) in v.iter_mut().enumerate() { *b = (i as u8).wrapping_mul(37); }
                v
            },
            &{
                let mut v = [0u8; 200];
                for (i, b) in v.iter_mut().enumerate() { *b = (i as u8).wrapping_mul(97).wrapping_add(13); }
                v
            },
        ];
        for raw in seeds {
            let mut u = Unstructured::new(raw);
            if let Ok(prog) = AutogradProgram::arbitrary(&mut u) {
                check_exact_refs(&prog, 4);
            }
        }
    }

    /// Programs must never be cut short by exceeds_cap: the generator bounds all
    /// shapes so the cap never fires for AutogradProgram.
    #[test]
    fn autograd_programs_never_exceed_cap() {
        use crate::ir::shape::MAX_TENSOR_ELEMENTS;
        let seeds: &[&[u8]] = &[
            &[0u8; 300],
            &[255u8; 300],
            &{
                let mut v = [0u8; 300];
                for (i, b) in v.iter_mut().enumerate() { *b = (i as u8).wrapping_mul(61); }
                v
            },
        ];
        for raw in seeds {
            let mut u = Unstructured::new(raw);
            if let Ok(prog) = AutogradProgram::arbitrary(&mut u) {
                let rows = (prog.rows as usize).clamp(1, 16);
                let cols = (prog.cols as usize).clamp(1, 16);
                let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];
                let mut leaf_count = 1usize;
                for op in &prog.ops {
                    let out = match op {
                        DiffOp::Leaf { rows: lr, cols: lc, seed } => {
                            if leaf_count < 4 {
                                leaf_count += 1;
                                Shape2((*lr as usize).clamp(1, 16), (*lc as usize).clamp(1, 16))
                            } else {
                                shapes[(*seed as usize) % shapes.len()]
                            }
                        }
                        DiffOp::Instr(_) => after_diff_op(&shapes, op).unwrap(),
                    };
                    assert!(
                        out.elements() <= MAX_TENSOR_ELEMENTS,
                        "autograd generator produced a shape {out} exceeding the cap — \
                         a MAX_DIM bound is wrong"
                    );
                    shapes.push(out);
                }
            }
        }
    }
}
