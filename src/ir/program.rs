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
    ///
    /// Deprecated on burn `main` ("burn-tch is deprecated ... use a CubeCL
    /// backend or `Device::flex()`"), so its days as the reference are numbered.
    LibTorch,
    /// CubeCL's CPU backend.  Compiles the same kernels as burn-cuda / rocm /
    /// wgpu and runs them on CPU, so it gives GPU-family coverage with no GPU —
    /// and unlike a real GPU backend it needs no panic hook, no hoisted device
    /// construction and no per-op tolerance.  Not deprecated, and measured
    /// correct on the `sign(NaN)` case the other CPU backends get wrong, which
    /// is why it takes the reference slot ahead of LibTorch.
    Cpu,
}

impl Backend {
    /// Every backend this fuzzer knows about, most-trustworthy first.
    ///
    /// The one place that ordering is written down: [`Backend::available`]
    /// filters this without reordering, so the reference side is whichever of
    /// these is compiled in first.
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

    /// Every backend this build can run, most-trustworthy first.
    ///
    /// This ordering *is* the default reference choice whenever `BACKENDS` is
    /// unset, so it leads with the backends both believed correct and not
    /// deprecated: CubeCL CPU first, then LibTorch — correct, but deprecated on
    /// burn `main` — then the two CPU backends known to be wrong on `sign(NaN)`.
    pub fn available() -> Vec<Self> {
        Backend::ALL
            .into_iter()
            .filter(|b| b.is_compiled_in())
            .collect()
    }
}

// ─── target selection ─────────────────────────────────────────────────────────

/// A framework the interpreter executes a program against.
///
/// [`Backend`] is an axis *inside* burn: 0.22 made the backend a property of
/// the device, so every burn backend shares one interpreter and one `Tensor<2>`
/// type.  `Target` is the axis above that — which framework runs the program at
/// all.  A non-burn target brings its own interpreter over the same
/// [`TensorInstr`](super::ops::TensorInstr) vocabulary, so the IR, the
/// generator, shape propagation, `values_diverge` and this selection logic are
/// all shared while only execution differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// burn, on one of its devices.
    Burn(Backend),
    /// Raw tch-rs: libtorch called directly, with no burn code in the path.
    ///
    /// On the **forward** pass this reaches the same libtorch kernels
    /// [`Backend::LibTorch`] does, so as a math oracle it adds little — but the
    /// difference between the two is then exactly burn's FFI/translation layer,
    /// and running both puts that bridge under test on its own.  That is the
    /// class the original 0.20.1 `swap_dims` bug belonged to — a shallow clone
    /// across the FFI boundary corrupting gradients, not a math error — and no
    /// pairing of burn-internal backends can isolate it.
    ///
    /// On the **backward** pass it is not the same implementation at all:
    /// `Device::libtorch().autodiff()` differentiates with burn-autodiff, so
    /// every burn backend shares one set of derivative formulas and no
    /// burn-vs-burn pairing can disagree about a derivative.  This target
    /// brings libtorch's own autograd, making it the only pairing here whose
    /// backward pass is implemented twice, independently — which matters, since
    /// `fuzz_autograd` is where every bug so far has come from.
    ///
    /// It is also the one oracle burn cannot deprecate out from under this
    /// fuzzer: `tch` is an independent project, and what burn `main` is
    /// dropping is the bridge, not the library.
    TchRaw,
    /// candle: an independent Rust ML framework, with no burn and no libtorch
    /// anywhere in the path.
    ///
    /// Every other entry here shares an implementation with some other entry.
    /// The four burn backends share burn-autodiff's derivative formulas;
    /// [`Backend::LibTorch`] and [`Target::TchRaw`] share libtorch's forward
    /// kernels.  candle shares neither: its kernels are its own and its
    /// autograd is its own, so it is the first target in this fuzzer that can
    /// disagree with *everything* else at once — and the first whose agreement
    /// with a burn backend is evidence rather than tautology.
    ///
    /// What that costs, and it is a real cost: a candle-vs-burn divergence no
    /// longer localises the bug.  `libtorch` vs `tch-raw` differ by exactly one
    /// thing (burn's bridge), so a divergence there names its own culprit;
    /// candle vs burn differ by two whole implementations, so triage has to
    /// decide which side is wrong.  Running candle *alongside* libtorch —
    /// `BACKENDS=tch-raw,candle,flex` — is what makes that decision cheap: two
    /// independent oracles agreeing against burn is a finding, and the one case
    /// where they split is candle's own bug.
    Candle,
}

impl Target {
    /// Every target this fuzzer knows about, most-trustworthy first.
    ///
    /// The one place this ordering is written down; [`Target::available`]
    /// filters it without reordering, so the reference side is whichever of
    /// these is compiled in first.
    ///
    /// Raw tch-rs leads: it is libtorch — whose documented special-value
    /// conventions are what the other backends are judged against — minus
    /// burn's bridge, so alone among these entries it cannot be wrong
    /// *because of* a burn bug.  candle follows for the same structural reason
    /// (no burn in the path, so no burn bug can reach it) but behind libtorch,
    /// because what a divergence is judged *against* is PyTorch's documented
    /// conventions and libtorch is where those are defined — candle is a
    /// younger implementation that has yet to earn that role.  burn's own
    /// LibTorch backend is libtorch's math plus the bridge, so it stays below
    /// CubeCL CPU, which is at least not deprecated.
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
            // Deliberately disjoint from "libtorch"/"tch"/"torch", which stay
            // burn's tch *backend*.  The two differ by burn's bridge and the
            // entire point is to run them against each other, so a name that
            // could mean either would defeat the comparison.
            "tch-raw" | "raw-tch" | "tch_raw" | "rawtch" => Some(Target::TchRaw),
            // No burn backend wraps candle in this build, so unlike `tch-raw`
            // there is no name to stay disjoint from — but burn *does* ship a
            // candle backend, so if one is ever wired in it must take a
            // distinct name rather than an alias of this one.
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

    /// The cargo feature that would add this target to a build.
    ///
    /// NdArray has no feature of its own — it is burn's default and always
    /// compiled in — so that arm is unreachable from the only caller, which
    /// asks this question only about targets that are *missing*.
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

    /// Every target this build can run, most-trustworthy first.
    ///
    /// This ordering *is* the default reference choice whenever `BACKENDS` is
    /// unset, so it leads with the targets both believed correct and not
    /// deprecated: raw tch-rs, then candle, then CubeCL CPU, then burn's
    /// LibTorch backend — correct, but deprecated on burn `main` — then the two
    /// CPU backends known to be wrong on `sign(NaN)`.
    pub fn available() -> Vec<Self> {
        Target::ALL
            .into_iter()
            .filter(|t| t.is_compiled_in())
            .collect()
    }
}

/// Targets to run, **reference side first** — every other target is compared
/// against the first, and divergences are reported as `<target> vs <reference>`.
///
/// Parsed from `BACKENDS`: a comma-separated list of names, or `all`.  Unset
/// runs every target compiled in, in [`Target::available`]'s order.  A single
/// entry runs the program without any comparison, which is useful for smoke and
/// throughput runs.
///
/// The variable keeps its original name so existing command lines and crash
/// reproductions keep working; what it selects is now a target rather than a
/// burn backend, which is a strict superset.  `libtorch` still means burn's tch
/// backend; `tch-raw` is the new non-burn one.
///
/// An unparseable or unavailable selection panics rather than silently dropping
/// a requested side: quietly comparing fewer targets than asked for would
/// manufacture exactly the false confidence this fuzzer exists to prevent.
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
