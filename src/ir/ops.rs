//! SSA instructions for fuzz IR.
//!
//! The register file is a `Vec<Tensor>`.  Each instruction produces exactly one
//! value and appends it.  Operands are [`Reg(u8)`] references resolved at
//! interpretation time as `raw % num_defined_regs`.

use std::fmt;
use arbitrary::Arbitrary;

// ─── register reference ──────────────────────────────────────────────────────

/// A fuzzer-generated register reference.
/// Resolved at interpretation time: `self.0 as usize % num_regs`.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub struct Reg(pub u8);

impl Reg {
    /// Resolve to a valid register index.  `num_regs` must be > 0.
    #[inline]
    pub fn resolve(&self, num_regs: usize) -> usize {
        (self.0 as usize) % num_regs
    }

    /// Pretty-print using the resolved index.
    pub fn name(&self, num_regs: usize) -> String {
        format!("r{}", self.resolve(num_regs))
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reg({})", self.0)
    }
}

// ─── SSA tensor instruction ──────────────────────────────────────────────────

/// SSA instruction for plain tensor programs.
/// Each instruction consumes register operand(s) and produces one new register.
#[derive(Arbitrary, Debug, Clone)]
pub enum TensorInstr {
    // --- binary: result = reg ⊕ reg ---
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    // --- unary ---
    Neg(Reg),
    Abs(Reg),
    Exp(Reg),
    Log(Reg),
    Sqrt(Reg),
    // --- activations ---
    Relu(Reg),
    Sigmoid(Reg),
    Tanh(Reg),
    // --- reductions ---
    SumAll(Reg),
    MeanAll(Reg),
    // --- dimensional reductions ---
    SumDim(Reg, u8),
    MeanDim(Reg, u8),
    // --- layout ---
    Transpose(Reg),
    Concat(Reg, Reg, u8),
    Repeat(Reg, u8, u8),
    // --- matmul ---
    Matmul(Reg, Reg),
    // --- guard ---
    Clamp(Reg),
}

impl TensorInstr {
    /// Pretty-print one SSA line.
    /// `num_regs` = how many registers are defined *before* this instruction.
    pub fn ssa_line(&self, out: &str, num_regs: usize) -> String {
        match self {
            TensorInstr::Add(a, b)    => format!("{out} = {} + {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Sub(a, b)    => format!("{out} = {} - {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Mul(a, b)    => format!("{out} = {} * {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Neg(r)       => format!("{out} = -{}", r.name(num_regs)),
            TensorInstr::Abs(r)       => format!("{out} = abs({})", r.name(num_regs)),
            TensorInstr::Exp(r)       => format!("{out} = exp({})", r.name(num_regs)),
            TensorInstr::Log(r)       => format!("{out} = log({})", r.name(num_regs)),
            TensorInstr::Sqrt(r)      => format!("{out} = sqrt({})", r.name(num_regs)),
            TensorInstr::Relu(r)      => format!("{out} = relu({})", r.name(num_regs)),
            TensorInstr::Sigmoid(r)   => format!("{out} = sigmoid({})", r.name(num_regs)),
            TensorInstr::Tanh(r)      => format!("{out} = tanh({})", r.name(num_regs)),
            TensorInstr::SumAll(r)    => format!("{out} = sum({})  # → [1,1]", r.name(num_regs)),
            TensorInstr::MeanAll(r)   => format!("{out} = mean({})  # → [1,1]", r.name(num_regs)),
            TensorInstr::SumDim(r, d)  => {
                let dim = *d as usize % 2;
                format!("{out} = sum({}, dim={dim})", r.name(num_regs))
            }
            TensorInstr::MeanDim(r, d) => {
                let dim = *d as usize % 2;
                format!("{out} = mean({}, dim={dim})", r.name(num_regs))
            }
            TensorInstr::Transpose(r) => format!("{out} = {}.T", r.name(num_regs)),
            TensorInstr::Concat(a, b, d) => {
                let dim = *d as usize % 2;
                format!("{out} = cat([{}, {}], dim={dim})", a.name(num_regs), b.name(num_regs))
            }
            TensorInstr::Repeat(r, d, c) => {
                let dim = *d as usize % 2;
                let count = (*c as usize).clamp(1, 4);
                format!("{out} = {}.repeat(dim={dim}, ×{count})", r.name(num_regs))
            }
            TensorInstr::Matmul(a, b) => format!("{out} = {} @ {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Clamp(r)     => format!("{out} = clamp({}, -1e6, 1e6)", r.name(num_regs)),
        }
    }
}

// ─── SSA diff instruction ────────────────────────────────────────────────────

/// SSA instruction for autograd programs.
///
/// `Leaf { seed, rows, cols }` introduces a new `requires_grad` input tensor
/// with its own shape.  The `seed` byte selects data from the seed pool (or
/// aliases a register when `max_leaves` is reached).  `Instr` wraps a
/// [`TensorInstr`] — all the same ops, no duplication.
#[derive(Arbitrary, Debug, Clone)]
pub enum DiffOp {
    /// Introduce a new leaf input with its own shape.
    Leaf {
        seed: u8,
        rows: u8,
        cols: u8,
    },
    /// Any tensor instruction (shared with plain-tensor programs).
    Instr(TensorInstr),
}

impl DiffOp {
    /// Pretty-print one SSA line.
    /// `num_regs` = how many registers are defined *before* this instruction.
    /// For `Leaf`, a placeholder is printed; callers should override.
    pub fn ssa_line(&self, out: &str, num_regs: usize) -> String {
        match self {
            DiffOp::Leaf { rows, cols, .. } => {
                let r = (*rows as usize).clamp(1, 16);
                let c = (*cols as usize).clamp(1, 16);
                format!("{out} = leaf({r}×{c})  [requires_grad]")
            }
            DiffOp::Instr(i) => i.ssa_line(out, num_regs),
        }
    }
}
