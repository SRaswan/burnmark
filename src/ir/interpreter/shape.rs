//! Interpreter-side shape resolution and output-shape computation.
//!
//! The canonical [`Shape2`] type lives in `ir::shape`.  This module adds the
//! `resolve_*` helpers (which need [`Reg`]) and the `after_*` methods that
//! compute output shapes for every instruction variant.

pub(crate) use crate::ir::shape::Shape2;
use crate::ir::ops::{Reg, TensorInstr, DiffOp};

// ─── output shape computation ────────────────────────────────────────────────

/// Given a [`TensorInstr`] and the current shape arena, compute the output shape.
pub(crate) fn after_tensor_instr(shapes: &[Shape2], instr: &TensorInstr) -> Shape2 {
    let n = shapes.len();
    match instr {
        TensorInstr::Add(a, b)
        | TensorInstr::Sub(a, b)
        | TensorInstr::Mul(a, b) => {
            let sa = shapes[a.resolve(n)];
            let sb = shapes[resolve_broadcast_compatible(shapes, a.resolve(n), b)];
            sa.broadcast_result(sb)
        }
        TensorInstr::Matmul(a, b) => {
            let sa = shapes[a.resolve(n)];
            match resolve_matmul_compatible(shapes, a.resolve(n), b) {
                Some(bi) => Shape2(sa.0, shapes[bi].1),
                None => sa, // demoted to passthrough
            }
        }
        TensorInstr::Neg(r)
        | TensorInstr::Abs(r)
        | TensorInstr::Exp(r)
        | TensorInstr::Log(r)
        | TensorInstr::Sqrt(r)
        | TensorInstr::Relu(r)
        | TensorInstr::Sigmoid(r)
        | TensorInstr::Tanh(r)
        | TensorInstr::Clamp(r) => shapes[r.resolve(n)],
        TensorInstr::SumAll(_) | TensorInstr::MeanAll(_) => Shape2(1, 1),
        TensorInstr::SumDim(r, d) | TensorInstr::MeanDim(r, d) => {
            let s = shapes[r.resolve(n)];
            match *d as usize % 2 {
                0 => Shape2(1, s.1),
                _ => Shape2(s.0, 1),
            }
        }
        TensorInstr::Transpose(r) => {
            let Shape2(r_, c_) = shapes[r.resolve(n)];
            Shape2(c_, r_)
        }
        TensorInstr::Concat(a, b, d) => {
            let dim = *d as usize % 2;
            let ai = a.resolve(n);
            let sa = shapes[ai];
            match resolve_concat_compatible(shapes, ai, b, dim) {
                Some(bi) => {
                    let sb = shapes[bi];
                    if dim == 0 { Shape2(sa.0 + sb.0, sa.1) }
                    else { Shape2(sa.0, sa.1 + sb.1) }
                }
                None => sa,
            }
        }
        TensorInstr::Repeat(r, d, c) => {
            let s = shapes[r.resolve(n)];
            let dim = *d as usize % 2;
            let count = (*c as usize).clamp(1, 4);
            if dim == 0 { Shape2(s.0 * count, s.1) }
            else { Shape2(s.0, s.1 * count) }
        }
    }
}

/// Given a [`DiffOp`], compute the output shape.  Returns `None` for `Leaf`
/// (handled by the main loop).
pub(crate) fn after_diff_op(shapes: &[Shape2], op: &DiffOp) -> Option<Shape2> {
    match op {
        DiffOp::Leaf { .. } => None,
        DiffOp::Instr(instr) => Some(after_tensor_instr(shapes, instr)),
    }
}

// ─── register resolution ────────────────────────────────────────────────────

/// For element-wise binary operations (Add/Sub/Mul), resolve operand `b` to a
/// register whose shape is **broadcast-compatible** with operand `a`.
/// Falls back to `a` itself when nothing compatible exists.
pub(crate) fn resolve_broadcast_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &Reg) -> usize {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.broadcast_compatible(shapes[b_idx]) {
        return b_idx;
    }
    for i in (0..n).rev() {
        if sa.broadcast_compatible(shapes[i]) {
            return i;
        }
    }
    a_idx // ultimate fallback: a ⊕ a
}

/// For matmul, resolve operand `b` to a register where
/// `shapes[a_idx].cols == shapes[b_idx].rows`.
/// Returns `None` when no compatible register exists.
pub(crate) fn resolve_matmul_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &Reg) -> Option<usize> {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.matmul_compatible(shapes[b_idx]) {
        return Some(b_idx);
    }
    for i in (0..n).rev() {
        if sa.matmul_compatible(shapes[i]) {
            return Some(i);
        }
    }
    None
}

/// For concat, resolve operand `b` to a register whose non-concat dimension
/// matches `shapes[a_idx]`.  Returns `None` when no compatible register exists.
pub(crate) fn resolve_concat_compatible(
    shapes: &[Shape2],
    a_idx: usize,
    b_raw: &Reg,
    dim: usize,
) -> Option<usize> {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.concat_compatible(shapes[b_idx], dim) {
        return Some(b_idx);
    }
    for i in (0..n).rev() {
        if sa.concat_compatible(shapes[i], dim) {
            return Some(i);
        }
    }
    None
}
