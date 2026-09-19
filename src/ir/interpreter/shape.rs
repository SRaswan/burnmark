//! Output-shape computation and operand resolution for every instruction variant.

pub(crate) use crate::ir::shape::Shape2;
use crate::ir::ops::{Reg, TensorInstr, DiffOp};

// ─── output shape computation ────────────────────────────────────────────────

pub(crate) fn after_tensor_instr(shapes: &[Shape2], instr: &TensorInstr) -> Shape2 {
    let n = shapes.len();
    match instr {
        TensorInstr::Add(a, b)
        | TensorInstr::Sub(a, b)
        | TensorInstr::Mul(a, b)
        | TensorInstr::Div(a, b) => {
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
        | TensorInstr::PowfScalar(r, _)
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

pub(crate) fn after_diff_op(shapes: &[Shape2], op: &DiffOp) -> Option<Shape2> {
    match op {
        DiffOp::Leaf { .. } => None,
        DiffOp::Instr(instr) => Some(after_tensor_instr(shapes, instr)),
    }
}

// ─── register resolution ────────────────────────────────────────────────────

/// Resolve `b` to a broadcast-compatible register with `a`. Falls back to `a`.
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
    a_idx 
}

/// Resolve `b` to a matmul-compatible register (`a.cols == b.rows`). Returns `None`
/// if none exists — callers pass through `a` unchanged in that case.
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

/// Resolve `b` to a concat-compatible register (non-concat dim must match `a`).
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
