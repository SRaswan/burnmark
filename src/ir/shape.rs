//! Core 2-D shape type shared across the IR.
//!
//! `Shape2` is the single source of truth for shape algebra.  Both the state-machine generator
//!  and the interpreter  import it from here — no duplication.

use std::fmt;

/// Upper bound on any single register's element count.
///
/// `AutogradProgram`'s generator (`generate.rs`) already keeps every shape
/// well under this via `MAX_DIM`, so this only ever bites the plain
/// `TensorProgram` path: its `Vec<TensorInstr>` comes straight from
/// `#[derive(Arbitrary)]` with no shape-aware builder, so chained
/// `Repeat`/`Concat`/`Matmul` can otherwise compound into a multi-gigabyte
/// single allocation (observed: a `malloc(4294967296)` OOM abort) — a fuzzer
/// generation-space gap, not anything resembling a real backend bug.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 20; // 1Mi elements (4MB as f32)

/// Lightweight 2-D shape tracked alongside every register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2(pub usize, pub usize);

impl Shape2 {
    /// Row count.
    #[inline]
    pub fn rows(self) -> usize { self.0 }
    /// Column count.
    #[inline]
    pub fn cols(self) -> usize { self.1 }

    /// Total element count, saturating so a pathological shape can't wrap.
    #[inline]
    pub fn elements(self) -> usize { self.0.saturating_mul(self.1) }

    /// `true` once this shape would allocate more than [`MAX_TENSOR_ELEMENTS`].
    #[inline]
    pub fn exceeds_cap(self) -> bool { self.elements() > MAX_TENSOR_ELEMENTS }

    #[inline]
    pub fn broadcast_compatible(self, other: Shape2) -> bool {
        (self.0 == other.0 || self.0 == 1 || other.0 == 1)
            && (self.1 == other.1 || self.1 == 1 || other.1 == 1)
    }

    #[inline]
    pub fn broadcast_result(self, other: Shape2) -> Shape2 {
        Shape2(self.0.max(other.0), self.1.max(other.1))
    }

    
    #[inline]
    pub fn matmul_compatible(self, other: Shape2) -> bool {
        self.1 == other.0
    }

    #[inline]
    pub fn concat_compatible(self, other: Shape2, dim: usize) -> bool {
        if dim == 0 { self.1 == other.1 } else { self.0 == other.0 }
    }
}

impl fmt::Display for Shape2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}×{}]", self.0, self.1)
    }
}
