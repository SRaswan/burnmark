//! Core 2-D shape type shared across the IR.
//!
//! `Shape2` is the single source of truth for shape algebra (broadcasting,
//! matmul compatibility, etc.).  Both the state-machine generator
//! (`ir::generate`) and the interpreter (`ir::interpreter`) import it from
//! here — no duplication.

use std::fmt;

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

    /// Can two 2-D shapes be element-wise combined under NumPy broadcasting?
    #[inline]
    pub fn broadcast_compatible(self, other: Shape2) -> bool {
        (self.0 == other.0 || self.0 == 1 || other.0 == 1)
            && (self.1 == other.1 || self.1 == 1 || other.1 == 1)
    }

    /// Result shape after broadcasting two compatible shapes.
    #[inline]
    pub fn broadcast_result(self, other: Shape2) -> Shape2 {
        Shape2(self.0.max(other.0), self.1.max(other.1))
    }

    /// Can `self` be left-multiplied by `other`?  i.e. `self @ other`
    /// requires `self.cols == other.rows`.
    #[inline]
    pub fn matmul_compatible(self, other: Shape2) -> bool {
        self.1 == other.0
    }

    /// Can `self` and `other` be concatenated along `dim`?
    /// dim 0 → cols must match.  dim 1 → rows must match.
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
