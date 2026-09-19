//! `Shape2` — the single source of truth for shape algebra across the IR.

use std::fmt;

/// Allocation cap per register. The `AutogradProgram` generator stays well under
/// this via `MAX_DIM`; the cap only fires on `TensorProgram`, which has no
/// shape-aware builder and can chain `Repeat`/`Concat`/`Matmul` into OOM.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 20; // 1Mi f32 elements = 4 MB

/// Lightweight 2-D shape tracked alongside every register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2(pub usize, pub usize);

impl Shape2 {
    #[inline] pub fn rows(self) -> usize { self.0 }
    #[inline] pub fn cols(self) -> usize { self.1 }
    /// Saturating to avoid wrapping on pathological shapes.
    #[inline] pub fn elements(self) -> usize { self.0.saturating_mul(self.1) }
    #[inline] pub fn exceeds_cap(self) -> bool { self.elements() > MAX_TENSOR_ELEMENTS }

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
