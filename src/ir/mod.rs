//! IR / AST module for fuzz programs.

pub mod ops;
pub mod program;
pub mod shape;
pub mod generate;

pub use program::{FuzzConfig, HarnessMode};
pub mod interpreter;
