//! The enumeration of persistence controls.

use num_derive::{FromPrimitive, ToPrimitive};

/// An enumeration of the various persistence controls used in processing blocks.
#[allow(missing_docs)]
#[repr(usize)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersistenceControl {
    /// Persistence control is disabled
    Disabled = 0,
    /// Valid 8 out of 8
    Valid8OutOf8 = 1,
    /// Valid 2 out of 3
    Valid2OutOf3 = 2,
    /// Valid 2 out of 4
    Valid2OutOf4 = 3,
    /// Valid 2 out of 8
    Valid2OutOf8 = 4,
    /// Valid 1 out of 2
    Valid1OutOf2 = 5,
    /// Valid 1 out of 5
    Valid1OutOf5 = 6,
    /// Valid 1 out of 8
    Valid1OutOf8 = 7,
    /// Indefinite
    Indefinitely = 8,
}
