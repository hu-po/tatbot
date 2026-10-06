//! Enumeration describing the different domains for timestamps acquired by a frame.
//!
//! **NOTE**: Arrival timestamps from the frame are always in system time, whereas other timestamps
//! (presentation and middle-exposure) will be in the frame's specified timestamp domain.

use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;
use std::{ffi::CStr, fmt::Display};

/// Enumeration of possible timestamp domains that frame timestamps are delivered in.
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2TimestampDomain {
    /// Timestamp is measured in relation to the device's internal clock
    HardwareClock = sys::rs2_timestamp_domain_RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK as i32,
    /// Timestamp was measured in relation to the OS (host) system clock
    SystemTime = sys::rs2_timestamp_domain_RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME as i32,
    /// Timestamp was measured in relation to the device's clock converted to the system clock.
    ///
    /// The timestamp is measured directly relative to the device's internal clock, and then
    /// converted to the OS (host) system clock by measuring the difference.
    GlobalTime = sys::rs2_timestamp_domain_RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME as i32,
    /* Not included since this just tells us the total number of domains
     *
     * Count = sys::rs2_timestamp_domain_RS2_TIMESTAMP_DOMAIN_COUNT, */
}

impl Rs2TimestampDomain {
    /// Get the timestamp domain variant as a `&CStr`
    pub fn as_cstr(&self) -> &'static CStr {
        unsafe {
            let ptr = sys::rs2_timestamp_domain_to_string(*self as sys::rs2_timestamp_domain);
            CStr::from_ptr(ptr)
        }
    }

    /// Get the timestamp domain variant as a `&str`
    pub fn as_str(&self) -> &'static str {
        self.as_cstr().to_str().unwrap()
    }
}

impl Display for Rs2TimestampDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        for i in 0..sys::rs2_timestamp_domain_RS2_TIMESTAMP_DOMAIN_COUNT as i32 {
            assert!(
                Rs2TimestampDomain::from_i32(i).is_some(),
                "Rs2TimestampDomain variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
