//! Measured takeover and landing targets carried from recovery.land_arm.
//! This module has no driver I/O. The caller must perform fresh connection,
//! golden application/reconnect, monitored commands, retries and verification.
use crate::{Error, Result};

pub const TAKEOVER_S: f64 = 0.5;
pub const STAGED_POSE_S: f64 = 4.0;
pub const SLEEP_POSE_S: f64 = 3.0;
pub const RETRY_DELAY_S: f64 = 2.0;
pub const CONFIGURE_TIMEOUT_S: f64 = 5.0;
pub const LANDING_DEADLINE_S: f64 = 45.0;
pub const LANDED_TOLERANCE_RAD: f64 = 0.20;
pub const LIMIT_MARGIN: f64 = 1e-4;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PositionLimit {
    pub min: f64,
    pub max: f64,
}

/// Preserve the SDK's raw error elsewhere; only these known healthy strings
/// count as healthy. A failed read is an error, never an empty observation.
pub fn controller_error(raw: &str) -> &str {
    let value = raw.trim();
    match value.to_ascii_lowercase().as_str() {
        "" | "no error" | "none" | "error state: none" => "",
        _ => value,
    }
}

fn validate_pose(pose: &[f64; 7]) -> Result<()> {
    if pose.iter().any(|value| !value.is_finite()) {
        return Err(Error("recovery pose must be finite".into()));
    }
    Ok(())
}
fn validate_limits(limits: &[PositionLimit; 7]) -> Result<()> {
    if limits.iter().any(|limit| {
        !limit.min.is_finite()
            || !limit.max.is_finite()
            || limit.max - limit.min < 2.0 * LIMIT_MARGIN
    }) {
        return Err(Error(
            "recovery limits cannot contain the hold margin".into(),
        ));
    }
    Ok(())
}

/// Test before takeover. If true, apply the golden and read live limits again
/// before constructing Targets; do not reuse boot limits after configuration.
pub fn needs_golden(pose: &[f64; 7], limits: &[PositionLimit; 7]) -> Result<bool> {
    validate_pose(pose)?;
    validate_limits(limits)?;
    Ok(pose
        .iter()
        .zip(limits)
        .any(|(q, limit)| *q < limit.min || *q > limit.max))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Targets {
    pub takeover: [f64; 7],
    pub staged: [f64; 7],
    pub sleep: [f64; 7],
}
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum Phase {
    Staged,
    Sleep,
}
impl Phase {
    pub fn duration(self) -> f64 {
        match self {
            Self::Staged => STAGED_POSE_S,
            Self::Sleep => SLEEP_POSE_S,
        }
    }
    pub fn target(self, targets: &Targets) -> [f64; 7] {
        match self {
            Self::Staged => targets.staged,
            Self::Sleep => targets.sleep,
        }
    }
}

impl Targets {
    /// Limits are those the controller actually enforces after any golden
    /// application. Six rotational joints use radians; axis 6 uses metres.
    ///
    /// The carriage rides at its measured value through takeover and the
    /// staged pose. `carriage_rest` is where it returns at Sleep: a
    /// qualified carriage's rest (the profile's `staged_positions[6]`), so a
    /// tripped, retracted pen lands closed; `None` keeps the measured value,
    /// which is all an unqualified carriage may be commanded.
    pub fn from_measured(
        measured: [f64; 7],
        staged_config: [f64; 7],
        limits: &[PositionLimit; 7],
        carriage_rest: Option<f64>,
    ) -> Result<Self> {
        validate_pose(&measured)?;
        validate_pose(&staged_config)?;
        validate_limits(limits)?;
        if carriage_rest.is_some_and(|rest| !rest.is_finite()) {
            return Err(Error("carriage rest must be finite".into()));
        }
        let takeover = std::array::from_fn(|i| {
            measured[i].clamp(limits[i].min + LIMIT_MARGIN, limits[i].max - LIMIT_MARGIN)
        });
        let mut staged = staged_config;
        staged[6] = takeover[6];
        let mut sleep = [0.0; 7];
        sleep[5] = staged[5];
        sleep[6] = carriage_rest.unwrap_or(takeover[6]);
        // The legacy routine lets the controller refuse these targets. Check
        // them before commanding instead, without clamping the intended pose.
        for pose in [&staged, &sleep] {
            if pose
                .iter()
                .zip(limits)
                .any(|(q, limit)| *q < limit.min || *q > limit.max)
            {
                return Err(Error("landing target outside controller limits".into()));
            }
        }
        Ok(Self {
            takeover,
            staged,
            sleep,
        })
    }
    /// Exactly the legacy final check: exclude the linear carriage from the
    /// rotational tolerance, but refuse non-finite feedback on every axis.
    pub fn verified(&self, measured: &[f64; 7]) -> Result<bool> {
        validate_pose(measured)?;
        Ok(measured[..6]
            .iter()
            .zip(&self.sleep[..6])
            .all(|(q, target)| (q - target).abs() <= LANDED_TOLERANCE_RAD))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> [PositionLimit; 7] {
        let mut limits = [PositionLimit {
            min: -3.0,
            max: 3.0,
        }; 7];
        limits[6] = PositionLimit {
            min: -0.006,
            max: 0.04,
        };
        limits
    }
    #[test]
    fn takeover_retains_retracted_carriage_and_cube_up_sleep() {
        let measured = [0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.032];
        let staged = [0.0, 0.2, -0.3, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.002];
        // An unqualified carriage keeps its measured value through every phase.
        let targets = Targets::from_measured(measured, staged, &limits(), None).unwrap();
        assert_eq!(targets.takeover, measured);
        assert_eq!(targets.staged[6], 0.032);
        assert_eq!(targets.sleep, [0.0, 0.0, 0.0, 0.0, 0.0, staged[5], 0.032]);
        let mut final_pose = targets.sleep;
        final_pose[6] = -0.004;
        assert!(targets.verified(&final_pose).unwrap());
        final_pose[2] = 0.201;
        assert!(!targets.verified(&final_pose).unwrap());
        final_pose[6] = f64::NAN;
        assert!(targets.verified(&final_pose).is_err());
        // A qualified carriage rides retracted to the staged pose and returns
        // to its rest at Sleep; a rest outside the live limits is refused.
        let landing = Targets::from_measured(measured, staged, &limits(), Some(staged[6])).unwrap();
        assert_eq!(landing.takeover, measured);
        assert_eq!(landing.staged[6], 0.032);
        assert_eq!(landing.sleep, [0.0, 0.0, 0.0, 0.0, 0.0, staged[5], 0.002]);
        assert!(Targets::from_measured(measured, staged, &limits(), Some(0.05)).is_err());
        assert!(Targets::from_measured(measured, staged, &limits(), Some(f64::NAN)).is_err());
    }
    #[test]
    fn boot_limit_requires_golden_and_hold_uses_reread_limits() {
        let mut measured = [0.0; 7];
        measured[6] = -0.0047;
        let mut boot = limits();
        boot[6].min = -0.004;
        assert!(needs_golden(&measured, &boot).unwrap());
        assert!(!needs_golden(&measured, &limits()).unwrap());
        let targets = Targets::from_measured(measured, [0.0; 7], &limits(), None).unwrap();
        assert_eq!(targets.takeover[6], measured[6]);
        // If golden application leaves a live limit unchanged, carry the
        // legacy clamp into every subsequent carriage target.
        let clamped = Targets::from_measured(measured, [0.0; 7], &boot, None).unwrap();
        assert_eq!(clamped.takeover[6], -0.004 + LIMIT_MARGIN);
        assert_eq!(clamped.takeover[6], clamped.sleep[6]);
    }
    #[test]
    fn invalid_limits_and_nonfinite_observations_never_create_targets() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut pose = [0.0; 7];
            pose[6] = bad;
            assert!(Targets::from_measured(pose, [0.0; 7], &limits(), None).is_err());
        }
        let mut invalid = limits();
        invalid[6].max = invalid[6].min;
        assert!(Targets::from_measured([0.0; 7], [0.0; 7], &invalid, None).is_err());
        let mut staged = [0.0; 7];
        staged[0] = 4.0;
        assert!(Targets::from_measured([0.0; 7], staged, &limits(), None).is_err());
    }
    #[test]
    fn healthy_vendor_strings_are_not_faults() {
        for raw in ["", "No error", " NONE ", "Error State: None"] {
            assert_eq!(controller_error(raw), "");
        }
        assert_eq!(controller_error(" motor 6 fault "), "motor 6 fault");
        assert_eq!(controller_error("No errors"), "No errors");
    }
}
