//! Seven-axis position and velocity feed-forward samples. Limits are the same
//! checked-in constants used by the C++ planner; this is not motion authority.
use crate::{Error, Result, Sample};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointSample {
    pub positions: [f64; 7],
    pub velocities: [f64; 7],
    pub dt_s: f64,
    pub pen: bool,
}
impl JointSample {
    pub(crate) fn angular(&self) -> Sample {
        Sample {
            joints: self.positions[..6].to_vec(),
            dt_s: self.dt_s,
            contact: if self.pen { 1.0 } else { 0.0 },
        }
    }
    /// Bound the drift between the planned seed and the feedback read immediately
    /// before dispatch. This is a position tolerance, not one planner tick of
    /// velocity: the former one-tick bound allowed 2.5 um of carriage drift, below
    /// the carriage encoder's resting noise, and refused a pen-up scan whose seed
    /// had drifted 3 um (run 9e00). A pen-down seed is held tighter because the
    /// carriage position is the pen pressure.
    pub fn check_seed(&self, requested: &[f64; 7], measured: &crate::Measured) -> Result<[f64; 7]> {
        if requested.iter().any(|q| !q.is_finite())
            || measured.joints.len() != 6
            || measured.mode != crate::Mode::Position
            || !crate::recovery::controller_error(&measured.error).is_empty()
        {
            return Err(Error("invalid stream seed/controller".into()));
        }
        let carriage = measured
            .carriage
            .as_ref()
            .ok_or_else(|| Error("stream seed carriage unavailable".into()))?;
        let mut actual = [0.0; 7];
        actual[..6].copy_from_slice(&measured.joints);
        actual[6] = carriage.position_m;
        if actual.iter().any(|q| !q.is_finite()) {
            return Err(Error("invalid stream seed/controller".into()));
        }
        let (joint_limit, carriage_limit) = if self.pen {
            (Self::SEED_DRIFT_PEN_DOWN_RAD, Self::SEED_DRIFT_PEN_DOWN_M)
        } else {
            (Self::SEED_DRIFT_PEN_UP_RAD, Self::SEED_DRIFT_PEN_UP_M)
        };
        let joint_drift = requested[..6]
            .iter()
            .zip(&actual[..6])
            .map(|(q, p)| (q - p).abs())
            .fold(0.0, f64::max);
        let carriage_drift = (requested[6] - actual[6]).abs();
        if joint_drift > joint_limit || carriage_drift > carriage_limit {
            return Err(Error(format!(
                "stream seed moved: joints {joint_drift:.5} rad (limit {joint_limit}), carriage {:.4} mm (limit {} mm); requested={requested:?}; actual={actual:?}",
                carriage_drift * 1e3,
                carriage_limit * 1e3
            )));
        }
        Ok(actual)
    }
    /// Seed drift the dispatch accepts between the plan and fresh feedback.
    /// The arm settles 1-2 mrad from a commanded pose under gravity, and a
    /// scan leg seeds from the previous leg's planned endpoint, so these are
    /// pose tolerances of the "still the same pose" kind, not encoder noise.
    /// Pen down, the carriage is the pen pressure and the contact cap still
    /// bounds it.
    pub const SEED_DRIFT_PEN_UP_RAD: f64 = 0.02;
    pub const SEED_DRIFT_PEN_UP_M: f64 = 0.002;
    pub const SEED_DRIFT_PEN_DOWN_RAD: f64 = 0.01;
    pub const SEED_DRIFT_PEN_DOWN_M: f64 = 0.0005;

    pub fn validate(&self, previous: Option<&Self>) -> Result<()> {
        self.validate_mode(previous, false)
    }

    /// Explicit air transfer only: six rotary axes stay fixed while the
    /// carriage moves through the existing pen-up travel range. This never
    /// authorizes paper contact and does not change the normal stream bounds.
    pub(crate) fn validate_transfer(&self, previous: Option<&Self>) -> Result<()> {
        if self.pen
            || self.velocities[..6].iter().any(|v| *v != 0.0)
            || previous.is_some_and(|old| old.positions[..6] != self.positions[..6])
        {
            return Err(Error(
                "air carriage transfer requires fixed rotary axes and pen up".into(),
            ));
        }
        self.validate_mode(previous, true)
    }

    pub fn carriage_transfer(seed: [f64; 7], target_m: f64) -> Result<Vec<Self>> {
        let c: serde_json::Value =
            serde_json::from_str(include_str!("../../../config/motion_constants.json"))
                .map_err(|e| Error(e.to_string()))?;
        let number = |group: &str, key: &str| {
            c[group][key]
                .as_f64()
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| Error("missing carriage transfer constant".into()))
        };
        let min = number("carriage_ik", "min_m")?;
        let max = number("carriage_ik", "max_m")?;
        if seed.iter().any(|v| !v.is_finite())
            || !target_m.is_finite()
            || !(min..=max).contains(&target_m)
        {
            return Err(Error(
                "carriage transfer must end inside the drawing reserve".into(),
            ));
        }
        let period = c["period_s"]
            .as_f64()
            .ok_or_else(|| Error("missing period".into()))?;
        let delta = target_m - seed[6];
        let duration = (1.875 * delta.abs() / number("planner", "max_carriage_velocity_m_s")?)
            .max((6.0 * delta.abs() / number("planner", "max_carriage_acceleration_m_s2")?).sqrt())
            .max(1.0);
        let ticks = (duration / period).ceil() as usize;
        let max_ticks = c["planner"]["max_ticks"]
            .as_u64()
            .ok_or_else(|| Error("missing transfer tick budget".into()))?
            as usize;
        if ticks >= max_ticks || !duration.is_finite() || period <= 0.0 {
            return Err(Error("carriage transfer exceeds stream budget".into()));
        }
        let duration = ticks as f64 * period;
        let mut samples = Vec::with_capacity(ticks + 1);
        for i in 0..=ticks {
            let u = i as f64 / ticks as f64;
            let mut positions = seed;
            positions[6] += delta * (10.0 * u.powi(3) - 15.0 * u.powi(4) + 6.0 * u.powi(5));
            if i == ticks {
                positions[6] = target_m;
            }
            let mut velocities = [0.0; 7];
            velocities[6] =
                delta * (30.0 * u.powi(2) - 60.0 * u.powi(3) + 30.0 * u.powi(4)) / duration;
            let sample = Self {
                positions,
                velocities,
                dt_s: period,
                pen: false,
            };
            sample.validate_transfer(samples.last())?;
            samples.push(sample);
        }
        Ok(samples)
    }

    fn validate_mode(&self, previous: Option<&Self>, transfer: bool) -> Result<()> {
        static CONSTANTS: OnceLock<serde_json::Value> = OnceLock::new();
        let c = CONSTANTS.get_or_init(|| {
            serde_json::from_str(include_str!("../../../config/motion_constants.json"))
                .expect("checked draw constants")
        });
        let number = |value: &serde_json::Value| {
            value
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| Error("missing stream constant".into()))
        };
        let period = number(&c["period_s"])?;
        let cap = number(
            &c["planner"][if self.pen {
                "max_joint_velocity_rad_s"
            } else {
                "max_joint_velocity_pen_up_rad_s"
            }],
        )?;
        let carriage_speed = number(&c["planner"]["max_carriage_velocity_m_s"])?;
        let carriage_acceleration = number(&c["planner"]["max_carriage_acceleration_m_s2"])?;
        // A pen-up scan may hold measured rest outside the drawing compliance
        // range. Preserve the measurement exactly; contact or carriage movement
        // still requires the original drawing bounds.
        let fixed_scan = !self.pen
            && self.velocities[6] == 0.0
            && previous.is_none_or(|old| {
                !old.pen && old.velocities[6] == 0.0 && old.positions[6] == self.positions[6]
            });
        let min = if fixed_scan || transfer {
            number(&c["carriage_ik"]["pen_up_min_m"])?
        } else {
            number(&c["carriage_ik"]["min_m"])?
        };
        let max = number(
            &c["carriage_ik"][if fixed_scan || transfer {
                "pen_up_max_m"
            } else {
                "max_m"
            }],
        )?;
        let old_velocity = previous.map_or(0.0, |s| s.velocities[6]);
        if let Some(old) = previous
            && ((self.positions[6] - old.positions[6]).abs() / period > carriage_speed + 1e-12
                || self.positions[..6]
                    .iter()
                    .zip(&old.positions[..6])
                    .any(|(q, p)| (q - p).abs() / period > cap + 1e-12))
        {
            return Err(Error(
                "seven-axis commanded step exceeds planner velocity".into(),
            ));
        }
        if self.dt_s != period
            || period <= 0.0
            || cap <= 0.0
            || self
                .positions
                .iter()
                .chain(&self.velocities)
                .any(|q| !q.is_finite())
            || self.velocities[..6].iter().any(|v| v.abs() > cap + 1e-12)
            || !(min..=max).contains(&self.positions[6])
            || self.velocities[6].abs() > carriage_speed + 1e-12
            || (self.velocities[6] - old_velocity).abs() / period > carriage_acceleration + 1e-12
        {
            return Err(Error(
                "seven-axis stream constants/velocity/carriage invariant".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn air_transfer_preserves_rotary_axes_and_original_contact_bounds() {
        for rest in [-0.0047, 0.0301, 0.002] {
            let seed = [0.1, 0.2, 0.3, 0.1, 0.2, 1.5, rest];
            let samples = JointSample::carriage_transfer(seed, 0.002).unwrap();
            assert_eq!(samples[0].positions, seed);
            assert!((samples.last().unwrap().positions[6] - 0.002).abs() < 1e-14);
            assert!(
                samples
                    .iter()
                    .all(|s| s.positions[..6] == seed[..6] && !s.pen)
            );
            for (i, sample) in samples.iter().enumerate() {
                sample
                    .validate_transfer(i.checked_sub(1).map(|j| &samples[j]))
                    .unwrap();
            }
            if rest != 0.002 {
                assert!(samples[1].validate(Some(&samples[0])).is_err());
            }
            let mut invalid = samples[1].clone();
            invalid.pen = true;
            assert!(invalid.validate_transfer(Some(&samples[0])).is_err());
            invalid = samples[1].clone();
            invalid.positions[0] += 1e-9;
            assert!(invalid.validate_transfer(Some(&samples[0])).is_err());
        }
        for target in [0.0, 0.03, f64::NAN] {
            assert!(JointSample::carriage_transfer([0.0; 7], target).is_err());
        }
        let mut outside = [0.0; 7];
        outside[6] = 0.035;
        assert!(JointSample::carriage_transfer(outside, 0.002).is_err());
    }
    #[test]
    fn resting_scan_keeps_carriage_fixed_and_refuses_contact_or_motion() {
        for rest in [
            0.0,
            -1.7546117305755615e-6,
            -0.006,
            -0.004719,
            0.03056,
            0.032,
            0.034,
        ] {
            let a = JointSample {
                positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rest],
                velocities: [0.0; 7],
                dt_s: 0.0025,
                pen: false,
            };
            a.validate(None).unwrap();
            a.validate(Some(&a)).unwrap();
            let mut contact = a.clone();
            contact.pen = true;
            assert!(contact.validate(Some(&a)).is_err());
            let mut moving = a.clone();
            moving.positions[6] += 1e-7;
            assert!(moving.validate(Some(&a)).is_err());
            moving = a.clone();
            moving.velocities[6] = 1e-7;
            assert!(moving.validate(None).is_err());
            // A stationary carriage may rest at the controller's -6 mm limit.
            let mut resting = a.clone();
            resting.positions[6] = -0.0012;
            resting.validate(None).unwrap();
            let mut outside = a.clone();
            outside.positions[6] = -0.0061;
            assert!(outside.validate(None).is_err());
            outside.positions[6] = 0.0341;
            assert!(outside.validate(None).is_err());
        }
    }

    #[test]
    #[allow(clippy::approx_constant)] // Measured encoder values, not mathematical constants.
    fn seed_check_tolerates_encoder_noise_and_refuses_real_drift() {
        let measured = |joints: [f64; 6], carriage_m: f64| crate::Measured {
            carriage: Some(crate::CarriageMeasured {
                position_m: carriage_m,
                target_m: Some(carriage_m),
                effort_n: 5.0,
            }),
            joints: joints.to_vec(),
            velocities: vec![0.0; 6],
            efforts: vec![0.0; 6],
            dynamics: None,
            mode: crate::Mode::Position,
            error: String::new(),
        };
        let requested = [0.001, 0.0998, 0.1001, -0.0124, -0.0006, 1.5707, -4.6e-5];
        let pen_up = JointSample {
            positions: requested,
            velocities: [0.0; 7],
            dt_s: 0.0025,
            pen: false,
        };
        // Run 9e00: 0.4 mrad on four joints and 3.3 um on the carriage is rest noise.
        let noisy = measured([0.001, 0.0994, 0.1001, -0.0128, -0.0010, 1.5703], -4.93e-5);
        assert_eq!(pen_up.check_seed(&requested, &noisy).unwrap()[6], -4.93e-5);
        let moved_joint = measured(
            [0.001, 0.0998, 0.1001, -0.0124, -0.0006, 1.5707 + 0.03],
            -4.6e-5,
        );
        assert!(pen_up.check_seed(&requested, &moved_joint).is_err());
        let moved_carriage = measured(
            [0.001, 0.0998, 0.1001, -0.0124, -0.0006, 1.5707],
            -4.6e-5 + 0.003,
        );
        assert!(pen_up.check_seed(&requested, &moved_carriage).is_err());
        // Pen down: the carriage is the pressure, so the same 0.3 mm is a refusal.
        let pen_down = JointSample {
            pen: true,
            ..pen_up.clone()
        };
        let slight = measured(
            [0.001, 0.0998, 0.1001, -0.0124, -0.0006, 1.5707],
            -4.6e-5 + 0.0008,
        );
        assert!(pen_up.check_seed(&requested, &slight).is_ok());
        assert!(pen_down.check_seed(&requested, &slight).is_err());
        let mut idle = noisy.clone();
        idle.mode = crate::Mode::Idle;
        assert!(pen_up.check_seed(&requested, &idle).is_err());
    }

    #[test]
    fn rejects_carriage_steps_and_velocity_acceleration_from_shared_constants() {
        let a = JointSample {
            positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
            velocities: [0.0; 7],
            dt_s: 0.0025,
            pen: true,
        };
        a.validate(None).unwrap();
        for case in 0..5 {
            let mut b = a.clone();
            match case {
                0 => b.positions[6] += 0.00001,
                1 => b.velocities[6] = 0.0001,
                2 => b.velocities[0] = 0.376,
                3 => b.positions[0] = 0.001,
                _ => b.positions[6] = f64::NAN,
            }
            assert!(b.validate(Some(&a)).is_err(), "case {case}");
        }
    }
}
