//! Carriage cap port from wxai_teleop.cpp. The reference policy is 800-sample
//! median baseline, slow baseline drift below 25% of cap, assessment below
//! 0.3 rad/s, and 40 consecutive over-cap/deflection ticks at 400 Hz.
use crate::{Error, Result};
#[derive(Debug, Clone)]
pub struct ContactCap {
    baseline_samples: Vec<f64>,
    baseline: Option<f64>,
    over_ticks: usize,
    pub cap_n: f64,
    pub deflect_m: f64,
}
impl Default for ContactCap {
    fn default() -> Self {
        Self {
            baseline_samples: Vec::with_capacity(800),
            baseline: None,
            over_ticks: 0,
            cap_n: 20.0,
            deflect_m: 0.002,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ContactObservation {
    pub effort_n: f64,
    pub position_m: f64,
    pub target_m: f64,
    pub arm_speed_rad_s: f64,
    pub aligning: bool,
    /// False while the carriage effort is known not to be a contact signal
    /// (a fixed carriage, whose effort swings with posture); the deflection
    /// screen remains the protection then.
    pub effort_assessable: bool,
}
#[derive(Debug, Clone, Copy)]
pub struct ContactEvaluation {
    pub armed: bool,
    /// The protection is judging: baseline armed and the arm slow enough.
    pub assessable: bool,
    /// That judgement counts the effort channel as contact (never for a
    /// fixed carriage); the deflection screen judges in every case.
    pub effort_assessable: bool,
    pub deflection_exceeded: bool,
    pub contact_n: f64,
    pub baseline_n: Option<f64>,
    pub deflection_m: f64,
    pub trip: bool,
}
impl ContactCap {
    /// Drop the armed baseline so the next 800 rest ticks arm a new one where
    /// the carriage now rests. Alignment never does this; only the completion
    /// of an air carriage transfer does, whose free-air sweep the session
    /// admitted with a model clearance receipt before streaming it: the
    /// carriage's holding effort at its new position is the only honest
    /// zero for the strokes that follow, and a baseline armed 30 mm away
    /// read 8-13 N of contact at rest on 2026-09-16.
    pub fn rearm(&mut self) {
        self.baseline = None;
        self.baseline_samples.clear();
        self.over_ticks = 0;
    }
    pub fn observe(&mut self, o: ContactObservation) -> Result<ContactEvaluation> {
        if [
            o.effort_n,
            o.position_m,
            o.target_m,
            o.arm_speed_rad_s,
            self.cap_n,
            self.deflect_m,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || self.cap_n <= 0.0
            || self.cap_n > 20.0
            || self.deflect_m <= 0.0
            || self.deflect_m > 0.002
            || o.arm_speed_rad_s < 0.0
        {
            return Err(Error(
                "invalid carriage contact measurement or widened cap".into(),
            ));
        }
        // A rest median must be contiguous and must exclude commanded motion.
        // Once armed, alignment never clears or replaces the protective baseline.
        if self.baseline.is_none() && o.aligning {
            self.baseline_samples.clear();
        }
        if self.baseline.is_none() && !o.aligning {
            self.baseline_samples.push(o.effort_n);
            if self.baseline_samples.len() >= 800 {
                self.baseline_samples.sort_by(f64::total_cmp);
                self.baseline = Some(self.baseline_samples[self.baseline_samples.len() / 2]);
                self.baseline_samples.clear();
            }
        }
        let mut contact = (o.effort_n - self.baseline.unwrap_or(0.0)).abs();
        let assessable = self.baseline.is_some() && o.arm_speed_rad_s < 0.3;
        let effort_assessable = assessable && o.effort_assessable;
        if effort_assessable && contact < self.cap_n * 0.25 {
            let baseline = self.baseline.as_mut().unwrap();
            *baseline += (o.effort_n - *baseline) * (0.0025 / 10.0);
            contact = (o.effort_n - *baseline).abs();
        }
        let deflection = o.position_m - o.target_m;
        let pushed = (effort_assessable && contact > self.cap_n) || deflection > self.deflect_m;
        self.over_ticks = if pushed {
            self.over_ticks.saturating_add(1)
        } else {
            0
        };
        Ok(ContactEvaluation {
            armed: self.baseline.is_some(),
            assessable,
            effort_assessable,
            deflection_exceeded: deflection > self.deflect_m,
            contact_n: contact,
            baseline_n: self.baseline,
            deflection_m: deflection,
            trip: self.over_ticks >= 40,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn observation(effort_n: f64) -> ContactObservation {
        ContactObservation {
            effort_n,
            position_m: 0.002,
            target_m: 0.002,
            arm_speed_rad_s: 0.0,
            aligning: false,
                effort_assessable: true,
        }
    }
    #[test]
    fn baseline_and_debounce_match_cpp() {
        let mut cap = ContactCap::default();
        for _ in 0..799 {
            assert!(!cap.observe(observation(3.0)).unwrap().armed);
        }
        assert!(cap.observe(observation(3.0)).unwrap().armed);
        for _ in 0..39 {
            assert!(!cap.observe(observation(24.0)).unwrap().trip);
        }
        assert!(cap.observe(observation(24.0)).unwrap().trip);
        assert!(!cap.observe(observation(3.0)).unwrap().trip);
        for _ in 0..40 {
            assert!(
                !cap.observe(ContactObservation {
                    arm_speed_rad_s: 0.3,
                    ..observation(100.0)
                })
                .unwrap()
                .trip
            );
        }
    }
    #[test]
    fn a_fixed_carriage_is_judged_by_deflection_while_its_effort_swings() {
        let mut cap = ContactCap::default();
        let fixed = |effort_n: f64| ContactObservation {
            effort_assessable: false,
            ..observation(effort_n)
        };
        for _ in 0..800 {
            cap.observe(fixed(-110.0)).unwrap();
        }
        for _ in 0..80 {
            let e = cap.observe(fixed(-85.0)).unwrap();
            assert!(e.armed && e.assessable && !e.effort_assessable && !e.trip);
            assert_eq!(e.baseline_n, Some(-110.0));
            assert!((e.contact_n - 25.0).abs() < 1e-9);
        }
        for i in 0..40 {
            let e = cap
                .observe(ContactObservation {
                    position_m: 0.005,
                    ..fixed(-110.0)
                })
                .unwrap();
            assert!(e.deflection_exceeded);
            assert_eq!(e.trip, i == 39);
        }
    }
    #[test]
    fn initial_rest_median_excludes_alignment_without_resetting_an_armed_cap() {
        let mut cap = ContactCap::default();
        for _ in 0..600 {
            assert!(!cap.observe(observation(-7.0)).unwrap().armed);
        }
        for _ in 0..1600 {
            assert!(
                !cap.observe(ContactObservation {
                    aligning: true,
                    ..observation(-7.0)
                })
                .unwrap()
                .armed
            );
        }
        for _ in 0..799 {
            assert!(!cap.observe(observation(3.0)).unwrap().armed);
        }
        let settled = cap.observe(observation(3.0)).unwrap();
        assert_eq!(settled.baseline_n, Some(3.0));
        for i in 0..40 {
            let observed = cap
                .observe(ContactObservation {
                    aligning: true,
                    ..observation(24.0)
                })
                .unwrap();
            assert!(observed.armed);
            assert_eq!(observed.baseline_n, Some(3.0));
            assert_eq!(observed.trip, i == 39);
        }
    }

    #[test]
    fn rearm_collects_a_fresh_baseline_where_the_carriage_now_rests() {
        let mut cap = ContactCap::default();
        for _ in 0..800 {
            cap.observe(observation(-7.0)).unwrap();
        }
        let before = cap.observe(observation(1.2)).unwrap();
        assert!(before.armed && (before.contact_n - 8.2).abs() < 1e-9);
        cap.rearm();
        let unarmed = cap.observe(observation(1.2)).unwrap();
        assert!(!unarmed.armed && !unarmed.assessable && !unarmed.trip);
        for _ in 0..799 {
            cap.observe(observation(1.2)).unwrap();
        }
        let after = cap.observe(observation(1.2)).unwrap();
        assert!(after.armed && after.contact_n.abs() < 1e-9);
        assert!((after.baseline_n.unwrap() - 1.2).abs() < 1e-9);
    }

    #[test]
    fn deflection_does_not_wait_for_baseline_and_nonfinite_refuses() {
        let mut cap = ContactCap::default();
        for _ in 0..39 {
            assert!(
                !cap.observe(ContactObservation {
                    position_m: 0.0041,
                    ..observation(0.0)
                })
                .unwrap()
                .trip
            );
        }
        assert!(
            cap.observe(ContactObservation {
                position_m: 0.0041,
                ..observation(0.0)
            })
            .unwrap()
            .trip
        );
        assert!(cap.observe(observation(f64::NAN)).is_err());
    }
}
