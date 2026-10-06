//! Measured-motion guard: a joint-velocity cap and a rolling overforce window.
use std::collections::VecDeque;
#[derive(Debug, Clone, PartialEq)]
pub struct Trip {
    pub reason: &'static str,
    pub joint: usize,
    pub observed: f64,
    pub limit: f64,
}
pub struct MotionGuard {
    pub velocity_limit: f64,
    pub overforce_limit: f64,
    pub window_s: f64,
    pub fraction: f64,
    pub minimum_samples: usize,
    samples: VecDeque<(f64, bool)>,
    last_s: Option<f64>,
}
impl MotionGuard {
    pub fn new(
        velocity_limit: f64,
        overforce_limit: f64,
        window_s: f64,
        fraction: f64,
        minimum_samples: usize,
    ) -> Self {
        Self {
            velocity_limit,
            overforce_limit,
            window_s,
            fraction,
            minimum_samples,
            samples: VecDeque::new(),
            last_s: None,
        }
    }
    pub fn reset(&mut self) {
        self.samples.clear();
        self.last_s = None;
    }
    pub fn observe(&mut self, now_s: f64, velocities: &[f64], efforts: &[f64]) -> Option<Trip> {
        let trip = |reason, joint, observed, limit| {
            Some(Trip {
                reason,
                joint,
                observed,
                limit,
            })
        };
        if !now_s.is_finite() || self.last_s.is_some_and(|last| now_s <= last) {
            return trip("invalid_clock", 0, now_s, 0.0);
        }
        if !self.velocity_limit.is_finite()
            || self.velocity_limit <= 0.0
            || !self.overforce_limit.is_finite()
            || self.overforce_limit <= 0.0
            || !self.window_s.is_finite()
            || self.window_s <= 0.0
            || !self.fraction.is_finite()
            || self.fraction <= 0.0
            || self.fraction > 1.0
            || self.minimum_samples == 0
        {
            return trip("invalid_limits", 0, 0.0, 0.0);
        }
        if velocities.is_empty() || velocities.len() != efforts.len() {
            return trip(
                "telemetry_width",
                0,
                velocities.len() as f64,
                efforts.len() as f64,
            );
        }
        for (i, (v, e)) in velocities.iter().zip(efforts).enumerate() {
            if !v.is_finite() || !e.is_finite() {
                return trip("non_finite_telemetry", i, *v, 0.0);
            }
        }
        let fastest = (1..velocities.len()).fold(0, |a, b| {
            if velocities[b].abs() > velocities[a].abs() {
                b
            } else {
                a
            }
        });
        if velocities[fastest].abs() > self.velocity_limit {
            return trip(
                "measured_velocity",
                fastest,
                velocities[fastest],
                self.velocity_limit,
            );
        }
        let loaded = (1..efforts.len()).fold(0, |a, b| {
            if efforts[b].abs() > efforts[a].abs() {
                b
            } else {
                a
            }
        });
        self.last_s = Some(now_s);
        self.samples
            .push_back((now_s, efforts[loaded].abs() > self.overforce_limit));
        while self
            .samples
            .front()
            .is_some_and(|s| s.0 < now_s - self.window_s)
        {
            self.samples.pop_front();
        }
        let ready = self.samples.len() >= self.minimum_samples
            && now_s - self.samples.front().unwrap().0 >= self.window_s * 0.8;
        let fraction =
            self.samples.iter().filter(|s| s.1).count() as f64 / self.samples.len() as f64;
        if ready && fraction >= self.fraction {
            return trip(
                "rolling_overforce",
                loaded,
                efforts[loaded],
                self.overforce_limit,
            );
        }
        None
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpp_window_readiness_and_reset() {
        let mut g = MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8);
        for i in 0..160 {
            assert!(
                g.observe(i as f64 * 0.0025, &[0.0; 6], &[10.0; 6])
                    .is_none()
            );
        }
        assert_eq!(
            g.observe(0.4, &[0.0; 6], &[10.0; 6]).unwrap().reason,
            "rolling_overforce"
        );
        g.reset();
        assert!(g.observe(0.5, &[1.0; 6], &[9.0; 6]).is_none());
        assert_eq!(
            g.observe(0.6, &[1.0001; 6], &[0.0; 6]).unwrap().reason,
            "measured_velocity"
        );
    }
    #[test]
    fn telemetry_and_clock_fail_closed() {
        let mut g = MotionGuard::new(1.0, 9.0, 0.5, 0.5, 8);
        assert_eq!(g.observe(0.0, &[], &[]).unwrap().reason, "telemetry_width");
        assert_eq!(
            g.observe(0.0, &[0.0], &[f64::NAN]).unwrap().reason,
            "non_finite_telemetry"
        );
        assert!(g.observe(0.0, &[0.0], &[0.0]).is_none());
        assert_eq!(
            g.observe(0.0, &[0.0], &[0.0]).unwrap().reason,
            "invalid_clock"
        );
    }
}
