//! Port of recovery.TrackingWatchdog: high error alone is not a fault.
//! A controller must also make less than 0.05 rad progress over >2 seconds.
use crate::{Error, Result};

#[derive(Default)]
pub struct TrackingWatchdog {
    anchor: Option<(f64, Vec<f64>)>,
    last_time: Option<f64>,
}
impl TrackingWatchdog {
    pub fn reset(&mut self) {
        self.anchor = None;
        self.last_time = None;
    }
    pub fn observe(&mut self, target: &[f64], measured: &[f64], now: f64) -> Result<()> {
        if target.is_empty()
            || target.len() != measured.len()
            || target.iter().chain(measured).any(|x| !x.is_finite())
            || !now.is_finite()
            || self.last_time.is_some_and(|last| now < last)
        {
            return Err(Error("invalid tracking watchdog input".into()));
        }
        self.last_time = Some(now);
        let error = target
            .iter()
            .zip(measured)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if error < 0.35 {
            self.anchor = None;
            return Ok(());
        }
        let Some((since, anchor)) = &self.anchor else {
            self.anchor = Some((now, measured.to_vec()));
            return Ok(());
        };
        if anchor.len() != measured.len() {
            return Err(Error("tracking watchdog telemetry width changed".into()));
        }
        if now - since <= 2.0 {
            return Ok(());
        }
        let progress = measured
            .iter()
            .zip(anchor)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if progress >= 0.05 {
            self.anchor = Some((now, measured.to_vec()));
            return Ok(());
        }
        Err(Error(format!(
            "tracking watchdog: error {error:.3} rad for {:.3} s, progress {progress:.3} rad",
            now - since
        )))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_threshold_window_and_progress_boundaries() {
        let mut w = TrackingWatchdog::default();
        w.observe(&[0.35], &[0.0], 0.0).unwrap();
        w.observe(&[0.35], &[0.0], 2.0).unwrap();
        assert!(w.observe(&[0.35], &[0.0], 2.001).is_err());
        w.reset();
        w.observe(&[0.5], &[0.0], 0.0).unwrap();
        w.observe(&[0.55], &[0.05], 2.5).unwrap();
        w.observe(&[0.55], &[0.05], 4.5).unwrap();
        assert!(w.observe(&[0.55], &[0.05], 4.501).is_err());
    }
    #[test]
    fn small_error_resets_window_and_bad_telemetry_refuses() {
        let mut w = TrackingWatchdog::default();
        w.observe(&[0.5], &[0.0], 0.0).unwrap();
        w.observe(&[0.34], &[0.0], 2.1).unwrap();
        w.observe(&[0.5], &[0.0], 5.0).unwrap();
        w.observe(&[0.5], &[0.0], 7.0).unwrap();
        assert!(w.observe(&[0.5], &[0.0], 7.1).is_err());
        assert!(w.observe(&[0.5], &[f64::NAN], 8.0).is_err());
        assert!(w.observe(&[], &[], 8.0).is_err());
    }
}
