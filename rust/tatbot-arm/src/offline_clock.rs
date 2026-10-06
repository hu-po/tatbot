//! A controller clock advanced only by an offline simulation's caller.
//!
//! Host time bounds transport/liveness waits; it never advances the simulated
//! world. A worker acknowledges a requested tick only after stepping its world
//! and publishing feedback. Live workers do not use this clock.

use crate::{Error, Result};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

pub const CONTROL_PERIOD: Duration = Duration::from_micros(2500);
const PERIOD_NS: u64 = 2_500_000;

#[derive(Debug, Default)]
struct Permits {
    requested: u64,
    completed: u64,
    closed: Option<String>,
}

#[derive(Debug)]
pub struct OfflineClock {
    origin: Instant,
    epoch_ns: u64,
    elapsed_ns: AtomicU64,
    worker_claimed: AtomicBool,
    permits: Mutex<Permits>,
    changed: Condvar,
    driver: Mutex<()>,
}

pub(crate) struct WorldOwner(Arc<OfflineClock>);

impl Drop for WorldOwner {
    fn drop(&mut self) {
        self.0.close("offline arm worker stopped");
    }
}

impl OfflineClock {
    pub fn new(epoch_ns: u64) -> Result<Arc<Self>> {
        if epoch_ns == 0 {
            return Err(Error(
                "offline clock needs an explicit nonzero timestamp epoch".into(),
            ));
        }
        Ok(Arc::new(Self {
            origin: Instant::now(),
            epoch_ns,
            elapsed_ns: AtomicU64::new(0),
            worker_claimed: AtomicBool::new(false),
            permits: Mutex::new(Permits::default()),
            changed: Condvar::new(),
            driver: Mutex::new(()),
        }))
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_ns.load(Ordering::Acquire))
    }

    pub fn timestamp_ns(&self) -> u64 {
        self.epoch_ns + self.elapsed_ns.load(Ordering::Acquire)
    }

    /// Monotonic time within this world, suitable only for comparing events
    /// stamped by this same clock. It is independent of host elapsed time.
    pub fn now(&self) -> Instant {
        self.origin + self.elapsed()
    }

    pub(crate) fn running(&self) -> bool {
        self.permits.lock().unwrap().closed.is_none()
    }

    pub(crate) fn claim_worker(&self) -> Result<()> {
        if !self.running() {
            return Err(Error("offline clock is closed".into()));
        }
        self.worker_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error("offline clock already has a world owner".into()))?;
        Ok(())
    }

    pub(crate) fn owner_guard(self: &Arc<Self>) -> WorldOwner {
        WorldOwner(self.clone())
    }

    /// Advance at least `duration`, rounded up to whole controller ticks.
    /// The independent wall timeout detects a lost worker/transport; it does
    /// not alter simulation time or make an unfinished step successful.
    pub fn advance(&self, duration: Duration, wall_timeout: Duration) -> Result<()> {
        let _driver = self
            .driver
            .try_lock()
            .map_err(|_| Error("offline clock already has a driver".into()))?;
        let ticks = duration.as_nanos().div_ceil(u128::from(PERIOD_NS));
        let ticks = u64::try_from(ticks).map_err(|_| Error("offline duration overflow".into()))?;
        if ticks == 0 || wall_timeout.is_zero() {
            return Err(Error(
                "offline advance needs positive duration and wall timeout".into(),
            ));
        }
        let deadline = Instant::now()
            .checked_add(wall_timeout)
            .ok_or_else(|| Error("offline wall timeout overflow".into()))?;
        let mut permits = self.permits.lock().unwrap();
        if let Some(reason) = &permits.closed {
            return Err(Error(reason.clone()));
        }
        let target = permits
            .completed
            .checked_add(ticks)
            .ok_or_else(|| Error("offline tick overflow".into()))?;
        target
            .checked_mul(PERIOD_NS)
            .and_then(|n| n.checked_add(self.epoch_ns))
            .ok_or_else(|| Error("offline timestamp overflow".into()))?;
        permits.requested = target;
        self.changed.notify_all();
        while permits.completed < target {
            if let Some(reason) = &permits.closed {
                return Err(Error(reason.clone()));
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                permits.closed = Some("offline worker acknowledgement timed out".into());
                self.changed.notify_all();
                return Err(Error("offline worker acknowledgement timed out".into()));
            };
            permits = self.changed.wait_timeout(permits, remaining).unwrap().0;
        }
        if let Some(reason) = &permits.closed {
            return Err(Error(reason.clone()));
        }
        Ok(())
    }

    pub(crate) fn wait_tick(
        &self,
        shutdown: &AtomicBool,
        mut emergency: impl FnMut() -> bool,
    ) -> bool {
        let mut permits = self.permits.lock().unwrap();
        loop {
            if shutdown.load(Ordering::Acquire) || permits.closed.is_some() {
                return false;
            }
            if permits.requested > permits.completed || emergency() {
                return true;
            }
            // The existing stop flag can be raised without owning this clock.
            // Poll it while paused; no simulation tick elapses during this wait.
            permits = self
                .changed
                .wait_timeout(permits, Duration::from_millis(5))
                .unwrap()
                .0;
        }
    }

    /// The world has advanced; source time now describes its new measurements.
    pub(crate) fn stepped(&self) -> Result<()> {
        let previous = self.elapsed_ns.load(Ordering::Acquire);
        let next = previous
            .checked_add(PERIOD_NS)
            .filter(|n| n.checked_add(self.epoch_ns).is_some())
            .ok_or_else(|| Error("offline timestamp overflow".into()))?;
        self.elapsed_ns.store(next, Ordering::Release);
        Ok(())
    }

    /// Called after publishing the feedback for the completed world step.
    pub(crate) fn acknowledge(&self) {
        let mut permits = self.permits.lock().unwrap();
        permits.completed = self.elapsed_ns.load(Ordering::Acquire) / PERIOD_NS;
        // An emergency may wake a paused worker without a caller permit.
        permits.requested = permits.requested.max(permits.completed);
        self.changed.notify_all();
    }

    pub(crate) fn close(&self, reason: impl Into<String>) {
        let mut permits = self.permits.lock().unwrap();
        permits.closed.get_or_insert_with(|| reason.into());
        self.changed.notify_all();
    }
}

#[derive(Clone, Debug, Default)]
pub enum Clock {
    #[default]
    Live,
    Offline(Arc<OfflineClock>),
}

impl Clock {
    pub fn now(&self) -> Instant {
        match self {
            Self::Live => Instant::now(),
            Self::Offline(clock) => clock.now(),
        }
    }

    pub fn since(&self, start: Instant) -> Duration {
        self.now().saturating_duration_since(start)
    }

    pub(crate) fn running(&self) -> bool {
        match self {
            Self::Live => true,
            Self::Offline(clock) => clock.running(),
        }
    }

    pub fn timestamp_ns(&self) -> Option<u64> {
        match self {
            Self::Live => None,
            Self::Offline(clock) => Some(clock.timestamp_ns()),
        }
    }
}
