//! Process lifetime ownership lease. Never unlink a locked inode. The hardware
//! launcher shares this contract with C++ teleop and LeRobot driver owners.
use crate::{Error, Result};
use std::{
    fs::{File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::Path,
};
pub const HARDWARE_LEASE: &str = "/tmp/tatbot-arm-driver.lock";

/// Proof of the fleet-wide lock, distinct from arbitrary mock/test leases.
/// Share one Arc between native arms owned by the same session.
pub struct HardwareLease {
    _lease: DriverLease,
}
impl HardwareLease {
    pub fn acquire() -> Result<Self> {
        Ok(Self {
            _lease: DriverLease::acquire(Path::new(HARDWARE_LEASE))?,
        })
    }
    #[cfg(all(test, feature = "trossen"))]
    pub(crate) fn for_test(path: &Path) -> Self {
        Self {
            _lease: DriverLease::acquire(path).unwrap(),
        }
    }
}

pub struct DriverLease {
    _file: File,
}
impl DriverLease {
    pub fn acquire(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(path)
            .map_err(|e| Error(format!("driver lease: {e}")))?;
        let metadata = file.metadata().map_err(|e| Error(e.to_string()))?;
        // SAFETY: getuid has no pointer arguments or preconditions.
        let uid = unsafe { libc::getuid() };
        if !metadata.is_file() || metadata.uid() != uid {
            return Err(Error(
                "driver lease must be a regular file owned by this user".into(),
            ));
        }
        // SAFETY: fd belongs to file and remains open for this object's lifetime.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(Error(format!(
                "driver busy: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self { _file: file })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusive_until_drop_and_rejects_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("arm.lock");
        let lease = DriverLease::acquire(&path).unwrap();
        assert!(DriverLease::acquire(&path).is_err());
        let output = std::process::Command::new("flock")
            .args(["-n"])
            .arg(&path)
            .arg("true")
            .status()
            .unwrap();
        assert!(!output.success());
        drop(lease);
        assert!(DriverLease::acquire(&path).is_ok());
        let link = root.path().join("link");
        std::os::unix::fs::symlink(path, &link).unwrap();
        assert!(DriverLease::acquire(&link).is_err());
    }
}
