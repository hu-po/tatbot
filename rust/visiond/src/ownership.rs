//! Lifetime camera/socket leases. Never unlink the locked inode.
use anyhow::{Context, Result, ensure};
use std::{
    fs::{File, OpenOptions},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};
#[derive(Debug)]
pub struct CameraLease {
    _file: File,
}
impl CameraLease {
    pub fn acquire(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .context("opening camera ownership lease")?;
        ensure!(
            file.metadata()?.is_file(),
            "camera lease is not a regular file"
        );
        // SAFETY: this file owns the descriptor for the lifetime of the lease.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "camera busy: another process owns {}",
            path.display()
        );
        Ok(Self { _file: file })
    }
    pub fn camera(identity: &str) -> Result<Self> {
        use sha2::{Digest, Sha256};
        Self::acquire(&std::env::temp_dir().join(format!(
            "tatbot-camera-{}.lock",
            hex::encode(Sha256::digest(identity.as_bytes()))
        )))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lease_is_exclusive_across_processes_and_released_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("camera.lock");
        let owner = CameraLease::acquire(&path).unwrap();
        assert!(
            !std::process::Command::new("flock")
                .arg("-n")
                .arg(&path)
                .arg("true")
                .status()
                .unwrap()
                .success()
        );
        drop(owner);
        assert!(CameraLease::acquire(&path).is_ok());
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(CameraLease::acquire(&link).is_err());
    }
}
