//! Bounded persistent estimator: a Python child on pipes, one request
//! outstanding, killed with its whole process group when dropped. No device
//! access or Rerun handle in Python; the child reads what its caller wrote.
use anyhow::{Context, Result, ensure};
use std::{
    io::{BufRead, BufReader, Read, Write},
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant},
};

const REPLY_LIMIT_BYTES: usize = 16 * 1024 * 1024;

pub struct Worker {
    child: Child,
    stdin: ChildStdin,
    replies: Receiver<Result<serde_json::Value, String>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    /// Spawn `python script args...` and wait for its `{"ready": true}` line.
    pub fn start(python: &Path, script: &Path, args: &[&Path], stop: &AtomicBool) -> Result<Self> {
        let mut child = Command::new(python)
            .arg(script)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0)
            .spawn()
            .with_context(|| format!("starting estimator {}", script.display()))?;
        let stdin = child.stdin.take().context("estimator stdin")?;
        let output = child.stdout.take().context("estimator stdout")?;
        let (sender, replies) = mpsc::sync_channel(1);
        let reader = std::thread::spawn(move || {
            let mut output = BufReader::new(output);
            loop {
                let mut bytes = Vec::new();
                let message = (|| -> Result<serde_json::Value> {
                    (&mut output)
                        .take(REPLY_LIMIT_BYTES as u64 + 1)
                        .read_until(b'\n', &mut bytes)?;
                    ensure!(
                        !bytes.is_empty()
                            && bytes.len() <= REPLY_LIMIT_BYTES
                            && bytes.last() == Some(&b'\n'),
                        "estimator ended or exceeded response limit"
                    );
                    Ok(serde_json::from_slice(&bytes)?)
                })()
                .map_err(|error| format!("{error:#}"));
                let failed = message.is_err();
                if sender.try_send(message).is_err() || failed {
                    break;
                }
            }
        });
        let worker = Self {
            child,
            stdin,
            replies,
            reader: Some(reader),
        };
        ensure!(
            worker.receive(stop, Duration::from_secs(20))?["ready"] == true,
            "estimator not ready"
        );
        Ok(worker)
    }

    fn receive(&self, stop: &AtomicBool, limit: Duration) -> Result<serde_json::Value> {
        let deadline = Instant::now() + limit;
        loop {
            ensure!(!stop.load(Ordering::Acquire), "estimator stopped");
            ensure!(Instant::now() < deadline, "estimator deadline exceeded");
            match self.replies.recv_timeout(Duration::from_millis(100)) {
                Ok(value) => return value.map_err(anyhow::Error::msg),
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// One request, one reply, within `limit`.
    pub fn request_within(
        &mut self,
        request: &serde_json::Value,
        stop: &AtomicBool,
        limit: Duration,
    ) -> Result<serde_json::Value> {
        let bytes = serde_json::to_vec(request)?;
        // Only one outstanding request. Even a worker that stopped reading
        // cannot fill the pipe with this small path-only message.
        ensure!(bytes.len() < 4000, "estimator request too large");
        self.stdin.write_all(&bytes)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        self.receive(stop, limit)
    }

    pub fn request(
        &mut self,
        request: &serde_json::Value,
        stop: &AtomicBool,
    ) -> Result<serde_json::Value> {
        self.request_within(request, stop, Duration::from_secs(8))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // This child and any descendants are independent of the SDK worker.
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let deadline = Instant::now() + Duration::from_millis(500);
            while !reader.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if reader.is_finished() {
                let _ = reader.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{os::unix::fs::PermissionsExt, sync::Arc};

    #[test]
    fn stopped_session_kills_unresponsive_estimator_without_waiting_for_reply() {
        let root = tempfile::tempdir().unwrap();
        let python = root.path().join("python");
        std::fs::write(
            &python,
            "#!/bin/sh\nprintf '{\"ready\":true}\\n'\nIFS= read -r line\nsleep 30\n",
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let mut worker = Worker::start(
            &python,
            Path::new("observer.py"),
            &[Path::new("binding"), root.path()],
            &stop,
        )
        .unwrap();
        let pid = worker.pid();
        let stopped = stop.clone();
        let signal = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            stopped.store(true, Ordering::Release);
        });
        let started = Instant::now();
        assert!(
            worker
                .request(&serde_json::json!({"captures":{}}), &stop)
                .is_err()
        );
        drop(worker);
        signal.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    }
}
