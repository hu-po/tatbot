//! EST1 protocol port of cpp/teleop/estop_monitor.cpp.
//! Snapshot writes run on a separate thread and never hold up heartbeat consumption.
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::{fs::OpenOptionsExt, io::AsRawFd},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI32, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
mod udp;
pub const OK: i32 = 0;
pub const PRESSED: i32 = 1;
pub const FAULT: i32 = 2;
#[derive(Debug)]
pub struct Protocol {
    pub state: i32,
    pub sequence: Option<u64>,
    raw: Option<u8>,
    stable: Option<u8>,
    count: u8,
    last_ms: u64,
    buffer: String,
}
impl Default for Protocol {
    fn default() -> Self {
        Self {
            state: FAULT,
            sequence: None,
            raw: None,
            stable: None,
            count: 0,
            last_ms: 0,
            buffer: String::new(),
        }
    }
}
impl Protocol {
    pub fn feed(&mut self, bytes: &[u8], now_ms: u64) {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        while let Some(n) = self.buffer.find('\n') {
            let line = self.buffer[..n].to_string();
            self.buffer.drain(..=n);
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() != 3 || fields[0] != "EST1" {
                continue;
            }
            let (Ok(seq), Ok(button)) = (fields[1].parse::<u64>(), fields[2].parse::<u8>()) else {
                continue;
            };
            if button > 1 {
                continue;
            }
            if self.sequence.is_some_and(|last| seq <= last) {
                self.raw = None;
                self.stable = None;
                self.count = 0;
            }
            self.sequence = Some(seq);
            self.last_ms = now_ms;
            if self.raw == Some(button) {
                self.count = (self.count + 1).min(3);
            } else {
                self.raw = Some(button);
                self.count = 1;
            }
            if self.count >= 3 {
                self.stable = Some(button);
            }
        }
        if self.buffer.len() > 1024 {
            self.buffer.clear();
        }
        self.tick(now_ms);
    }
    pub fn tick(&mut self, now_ms: u64) {
        if now_ms.saturating_sub(self.last_ms) > 100 {
            self.state = FAULT;
        } else if self.stable == Some(0) {
            self.state = PRESSED;
        } else if self.stable == Some(1) {
            self.state = OK;
        }
    }
}
#[derive(Clone, Serialize)]
pub struct Snapshot {
    pub schema: &'static str,
    pub device: String,
    pub state: &'static str,
    pub engaged: bool,
    pub heartbeat_age_ms: Option<u64>,
    pub last_sequence: Option<u64>,
    pub pid: u32,
    pub updated_unix: f64,
}
fn open(device: &Path) -> std::io::Result<File> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOCTTY | libc::O_NONBLOCK)
        .open(device)?;
    // SAFETY: fd belongs to f; termios is initialized by tcgetattr before use.
    unsafe {
        if libc::isatty(f.as_raw_fd()) == 1 {
            let mut t = std::mem::zeroed();
            if libc::tcgetattr(f.as_raw_fd(), &mut t) == 0 {
                libc::cfmakeraw(&mut t);
                libc::tcsetattr(f.as_raw_fd(), libc::TCSANOW, &t);
            }
        }
    }
    Ok(f)
}
pub struct Monitor {
    pub state: Arc<AtomicI32>,
    latest: Arc<Mutex<Option<(Snapshot, Instant)>>>,
    stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    writer: Option<JoinHandle<()>>,
    snapshot_path: PathBuf,
    device: PathBuf,
}
type Publisher = Box<dyn Fn(&Path, &Snapshot) -> std::io::Result<()> + Send>;
impl Monitor {
    pub fn start(device: PathBuf, snapshot_path: PathBuf) -> std::io::Result<Self> {
        Self::start_with_publisher(device, snapshot_path, Box::new(write_snapshot))
    }
    fn start_with_publisher(
        device: PathBuf,
        snapshot_path: PathBuf,
        publish: Publisher,
    ) -> std::io::Result<Self> {
        Self::start_on_state(
            device,
            snapshot_path,
            Arc::new(AtomicI32::new(FAULT)),
            publish,
        )
    }
    /// Bind the reader to the atomic already consumed by a control worker.
    /// Failure to open the device leaves that worker stopped.
    pub fn bind(
        device: PathBuf,
        snapshot_path: PathBuf,
        state: Arc<AtomicI32>,
    ) -> std::io::Result<Self> {
        Self::start_on_state(device, snapshot_path, state, Box::new(write_snapshot))
    }
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.latest
            .lock()
            .unwrap()
            .as_ref()
            .map(|(snapshot, captured)| {
                let mut snapshot = snapshot.clone();
                snapshot.heartbeat_age_ms = snapshot
                    .heartbeat_age_ms
                    .map(|age| age.saturating_add(captured.elapsed().as_millis() as u64));
                snapshot
            })
    }
    /// Nonblocking read for owner-thread guards; contention is missing evidence.
    pub fn snapshot_reader(&self) -> impl Fn() -> Option<Snapshot> + Send + 'static {
        let latest = self.latest.clone();
        move || {
            latest
                .try_lock()
                .ok()?
                .as_ref()
                .map(|(snapshot, captured)| {
                    let mut snapshot = snapshot.clone();
                    snapshot.heartbeat_age_ms = snapshot
                        .heartbeat_age_ms
                        .map(|age| age.saturating_add(captured.elapsed().as_millis() as u64));
                    snapshot
                })
        }
    }
    fn start_on_state(
        device: PathBuf,
        snapshot_path: PathBuf,
        state: Arc<AtomicI32>,
        publish: Publisher,
    ) -> std::io::Result<Self> {
        state.store(FAULT, Ordering::SeqCst);
        if device.to_string_lossy().starts_with("udp://") {
            return udp::start(device, snapshot_path, state, publish);
        }
        let initial = open(&device)?;
        let stop = Arc::new(AtomicBool::new(false));
        let latest: Arc<Mutex<Option<(Snapshot, Instant)>>> = Arc::new(Mutex::new(None));
        let (reader_state, reader_stop, reader_latest, reader_device) =
            (state.clone(), stop.clone(), latest.clone(), device.clone());
        let reader = thread::spawn(move || {
            let start = Instant::now();
            let mut p = Protocol::default();
            let mut file = Some(initial);
            let mut reopen = Instant::now();
            while !reader_stop.load(Ordering::SeqCst) {
                let now = start.elapsed().as_millis() as u64;
                if let Some(f) = &mut file {
                    let mut fd = libc::pollfd {
                        fd: f.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // SAFETY: poll receives one initialized pollfd for a live file.
                    let ready = unsafe { libc::poll(&mut fd, 1, 20) };
                    if ready > 0 {
                        let mut bytes = [0; 256];
                        match f.read(&mut bytes) {
                            Ok(0) => {
                                file = None;
                                p.state = FAULT;
                            }
                            Ok(n) => p.feed(&bytes[..n], start.elapsed().as_millis() as u64),
                            Err(e)
                                if [
                                    std::io::ErrorKind::WouldBlock,
                                    std::io::ErrorKind::Interrupted,
                                ]
                                .contains(&e.kind()) => {}
                            Err(_) => {
                                file = None;
                                p.state = FAULT;
                            }
                        }
                    }
                    if file.is_some() {
                        p.tick(start.elapsed().as_millis() as u64);
                    }
                } else {
                    p.state = FAULT;
                    if reopen.elapsed() > Duration::from_millis(500) {
                        reopen = Instant::now();
                        file = open(&reader_device).ok();
                        if file.is_some() {
                            p = Protocol {
                                last_ms: now,
                                ..Protocol::default()
                            };
                        }
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                reader_state.store(p.state, Ordering::SeqCst);
                let snap = Snapshot {
                    schema: "tatbot.estop-status/1",
                    device: reader_device.to_string_lossy().into(),
                    state: match p.state {
                        OK => "ok",
                        PRESSED => "pressed",
                        _ => "fault",
                    },
                    engaged: p.state != OK,
                    heartbeat_age_ms: p.sequence.map(|_| now.saturating_sub(p.last_ms)),
                    last_sequence: p.sequence,
                    pid: std::process::id(),
                    updated_unix: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                };
                *reader_latest.lock().unwrap() = Some((snap, Instant::now()));
            }
            reader_state.store(FAULT, Ordering::SeqCst);
        });
        let writer = spawn_writer(stop.clone(), latest.clone(), snapshot_path.clone(), publish);
        Ok(Self {
            state,
            latest,
            stop,
            reader: Some(reader),
            writer: Some(writer),
            snapshot_path,
            device,
        })
    }
}
fn spawn_writer(
    stop: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<(Snapshot, Instant)>>>,
    path: PathBuf,
    publish: Publisher,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(Ordering::SeqCst) {
            let snapshot = latest.lock().unwrap().clone();
            if let Some((snapshot, _)) = snapshot {
                let _ = publish(&path, &snapshot);
            }
            thread::sleep(Duration::from_millis(20));
        }
    })
}
fn write_snapshot(path: &Path, s: &Snapshot) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("snapshot parent"))?;
    std::fs::create_dir_all(parent)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer(&mut temp, s)?;
    temp.write_all(b"\n")?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.reader.take() {
            let _ = t.join();
        }
        if let Some(t) = self.writer.take() {
            let _ = t.join();
        }
        if let Ok(bytes) = std::fs::read(&self.snapshot_path)
            && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && v["pid"] == std::process::id()
            && v["device"] == self.device.to_string_lossy().as_ref()
        {
            let _ = std::fs::remove_file(&self.snapshot_path);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn carried_protocol_debounce_malformed_reset_timeout() {
        let mut p = Protocol::default();
        assert_eq!(p.state, FAULT);
        p.feed(b"EST1 0 1\nEST1 1 1\n", 10);
        assert_eq!(p.state, FAULT);
        p.feed(b"EST1 2 1\n", 20);
        assert_eq!(p.state, OK);
        p.feed(b"EST1 99 1 trailing\nnot-a-frame\n", 30);
        assert_eq!(p.sequence, Some(2));
        p.feed(b"EST1 3 0\nEST1 4 0\nEST1 5 0\n", 40);
        assert_eq!(p.state, PRESSED);
        p.feed(b"EST1 6 1\nEST1 7 1\nEST1 8 1\n", 50);
        assert_eq!(p.state, OK);
        p.feed(b"EST1 0 0\nEST1 1 0\n", 60);
        assert_eq!(p.state, OK);
        p.feed(b"EST1 2 0\n", 70);
        assert_eq!(p.state, PRESSED);
        p.tick(170);
        assert_eq!(p.state, PRESSED);
        p.tick(171);
        assert_eq!(p.state, FAULT);
    }
    #[test]
    fn fragmented_lines_and_oversized_garbage() {
        let mut p = Protocol::default();
        p.feed(b"EST", 1);
        p.feed(b"1 0 1\nEST1 1 1\nEST1 2 1\n", 2);
        assert_eq!(p.state, OK);
        p.feed(&vec![b'x'; 2048], 3);
        assert!(p.buffer.is_empty());
        p.tick(103);
        assert_eq!(p.state, FAULT);
    }
}

#[cfg(test)]
mod threaded_tests {
    use super::*;
    use std::os::fd::FromRawFd;
    use std::os::unix::fs::PermissionsExt;
    fn pty() -> (File, PathBuf) {
        // SAFETY: libc allocates the PTY; ownership of master transfers exactly once to File.
        unsafe {
            let fd = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(fd >= 0);
            assert_eq!(libc::grantpt(fd), 0);
            assert_eq!(libc::unlockpt(fd), 0);
            // c_char is i8 on x86_64 Linux but u8 on aarch64, so the
            // buffer must follow libc rather than hardcode a signedness.
            let mut name = [0 as libc::c_char; 256];
            assert_eq!(libc::ptsname_r(fd, name.as_mut_ptr(), name.len()), 0);
            let path = std::ffi::CStr::from_ptr(name.as_ptr())
                .to_str()
                .unwrap()
                .into();
            (File::from_raw_fd(fd), path)
        }
    }
    fn frames(f: &mut File, seq: &mut u64, button: u8) {
        for _ in 0..3 {
            writeln!(f, "EST1 {} {button}", *seq).unwrap();
            *seq += 1;
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn wait(m: &Monitor, state: i32) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(400) {
            if m.state.load(Ordering::SeqCst) == state {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(m.state.load(Ordering::SeqCst), state);
    }
    #[test]
    fn carried_pty_snapshot_press_timeout_and_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("status.json");
        let (mut master, device) = pty();
        let monitor = Monitor::start(device, path.clone()).unwrap();
        assert_eq!(monitor.state.load(Ordering::SeqCst), FAULT);
        let mut seq = 0;
        frames(&mut master, &mut seq, 1);
        wait(&monitor, OK);
        for _ in 0..10 {
            frames(&mut master, &mut seq, 1);
        }
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["schema"], "tatbot.estop-status/1");
        assert_eq!(value["state"], "ok");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        frames(&mut master, &mut seq, 0);
        wait(&monitor, PRESSED);
        frames(&mut master, &mut seq, 1);
        wait(&monitor, OK);
        wait(&monitor, FAULT);
        drop(monitor);
        assert!(!path.exists());
    }
    #[test]
    fn carried_snapshot_failure_cannot_block_monitor() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("file");
        std::fs::write(&parent, b"not directory").unwrap();
        let (mut master, device) = pty();
        let monitor = Monitor::start(device, parent.join("status.json")).unwrap();
        let mut seq = 0;
        frames(&mut master, &mut seq, 1);
        wait(&monitor, OK);
        frames(&mut master, &mut seq, 0);
        wait(&monitor, PRESSED);
        wait(&monitor, FAULT);
    }
    #[test]
    fn carried_unplug_replug_resets_protocol() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("device");
        let (mut master, device) = pty();
        std::os::unix::fs::symlink(device, &link).unwrap();
        let monitor = Monitor::start(link.clone(), dir.path().join("status.json")).unwrap();
        let mut seq = 0;
        frames(&mut master, &mut seq, 1);
        wait(&monitor, OK);
        drop(master);
        wait(&monitor, FAULT);
        let (mut replacement, device) = pty();
        let next = dir.path().join("next");
        std::os::unix::fs::symlink(device, &next).unwrap();
        std::fs::rename(next, link).unwrap();
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            frames(&mut replacement, &mut seq, 1);
            if monitor.state.load(Ordering::SeqCst) == OK {
                return;
            }
        }
        panic!("replug did not recover");
    }
}

#[cfg(test)]
mod blocked_writer_tests {
    use super::*;
    use std::os::fd::FromRawFd;
    #[test]
    fn blocked_publisher_never_delays_atomic_press_or_timeout() {
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: each allocated descriptor is transferred once to an owning File.
        let (mut master, device) = unsafe {
            let fd = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(fd >= 0);
            assert_eq!(libc::grantpt(fd), 0);
            assert_eq!(libc::unlockpt(fd), 0);
            // c_char is i8 on x86_64 Linux but u8 on aarch64, so the
            // buffer must follow libc rather than hardcode a signedness.
            let mut name = [0 as libc::c_char; 256];
            assert_eq!(libc::ptsname_r(fd, name.as_mut_ptr(), name.len()), 0);
            (
                File::from_raw_fd(fd),
                PathBuf::from(std::ffi::CStr::from_ptr(name.as_ptr()).to_str().unwrap()),
            )
        };
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let monitor = Monitor::start_with_publisher(
            device,
            dir.path().join("status.json"),
            Box::new(move |_, _| {
                let _ = entered_tx.try_send(());
                let _ = release_rx.recv_timeout(Duration::from_secs(2));
                Ok(())
            }),
        )
        .unwrap();
        let frames = |f: &mut File, seq: u64, button: u8| {
            for i in seq..seq + 3 {
                writeln!(f, "EST1 {i} {button}").unwrap();
                thread::sleep(Duration::from_millis(10));
            }
        };
        frames(&mut master, 0, 1);
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(monitor.state.load(Ordering::SeqCst), OK);
        frames(&mut master, 3, 0);
        assert_eq!(monitor.state.load(Ordering::SeqCst), PRESSED);
        thread::sleep(Duration::from_millis(130));
        assert_eq!(monitor.state.load(Ordering::SeqCst), FAULT);
        release_tx.send(()).unwrap();
    }
}
