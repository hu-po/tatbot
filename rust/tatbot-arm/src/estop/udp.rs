//! Relay transport: the same sender, datagram, debounce and silence rules as
//! the Python E-stop monitor and ROS EST1 reader. Serial parsing stays separate.
use super::*;
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};

const TIMEOUT_MS: u64 = 150;
const MAX_FRAME: usize = 128;

#[derive(Default)]
struct Frames {
    sequence: Option<u64>,
    last_ms: Option<u64>,
    raw: Option<u8>,
    stable: Option<u8>,
    count: u8,
}
impl Frames {
    fn stale(&self, now: u64) -> bool {
        self.last_ms
            .is_none_or(|last| now.saturating_sub(last) > TIMEOUT_MS)
    }
    fn accept(&mut self, sequence: u64, button: u8, now: u64) {
        let stale = self.stale(now);
        if !stale && self.sequence.is_some_and(|last| sequence <= last) {
            return;
        }
        if stale {
            self.raw = None;
            self.stable = None;
            self.count = 0;
        }
        self.sequence = Some(sequence);
        self.last_ms = Some(now);
        if self.raw == Some(button) {
            self.count = (self.count + 1).min(3);
        } else {
            self.raw = Some(button);
            self.count = 1;
        }
        if self.count == 3 {
            self.stable = Some(button);
        }
    }
    fn state(&self, now: u64) -> i32 {
        if self.stale(now) {
            return FAULT;
        }
        match self.stable {
            Some(0) => PRESSED,
            Some(1) => OK,
            _ => FAULT,
        }
    }
}

fn parse_frame(data: &[u8]) -> Option<(u64, u8)> {
    if data.len() > MAX_FRAME {
        return None;
    }
    let text = std::str::from_utf8(data).ok()?;
    let text = text.strip_suffix('\n').unwrap_or(text);
    let text = text.strip_suffix('\r').unwrap_or(text);
    let fields: Vec<_> = text.split(' ').collect();
    if fields.len() != 3
        || fields[0] != "EST1"
        || fields[1].is_empty()
        || fields[1].len() > 18
        || !fields[1].bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let button = match fields[2] {
        "0" => 0,
        "1" => 1,
        _ => return None,
    };
    Some((fields[1].parse().ok()?, button))
}

fn addresses(device: &Path) -> std::io::Result<(SocketAddrV4, Ipv4Addr)> {
    let text = device.to_string_lossy();
    let invalid = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "UDP e-stop needs udp://[IPv4]:PORT?from=<relay IPv4>",
        )
    };
    let (listen, source) = text
        .strip_prefix("udp://")
        .and_then(|s| s.split_once("?from="))
        .ok_or_else(invalid)?;
    let listen = if listen.starts_with(':') {
        format!("0.0.0.0{listen}")
    } else {
        listen.to_owned()
    };
    Ok((
        listen.parse().map_err(|_| invalid())?,
        source.parse().map_err(|_| invalid())?,
    ))
}

pub(super) fn start(
    device: PathBuf,
    snapshot_path: PathBuf,
    state: Arc<AtomicI32>,
    publish: Publisher,
) -> std::io::Result<Monitor> {
    let (address, source) = addresses(&device)?;
    // No address reuse: a second reader must not receive part of the heartbeat stream.
    let socket = UdpSocket::bind(address)?;
    start_socket(socket, source, device, snapshot_path, state, publish)
}

fn start_socket(
    socket: UdpSocket,
    source: Ipv4Addr,
    device: PathBuf,
    snapshot_path: PathBuf,
    state: Arc<AtomicI32>,
    publish: Publisher,
) -> std::io::Result<Monitor> {
    socket.set_nonblocking(true)?;
    let stop = Arc::new(AtomicBool::new(false));
    let latest = Arc::new(Mutex::new(None));
    let (reader_stop, reader_state, reader_latest, reader_device) =
        (stop.clone(), state.clone(), latest.clone(), device.clone());
    let reader = thread::spawn(move || {
        let started = Instant::now();
        let mut frames = Frames::default();
        while !reader_stop.load(Ordering::SeqCst) {
            let mut fd = libc::pollfd {
                fd: socket.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one initialized pollfd refers to this thread's live socket.
            if unsafe { libc::poll(&mut fd, 1, 20) } > 0 {
                for _ in 0..64 {
                    let mut data = [0; MAX_FRAME + 1];
                    let Ok((count, sender)) = socket.recv_from(&mut data) else {
                        break;
                    };
                    if sender.ip() == source
                        && let Some((sequence, button)) = parse_frame(&data[..count])
                    {
                        frames.accept(sequence, button, started.elapsed().as_millis() as u64);
                    }
                }
            }
            let now = started.elapsed().as_millis() as u64;
            let current = frames.state(now);
            reader_state.store(current, Ordering::SeqCst);
            let snapshot = Snapshot {
                schema: "tatbot.estop-status/1",
                device: reader_device.to_string_lossy().into(),
                state: match current {
                    OK => "ok",
                    PRESSED => "pressed",
                    _ => "fault",
                },
                engaged: current != OK,
                heartbeat_age_ms: frames.last_ms.map(|last| now.saturating_sub(last)),
                last_sequence: frames.sequence,
                pid: std::process::id(),
                updated_unix: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            };
            *reader_latest.lock().unwrap() = Some((snapshot, Instant::now()));
        }
        reader_state.store(FAULT, Ordering::SeqCst);
    });
    let writer = spawn_writer(stop.clone(), latest.clone(), snapshot_path.clone(), publish);
    Ok(Monitor {
        state,
        latest,
        stop,
        reader: Some(reader),
        writer: Some(writer),
        snapshot_path,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_debounce_replay_silence_and_restart() {
        let mut frames = Frames::default();
        assert_eq!(frames.state(0), FAULT);
        frames.accept(100, 1, 10);
        frames.accept(101, 1, 20);
        assert_eq!(frames.state(20), FAULT);
        frames.accept(102, 1, 30);
        assert_eq!(frames.state(180), OK);
        frames.accept(102, 0, 100);
        frames.accept(101, 0, 170);
        assert_eq!(frames.last_ms, Some(30));
        assert_eq!(frames.state(181), FAULT);
        // Silence permits a restarted sender to reseed, but still needs debounce.
        frames.accept(0, 0, 181);
        frames.accept(1, 0, 191);
        assert_eq!(frames.state(191), FAULT);
        frames.accept(2, 0, 201);
        assert_eq!(frames.state(201), PRESSED);
        frames.accept(3, 1, 211);
        frames.accept(4, 1, 221);
        assert_eq!(frames.state(221), PRESSED);
        frames.accept(5, 1, 231);
        assert_eq!(frames.state(231), OK);
    }

    #[test]
    fn only_one_strict_bounded_frame_per_datagram() {
        for frame in [b"EST1 12 1".as_slice(), b"EST1 12 1\n", b"EST1 12 1\r\n"] {
            assert_eq!(parse_frame(frame), Some((12, 1)));
        }
        for frame in [
            b"EST1  12 1".as_slice(),
            b"EST1 -1 1",
            b"EST1 +1 1",
            b"EST1 1234567890123456789 1",
            b"EST1 1 2",
            b"EST1 1 1 extra",
            b"EST1 1 1\nEST1 2 1\n",
            b"EST1 1 1\n\n",
            b"EST1 1 1\x00",
            b"EST1 1 1\xff",
        ] {
            assert_eq!(parse_frame(frame), None, "{frame:?}");
        }
        assert_eq!(parse_frame(&[b'x'; MAX_FRAME + 1]), None);
    }

    #[test]
    fn normalized_uri_requires_a_literal_sender() {
        let (listen, source) = addresses(Path::new("udp://:7640?from=192.0.2.9")).unwrap();
        assert_eq!(listen, "0.0.0.0:7640".parse().unwrap());
        assert_eq!(source, Ipv4Addr::new(192, 0, 2, 9));
        for device in [
            "udp://:7640",
            "udp://:7640?from=relay-role",
            "udp://:7640?from=",
            "udp://host:7640?from=192.0.2.9",
            "udp://:7640?from=192.0.2.9&from=192.0.2.10",
        ] {
            assert!(addresses(Path::new(device)).is_err(), "{device}");
        }
    }

    fn fixture(
        publish: Publisher,
    ) -> (tempfile::TempDir, Monitor, std::net::SocketAddr, UdpSocket) {
        let dir = tempfile::tempdir().unwrap();
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let address = socket.local_addr().unwrap();
        let device = PathBuf::from(format!("udp://{address}?from=127.0.0.1"));
        let monitor = start_socket(
            socket,
            Ipv4Addr::LOCALHOST,
            device,
            dir.path().join("status.json"),
            Arc::new(AtomicI32::new(FAULT)),
            publish,
        )
        .unwrap();
        (
            dir,
            monitor,
            address,
            UdpSocket::bind("127.0.0.1:0").unwrap(),
        )
    }

    fn send(socket: &UdpSocket, address: std::net::SocketAddr, seq: u64, button: u8) {
        for i in seq..seq + 3 {
            socket
                .send_to(format!("EST1 {i} {button}\n").as_bytes(), address)
                .unwrap();
        }
    }

    fn wait(monitor: &Monitor, state: i32) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(500) {
            if monitor.state.load(Ordering::SeqCst) == state {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(monitor.state.load(Ordering::SeqCst), state);
    }

    #[test]
    fn socket_sender_filter_snapshot_timeout_and_exclusive_cleanup() {
        let (dir, monitor, address, sender) = fixture(Box::new(write_snapshot));
        assert_eq!(monitor.state.load(Ordering::SeqCst), FAULT);
        let wrong = UdpSocket::bind("127.0.0.2:0").unwrap();
        send(&wrong, address, 100, 1);
        sender
            .send_to(b"EST1 100 1\nEST1 101 1\nEST1 102 1\n", address)
            .unwrap();
        sender.send_to(&[b'x'; MAX_FRAME + 100], address).unwrap();
        thread::sleep(Duration::from_millis(40));
        assert_eq!(monitor.state.load(Ordering::SeqCst), FAULT);
        assert_eq!(monitor.snapshot().unwrap().last_sequence, None);
        // Invalid and foreign frames must not consume sequence numbers either.
        send(&sender, address, 0, 1);
        wait(&monitor, OK);
        assert!(Monitor::start(monitor.device.clone(), dir.path().join("second.json")).is_err());
        send(&sender, address, 3, 0);
        wait(&monitor, PRESSED);
        send(&sender, address, 6, 1);
        wait(&monitor, OK);
        wait(&monitor, FAULT);
        assert_eq!(monitor.snapshot().unwrap().last_sequence, Some(8));
        let path = dir.path().join("status.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["schema"], "tatbot.estop-status/1");
        drop(monitor);
        assert!(!path.exists());
        UdpSocket::bind(address).unwrap();
    }

    #[test]
    fn blocked_snapshot_publisher_does_not_delay_press_or_dropout() {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let (_dir, monitor, address, sender) = fixture(Box::new(move |_, _| {
            let _ = entered_tx.try_send(());
            let _ = release_rx.recv_timeout(Duration::from_secs(2));
            Ok(())
        }));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        send(&sender, address, 0, 1);
        wait(&monitor, OK);
        send(&sender, address, 3, 0);
        wait(&monitor, PRESSED);
        wait(&monitor, FAULT);
        // Stop the publisher before releasing it, so teardown cannot block a second time.
        monitor.stop.store(true, Ordering::SeqCst);
        release_tx.send(()).unwrap();
        drop(monitor);
    }
}
