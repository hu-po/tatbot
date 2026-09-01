//! Continuous PoE camera monitoring with a Prometheus metrics endpoint.
//!
//! This is the always-on service-mode counterpart to the bounded capture
//! commands: it subscribes to the (cheap, camera-ASIC-encoded) substreams of
//! every configured PoE camera, discards payloads, and keeps per-camera
//! health counters that a Prometheus scrape can alert on. No pixels are
//! recorded and nothing crosses the network except the encoded substreams
//! and the tiny metrics responses.

use std::{
    collections::BTreeMap,
    env,
    io::{Read, Write},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

use crate::{
    TimestampDomain, VisionConfig,
    gstreamer_backend::{PoeRtspCapture, PoeStream},
};

#[derive(Debug, Default, Clone)]
struct CameraStats {
    frames_total: u64,
    bytes_total: u64,
    errors_total: u64,
    reconnects_total: u64,
    last_frame_unix_s: Option<f64>,
    last_source_domain_camera_ntp: bool,
    last_clock_offset_large: bool,
}

enum Event {
    Frame {
        sensor: String,
        bytes: u64,
        camera_ntp: bool,
        offset_large: bool,
    },
    Error {
        sensor: String,
    },
    Reconnect {
        sensor: String,
    },
}

fn now_unix_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

fn render_metrics(stats: &BTreeMap<String, CameraStats>, process_start_unix_s: f64) -> String {
    let mut out = String::new();
    out.push_str(
        "# HELP tatbot_vision_build_info Build identity for the running vision monitor.\n",
    );
    out.push_str("# TYPE tatbot_vision_build_info gauge\n");
    out.push_str(&format!(
        "tatbot_vision_build_info{{commit=\"{}\"}} 1\n",
        env!("TATBOT_BUILD_GIT_SHA")
    ));
    out.push_str("# HELP tatbot_vision_process_start_time_seconds Unix start time of the running vision monitor.\n");
    out.push_str("# TYPE tatbot_vision_process_start_time_seconds gauge\n");
    out.push_str(&format!(
        "tatbot_vision_process_start_time_seconds {process_start_unix_s:.3}\n"
    ));
    out.push_str("# HELP tatbot_vision_frames_total Frames received per camera.\n");
    out.push_str("# TYPE tatbot_vision_frames_total counter\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_frames_total{{camera=\"{camera}\"}} {}\n",
            s.frames_total
        ));
    }
    out.push_str("# HELP tatbot_vision_bytes_total Encoded payload bytes received per camera.\n");
    out.push_str("# TYPE tatbot_vision_bytes_total counter\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_bytes_total{{camera=\"{camera}\"}} {}\n",
            s.bytes_total
        ));
    }
    out.push_str("# HELP tatbot_vision_errors_total Capture errors per camera.\n");
    out.push_str("# TYPE tatbot_vision_errors_total counter\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_errors_total{{camera=\"{camera}\"}} {}\n",
            s.errors_total
        ));
    }
    out.push_str("# HELP tatbot_vision_reconnects_total Pipeline reconnects per camera.\n");
    out.push_str("# TYPE tatbot_vision_reconnects_total counter\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_reconnects_total{{camera=\"{camera}\"}} {}\n",
            s.reconnects_total
        ));
    }
    out.push_str(
        "# HELP tatbot_vision_last_frame_age_seconds Seconds since the last frame arrived.\n",
    );
    out.push_str("# TYPE tatbot_vision_last_frame_age_seconds gauge\n");
    let now = now_unix_s();
    for (camera, s) in stats {
        let age = s
            .last_frame_unix_s
            .map(|t| (now - t).max(0.0))
            .unwrap_or(f64::INFINITY);
        if age.is_finite() {
            out.push_str(&format!(
                "tatbot_vision_last_frame_age_seconds{{camera=\"{camera}\"}} {age:.3}\n"
            ));
        } else {
            out.push_str(&format!(
                "tatbot_vision_last_frame_age_seconds{{camera=\"{camera}\"}} +Inf\n"
            ));
        }
    }
    out.push_str(
        "# HELP tatbot_vision_camera_ntp Last frame carried a camera-NTP timestamp (1) or fell back to host time (0).\n",
    );
    out.push_str("# TYPE tatbot_vision_camera_ntp gauge\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_camera_ntp{{camera=\"{camera}\"}} {}\n",
            u8::from(s.last_source_domain_camera_ntp)
        ));
    }
    out.push_str(
        "# HELP tatbot_vision_clock_offset_large Last frame needed a large (>1s) software clock correction.\n",
    );
    out.push_str("# TYPE tatbot_vision_clock_offset_large gauge\n");
    for (camera, s) in stats {
        out.push_str(&format!(
            "tatbot_vision_clock_offset_large{{camera=\"{camera}\"}} {}\n",
            u8::from(s.last_clock_offset_large)
        ));
    }
    out
}

fn serve_metrics(
    listener: TcpListener,
    stats: Arc<Mutex<BTreeMap<String, CameraStats>>>,
    running: Arc<AtomicBool>,
    process_start_unix_s: f64,
) {
    listener
        .set_nonblocking(true)
        .expect("metrics listener nonblocking");
    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((mut socket, _)) => {
                let _ = socket.set_read_timeout(Some(Duration::from_millis(500)));
                let mut request = [0_u8; 1024];
                let _ = socket.read(&mut request);
                let body = {
                    let stats = stats.lock().expect("metrics stats mutex");
                    render_metrics(&stats, process_start_unix_s)
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes());
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(_) => thread::sleep(Duration::from_millis(100)),
        }
    }
}

pub fn run(
    config: VisionConfig,
    stream: PoeStream,
    bind_host: &str,
    port: u16,
    duration_seconds: u64,
) -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let stats = Arc::new(Mutex::new(BTreeMap::<String, CameraStats>::new()));
    {
        let mut stats = stats.lock().expect("stats mutex");
        for camera in &config.cameras.poe {
            stats.insert(camera.name.clone(), CameraStats::default());
        }
    }

    let listener = TcpListener::bind((bind_host, port))
        .with_context(|| format!("binding metrics endpoint on {bind_host}:{port}"))?;
    tracing::info!(bind_host, port, "serving Prometheus metrics at /metrics");
    let metrics_stats = Arc::clone(&stats);
    let metrics_running = Arc::clone(&running);
    let process_start_unix_s = now_unix_s();
    let metrics_thread = thread::spawn(move || {
        serve_metrics(
            listener,
            metrics_stats,
            metrics_running,
            process_start_unix_s,
        )
    });

    let deadline =
        (duration_seconds > 0).then(|| Instant::now() + Duration::from_secs(duration_seconds));
    let expired = |deadline: Option<Instant>| deadline.is_some_and(|value| Instant::now() >= value);

    let (sender, receiver) = mpsc::channel::<Event>();
    let mut workers = Vec::new();
    for camera in config.cameras.poe.clone() {
        let sender = sender.clone();
        let running = Arc::clone(&running);
        workers.push(thread::spawn(move || {
            let sensor = camera.name.clone();
            let mut capture: Option<PoeRtspCapture> = None;
            let mut had_capture = false;
            while running.load(Ordering::Relaxed) && !expired(deadline) {
                if capture.is_none() {
                    match env::var(&camera.password_env)
                        .with_context(|| {
                            format!("missing password environment variable {}", camera.password_env)
                        })
                        .and_then(|password| {
                            PoeRtspCapture::new_monitor(camera.clone(), stream, &password)
                        }) {
                        Ok(value) => {
                            if had_capture {
                                let _ = sender.send(Event::Reconnect {
                                    sensor: sensor.clone(),
                                });
                            }
                            had_capture = true;
                            capture = Some(value);
                        }
                        Err(error) => {
                            tracing::warn!(sensor = %sensor, %error, "pipeline setup failed");
                            let _ = sender.send(Event::Error {
                                sensor: sensor.clone(),
                            });
                            thread::sleep(Duration::from_secs(2));
                            continue;
                        }
                    }
                }
                match capture
                    .as_mut()
                    .expect("capture initialized")
                    .next_observation(Duration::from_millis(1500))
                {
                    Ok(Some(frame)) => {
                        let camera_ntp =
                            matches!(frame.source_domain, TimestampDomain::CameraNtp);
                        let _ = sender.send(Event::Frame {
                            sensor: sensor.clone(),
                            bytes: frame.payload_bytes,
                            camera_ntp,
                            offset_large: frame.clock_offset_large,
                        });
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(sensor = %sensor, %error, "capture error; rebuilding pipeline");
                        let _ = sender.send(Event::Error {
                            sensor: sensor.clone(),
                        });
                        if let Some(value) = capture.take() {
                            let _ = value.stop();
                        }
                        thread::sleep(Duration::from_secs(2));
                    }
                }
            }
            if let Some(value) = capture.take() {
                let _ = value.stop();
            }
        }));
    }
    drop(sender);

    // Aggregate worker events until every worker exits (deadline reached) or
    // the process is terminated externally (systemd stop / Ctrl-C kills us;
    // gstreamer pipelines die with the process).
    while let Ok(event) = receiver.recv() {
        let mut stats = stats.lock().expect("stats mutex");
        match event {
            Event::Frame {
                sensor,
                bytes,
                camera_ntp,
                offset_large,
            } => {
                let entry = stats.entry(sensor).or_default();
                entry.frames_total += 1;
                entry.bytes_total += bytes;
                entry.last_frame_unix_s = Some(now_unix_s());
                entry.last_source_domain_camera_ntp = camera_ntp;
                entry.last_clock_offset_large = offset_large;
            }
            Event::Error { sensor } => {
                stats.entry(sensor).or_default().errors_total += 1;
            }
            Event::Reconnect { sensor } => {
                stats.entry(sensor).or_default().reconnects_total += 1;
            }
        }
    }

    running.store(false, Ordering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
    let _ = metrics_thread.join();
    let stats = stats.lock().expect("stats mutex");
    println!("{}", render_metrics(&stats, process_start_unix_s));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prometheus_exposition() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "camera1".to_string(),
            CameraStats {
                frames_total: 42,
                bytes_total: 1000,
                errors_total: 1,
                reconnects_total: 0,
                last_frame_unix_s: Some(now_unix_s()),
                last_source_domain_camera_ntp: true,
                last_clock_offset_large: true,
            },
        );
        let text = render_metrics(&stats, 1234.5);
        assert!(text.contains("tatbot_vision_build_info{commit="));
        assert!(text.contains("tatbot_vision_process_start_time_seconds 1234.500"));
        assert!(text.contains("tatbot_vision_frames_total{camera=\"camera1\"} 42"));
        assert!(text.contains("tatbot_vision_camera_ntp{camera=\"camera1\"} 1"));
        assert!(text.contains("tatbot_vision_last_frame_age_seconds{camera=\"camera1\"}"));
    }

    #[test]
    fn never_seen_camera_reports_infinite_age() {
        let mut stats = BTreeMap::new();
        stats.insert("camera9".to_string(), CameraStats::default());
        let text = render_metrics(&stats, 1234.5);
        assert!(text.contains("tatbot_vision_last_frame_age_seconds{camera=\"camera9\"} +Inf"));
    }
}
