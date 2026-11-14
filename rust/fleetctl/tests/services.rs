use std::os::unix::fs::PermissionsExt;
use tatbot_bus::{Producer, service::ServiceLease, transport::Bus};
use zenoh::Wait;
fn local_router() -> (zenoh::Session, String) {
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("tcp/{}", socket.local_addr().unwrap());
    let mut config = zenoh::Config::default();
    config.insert_json5("mode", "\"router\"").unwrap();
    config
        .insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    config
        .insert_json5(
            "listen/endpoints",
            &serde_json::to_string(&[&endpoint]).unwrap(),
        )
        .unwrap();
    drop(socket);
    (zenoh::open(config).wait().unwrap(), endpoint)
}
#[test]
fn completed_reply_stream_is_success_but_duplicate_is_refused() {
    let (_router, endpoint) = local_router();
    let bus = Bus::open_with(std::slice::from_ref(&endpoint), None).unwrap();
    let producer = Producer {
        node: "test".into(),
        pid: 1,
        sha: "test-sha".into(),
        run_id: "test".into(),
    };
    let _lease = ServiceLease::declare(&bus, producer.clone(), "camera", vec![]).unwrap();
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_fleetctl"))
            .args([
                "--connect",
                &endpoint,
                "services",
                "--expected-sha",
                "test-sha",
                "--require",
                "test/camera",
            ])
            .env("TATBOT_BUS_CREDENTIALS", "") // the test router takes none
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["complete"], true);
    let second = Bus::open_with(std::slice::from_ref(&endpoint), None).unwrap();
    let _duplicate = ServiceLease::declare(&second, producer, "camera", vec![]).unwrap();
    let result = run();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("duplicate service producers"));
    assert!(result.stdout.is_empty());
}
/// Runs `advertise` under a `systemctl` that reports the unit active once, then
/// exits `later_code` on every poll after that.
fn advertise_until_systemctl_exits(later_code: u8) -> std::process::Output {
    let (_router, endpoint) = local_router();
    let dir = std::env::temp_dir().join(format!(
        "fleetctl-advertise-{}-{later_code}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let systemctl = dir.join("systemctl");
    std::fs::write(
        &systemctl,
        format!("#!/bin/sh\n[ -e \"$0.seen\" ] && exit {later_code}\n: > \"$0.seen\"\n"),
    )
    .unwrap();
    std::fs::set_permissions(&systemctl, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fleetctl"))
        .args(["--connect", &endpoint, "advertise"])
        .args(["--node", "test", "--service", "zenohd"])
        .args(["--unit", "tatbot-zenohd.service"])
        .env("PATH", &dir)
        .env("TATBOT_BUS_CREDENTIALS", "") // the test router takes none
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    output
}
#[test]
fn advertise_ends_cleanly_when_the_unit_stops() {
    let result = advertise_until_systemctl_exits(3);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
#[test]
fn advertise_fails_when_systemctl_cannot_answer_so_its_unit_restarts_it() {
    let result = advertise_until_systemctl_exits(1);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("systemctl"));
}
