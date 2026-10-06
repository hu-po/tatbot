use tatbot_bus::{Producer, service::ServiceLease, transport::Bus};
use zenoh::Wait;
#[test]
fn completed_reply_stream_is_success_but_duplicate_is_refused() {
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
    let _router = zenoh::open(config).wait().unwrap();
    let bus = Bus::open(std::slice::from_ref(&endpoint), &[]).unwrap();
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
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["complete"], true);
    let second = Bus::open(std::slice::from_ref(&endpoint), &[]).unwrap();
    let _duplicate = ServiceLease::declare(&second, producer, "camera", vec![]).unwrap();
    let result = run();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("duplicate service producers"));
}
