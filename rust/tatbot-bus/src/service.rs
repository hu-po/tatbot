//! Liveliness tokens advertise presence; a query supplies immutable provenance.
//! Capture or tracking freshness is a separate data guard, never inferred here.
use crate::{Envelope, Liveliness, Producer, Stamp, transport::Bus};
use zenoh::Wait;
pub struct ServiceLease {
    pub metrics: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, f64>>>,
    _token: zenoh::liveliness::LivelinessToken,
    _queryable: zenoh::query::Queryable<()>,
}
fn segment(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
}
impl ServiceLease {
    pub fn declare(
        bus: &Bus,
        producer: Producer,
        service: &str,
        schemas: Vec<String>,
    ) -> zenoh::Result<Self> {
        if !segment(&producer.node) || !segment(service) || producer.sha.is_empty() {
            return Err("invalid service identity".into());
        }
        let key = format!("tatbot/alive/{}/{}", producer.node, service);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        let body = Envelope {
            schema: "tatbot.liveliness/1".into(),
            payload: Liveliness {
                sha: producer.sha.clone(),
                schemas,
                pid: producer.pid,
                started_unix_ms: now,
                metrics: Default::default(),
            },
            producer,
            stamp: Stamp {
                mono_ns: 0,
                wall_ns: now * 1_000_000,
                basis: "host".into(),
            },
            seq: 0,
        };
        let metrics = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
        let observed = metrics.clone();
        let reply_key = key.clone();
        let queryable = bus
            .session
            .declare_queryable(key.clone())
            .callback(move |q| {
                let mut value = body.clone();
                value.payload.metrics = observed.lock().unwrap().clone();
                if let Ok(json) = serde_json::to_vec(&value) {
                    let _ = q.reply(reply_key.clone(), json).wait();
                }
            })
            .wait()?;
        let token = bus.session.liveliness().declare_token(key).wait()?;
        Ok(Self {
            metrics,
            _token: token,
            _queryable: queryable,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routed_clients_discover_presence_and_provenance() {
        let mut config = zenoh::Config::default();
        config.insert_json5("mode", "\"router\"").unwrap();
        config
            .insert_json5("scouting/multicast/enabled", "false")
            .unwrap();
        config
            .insert_json5("scouting/gossip/enabled", "false")
            .unwrap();
        let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoints = vec![format!("tcp/{}", socket.local_addr().unwrap())];
        config
            .insert_json5(
                "listen/endpoints",
                &serde_json::to_string(&endpoints).unwrap(),
            )
            .unwrap();
        drop(socket);
        let _router = zenoh::open(config).wait().unwrap();
        let publisher = Bus::open(&endpoints, &[]).unwrap();
        let _lease = ServiceLease::declare(
            &publisher,
            Producer {
                node: "test".into(),
                pid: 1,
                sha: "routed-sha".into(),
                run_id: "test".into(),
            },
            "viewer",
            vec![],
        )
        .unwrap();
        let client = Bus::open(&endpoints, &[]).unwrap();
        let replies = client
            .session
            .liveliness()
            .get("tatbot/alive/*/*")
            .timeout(std::time::Duration::from_secs(2))
            .wait()
            .unwrap();
        let sample = replies
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap()
            .unwrap();
        assert_eq!(
            sample.result().unwrap().key_expr().as_str(),
            "tatbot/alive/test/viewer"
        );
        let replies = client
            .session
            .get("tatbot/alive/test/viewer")
            .wait()
            .unwrap();
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap()
            .unwrap();
        let body: Envelope<Liveliness> =
            serde_json::from_slice(&reply.result().unwrap().payload().to_bytes()).unwrap();
        assert_eq!(body.payload.sha, "routed-sha");
    }
    #[test]
    fn token_and_sha_query_have_the_same_lifetime() {
        let bus = Bus::open(&[], &[]).unwrap();
        let observer = bus
            .session
            .liveliness()
            .declare_subscriber("tatbot/alive/*/*")
            .wait()
            .unwrap();
        let lease = ServiceLease::declare(
            &bus,
            Producer {
                node: "mock".into(),
                pid: 1,
                sha: "test-sha".into(),
                run_id: "test".into(),
            },
            "visiond",
            vec!["tatbot.frame-set/1".into()],
        )
        .unwrap();
        let sample = observer
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(sample.kind(), zenoh::sample::SampleKind::Put);
        let reply = bus
            .session
            .get("tatbot/alive/mock/visiond")
            .wait()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let body: Envelope<Liveliness> =
            serde_json::from_slice(&reply.result().unwrap().payload().to_bytes()).unwrap();
        assert_eq!(body.payload.sha, "test-sha");
        drop(lease);
        assert_eq!(
            observer
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .kind(),
            zenoh::sample::SampleKind::Delete
        );
    }
}
