//! TCP-only Zenoh transport. Scouting is disabled; endpoints are explicit.
use crate::Envelope;
use serde::{Serialize, de::DeserializeOwned};
use zenoh::Wait;
pub struct Bus {
    pub session: zenoh::Session,
}
impl Bus {
    pub fn open(connect: &[String], listen: &[String]) -> zenoh::Result<Self> {
        let mut config = zenoh::Config::default();
        // Fleet services attach to the explicit router. Peer mode depends on
        // peer discovery/routing that the fleet deliberately disables.
        if !connect.is_empty() && listen.is_empty() {
            config.insert_json5("mode", "\"client\"")?;
        }
        config.insert_json5("scouting/multicast/enabled", "false")?;
        config.insert_json5("connect/timeout_ms", "2000")?;
        config.insert_json5("connect/exit_on_failure", "true")?;
        config.insert_json5("connect/endpoints", &serde_json::to_string(connect)?)?;
        config.insert_json5("listen/endpoints", &serde_json::to_string(listen)?)?;
        Ok(Self {
            session: zenoh::open(config).wait()?,
        })
    }
    pub fn publish<T: Serialize>(&self, key: &str, value: &Envelope<T>) -> zenoh::Result<()> {
        self.session.put(key, serde_json::to_vec(value)?).wait()
    }
}
pub fn decode<T: DeserializeOwned>(bytes: &[u8], expected: &str) -> zenoh::Result<Envelope<T>> {
    let value: Envelope<T> = serde_json::from_slice(bytes)?;
    if value.schema != expected {
        return Err(format!("unknown schema {}; expected {expected}", value.schema).into());
    }
    Ok(value)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Producer, Stamp};
    #[test]
    fn zenoh_pubsub_and_queryable_roundtrip() {
        // Session-local transport test has no fleet endpoint and never discovers peers.
        let bus = Bus::open(&[], &[]).unwrap();
        let subscriber = bus
            .session
            .declare_subscriber("tatbot/test/value")
            .wait()
            .unwrap();
        let msg = Envelope {
            schema: "tatbot.test/1".into(),
            producer: Producer {
                node: "mock".into(),
                pid: 0,
                sha: "test".into(),
                run_id: "test".into(),
            },
            stamp: Stamp {
                mono_ns: 0,
                wall_ns: 0,
                basis: "mock".into(),
            },
            seq: 1,
            payload: 42u64,
        };
        bus.publish("tatbot/test/value", &msg).unwrap();
        let sample = subscriber
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(
            decode::<u64>(&sample.payload().to_bytes(), "tatbot.test/1").unwrap(),
            msg
        );
        assert!(decode::<u64>(&sample.payload().to_bytes(), "tatbot.test/2").is_err());
        let _queryable = bus
            .session
            .declare_queryable("tatbot/session/ctl/status")
            .callback(|q| {
                q.reply("tatbot/session/ctl/status", b"idle".as_slice())
                    .wait()
                    .unwrap();
            })
            .wait()
            .unwrap();
        let replies = bus.session.get("tatbot/session/ctl/status").wait().unwrap();
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(
            reply.result().unwrap().payload().to_bytes().as_ref(),
            b"idle"
        );
    }
}
