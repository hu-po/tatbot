//! TCP-only Zenoh transport. Scouting is disabled; endpoints are explicit.
use crate::Envelope;
use serde::{Serialize, de::DeserializeOwned};
use zenoh::Wait;
pub struct Bus {
    pub session: zenoh::Session,
}
/// The fleet bus credential: the `user:password` line of the file
/// `$TATBOT_BUS_CREDENTIALS` names, else `~/.config/tatbot/bus-credentials`.
/// The router admits only it once its node holds the file; without one the
/// bus is open, as it was. An empty variable means no credential.
#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    pub user: String,
    pub password: String,
}
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Credentials {{ user: {:?}, password: <redacted> }}",
            self.user
        )
    }
}
impl Credentials {
    pub fn load() -> zenoh::Result<Option<Self>> {
        let path = match std::env::var_os("TATBOT_BUS_CREDENTIALS") {
            Some(path) if path.is_empty() => return Ok(None),
            Some(path) => std::path::PathBuf::from(path),
            None => match std::env::var_os("HOME") {
                Some(home) => std::path::Path::new(&home).join(".config/tatbot/bus-credentials"),
                None => return Ok(None),
            },
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display()).into()),
        };
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(&path)?.permissions().mode() & 0o077 != 0 {
            return Err(format!("{} is readable by others: chmod 600 it", path.display()).into());
        }
        Self::parse(&text)
            .map(Some)
            .ok_or_else(|| format!("{}: no user:password line", path.display()).into())
    }
    /// The first non-blank line, `user:password` with both trimmed and non-empty:
    /// zenohd reads the same file as its dictionary and refuses anything else.
    pub fn parse(text: &str) -> Option<Self> {
        let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
        let (user, password) = line.split_once(':')?;
        let (user, password) = (user.trim(), password.trim());
        (!user.is_empty() && !password.is_empty()).then(|| Self {
            user: user.into(),
            password: password.into(),
        })
    }
}
impl Bus {
    /// A session on the fleet bus: a client of `connect` (the router), or with no
    /// endpoint a peer that only this process's other sessions reach (tests). It
    /// never listens: the router is the only listener the fleet runs. A client
    /// presents this node's fleet credential when it holds one.
    pub fn open(connect: &[String]) -> zenoh::Result<Self> {
        let credentials = if connect.is_empty() {
            None
        } else {
            Credentials::load()?
        };
        Self::open_with(connect, credentials.as_ref())
    }
    /// `open` with the credential stated: a test's own router needs none.
    pub fn open_with(connect: &[String], credentials: Option<&Credentials>) -> zenoh::Result<Self> {
        let mut config = zenoh::Config::default();
        // Fleet services attach to the explicit router. Peer mode depends on
        // peer discovery/routing that the fleet deliberately disables.
        if !connect.is_empty() {
            config.insert_json5("mode", "\"client\"")?;
        }
        config.insert_json5("scouting/multicast/enabled", "false")?;
        config.insert_json5("connect/timeout_ms", "2000")?;
        config.insert_json5("connect/exit_on_failure", "true")?;
        config.insert_json5("connect/endpoints", &serde_json::to_string(connect)?)?;
        config.insert_json5("listen/endpoints", "[]")?;
        if let Some(c) = credentials {
            config.insert_json5(
                "transport/auth/usrpwd/user",
                &serde_json::to_string(&c.user)?,
            )?;
            config.insert_json5(
                "transport/auth/usrpwd/password",
                &serde_json::to_string(&c.password)?,
            )?;
        }
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
        let bus = Bus::open(&[]).unwrap();
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
    #[test]
    fn credentials_are_the_first_user_password_line() {
        let c = Credentials::parse("\n tatbot : pa:ss \nother:x\n").unwrap();
        assert_eq!((c.user.as_str(), c.password.as_str()), ("tatbot", "pa:ss"));
        assert!(!format!("{c:?}").contains("pa:ss"));
        for text in [
            "",
            "# a comment, which zenohd refuses",
            "tatbot",
            "tatbot:",
            ":secret",
        ] {
            assert_eq!(Credentials::parse(text), None, "{text:?}");
        }
    }
    #[test]
    fn a_router_with_the_fleet_dictionary_admits_only_its_credential() {
        let dir = std::env::temp_dir().join(format!("tatbot-bus-auth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dictionary = dir.join("bus-credentials");
        std::fs::write(&dictionary, "tatbot:fleet-secret\n").unwrap();
        let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp/{}", socket.local_addr().unwrap());
        drop(socket);
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
        config
            .insert_json5(
                "transport/auth/usrpwd/dictionary_file",
                &serde_json::to_string(&dictionary).unwrap(),
            )
            .unwrap();
        let _router = zenoh::open(config).wait().unwrap();
        let connect = std::slice::from_ref(&endpoint);
        let fleet = Credentials::parse("tatbot:fleet-secret").unwrap();
        let wrong = Credentials::parse("tatbot:guess").unwrap();
        assert!(Bus::open_with(connect, Some(&fleet)).is_ok());
        assert!(Bus::open_with(connect, None).is_err());
        assert!(Bus::open_with(connect, Some(&wrong)).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
