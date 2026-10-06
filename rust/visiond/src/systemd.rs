//! Readiness signalling for units run as `Type=notify`.
//!
//! `After=` orders process *starts*. A `Type=simple` unit counts as started
//! the moment it is exec'd, so a consumer ordered after the capture owner
//! still races its socket bind -- ordering that cannot mean what it looks
//! like it means. `Type=notify` closes the gap: the owner declares itself
//! ready once the listener is bound, and systemd holds every `After=`
//! consumer until that datagram arrives.
//!
//! The protocol is one datagram of newline-separated `KEY=value` fields on
//! `$NOTIFY_SOCKET`, so this needs no libsystemd dependency. A leading `@`
//! names a socket in the abstract namespace; a leading `/` is a filesystem
//! path. Nothing else is addressable, and neither is a relative path.

use std::{
    ffi::OsStr,
    os::{
        linux::net::SocketAddrExt,
        unix::{
            ffi::OsStrExt,
            net::{SocketAddr, UnixDatagram},
        },
    },
    path::Path,
};

use anyhow::{Context, Result};

/// Declare this process ready to serve, naming what became available.
///
/// Returns whether a datagram was sent. `false` means `NOTIFY_SOCKET` is
/// unset -- every operator-invoked run, and every unit that is not
/// `Type=notify`. A delivery failure is reported on stderr and swallowed:
/// systemd will fail the start on its own timeout, and aborting a capture
/// that is otherwise up would turn a reporting problem into an outage.
pub fn notify_ready(status: &str) -> bool {
    ready_with(std::env::var_os("NOTIFY_SOCKET").as_deref(), status)
}

/// Declare a capture process ready, naming the socket its consumers wait on.
///
/// Both capture paths call this at the same point -- the listener is bound,
/// the cameras are not yet connected. A subscriber needs the listener; one
/// camera slow to come up must not hold the unit's start open.
pub fn notify_socket_ready(socket: Option<&Path>) -> bool {
    notify_ready(&match socket {
        Some(path) => format!("frame socket bound at {}", path.display()),
        None => "capturing without a frame socket".into(),
    })
}

fn ready_with(address: Option<&OsStr>, status: &str) -> bool {
    let Some(address) = address else {
        return false;
    };
    // A status field is one line by construction; fold anything else so a
    // stray newline cannot forge a second field.
    let status = status.replace(['\n', '\r'], " ");
    match notify(address, &format!("READY=1\nSTATUS={status}\n")) {
        Ok(()) => true,
        Err(error) => {
            eprintln!("systemd readiness not delivered: {error:#}");
            false
        }
    }
}

fn notify(address: &OsStr, message: &str) -> Result<()> {
    let address = notify_address(address)?;
    let socket = UnixDatagram::unbound().context("opening a notify datagram socket")?;
    let sent = socket
        .send_to_addr(message.as_bytes(), &address)
        .context("sending the readiness datagram")?;
    anyhow::ensure!(
        sent == message.len(),
        "readiness datagram truncated to {sent} of {} bytes",
        message.len()
    );
    Ok(())
}

fn notify_address(value: &OsStr) -> Result<SocketAddr> {
    match value.as_bytes() {
        [b'@', name @ ..] => {
            SocketAddr::from_abstract_name(name).context("abstract notify socket name")
        }
        [b'/', ..] => SocketAddr::from_pathname(Path::new(value)).context("notify socket path"),
        _ => anyhow::bail!(
            "NOTIFY_SOCKET must be an absolute path or an @-prefixed abstract name, got {:?}",
            value
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn receive(listener: &UnixDatagram) -> String {
        let mut buffer = [0_u8; 256];
        let read = listener.recv(&mut buffer).unwrap();
        String::from_utf8(buffer[..read].to_vec()).unwrap()
    }

    #[test]
    fn readiness_reaches_a_path_notify_socket() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notify");
        let listener = UnixDatagram::bind(&path).unwrap();
        listener
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();

        assert!(ready_with(Some(path.as_os_str()), "frame socket bound"));
        assert_eq!(receive(&listener), "READY=1\nSTATUS=frame socket bound\n");
    }

    #[test]
    fn readiness_reaches_an_abstract_notify_socket() {
        // systemd hands out an abstract name on most systems, so the '@' form
        // is the one that actually runs in production -- not the path form.
        let name = format!("tatbot-notify-test-{}", std::process::id());
        let listener =
            UnixDatagram::bind_addr(&SocketAddr::from_abstract_name(&name).unwrap()).unwrap();
        listener
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();

        let address = format!("@{name}");
        assert!(ready_with(Some(OsStr::new(&address)), "frame socket bound"));
        assert_eq!(receive(&listener), "READY=1\nSTATUS=frame socket bound\n");
    }

    #[test]
    fn a_status_line_cannot_forge_a_second_field() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notify");
        let listener = UnixDatagram::bind(&path).unwrap();
        listener
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();

        assert!(ready_with(Some(path.as_os_str()), "bound\nMAINPID=1"));
        assert_eq!(receive(&listener), "READY=1\nSTATUS=bound MAINPID=1\n");
    }

    #[test]
    fn an_unset_notify_socket_is_not_an_error() {
        // The operator-invoked case: `tatbot vision ...` by hand sends
        // nothing and must not warn about it.
        assert!(!ready_with(None, "frame socket bound"));
    }

    #[test]
    fn notify_rejects_an_address_that_is_neither_a_path_nor_abstract() {
        let error = notify_address(OsStr::new("notify.sock")).unwrap_err();
        assert!(error.to_string().contains("absolute path"), "{error:#}");
        assert!(notify_address(OsStr::new("")).is_err());
    }
}
