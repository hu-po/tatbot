//! Fleet service presence on the bus: `advertise` holds a liveliness token
//! while a supervised unit is active, `services` lists and checks them.
//! No arm or serial device can be opened here.
use clap::{Parser, Subcommand};
use std::time::Duration;
use zenoh::Wait;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    connect: Vec<String>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Services {
        #[arg(long)]
        expected_sha: Option<String>,
        #[arg(long)]
        require: Vec<String>,
    },
    Advertise {
        #[arg(long)]
        node: String,
        #[arg(long)]
        service: String,
        #[arg(long)]
        unit: String,
    },
}
fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(!args.connect.is_empty(), "explicit bus endpoint required");
    let bus =
        tatbot_bus::transport::Bus::open(&args.connect, &[]).map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Command::Advertise {
        node,
        service,
        unit,
    } = &args.command
    {
        anyhow::ensure!(
            !unit.starts_with('-') && unit.ends_with(".service"),
            "invalid supervised unit"
        );
        let active = || {
            std::process::Command::new("systemctl")
                .args(["is-active", "--quiet", unit])
                .status()
                .is_ok_and(|s| s.success())
        };
        anyhow::ensure!(active(), "supervised service is not active");
        let _lease = tatbot_bus::service::ServiceLease::declare(
            &bus,
            tatbot_bus::Producer {
                node: node.clone(),
                pid: std::process::id(),
                sha: option_env!("TATBOT_SOURCE_COMMIT")
                    .unwrap_or("development")
                    .into(),
                run_id: std::env::var("TATBOT_RUN_ID").unwrap_or_else(|_| "service".into()),
            },
            service,
            Vec::new(),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        while active() {
            std::thread::sleep(Duration::from_secs(1));
        }
        return Ok(());
    }
    if let Command::Services {
        expected_sha,
        require,
    } = &args.command
    {
        let replies = bus
            .session
            .liveliness()
            .get("tatbot/alive/*/*")
            .timeout(Duration::from_secs(2))
            .wait()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut keys = std::collections::BTreeSet::new();
        while let Ok(Some(reply)) = replies.recv_timeout(Duration::from_secs(3)) {
            if let Ok(sample) = reply.result() {
                keys.insert(sample.key_expr().as_str().to_owned());
            }
        }
        let mut services = Vec::new();
        let mut mismatches = Vec::new();
        let mut missing = Vec::new();
        for name in require {
            if !keys.contains(&format!("tatbot/alive/{name}")) {
                missing.push(name.clone());
            }
        }
        for key in keys {
            let response = bus
                .session
                .get(key.clone())
                .target(zenoh::query::QueryTarget::All)
                .consolidation(zenoh::query::ConsolidationMode::None)
                .timeout(Duration::from_secs(2))
                .wait()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut replies = response.into_iter();
            let reply = replies
                .next()
                .ok_or_else(|| anyhow::anyhow!("liveliness metadata unavailable: {key}"))?;
            let sample = reply.result().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            anyhow::ensure!(
                replies.next().is_none(),
                "duplicate service producers: {key}"
            );
            let value: tatbot_bus::Envelope<tatbot_bus::Liveliness> =
                serde_json::from_slice(&sample.payload().to_bytes())?;
            anyhow::ensure!(
                value.schema == "tatbot.liveliness/1"
                    && value.payload.sha == value.producer.sha
                    && value.payload.pid == value.producer.pid
                    && key.starts_with(&format!("tatbot/alive/{}/", value.producer.node)),
                "invalid service provenance"
            );
            if (require.is_empty()
                || require
                    .iter()
                    .any(|name| key == format!("tatbot/alive/{name}")))
                && expected_sha
                    .as_ref()
                    .is_some_and(|sha| sha != &value.payload.sha)
            {
                mismatches.push(key.clone());
            }
            services.push(serde_json::json!({"key":key,"value":value}));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"schema":"tatbot.services/1","services":services,"missing":missing,"sha_mismatches":mismatches,"complete":missing.is_empty() && mismatches.is_empty()})
            )?
        );
        if !mismatches.is_empty() {
            std::process::exit(3);
        }
        if !missing.is_empty() {
            std::process::exit(5);
        }
        return Ok(());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        println!(
            "{}",
            serde_json::json!({"schema":"tatbot.session-status/1","available":false,"reason":format!("{e:#}")})
        );
        std::process::exit(5);
    }
}
