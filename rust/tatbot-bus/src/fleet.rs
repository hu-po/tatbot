//! Fleet roles, resolved from `config/nodes.json` at compile time (build.rs
//! falls back to `config/examples/nodes.json` where a checkout has none).
//!
//! No crate names a node: a deployment describing a different fleet in its own
//! `config/nodes.json` gets its own answer without a patch.

use std::sync::OnceLock;

const NODES_JSON: &str = include_str!(env!("TATBOT_NODES_JSON"));

/// The node carrying `role` — its `hostname` alias where it has one. `None`
/// when the map names no such node.
pub fn node_with_role(role: &str) -> Option<&'static str> {
    static NODES: OnceLock<serde_json::Value> = OnceLock::new();
    let nodes =
        NODES.get_or_init(|| serde_json::from_str(NODES_JSON).unwrap_or(serde_json::Value::Null));
    nodes.as_object()?.iter().find_map(|(name, value)| {
        if name.starts_with("//") || name.starts_with("__") {
            return None;
        }
        let carries = value
            .get("roles")?
            .as_array()?
            .iter()
            .any(|r| r.as_str() == Some(role));
        if !carries {
            return None;
        }
        Some(
            value
                .get("hostname")
                .and_then(|v| v.as_str())
                .unwrap_or(name.as_str()),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_a_role_and_refuses_an_unknown_one() {
        assert!(node_with_role("arm").is_some());
        assert!(node_with_role("no-such-role").is_none());
    }
}
