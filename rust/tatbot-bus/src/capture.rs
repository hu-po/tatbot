//! The configured arm registry: each arm's non-authorizing physical identity
//! binding, parsed from the source build's `config/arms.json`.
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

/// Non-authorizing physical identity binding from the source build's
/// `config/arms.json`. A renamed session arm keeps its measured controller,
/// workspace and URDF references; capture never guesses them from its ID.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmBinding {
    pub control_role: String,
    pub profile_ip_field: String,
    pub controller_config: String,
    pub sdk_end_effector: String,
    pub workspace_section: String,
    pub urdf_prefix: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmRegistry {
    schema: String,
    arms: BTreeMap<String, ArmBinding>,
}

fn arm_token(value: &str) -> bool {
    value.len() <= 64
        && value
            .bytes()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

impl ArmRegistry {
    /// Parse and validate all identities before accepting any one of them.
    /// This checks reference uniqueness, not installed hardware or calibration.
    pub fn parse(json: &str) -> Result<Self, String> {
        let registry: Self =
            serde_json::from_str(json).map_err(|error| format!("invalid arm registry: {error}"))?;
        if registry.schema != "tatbot.arms/1" || registry.arms.is_empty() {
            return Err("arm registry requires tatbot.arms/1 and at least one arm".into());
        }
        let mut profile_ips = BTreeSet::new();
        let mut controller_configs = BTreeSet::new();
        let mut workspace_sections = BTreeSet::new();
        let mut urdf_prefixes = BTreeSet::new();
        for (id, arm) in &registry.arms {
            if !arm_token(id)
                || !arm_token(&arm.control_role)
                || !arm_token(&arm.profile_ip_field)
                || !arm_token(&arm.workspace_section)
                || !arm_token(&arm.urdf_prefix)
                || arm.workspace_section != arm.urdf_prefix
                || arm.sdk_end_effector.trim().is_empty()
            {
                return Err(format!("invalid arm binding: {id}"));
            }
            let parts: Vec<_> = arm.controller_config.split('/').collect();
            if parts.len() < 2
                || parts[0] != "config"
                || parts
                    .iter()
                    .any(|part| part.is_empty() || *part == "." || *part == "..")
                || !parts.last().is_some_and(|name| {
                    name.strip_suffix(".yaml")
                        .is_some_and(|stem| !stem.is_empty())
                })
            {
                return Err(format!("invalid controller config for arm {id}"));
            }
            if !profile_ips.insert(&arm.profile_ip_field)
                || !controller_configs.insert(&arm.controller_config)
                || !workspace_sections.insert(&arm.workspace_section)
                || !urdf_prefixes.insert(&arm.urdf_prefix)
            {
                return Err(format!("duplicate physical arm reference: {id}"));
            }
        }
        Ok(registry)
    }

    pub fn binding(&self, id: &str) -> Result<&ArmBinding, String> {
        self.arms
            .get(id)
            .ok_or_else(|| format!("unknown configured arm {id:?}"))
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.arms.keys().map(String::as_str)
    }

    pub fn binding_for_prefix(&self, prefix: &str) -> Result<&ArmBinding, String> {
        self.arms
            .values()
            .find(|arm| arm.urdf_prefix == prefix)
            .ok_or_else(|| format!("unknown configured URDF arm prefix {prefix:?}"))
    }
}

/// The same arm registry as the source build, like the fleet map in `fleet`.
/// A changed registry requires rebuilding the services that validate captures.
pub fn configured_arms() -> Result<&'static ArmRegistry, String> {
    static ARMS: OnceLock<Result<ArmRegistry, String>> = OnceLock::new();
    ARMS.get_or_init(|| ArmRegistry::parse(include_str!("../../../config/arms.json")))
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_value(bindings: &[(&str, &str)]) -> serde_json::Value {
        let arms: serde_json::Map<String, serde_json::Value> = bindings
            .iter()
            .enumerate()
            .map(|(index, (id, prefix))| {
                (
                    (*id).into(),
                    serde_json::json!({
                        "control_role":"arm",
                        "profile_ip_field":format!("arm_{index}_ip"),
                        "controller_config":format!("config/trossen/arm_{index}.yaml"),
                        "sdk_end_effector":format!("arm_{index}_ee"),
                        "workspace_section":prefix,
                        "urdf_prefix":prefix,
                    }),
                )
            })
            .collect();
        serde_json::json!({"schema":"tatbot.arms/1", "arms":arms})
    }

    fn fixture_registry(bindings: &[(&str, &str)]) -> ArmRegistry {
        ArmRegistry::parse(&fixture_value(bindings).to_string()).unwrap()
    }

    #[test]
    fn one_two_and_three_configured_arms_accept_renamed_ids_only() {
        let bindings = [("starboard", "right"), ("port", "left"), ("spare", "third")];
        for count in 1..=3 {
            let arms = fixture_registry(&bindings[..count]);
            for (id, prefix) in &bindings[..count] {
                assert_eq!(arms.binding(id).unwrap().urdf_prefix, *prefix);
                assert_eq!(
                    arms.binding_for_prefix(prefix).unwrap().urdf_prefix,
                    *prefix
                );
            }
            assert!(arms.binding("right").is_err());
            assert_eq!(arms.ids().count(), count);
        }
    }

    #[test]
    fn the_configured_registry_parses() {
        assert!(configured_arms().unwrap().ids().next().is_some());
    }

    #[test]
    fn arm_registry_rejects_ambiguous_or_unsafe_bindings() {
        let mut value = fixture_value(&[("starboard", "right"), ("port", "left")]);
        assert!(ArmRegistry::parse(&value.to_string()).is_ok());
        value["arms"]["port"]["profile_ip_field"] = "arm_0_ip".into();
        assert!(ArmRegistry::parse(&value.to_string()).is_err());
        value["arms"]["port"]["profile_ip_field"] = "arm_1_ip".into();
        value["arms"]["port"]["controller_config"] = "../other.yaml".into();
        assert!(ArmRegistry::parse(&value.to_string()).is_err());
        value["arms"]["port"]["controller_config"] = "config/trossen/arm_1.yaml".into();
        value["arms"]["port"]["workspace_section"] = "right".into();
        assert!(ArmRegistry::parse(&value.to_string()).is_err());
    }
}
