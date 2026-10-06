//! Recovery profile input and cold native worker construction. Parsing a profile
//! is not hardware qualification and construction does not connect a controller.
use crate::{Error, Result, contact::ContactCap, recovery::PositionLimit};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
};

/// Rolling joint-effort trip for the worker's motion guard. The shoulder
/// holds 6-8.5 Nm of gravity load at the staged pose alone, so the 9 Nm
/// square-probe abort was one reach away from tripping an ordinary scan.
pub const OVERFORCE_NM: f64 = 15.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryProfile {
    pub staged_positions: [f64; 7],
    pub max_joint_velocity: f64,
    pub carriage_contact_cap_n: f64,
    pub carriage_contact_deflect_m: f64,
    pub carriage_retract_m: f64,
}
impl RecoveryProfile {
    /// Read the same profile selected by the world without opening a driver
    /// or requiring the private controller/address inventory. The small public
    /// fixture declares only a staged pose and uses the mock's stated limits.
    pub fn simulation_from_source(bytes: &[u8]) -> Result<Self> {
        let document: serde_yaml::Value = serde_yaml::from_slice(bytes).map_err(error)?;
        let follower = document["follower"]
            .as_mapping()
            .ok_or_else(|| error("follower profile missing"))?;
        if !follower.contains_key("staged_positions") {
            return Err(error("follower staged pose missing"));
        }
        let mut value = serde_yaml::to_value(Self::simulation()?).map_err(error)?;
        value.as_mapping_mut().unwrap().extend(follower.clone());
        let profile: Self = serde_yaml::from_value(value).map_err(error)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn simulation() -> Result<Self> {
        let value: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../../../config/examples/tatbot-sim.yaml"))
                .map_err(error)?;
        Ok(Self {
            staged_positions: serde_yaml::from_value(value["follower"]["staged_positions"].clone())
                .map_err(error)?,
            max_joint_velocity: 1.0,
            carriage_contact_cap_n: ContactCap::default().cap_n,
            carriage_contact_deflect_m: ContactCap::default().deflect_m,
            carriage_retract_m: 0.032,
        })
    }
    fn validate(&self) -> Result<()> {
        let cap = ContactCap::default();
        if self.staged_positions.iter().any(|q| !q.is_finite())
            || !self.max_joint_velocity.is_finite()
            || self.max_joint_velocity <= 0.0
            || self.carriage_contact_cap_n != cap.cap_n
            || self.carriage_contact_deflect_m != cap.deflect_m
            || self.carriage_retract_m != 0.032
        {
            return Err(Error(
                "profile differs from supported recovery/contact contract".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Deserialize)]
struct Document {
    follower: RecoveryProfile,
}
/// `<role>.carriage_qualified` is required for the role being loaded: a
/// profile that does not say whether this carriage's effort channel and
/// retract are qualified is refused rather than assumed either way.
fn carriage_qualified(profile: &[u8], role: ControllerRole) -> Result<bool> {
    let document: serde_yaml::Value = serde_yaml::from_slice(profile).map_err(error)?;
    document[role.name()]["carriage_qualified"]
        .as_bool()
        .ok_or_else(|| {
            error(format!(
                "tatbot.yaml {}.carriage_qualified must be true or false",
                role.name()
            ))
        })
}
#[derive(Clone, Debug, Deserialize)]
struct Limit {
    position_min: f64,
    position_max: f64,
    velocity_max: f64,
}
#[derive(Deserialize)]
struct Golden {
    manual_ip: Ipv4Addr,
    joint_limits: Vec<Limit>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerRole {
    Leader,
    Follower,
}
impl ControllerRole {
    fn golden_name(self) -> &'static str {
        match self {
            Self::Leader => "leader.yaml",
            Self::Follower => "follower.yaml",
        }
    }
    /// The profile block this role reads in `tatbot.yaml`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::Follower => "follower",
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct NativeProfile {
    pub role: ControllerRole,
    pub recovery: RecoveryProfile,
    pub address: Ipv4Addr,
    pub profile_sha256: String,
    pub golden_sha256: String,
    pub profile_path: PathBuf,
    pub golden_path: PathBuf,
    pub angular_envelope: f64,
    pub position_limits: [PositionLimit; 7],
    /// `tatbot.yaml <role>.carriage_qualified`: this carriage's effort channel
    /// counts as contact and its trip retract and timed moves are permitted.
    /// False keeps the deflection screen and holds on a trip. Flipping it is
    /// a bench receipt (the datum measured, a native retract exercised on
    /// that carriage), never an edit made to run a program.
    pub carriage_qualified: bool,
}
fn error(e: impl std::fmt::Display) -> Error {
    Error(format!("recovery profile: {e}"))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
impl NativeProfile {
    /// Existing callers retain the follower contract.
    pub fn load(root: &Path) -> Result<Self> {
        Self::load_role(root, ControllerRole::Follower)
    }
    /// The common staged pose comes from the same profile used by two-arm
    /// teleop landing. Recovery targets carry the measured carriage to the
    /// staged pose; the role's `carriage_qualified` says whether its retract
    /// is permitted and whether it returns to the staged rest at Sleep.
    /// Parsing is not powered acceptance.
    pub fn load_role(root: &Path, role: ControllerRole) -> Result<Self> {
        let profile_path = root.join("tatbot.yaml").canonicalize().map_err(error)?;
        let golden_path = root
            .join(role.golden_name())
            .canonicalize()
            .map_err(error)?;
        let profile = std::fs::read(&profile_path).map_err(error)?;
        let golden = std::fs::read(&golden_path).map_err(error)?;
        let document: Document = serde_yaml::from_slice(&profile).map_err(error)?;
        document.follower.validate()?;
        let carriage_qualified = carriage_qualified(&profile, role)?;
        let controller: Golden = serde_yaml::from_slice(&golden).map_err(error)?;
        if controller.joint_limits.len() != 7
            || controller.address_invalid()
            || controller.joint_limits.iter().any(|l| {
                !l.position_min.is_finite()
                    || !l.position_max.is_finite()
                    || l.position_min >= l.position_max
                    || !l.velocity_max.is_finite()
                    || l.velocity_max <= 0.0
            })
        {
            return Err(Error(
                "invalid controller address or seven-axis limits".into(),
            ));
        }
        if controller.joint_limits[..6]
            .iter()
            .any(|l| document.follower.max_joint_velocity > l.velocity_max)
        {
            return Err(Error("profile velocity exceeds controller golden".into()));
        }
        let limits = std::array::from_fn(|i| PositionLimit {
            min: controller.joint_limits[i].position_min,
            max: controller.joint_limits[i].position_max,
        });
        // The staged pose and the carriage rest it carries must sit inside
        // the controller's own limits before any landing is planned from them.
        crate::recovery::Targets::from_measured(
            document.follower.staged_positions,
            document.follower.staged_positions,
            &limits,
            Some(document.follower.staged_positions[6]),
        )?;
        let angular_envelope = controller.joint_limits[..6]
            .iter()
            .map(|l| l.position_min.abs().max(l.position_max.abs()))
            .fold(0.0_f64, f64::max);
        Ok(Self {
            role,
            recovery: document.follower,
            address: controller.manual_ip,
            profile_sha256: digest(&profile),
            golden_sha256: digest(&golden),
            profile_path,
            golden_path,
            angular_envelope,
            position_limits: limits,
            carriage_qualified,
        })
    }
    /// Copy exact validated inputs into a fresh run directory before creating a
    /// driver. Concurrent checkout changes cannot silently replace its golden.
    pub fn snapshot(&self, directory: &Path) -> Result<Self> {
        let profile = std::fs::read(&self.profile_path).map_err(error)?;
        let golden = std::fs::read(&self.golden_path).map_err(error)?;
        if digest(&profile) != self.profile_sha256 || digest(&golden) != self.golden_sha256 {
            return Err(Error("recovery profile changed after validation".into()));
        }
        std::fs::create_dir(directory).map_err(error)?;
        use std::io::Write;
        for (name, bytes) in [("tatbot.yaml", profile), (self.role.golden_name(), golden)] {
            let mut file = std::fs::File::create_new(directory.join(name)).map_err(error)?;
            file.write_all(&bytes).map_err(error)?;
            file.sync_all().map_err(error)?;
        }
        Self::load_role(directory, self.role)
    }
    /// Run the recovery contract against a cold mock controller using this
    /// profile's limits and control caps. Never opens an SDK or serial device.
    pub fn spawn_mock_recovery(
        &self,
        seed: [f64; 7],
        stop: std::sync::Arc<std::sync::atomic::AtomicI32>,
        policy: crate::ContactPolicy,
    ) -> Result<crate::worker::Worker> {
        let velocity = self.recovery.max_joint_velocity;
        let mut backend = crate::MockArm::recovery_seed(self.position_limits, seed)?;
        backend.contact_policy = policy;
        backend.carriage_qualified = self.carriage_qualified;
        crate::worker::Worker::spawn(
            crate::Control {
                backend,
                estop: stop,
                max_velocity: velocity,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: self.angular_envelope,
            },
            std::time::Duration::from_micros(2500),
            crate::motion_guard::MotionGuard::new(velocity, OVERFORCE_NM, 0.5, 0.5, 8),
        )
    }
    #[cfg(feature = "trossen")]
    pub fn spawn(
        self,
        stop: std::sync::Arc<std::sync::atomic::AtomicI32>,
        lease: std::sync::Arc<crate::lease::HardwareLease>,
        policy: crate::ContactPolicy,
    ) -> Result<crate::worker::Worker> {
        self.spawn_native(stop, lease, policy, None)
    }
    /// The same cold native worker, optionally forwarding every tick's
    /// feedback to a recorder (the hand-guiding owner retains all of it).
    #[cfg(feature = "trossen")]
    pub fn spawn_native(
        self,
        stop: std::sync::Arc<std::sync::atomic::AtomicI32>,
        lease: std::sync::Arc<crate::lease::HardwareLease>,
        policy: crate::ContactPolicy,
        telemetry: Option<std::sync::mpsc::SyncSender<crate::worker::TelemetrySample>>,
    ) -> Result<crate::worker::Worker> {
        use crate::{
            Control,
            motion_guard::MotionGuard,
            trossen::{Config, Role, TrossenArm},
            worker::Worker,
        };
        let profile = Self::load_role(
            self.profile_path
                .parent()
                .ok_or_else(|| error("profile parent missing"))?,
            self.role,
        )?;
        if profile.profile_sha256 != self.profile_sha256
            || profile.golden_sha256 != self.golden_sha256
        {
            return Err(error("profile changed before worker construction"));
        }
        let velocity = profile.recovery.max_joint_velocity;
        let construct = move || {
            Ok(Control {
                backend: TrossenArm::new(
                    Config {
                        address: profile.address,
                        role: match profile.role {
                            ControllerRole::Leader => Role::Leader,
                            ControllerRole::Follower => Role::Follower,
                        },
                        golden: profile.golden_path,
                        contact: policy,
                        carriage_qualified: profile.carriage_qualified,
                    },
                    stop.clone(),
                    lease,
                )?,
                estop: stop,
                max_velocity: velocity,
                max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                max_contact: 1.0,
                envelope: profile.angular_envelope,
            })
        };
        let guard = MotionGuard::new(velocity, OVERFORCE_NM, 0.5, 0.5, 8);
        match telemetry {
            Some(sender) => Worker::spawn_recording(construct, guard, sender),
            None => Worker::spawn_with(construct, std::time::Duration::from_micros(2500), guard),
        }
        .map(Worker::bound_native_shutdown)
    }
    /// The controller role the physical-arm registry names for this arm.
    pub fn role_named(name: &str) -> Result<ControllerRole> {
        match name {
            "leader" => Ok(ControllerRole::Leader),
            "follower" => Ok(ControllerRole::Follower),
            other => Err(error(format!("unknown controller role {other:?}"))),
        }
    }
    /// Mock worker for the hand-guiding owner: this profile's limits and
    /// caps around a seeded offline controller; never an SDK or serial device.
    pub fn spawn_mock_guide(
        &self,
        seed: [f64; 7],
        stop: std::sync::Arc<std::sync::atomic::AtomicI32>,
        policy: crate::ContactPolicy,
        telemetry: std::sync::mpsc::SyncSender<crate::worker::TelemetrySample>,
    ) -> Result<crate::worker::Worker> {
        let velocity = self.recovery.max_joint_velocity;
        let mut backend = crate::MockArm::recovery_seed(self.position_limits, seed)?;
        backend.contact_policy = policy;
        backend.carriage_qualified = self.carriage_qualified;
        let envelope = self.angular_envelope;
        crate::worker::Worker::spawn_recording(
            move || {
                Ok(crate::Control {
                    backend,
                    estop: stop,
                    max_velocity: velocity,
                    max_tracking_error: crate::TRACKING_ERROR_LIMIT_RAD,
                    max_contact: 1.0,
                    envelope,
                })
            },
            crate::motion_guard::MotionGuard::new(velocity, OVERFORCE_NM, 0.5, 0.5, 8),
            telemetry,
        )
    }
}
impl Golden {
    fn address_invalid(&self) -> bool {
        self.manual_ip.is_unspecified() || self.manual_ip.is_multicast()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// False, with a note, in a checkout that omits the deployment's `rel`
    /// (the public export): a test that reads it returns early there.
    fn deployed(rel: &str) -> bool {
        let present = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel).is_file();
        if !present {
            eprintln!("skipped: needs the deployment's {rel}");
        }
        present
    }
    fn source() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/trossen")
    }
    #[test]
    fn physics_profile_preserves_the_selected_limits_and_contact_contract() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        let bytes = std::fs::read(source().join("tatbot.yaml")).unwrap();
        let native = NativeProfile::load(&source()).unwrap().recovery;
        let physics = RecoveryProfile::simulation_from_source(&bytes).unwrap();
        assert_eq!(
            serde_json::to_value(native).unwrap(),
            serde_json::to_value(physics).unwrap()
        );
        let public = include_bytes!("../../../config/examples/tatbot-sim.yaml");
        assert_eq!(
            serde_json::to_value(RecoveryProfile::simulation_from_source(public).unwrap()).unwrap(),
            serde_json::to_value(RecoveryProfile::simulation().unwrap()).unwrap()
        );
        let mut changed: serde_yaml::Value = serde_yaml::from_slice(public).unwrap();
        changed["follower"]["carriage_contact_cap_n"] = serde_yaml::to_value(19.0).unwrap();
        assert!(
            RecoveryProfile::simulation_from_source(
                serde_yaml::to_string(&changed).unwrap().as_bytes()
            )
            .is_err()
        );
    }
    #[test]
    fn mock_profile_limits_are_explicit_and_validate_without_changing_defaults() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        use crate::ArmBackend;
        let profile = NativeProfile::load(&source()).unwrap();
        let mut arm = crate::MockArm::with_recovery_limits(profile.position_limits).unwrap();
        for (actual, expected) in arm
            .recovery_limits()
            .unwrap()
            .iter()
            .zip(profile.position_limits)
        {
            assert_eq!(actual.min, expected.min);
            assert_eq!(actual.max, expected.max);
        }
        let defaults = crate::MockArm::default().recovery_limits().unwrap();
        assert_eq!(defaults[0].min, -1.0);
        assert_eq!(defaults[0].max, 1.0);
        let mut invalid = profile.position_limits;
        invalid[6].min = f64::NAN;
        assert!(crate::MockArm::with_recovery_limits(invalid).is_err());
    }
    #[test]
    fn leader_profile_retains_its_own_golden_and_unqualified_carriage() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        let profile = NativeProfile::load_role(&source(), ControllerRole::Leader).unwrap();
        let follower = NativeProfile::load(&source()).unwrap();
        assert_ne!(profile.golden_sha256, follower.golden_sha256);
        assert_ne!(profile.address, follower.address);
        assert!(!profile.carriage_qualified && follower.carriage_qualified);
        assert_eq!(
            profile.recovery.staged_positions,
            follower.recovery.staged_positions
        );
        let dir = tempfile::tempdir().unwrap();
        let retained = profile.snapshot(&dir.path().join("profile")).unwrap();
        assert_eq!(retained.role, ControllerRole::Leader);
        assert_eq!(retained.golden_sha256, profile.golden_sha256);
        assert!(retained.golden_path.ends_with("leader.yaml"));
        assert!(!retained.carriage_qualified);
        assert_eq!(
            serde_json::to_value(&retained).unwrap()["carriage_qualified"],
            serde_json::Value::Bool(false)
        );
    }
    /// The key is per role and required: a profile that omits it, or gives it
    /// a non-boolean, is refused for that role; the other role still loads.
    #[test]
    fn carriage_qualified_is_required_per_role_and_carried_into_the_spawned_backend() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        use crate::ArmBackend;
        let base = NativeProfile::load(&source()).unwrap();
        for (replacement, refused) in [
            ("  carriage_qualified: false\n", ControllerRole::Leader),
            ("  carriage_qualified: true\n", ControllerRole::Follower),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().to_path_buf();
            for name in ["tatbot.yaml", "leader.yaml", "follower.yaml"] {
                std::fs::copy(source().join(name), root.join(name)).unwrap();
            }
            let profile_path = root.join("tatbot.yaml");
            let text = std::fs::read_to_string(&profile_path).unwrap();
            assert_eq!(text.matches(replacement).count(), 1, "{replacement:?}");
            std::fs::write(&profile_path, text.replace(replacement, "")).unwrap();
            let error = NativeProfile::load_role(&root, refused)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("{}.carriage_qualified", refused.name())),
                "{error}"
            );
            let other = match refused {
                ControllerRole::Leader => ControllerRole::Follower,
                ControllerRole::Follower => ControllerRole::Leader,
            };
            assert!(NativeProfile::load_role(&root, other).is_ok());
            std::fs::write(
                &profile_path,
                text.replace(
                    replacement,
                    &replacement.replace("true", "yes-ish").replace("false", "0"),
                ),
            )
            .unwrap();
            assert!(NativeProfile::load_role(&root, refused).is_err());
        }
        let leader = NativeProfile::load_role(&source(), ControllerRole::Leader).unwrap();
        let mut backend =
            crate::MockArm::recovery_seed(leader.position_limits, leader.recovery.staged_positions)
                .unwrap();
        backend.carriage_qualified = leader.carriage_qualified;
        backend.contact_policy = crate::ContactPolicy::Contact;
        assert!(!backend.carriage_qualified());
        let refused = backend
            .retract_carriage(0.032, 0.6)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("carriage_qualified"), "{refused}");
        assert!(
            backend
                .move_carriage(0.0, 2.0)
                .unwrap_err()
                .to_string()
                .contains("carriage_qualified")
        );
        let mut follower =
            crate::MockArm::recovery_seed(base.position_limits, base.recovery.staged_positions)
                .unwrap();
        follower.carriage_qualified = base.carriage_qualified;
        assert!(follower.carriage_qualified());
        assert!(follower.retract_carriage(0.032, 0.6).is_ok());
    }
    #[test]
    fn actual_profile_preserves_staged_pose_and_cap_contract() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        let profile = NativeProfile::load(&source()).unwrap();
        assert_eq!(
            profile.recovery.staged_positions[5],
            std::f64::consts::FRAC_PI_2
        );
        assert_eq!(profile.recovery.carriage_retract_m, 0.032);
        assert_eq!(profile.recovery.carriage_contact_cap_n, 20.0);
        let temp = tempfile::tempdir().unwrap();
        let snapshot = profile.snapshot(&temp.path().join("profile")).unwrap();
        assert_eq!(profile.golden_sha256, snapshot.golden_sha256);
        std::fs::write(&snapshot.profile_path, "changed").unwrap();
        assert!(snapshot.snapshot(&temp.path().join("changed")).is_err());
    }
    #[test]
    fn changed_safety_settings_and_missing_values_refuse() {
        if !deployed("../../config/trossen/tatbot.yaml") {
            return;
        }
        let base = NativeProfile::load(&source()).unwrap();
        for replacement in [
            "carriage_contact_cap_n: 21.0",
            "carriage_contact_cap_n: .nan",
            "# cap omitted",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let snapshot = base.snapshot(&temp.path().join("profile")).unwrap();
            let text = std::fs::read_to_string(&snapshot.profile_path)
                .unwrap()
                .replace("carriage_contact_cap_n: 20.0", replacement);
            std::fs::write(&snapshot.profile_path, text).unwrap();
            assert!(NativeProfile::load(snapshot.profile_path.parent().unwrap()).is_err());
        }
    }
    #[cfg(feature = "trossen")]
    #[test]
    fn native_factory_is_cold_and_retains_lease_until_worker_drop() {
        use std::sync::{Arc, atomic::AtomicI32};
        let temp = tempfile::tempdir().unwrap();
        let lock = temp.path().join("lease");
        let lease = Arc::new(crate::lease::HardwareLease::for_test(&lock));
        let worker = NativeProfile::load(&source())
            .unwrap()
            .spawn(
                Arc::new(AtomicI32::new(crate::estop::FAULT)),
                lease.clone(),
                crate::ContactPolicy::Contact,
            )
            .unwrap();
        drop(lease);
        assert!(crate::lease::DriverLease::acquire(&lock).is_err());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while worker.status().fault.is_none() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(worker.status().measured.is_none());
        assert_eq!(worker.status().commands, 0);
        drop(worker);
        assert!(crate::lease::DriverLease::acquire(&lock).is_ok());
    }
}
