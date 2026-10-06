//! Pinned vendor ABI only. This crate never supplies launch, fixture, driver
//! ownership, E-stop or trajectory qualification. Callers must provide those
//! through tatbot-arm and the gated CLI. No hardware backend is enabled here.
pub const SDK_VERSION: &str = "1.8.5";
pub const SDK_AVAILABLE: bool = cfg!(feature = "sdk");
#[cfg(feature = "sdk")]
pub type DriverPtr = cxx::UniquePtr<ffi::Driver>;

#[cfg(feature = "sdk")]
#[cxx::bridge(namespace = "tatbot::trossen")]
pub mod ffi {
    /// Controller limits: six rotational joints followed by the linear carriage.
    struct JointLimit {
        position_min: f64,
        position_max: f64,
        position_tolerance: f64,
        velocity_max: f64,
        velocity_tolerance: f64,
        effort_max: f64,
        effort_tolerance: f64,
    }
    struct Measurement {
        positions: Vec<f64>,
        velocities: Vec<f64>,
        accelerations: Vec<f64>,
        efforts: Vec<f64>,
        external_efforts: Vec<f64>,
        compensation_efforts: Vec<f64>,
        modes: Vec<u8>,
        error: String,
    }
    /// The end-effector properties the controller compensates, as configured.
    struct EndEffectorReadback {
        palm_mass_kg: f64,
        finger_left_mass_kg: f64,
        finger_right_mass_kg: f64,
        palm_origin_xyz_m: Vec<f64>,
        palm_inertia: Vec<f64>,
        offset_finger_left_m: f64,
        offset_finger_right_m: f64,
        pitch_circle_radius_m: f64,
        t_flange_tool: Vec<f64>,
    }
    unsafe extern "C++" {
        include!("trossen-arm-sys/include/bridge.h");
        type Driver;
        fn make_driver() -> Result<UniquePtr<Driver>>;
        fn configure(
            self: Pin<&mut Driver>,
            ip: &str,
            follower: bool,
            clear_error: bool,
            timeout_s: f64,
        ) -> Result<()>;
        /// Joint feedback from the SDK's cached UDP output; modes and error
        /// re-queried over TCP only when older than `max_status_age_s`.
        fn measure(self: Pin<&mut Driver>, max_status_age_s: f64) -> Result<Measurement>;
        fn limits(self: Pin<&mut Driver>) -> Result<Vec<JointLimit>>;
        fn position_mode(self: Pin<&mut Driver>) -> Result<()>;
        fn idle_mode(self: Pin<&mut Driver>) -> Result<()>;
        /// Seven per-joint modes: 0 idle, 1 position, 3 external effort.
        fn joint_modes(self: Pin<&mut Driver>, modes: &[u8]) -> Result<()>;
        fn zero_arm_external_efforts(self: Pin<&mut Driver>) -> Result<()>;
        fn joint_position(
            self: Pin<&mut Driver>,
            index: u8,
            position: f64,
            goal_s: f64,
        ) -> Result<()>;
        fn end_effector(self: Pin<&mut Driver>) -> Result<EndEffectorReadback>;
        fn positions(
            self: Pin<&mut Driver>,
            q: &[f64],
            v: &[f64],
            a: &[f64],
            goal_s: f64,
        ) -> Result<()>;
        // Returns the exact measured target submitted, only after SDK acceptance.
        fn hold(self: Pin<&mut Driver>) -> Result<Vec<f64>>;
        fn load_config(self: Pin<&mut Driver>, path: &str) -> Result<()>;
        fn cleanup(self: Pin<&mut Driver>) -> Result<()>;
        // Pure validation is available without constructing/configuring a driver.
        fn validate_command(q: &[f64], v: &[f64], a: &[f64], goal_s: f64) -> Result<()>;
    }
}

#[cfg(all(test, feature = "sdk"))]
mod tests {
    use super::ffi;
    #[test]
    fn unconfigured_vendor_object_refuses_control_without_connecting() {
        let mut driver = ffi::make_driver().unwrap();
        assert!(driver.pin_mut().measure(0.0).is_err());
        assert!(driver.pin_mut().limits().is_err());
        assert!(driver.pin_mut().position_mode().is_err());
        assert!(driver.pin_mut().idle_mode().is_err());
        assert!(driver.pin_mut().hold().is_err());
        assert!(
            driver
                .pin_mut()
                .joint_modes(&[3, 3, 3, 3, 3, 3, 1])
                .is_err()
        );
        assert!(driver.pin_mut().zero_arm_external_efforts().is_err());
        assert!(driver.pin_mut().joint_position(6, 0.0, 0.0).is_err());
        assert!(driver.pin_mut().end_effector().is_err());
        // Argument validation precedes the connection check.
        assert!(
            driver
                .pin_mut()
                .joint_modes(&[3, 3, 3, 3, 3, 3])
                .unwrap_err()
                .to_string()
                .contains("seven")
        );
        assert!(
            driver
                .pin_mut()
                .joint_modes(&[2, 2, 2, 2, 2, 2, 1])
                .unwrap_err()
                .to_string()
                .contains("external_effort")
        );
        assert!(
            driver
                .pin_mut()
                .joint_position(7, 0.0, 0.0)
                .unwrap_err()
                .to_string()
                .contains("0..6")
        );
        assert!(driver.pin_mut().load_config("/not-a-config").is_err());
        let zero = [0.0; 7];
        assert!(
            driver
                .pin_mut()
                .positions(&zero, &zero, &zero, 0.0025)
                .is_err()
        );
        for ip in ["", "not-an-address", "127.0.0.1\0suffix"] {
            assert!(driver.pin_mut().configure(ip, true, false, 1.0).is_err());
        }
        for timeout in [0.0, -1.0, 21.0, f64::NAN] {
            // Invalid durations refuse before the vendor configure call. Only
            // loopback is named even if this validation were to regress.
            assert!(
                driver
                    .pin_mut()
                    .configure("127.0.0.1", true, false, timeout)
                    .is_err()
            );
        }
        assert!(driver.pin_mut().position_mode().is_err());
    }
    #[test]
    fn native_argument_errors_cross_the_bridge_without_a_driver() {
        let zero = [0.0; 7];
        assert!(ffi::validate_command(&zero, &zero, &zero, 0.0025).is_ok());
        assert!(ffi::validate_command(&zero, &zero, &zero, 0.0).is_ok());
        for duration in [-1.0, 61.0, f64::NAN, f64::INFINITY] {
            assert!(ffi::validate_command(&zero, &zero, &zero, duration).is_err());
        }
        for field in 0..3 {
            for width in [0, 6, 8] {
                let bad = vec![0.0; width];
                let mut inputs: [&[f64]; 3] = [&zero, &zero, &zero];
                inputs[field] = &bad;
                assert!(ffi::validate_command(inputs[0], inputs[1], inputs[2], 0.0025).is_err());
            }
            let mut bad = zero;
            bad[6] = f64::NAN;
            let mut inputs: [&[f64]; 3] = [&zero, &zero, &zero];
            inputs[field] = &bad;
            let error = ffi::validate_command(inputs[0], inputs[1], inputs[2], 0.0025).unwrap_err();
            assert!(error.to_string().contains("finite"));
        }
    }
}
