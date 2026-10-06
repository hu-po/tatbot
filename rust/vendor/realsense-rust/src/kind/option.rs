//! Enumeration of RS2 sensor options.
//!
//! Not all options apply to every sensor. In order to retrieve the correct options,
//! one must iterate over the `sensor` object for option compatibility.
//!
//! Notice that this option refers to the `sensor`, not the device. However, the device
//! used also matters; sensors that are alike across devices are not guaranteed to share
//! the same sensor options. Again, it is up to the user to query whether an option
//! is supported by the sensor before attempting to set it. Failure to do so may cause
//! an error in operation.

use super::Rs2Exception;
use num_derive::{FromPrimitive, ToPrimitive};
use realsense_sys as sys;
use std::{ffi::CStr, fmt::Display};
use thiserror::Error;

/// Occur when an option cannot be set.
#[derive(Error, Debug)]
pub enum OptionSetError {
    /// The requested option is not supported by this sensor.
    #[error("Option not supported on this sensor.")]
    OptionNotSupported,
    /// The requested option is read-only and cannot be set.
    #[error("Option is read only.")]
    OptionIsReadOnly,
    /// The requested option could not be set. Reason is reported by the sensor.
    #[error("Could not set option. Type: {0}; Reason: {1}")]
    CouldNotSetOption(Rs2Exception, String),
}

/// The enumeration of options available in the RealSense SDK.
///
/// The majority of the options presented have a specific range of valid values. Run
/// `sensor.get_option_range(Rs2Option::_)` to retrieve possible values of an Option type for your sensor.
/// Setting a bad value will lead to a no-op at best, and a malfunction at worst.
///
/// # Deprecated Options
///
/// `AmbientLight`
///
/// - Equivalent to `RS2_OPTION_AMBIENT_LIGHT`
/// - Replacement: [Rs2Option::DigitalGain].
/// - Old Description: "Change the depth ambient light see rs2_ambient_light for values".
///
/// `ZeroOrderEnabled`
///
/// - Equivalent to `RS2_OPTION_ZERO_ORDER_ENABLED`
/// - Replacement: N/A.
/// - Old Description: "Toggle Zero-Order mode."
///
/// `ZeroOrderPointX`
///
/// - Equivalent to `RS2_OPTION_ZERO_ORDER_POINT_X`
/// - Replacement: N/A.
/// - Old Description: "Get the Zero order point x."
///
/// `ZeroOrderPointY`
///
/// - Equivalent to `RS2_OPTION_ZERO_ORDER_POINT_Y`
/// - Replacement: N/A.
/// - Old Description: "Get the Zero order point y."
///
/// `Trigger camera accuracy health`
///
/// - Deprecated as of 2.46 (not officially released, so technically 2.47)
/// - Old Description: "Enable Depth & color frame sync with periodic calibration for proper
///   alignment"
///
/// `Reset camera accuracy health`
///
/// - Deprecated as of 2.46 (not officially released, so technically 2.47)
/// - Old Description: "Reset Camera Accuracy metric (if affected by TriggerCameraAccuracyHealth
///   option)."
#[repr(i32)]
#[derive(FromPrimitive, ToPrimitive, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rs2Option {
    /// Enable/disable color backlight compensation.
    BacklightCompensation = sys::rs2_option_RS2_OPTION_BACKLIGHT_COMPENSATION as i32,
    /// Set color image brightness.
    Brightness = sys::rs2_option_RS2_OPTION_BRIGHTNESS as i32,
    /// Set color image contrast.
    Contrast = sys::rs2_option_RS2_OPTION_CONTRAST as i32,
    /// Set exposure time of color camera. Setting any value will disable auto exposure.
    Exposure = sys::rs2_option_RS2_OPTION_EXPOSURE as i32,
    /// Set color image gain.
    Gain = sys::rs2_option_RS2_OPTION_GAIN as i32,
    /// Set color image gamma setting.
    Gamma = sys::rs2_option_RS2_OPTION_GAMMA as i32,
    /// Set color image hue.
    Hue = sys::rs2_option_RS2_OPTION_HUE as i32,
    /// Set color image saturation.
    Saturation = sys::rs2_option_RS2_OPTION_SATURATION as i32,
    /// Set color image sharpness.
    Sharpness = sys::rs2_option_RS2_OPTION_SHARPNESS as i32,
    /// Set white balance of color image. Setting any value will disable auto white balance.
    WhiteBalance = sys::rs2_option_RS2_OPTION_WHITE_BALANCE as i32,
    /// Enable/disable color image auto-exposure.
    EnableAutoExposure = sys::rs2_option_RS2_OPTION_ENABLE_AUTO_EXPOSURE as i32,
    /// Enable/disable color image auto-white-balance
    EnableAutoWhiteBalance = sys::rs2_option_RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE as i32,
    /// Set the visual preset on the sensor. `sensor.get_option_range()` provides
    /// access to several recommend sets of option presets for a depth camera. The preset
    /// selection varies between devices and sensors.
    VisualPreset = sys::rs2_option_RS2_OPTION_VISUAL_PRESET as i32,
    /// Set the power of the laser emitter, with 0 meaning projector off.
    LaserPower = sys::rs2_option_RS2_OPTION_LASER_POWER as i32,
    /// Set the number of patterns projected per frame. The higher the accuracy value,
    /// the more patterns projected. Increasing the number of patterns helps to achieve
    /// better accuracy. Note that this control affects Depth FPS.
    Accuracy = sys::rs2_option_RS2_OPTION_ACCURACY as i32,
    /// Set the motion vs. range trade-off. Lower values allow for better motion sensitivity.
    /// Higher values allow for better depth range.
    MotionRange = sys::rs2_option_RS2_OPTION_MOTION_RANGE as i32,
    /// Set the filter to apply to each depth frame. Each one of the filter is optimized per the
    /// application requirements.
    FilterOption = sys::rs2_option_RS2_OPTION_FILTER_OPTION as i32,
    /// Set the confidence level threshold used by the Depth algorithm pipe.
    /// This determines whether a pixel will get a valid range or will be marked as invalid.
    ConfidenceThreshold = sys::rs2_option_RS2_OPTION_CONFIDENCE_THRESHOLD as i32,
    /// Enable/disable emitters. Emitter selection:
    ///
    /// - `0`: disable all emitters
    /// - `1`: enable laser
    /// - `2`: enable auto laser
    /// - `3`: enable LED
    EmitterEnabled = sys::rs2_option_RS2_OPTION_EMITTER_ENABLED as i32,
    /// Set the number of frames the user is allowed to keep per stream.
    /// Trying to hold on to more frames will cause frame drops.
    FramesQueueSize = sys::rs2_option_RS2_OPTION_FRAMES_QUEUE_SIZE as i32,
    /// Get the total number of detected frame drops from all streams.
    TotalFrameDrops = sys::rs2_option_RS2_OPTION_TOTAL_FRAME_DROPS as i32,
    /// Set the auto-exposure mode:
    ///
    /// - Static
    /// - Anti-Flicker
    /// - Hybrid
    AutoExposureMode = sys::rs2_option_RS2_OPTION_AUTO_EXPOSURE_MODE as i32,
    /// Set the power line frequency control for anti-flickering:
    ///
    /// - Off
    /// - 50Hz
    /// - 60Hz
    /// - Auto
    PowerLineFrequency = sys::rs2_option_RS2_OPTION_POWER_LINE_FREQUENCY as i32,
    /// Get the current Temperature of the ASIC.
    AsicTemperature = sys::rs2_option_RS2_OPTION_ASIC_TEMPERATURE as i32,
    /// Enable/disable error handling.
    ErrorPollingEnabled = sys::rs2_option_RS2_OPTION_ERROR_POLLING_ENABLED as i32,
    /// Get the Current Temperature of the projector.
    ProjectorTemperature = sys::rs2_option_RS2_OPTION_PROJECTOR_TEMPERATURE as i32,
    /// Enable/disable trigger to be outputed from the camera to any external device on
    /// every depth frame.
    OutputTriggerEnabled = sys::rs2_option_RS2_OPTION_OUTPUT_TRIGGER_ENABLED as i32,
    /// Get the current Motion-Module Temperature.
    MotionModuleTemperature = sys::rs2_option_RS2_OPTION_MOTION_MODULE_TEMPERATURE as i32,
    /// Set the number of meters represented by a single depth unit.
    DepthUnits = sys::rs2_option_RS2_OPTION_DEPTH_UNITS as i32,
    /// Enable/Disable automatic correction of the motion data.
    EnableMotionCorrection = sys::rs2_option_RS2_OPTION_ENABLE_MOTION_CORRECTION as i32,
    /// Allows sensor to dynamically ajust the frame rate depending on lighting conditions.
    AutoExposurePriority = sys::rs2_option_RS2_OPTION_AUTO_EXPOSURE_PRIORITY as i32,
    /// Set the color scheme for data visualization.
    ColorScheme = sys::rs2_option_RS2_OPTION_COLOR_SCHEME as i32,
    /// Enable/disable histogram equalization post-processing on the depth data.
    HistogramEqualizationEnabled = sys::rs2_option_RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED as i32,
    /// Set the Minimal distance to the target.
    MinDistance = sys::rs2_option_RS2_OPTION_MIN_DISTANCE as i32,
    /// Set the Maximum distance to the target.
    MaxDistance = sys::rs2_option_RS2_OPTION_MAX_DISTANCE as i32,
    /// Get the texture mapping stream unique ID.
    TextureSource = sys::rs2_option_RS2_OPTION_TEXTURE_SOURCE as i32,
    /// Set the 2D-filter effect. The specific interpretation is given within the context of the filter.
    FilterMagnitude = sys::rs2_option_RS2_OPTION_FILTER_MAGNITUDE as i32,
    /// Set the 2D-filter parameter that controls the weight/radius for smoothing.
    FilterSmoothAlpha = sys::rs2_option_RS2_OPTION_FILTER_SMOOTH_ALPHA as i32,
    /// Set the 2D-filter range/validity threshold.
    FilterSmoothDelta = sys::rs2_option_RS2_OPTION_FILTER_SMOOTH_DELTA as i32,
    /// Enhance depth data post-processing with holes filling where appropriate.
    HolesFill = sys::rs2_option_RS2_OPTION_HOLES_FILL as i32,
    /// Get the distance in mm between the first and the second imagers in stereo-based depth cameras.
    StereoBaseline = sys::rs2_option_RS2_OPTION_STEREO_BASELINE as i32,
    /// Allows dynamically ajust the converge step value of the target exposure in
    /// the Auto-Exposure algorithm.
    AutoExposureConvergeStep = sys::rs2_option_RS2_OPTION_AUTO_EXPOSURE_CONVERGE_STEP as i32,
    /// Impose Inter-camera HW synchronization mode. Applicable for D400/L500/Rolling Shutter SKUs.
    InterCamSyncMode = sys::rs2_option_RS2_OPTION_INTER_CAM_SYNC_MODE as i32,
    /// Select a stream to process.
    StreamFilter = sys::rs2_option_RS2_OPTION_STREAM_FILTER as i32,
    /// Select a stream format to process.
    StreamFormatFilter = sys::rs2_option_RS2_OPTION_STREAM_FORMAT_FILTER as i32,
    /// Select a stream index to process.
    StreamIndexFilter = sys::rs2_option_RS2_OPTION_STREAM_INDEX_FILTER as i32,
    /// When supported, this option make the camera to switch the emitter state every frame.
    /// 0 for disabled, 1 for enabled.
    EmitterOnOff = sys::rs2_option_RS2_OPTION_EMITTER_ON_OFF as i32,
    /// Get the LDD temperature.
    LldTemperature = sys::rs2_option_RS2_OPTION_LLD_TEMPERATURE as i32,
    /// Get the MC temperature.
    McTemperature = sys::rs2_option_RS2_OPTION_MC_TEMPERATURE as i32,
    /// Get the MA temperature.
    MaTemperature = sys::rs2_option_RS2_OPTION_MA_TEMPERATURE as i32,
    /// Hardware stream configuration.
    HardwarePreset = sys::rs2_option_RS2_OPTION_HARDWARE_PRESET as i32,
    /// Enable/disable global time.
    GlobalTimeEnabled = sys::rs2_option_RS2_OPTION_GLOBAL_TIME_ENABLED as i32,
    /// Get the APD temperature.
    ApdTemperature = sys::rs2_option_RS2_OPTION_APD_TEMPERATURE as i32,
    /// Enable/disable an internal map.
    EnableMapping = sys::rs2_option_RS2_OPTION_ENABLE_MAPPING as i32,
    /// Enable/disable appearance-based relocalization.
    EnableRelocalization = sys::rs2_option_RS2_OPTION_ENABLE_RELOCALIZATION as i32,
    /// Enable/disable position jumping.
    EnablePoseJumping = sys::rs2_option_RS2_OPTION_ENABLE_POSE_JUMPING as i32,
    /// Enable/disable dynamic calibration.
    EnableDynamicCalibration = sys::rs2_option_RS2_OPTION_ENABLE_DYNAMIC_CALIBRATION as i32,
    /// Get the offset from sensor to depth origin in millimeters.
    DepthOffset = sys::rs2_option_RS2_OPTION_DEPTH_OFFSET as i32,
    /// Set the power of the LED (light emitting diode), with 0 meaning off
    LedPower = sys::rs2_option_RS2_OPTION_LED_POWER as i32,
    /// Preserve the previous map when starting.
    EnableMapPreservation = sys::rs2_option_RS2_OPTION_ENABLE_MAP_PRESERVATION as i32,
    /// Enable/disable sensor shutdown when a free-fall is detected (on by default).
    FreefallDetectionEnabled = sys::rs2_option_RS2_OPTION_FREEFALL_DETECTION_ENABLED as i32,
    /// Changes the exposure time of Avalanche Photo Diode in the receiver.
    AvalanchePhotoDiode = sys::rs2_option_RS2_OPTION_AVALANCHE_PHOTO_DIODE as i32,
    /// Changes the amount of sharpening in the post-processed image.
    PostProcessingSharpening = sys::rs2_option_RS2_OPTION_POST_PROCESSING_SHARPENING as i32,
    /// Changes the amount of sharpening in the pre-processed image.
    PreProcessingSharpening = sys::rs2_option_RS2_OPTION_PRE_PROCESSING_SHARPENING as i32,
    /// Control edges and background noise.
    NoiseFiltering = sys::rs2_option_RS2_OPTION_NOISE_FILTERING as i32,
    /// Enable/disable pixel invalidation.
    InvalidationBypass = sys::rs2_option_RS2_OPTION_INVALIDATION_BYPASS as i32,
    /// Change the depth digital gain see rs2_digital_gain for values.
    DigitalGain = sys::rs2_option_RS2_OPTION_DIGITAL_GAIN as i32,
    /// The resolution mode: see rs2_sensor_mode for values.
    SensoeMode = sys::rs2_option_RS2_OPTION_SENSOR_MODE as i32,
    /// Enable/disable Laser On constantly (GS SKU Only).
    EmitterAlwaysOn = sys::rs2_option_RS2_OPTION_EMITTER_ALWAYS_ON as i32,
    /// Depth Thermal Compensation for selected D400 SKUs.
    ThermalCompensation = sys::rs2_option_RS2_OPTION_THERMAL_COMPENSATION as i32,
    /// Set host performance mode to optimize device settings so host can keep up with workload.
    /// Take USB transaction granularity as an example. Setting option to low performance host leads
    /// to larger USB transaction sizes and a reduced number of transactions. This improves performance
    /// and stability if the host machine is relatively weak compared to the workload.
    HostPerformance = sys::rs2_option_RS2_OPTION_HOST_PERFORMANCE as i32,
    /// Enable/disable HDR.
    HdrEnabled = sys::rs2_option_RS2_OPTION_HDR_ENABLED as i32,
    /// Get HDR Sequence name.
    SequenceName = sys::rs2_option_RS2_OPTION_SEQUENCE_NAME as i32,
    /// Get HDR Sequence size.
    SequenceSize = sys::rs2_option_RS2_OPTION_SEQUENCE_SIZE as i32,
    /// Get HDR Sequence ID - 0 is not HDR; sequence ID for HDR configuration starts from 1.
    SequenceId = sys::rs2_option_RS2_OPTION_SEQUENCE_ID as i32,
    /// Get Humidity temperature [in Celsius].
    HumidityTemperature = sys::rs2_option_RS2_OPTION_HUMIDITY_TEMPERATURE as i32,
    /// Enable/disable the maximum usable depth sensor range given the amount of ambient light in the scene.
    EnableMaxUsableRange = sys::rs2_option_RS2_OPTION_ENABLE_MAX_USABLE_RANGE as i32,
    /// Enable/disable the alternate IR, When enabling alternate IR, the IR image is holding the amplitude of the depth correlation.
    AlternateIr = sys::rs2_option_RS2_OPTION_ALTERNATE_IR as i32,
    /// Get an estimation of the noise on the IR image.
    NoiseEstimation = sys::rs2_option_RS2_OPTION_NOISE_ESTIMATION as i32,
    /// Enable/disable data collection for calculating IR pixel reflectivity.
    EnableIrReflectivity = sys::rs2_option_RS2_OPTION_ENABLE_IR_REFLECTIVITY as i32,
    /// Auto exposure limit in microseconds.
    ///
    /// Default is 0 which means full exposure range. If the requested exposure limit is greater
    /// than frame time, it will be set to frame time at runtime. Setting will not take effect
    /// until next streaming session.
    AutoExposureLimit = sys::rs2_option_RS2_OPTION_AUTO_EXPOSURE_LIMIT as i32,
    /// Auto gain limits ranging from 16 to 248.
    ///
    /// Default is 0 which means full gain. If the requested gain limit is less than 16, it will be
    /// set to 16. If the requested gain limit is greater than 248, it will be set to 248. Setting
    /// will not take effect until next streaming session.
    AutoGainLimit = sys::rs2_option_RS2_OPTION_AUTO_GAIN_LIMIT as i32,
    /// Enable receiver sensitivity according to ambient light, bounded by the Receiver GAin
    /// control.
    AutoReceiverSensitivity = sys::rs2_option_RS2_OPTION_AUTO_RX_SENSITIVITY as i32,
    /// Changes the transmistter frequency frequencies increasing effective range over sharpness.
    TransmitterFrequency = sys::rs2_option_RS2_OPTION_TRANSMITTER_FREQUENCY as i32,
    /// Enables vertical binning which increases the maximal sensed distance.
    VerticalBinning = sys::rs2_option_RS2_OPTION_VERTICAL_BINNING as i32,
    /// Control receiver sensitivity to incoming light, both projected and ambient (same as APD on L515).
    ReceiverSensitivity = sys::rs2_option_RS2_OPTION_RECEIVER_SENSITIVITY as i32,
    /// Enable / disable color image auto-exposure
    AutoExposureLimitToggle = sys::rs2_option_RS2_OPTION_AUTO_EXPOSURE_LIMIT_TOGGLE as i32,
    /// Enable / disable color image auto-gain
    AutoGainLimitToggle = sys::rs2_option_RS2_OPTION_AUTO_GAIN_LIMIT_TOGGLE as i32,
    /// Select emitter (laser projector) frequency, see rs2_emitter_frequency for values
    EmitterFrequency = sys::rs2_option_RS2_OPTION_EMITTER_FREQUENCY as i32,
    /// Select depth sensor auto exposure mode see rs2_depth_auto_exposure_mode for values
    DepthAutoExposureMode = sys::rs2_option_RS2_OPTION_DEPTH_AUTO_EXPOSURE_MODE as i32,
    /// Temperature of the Optical Head Sensor
    OhmTemperature = sys::rs2_option_RS2_OPTION_OHM_TEMPERATURE as i32,
    /// Temperature of PVT SOC
    SocPvtTemperature = sys::rs2_option_RS2_OPTION_SOC_PVT_TEMPERATURE as i32,
    /// Control of the gyro sensitivity level, see rs2_gyro_sensitivity for values
    GyroSensitivity = sys::rs2_option_RS2_OPTION_GYRO_SENSITIVITY as i32,
    /// The rectangular area used from the streaming profile
    RegionOfInterest = sys::rs2_option_RS2_OPTION_REGION_OF_INTEREST as i32,
    /// Rotation
    Rotation = sys::rs2_option_RS2_OPTION_ROTATION as i32,
    // Not included since this just tells us the total number of options.
    //
    // Count = sys::rs2_option_RS2_OPTION_COUNT as i32,
}

impl Rs2Option {
    /// Get the option as a CStr.
    pub fn to_cstr(self) -> &'static CStr {
        unsafe {
            let ptr = sys::rs2_option_to_string(self as sys::rs2_option);
            CStr::from_ptr(ptr)
        }
    }

    /// Get the option as a str.
    pub fn to_str(self) -> &'static str {
        self.to_cstr().to_str().unwrap()
    }
}

impl Display for Rs2Option {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

/// The range of available values of a supported option.
pub struct Rs2OptionRange {
    /// The minimum value which will be accepted for this option
    pub min: f32,
    /// The maximum value which will be accepted for this option
    pub max: f32,
    /// The granularity of options which accept discrete values, or zero if the option accepts
    /// continuous values
    pub step: f32,
    /// The default value of the option
    pub default: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn all_variants_exist() {
        let deprecated_options = [
            sys::rs2_option_RS2_OPTION_ZERO_ORDER_POINT_X as i32,
            sys::rs2_option_RS2_OPTION_ZERO_ORDER_POINT_Y as i32,
            sys::rs2_option_RS2_OPTION_ZERO_ORDER_ENABLED as i32,
            sys::rs2_option_RS2_OPTION_AMBIENT_LIGHT as i32,
            sys::rs2_option_RS2_OPTION_TRIGGER_CAMERA_ACCURACY_HEALTH as i32,
            sys::rs2_option_RS2_OPTION_RESET_CAMERA_ACCURACY_HEALTH as i32,
        ];

        for i in 0..sys::rs2_option_RS2_OPTION_COUNT as i32 {
            if deprecated_options.iter().any(|x| x == &i) {
                continue;
            }

            assert!(
                Rs2Option::from_i32(i).is_some(),
                "Rs2Option variant for ordinal {} does not exist.",
                i,
            );
        }
    }
}
