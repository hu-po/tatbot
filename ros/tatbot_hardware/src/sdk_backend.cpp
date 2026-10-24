// Copyright 2025 Trossen Robotics
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// The SDK calls of trossen_arm_ros (jazzy branch, trossen_arm_hardware/src/interface.cpp),
// forked for tatbot: position mode with the controller's velocity command as feed-forward
// (upstream sends none), no idle on deactivate (the adapter holds instead), no cleanup() unless
// landed (cleanup idles the controller; the adapter leaks a held session instead, and closes a
// landed one at once), and never load_configs_from_file (it rewrites the controller's EEPROM
// network settings). Since 2026-10-07 the controllers run firmware v1.11.1 and this package fetches SDK
// v1.11.0: the dated presets name the 2025-05-09 models the arms always ran with (the undated follower now
// means 2026-06-26), the palm carries the fitted tool's identified load, the 2025-05-09 position gains are
// set at connect (the firmware boots with 2026-03-17's, which sagged the tool 30-38 mm), and the carriage's
// position offset keeps the 1.8.4 frame (1.11 reads the carriage 0.70 mm higher). The firmware idles when its
// connection is lost, and on Trossen's idle gains the loaded arm then falls: 34 mm in 0.22 s held in the air
// (bench 2026-10-08), into the skin in three drops that day; the arm joints' idle velocity integrals are scaled
// (config/trossen/tatbot.yaml follower.idle_velocity_ki_scale) so idle catches it within a few millimetres.
#include "tatbot_hardware/backend.hpp"
#include "tatbot_hardware/sdk_options.hpp"

#ifdef TATBOT_HAVE_SDK
#include <libtrossen_arm/trossen_arm.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

// The SDK logs through spdlog (its static copy, which the process's own libspdlog interposes); only this
// one call is needed, so it is declared rather than pulling in spdlog's headers.
namespace spdlog
{
void drop(const std::string & name);
}  // namespace spdlog

namespace tatbot_hardware
{
namespace
{
constexpr double kConnectTimeoutS = 5.0;   // arm_recover CONFIGURE_TIMEOUT_S; the SDK default is 20 s

class SdkBackend : public Backend
{
public:
  SdkBackend(std::string ip, SdkOptions options)
  : ip_(std::move(ip)), options_(std::move(options)), end_effector_(sdk_end_effector(options_)),
    gains_(sdk_gains(options_)), q_(kJoints, 0.0),
    qd_(std::vector<double>(kJoints, 0.0))
  {}

  // Reached only when landed or never commanded (TatbotArm::release leaks a held session).
  ~SdkBackend() override {disconnect();}

  void disconnect() override
  {
    // The SDK's cleanup() idles every motor and closes both sockets. It throws on an already-closed
    // socket and its destructor calls cleanup() again; a driver whose cleanup failed is leaked on
    // purpose (arm_recover quiet_cleanup).
    if (!driver_) {return;}
    try {
      driver_->cleanup();
      driver_.reset();
    } catch (const std::exception &) {
      (void)driver_.release();
    }
  }

  void connect() override
  {
    // Both arms in one process (the stack's ros2_control_node): a configure that reaches the controller
    // registers the SDK's default logger, "trossen_arm_driver", and spdlog refuses a second logger of one
    // name, so the second arm failed with "logger with name 'trossen_arm_driver' already exists"
    // (2026-09-30, arms right,left). The caller serializes connects (connect_mutex): dropping the default
    // name and this arm's own (a reconnect after a failed configure) lets each configure register its
    // logger, and a driver keeps the one it holds.
    spdlog::drop(trossen_arm::TrossenArmDriver::get_default_logger_name());
    spdlog::drop(trossen_arm::TrossenArmDriver::get_logger_name(trossen_arm::Model::wxai_v0, ip_));
    auto driver = std::make_unique<trossen_arm::TrossenArmDriver>();
    try {
      driver->configure(trossen_arm::Model::wxai_v0, end_effector_, ip_, true, kConnectTimeoutS);
    } catch (...) {
      (void)driver.release();   // half torn down; never cleaned (arm_recover fresh_session)
      throw;
    }
    driver_ = std::move(driver);
    // Volatile, reset at every controller boot: the gains. Persistent: the joint characteristics, so the
    // carriage offset is written only when the controller's differs (a reflashed or replaced controller).
    driver_->set_motor_parameters(gains_);
    std::ostringstream note;
    note << options_.end_effector << ", palm " << std::fixed << std::setprecision(4) << end_effector_.palm.mass
         << " kg, gains " << options_.motor_parameters << std::setprecision(1) << ", idle ki x";
    for (size_t j = 0; j < options_.idle_velocity_ki_scale.size(); ++j) {
      note << (j ? "/" : "") << options_.idle_velocity_ki_scale[j];
    }
    for (auto [label, scales] : {std::pair{", position kp x", &options_.position_kp_scale},
        std::pair{", velocity kp x", &options_.velocity_kp_scale}, std::pair{", velocity ki x", &options_.velocity_ki_scale}})
    {
      if (std::any_of(scales->begin(), scales->end(), [](double s) {return s != 1.0;})) {
        note << label;
        for (size_t j = 0; j < scales->size(); ++j) {note << (j ? "/" : "") << (*scales)[j];}
      }
    }
    if (std::isfinite(options_.carriage_position_offset_m)) {
      auto characteristics = driver_->get_joint_characteristics();
      const double had = characteristics.back().position_offset;
      note << ", carriage offset " << std::setprecision(2) << options_.carriage_position_offset_m * 1e3 << " mm";
      if (std::fabs(had - options_.carriage_position_offset_m) > 1e-7) {
        characteristics.back().position_offset = options_.carriage_position_offset_m;
        driver_->set_joint_characteristics(characteristics);
        note << " (was " << had * 1e3 << " mm, written)";
      }
    }
    // The friction terms live in the controller's EEPROM: written only where ros_tuning names one that differs.
    {
      auto characteristics = driver_->get_joint_characteristics();
      bool changed = false;
      for (size_t j = 0; j < options_.friction_transition_velocity.size() && j < characteristics.size(); ++j) {
        for (auto [want, have, name] : {
            std::tuple{options_.friction_transition_velocity[j], &characteristics[j].friction_transition_velocity,
              "transition"},
            std::tuple{options_.friction_constant_term[j], &characteristics[j].friction_constant_term, "constant"}})
        {
          if (std::isfinite(want) && std::fabs(*have - want) > 1e-7) {
            note << ", j" << j << " friction " << name << " " << std::setprecision(4) << *have << " -> " << want;
            *have = want;
            changed = true;
          }
        }
      }
      if (changed) {driver_->set_joint_characteristics(characteristics);}
    }
    description_ = note.str();
  }

  std::string snapshot() override
  {
    if (!driver_) {return "";}
    auto list = [](const std::vector<double> & v) {
        std::ostringstream o;
        o << std::setprecision(9) << "[";
        for (size_t i = 0; i < v.size(); ++i) {o << (i ? "," : "") << v[i];}
        o << "]";
        return o.str();
      };
    std::ostringstream o;
    o << std::setprecision(9) << "{\"driver\":\"" << driver_->get_driver_version() << "\",\"controller\":\""
      << driver_->get_controller_version() << "\",\"motor_parameters\":[";
    const auto gains = driver_->get_motor_parameters();
    for (size_t j = 0; j < gains.size(); ++j) {
      o << (j ? "," : "") << "{";
      bool first = true;
      for (const auto & [mode, p] : gains[j]) {
        o << (first ? "" : ",") << "\"" << static_cast<int>(mode) << "\":{\"position\":["
          << p.position.kp << "," << p.position.ki << "," << p.position.kd << "," << p.position.imax
          << "],\"velocity\":[" << p.velocity.kp << "," << p.velocity.ki << "," << p.velocity.kd << ","
          << p.velocity.imax << "]}";
        first = false;
      }
      o << "}";
    }
    o << "],\"effort_correction\":" << list(driver_->get_effort_corrections())
      << ",\"friction_transition_velocity\":" << list(driver_->get_friction_transition_velocities())
      << ",\"friction_constant_term\":" << list(driver_->get_friction_constant_terms())
      << ",\"friction_coulomb_coef\":" << list(driver_->get_friction_coulomb_coefs())
      << ",\"friction_viscous_coef\":" << list(driver_->get_friction_viscous_coefs())
      << ",\"position_offset\":" << list(driver_->get_position_offsets()) << ",\"joint_limits\":[";
    const auto limits = driver_->get_joint_limits();
    for (size_t j = 0; j < limits.size(); ++j) {
      const auto & l = limits[j];
      o << (j ? "," : "") << "[" << l.position_min << "," << l.position_max << "," << l.position_tolerance << ","
        << l.velocity_max << "," << l.velocity_tolerance << "," << l.effort_max << "," << l.effort_tolerance << "]";
    }
    o << "],\"joint_limit_fields\":\"position_min,position_max,position_tolerance,velocity_max,velocity_tolerance,"
         "effort_max,effort_tolerance\",\"modes_key\":\"0 idle,1 position,2 velocity,3 external_effort,4 effort\"}";
    return o.str();
  }

  std::string describe() const override {return description_;}

  Feedback read() override
  {
    if (!driver_) {throw std::runtime_error("no SDK session");}
    const auto output = driver_->get_robot_output();
    const auto & all = output.joint.all;
    Feedback fb;
    if (all.positions.size() != kJoints || all.velocities.size() != kJoints ||
      all.external_efforts.size() != kJoints)
    {
      throw std::runtime_error("SDK reported a joint count other than 7");
    }
    std::copy(all.positions.begin(), all.positions.end(), fb.q.begin());
    std::copy(all.velocities.begin(), all.velocities.end(), fb.qd.begin());
    // External effort (total minus the controller's gravity and friction compensation), as
    // rust/tatbot-arm reads it: the carriage contact cap was tuned on it.
    std::copy(all.external_efforts.begin(), all.external_efforts.end(), fb.effort.begin());
    if (all.compensation_efforts.size() == kJoints) {
      std::copy(all.compensation_efforts.begin(), all.compensation_efforts.end(), fb.compensation.begin());
    }
    if (all.rotor_temperatures.size() == kJoints) {
      std::copy(all.rotor_temperatures.begin(), all.rotor_temperatures.end(), fb.rotor_temp.begin());
    }
    fb.controller_id = output.header.id;
    fb.controller_us = output.header.timestamp;
    return fb;
  }

  void command(const Vec7 & q, const Vec7 & qd) override
  {
    std::copy(q.begin(), q.end(), q_.begin());   // preallocated: no allocation on the control thread
    std::copy(qd.begin(), qd.end(), qd_->begin());
    driver_->set_all_positions(q_, 0.0, false, qd_);
  }

  std::pair<double, double> set_carriage_limits(double min_m, double max_m) override
  {
    // set_joint_limits is volatile (unlike load_configs_from_file, it never touches EEPROM).
    auto limits = driver_->get_joint_limits();
    if (limits.size() != kJoints) {throw std::runtime_error("SDK reported joint limits other than 7");}
    const std::pair<double, double> before{limits.back().position_min, limits.back().position_max};
    limits.back().position_min = min_m;
    limits.back().position_max = max_m;
    driver_->set_joint_limits(limits);
    return before;
  }

  void set_position_mode() override {driver_->set_all_modes(trossen_arm::Mode::position);}

  std::string error() override
  {
    if (!driver_ || !driver_->get_is_configured()) {return "";}
    const std::string raw = driver_->get_error_information();
    const auto begin = raw.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {return "";}
    const std::string value = raw.substr(begin, raw.find_last_not_of(" \t\r\n") - begin + 1);
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});
    if (lower == "no error" || lower == "none" || lower == "error state: none") {return "";}
    return value;
  }

private:
  std::string ip_;
  SdkOptions options_;
  trossen_arm::EndEffector end_effector_;
  std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> gains_;
  std::string description_;
  std::unique_ptr<trossen_arm::TrossenArmDriver> driver_;
  std::vector<double> q_;
  std::optional<std::vector<double>> qd_;
};
}  // namespace

trossen_arm::EndEffector sdk_end_effector(const SdkOptions & options)
{
  using trossen_arm::StandardEndEffector;
  static const std::map<std::string, trossen_arm::EndEffector> presets{
    {"wxai_v0_leader_20250509", StandardEndEffector::wxai_v0_leader_20250509},
    {"wxai_v0_follower_20250509", StandardEndEffector::wxai_v0_follower_20250509},
    {"wxai_v0_base_20250509", StandardEndEffector::wxai_v0_base_20250509},
    {"wxai_v0_leader_20260626", StandardEndEffector::wxai_v0_leader_20260626},
    {"wxai_v0_follower_20260626", StandardEndEffector::wxai_v0_follower_20260626},
    {"wxai_v0_base_20260626", StandardEndEffector::wxai_v0_base_20260626}};
  const auto it = presets.find(options.end_effector);
  if (it == presets.end()) {
    throw std::invalid_argument("unknown end_effector '" + options.end_effector +
            "' (a dated Trossen preset, e.g. wxai_v0_follower_20250509)");
  }
  trossen_arm::EndEffector ee = it->second;
  if (std::isfinite(options.palm_mass_kg)) {
    ee.palm.mass = options.palm_mass_kg;
    ee.palm.origin_xyz = options.palm_com_m;
  }
  return ee;
}

const std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> & sdk_motor_parameters(
  const std::string & name)
{
  using trossen_arm::StandardMotorParameters;
  if (name == "wxai_v0_20250509") {return StandardMotorParameters::wxai_v0_20250509;}
  if (name == "wxai_v0_20260317") {return StandardMotorParameters::wxai_v0_20260317;}
  throw std::invalid_argument("unknown motor_parameters '" + name + "' (wxai_v0_20250509 or wxai_v0_20260317)");
}

std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> sdk_gains(const SdkOptions & options)
{
  auto gains = sdk_motor_parameters(options.motor_parameters);
  for (size_t j = 0; j < options.idle_velocity_ki_scale.size(); ++j) {
    const double scale = options.idle_velocity_ki_scale[j];
    if (!(std::isfinite(scale) && scale > 0)) {
      throw std::invalid_argument("idle_velocity_ki_scale must be finite and positive");
    }
    gains.at(j).at(trossen_arm::Mode::idle).velocity.ki *= scale;
  }
  for (size_t j = 0; j < options.position_kp_scale.size(); ++j) {
    for (double scale : {options.position_kp_scale[j], options.velocity_kp_scale[j], options.velocity_ki_scale[j]}) {
      if (!(std::isfinite(scale) && scale > 0)) {
        throw std::invalid_argument("ros_tuning gain scales must be finite and positive");
      }
    }
    auto & position = gains.at(j).at(trossen_arm::Mode::position);
    position.position.kp *= options.position_kp_scale[j];
    position.velocity.kp *= options.velocity_kp_scale[j];
    position.velocity.ki *= options.velocity_ki_scale[j];
  }
  return gains;
}

std::unique_ptr<Backend> make_sdk_backend(const std::string & ip, const SdkOptions & options)
{
  return std::make_unique<SdkBackend>(ip, options);
}

}  // namespace tatbot_hardware

#else

namespace tatbot_hardware
{
std::unique_ptr<Backend> make_sdk_backend(const std::string &, const SdkOptions &) {return nullptr;}
}  // namespace tatbot_hardware

#endif
