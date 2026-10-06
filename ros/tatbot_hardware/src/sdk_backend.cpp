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
// landed (cleanup idles the controller; the adapter leaks a held session instead), and never
// load_configs_from_file (it rewrites the controller's EEPROM network settings). Upstream builds
// against SDK v1.9.0; every call here exists in v1.8.5, which this package fetches.
#include "tatbot_hardware/backend.hpp"

#ifdef TATBOT_HAVE_SDK
#include <libtrossen_arm/trossen_arm.hpp>

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
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
  SdkBackend(std::string ip, trossen_arm::EndEffector end_effector)
  : ip_(std::move(ip)), end_effector_(end_effector), q_(kJoints, 0.0), qd_(std::vector<double>(kJoints, 0.0))
  {}

  ~SdkBackend() override
  {
    // Reached only when landed or never commanded (TatbotArm::release leaks a held session).
    // The SDK's cleanup() throws on an already-closed socket and its destructor calls cleanup()
    // again; a driver whose cleanup failed is leaked on purpose (arm_recover quiet_cleanup).
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
  }

  Feedback read() override
  {
    const auto & all = driver_->get_robot_output().joint.all;
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
  void set_idle() override {driver_->set_all_modes(trossen_arm::Mode::idle);}

  std::string error() override
  {
    if (!driver_->get_is_configured()) {return "";}
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
  trossen_arm::EndEffector end_effector_;
  std::unique_ptr<trossen_arm::TrossenArmDriver> driver_;
  std::vector<double> q_;
  std::optional<std::vector<double>> qd_;
};
}  // namespace

bool sdk_available() {return true;}

std::unique_ptr<Backend> make_sdk_backend(const std::string & ip, const std::string & end_effector)
{
  trossen_arm::EndEffector ee = trossen_arm::StandardEndEffector::wxai_v0_follower;
  if (end_effector == "wxai_v0_leader") {
    ee = trossen_arm::StandardEndEffector::wxai_v0_leader;
  } else if (end_effector == "wxai_v0_base") {
    ee = trossen_arm::StandardEndEffector::wxai_v0_base;
  } else if (end_effector != "wxai_v0_follower") {
    throw std::invalid_argument("unknown end_effector '" + end_effector + "'");
  }
  return std::make_unique<SdkBackend>(ip, ee);
}

}  // namespace tatbot_hardware

#else

namespace tatbot_hardware
{
bool sdk_available() {return false;}
std::unique_ptr<Backend> make_sdk_backend(const std::string &, const std::string &) {return nullptr;}
}  // namespace tatbot_hardware

#endif
