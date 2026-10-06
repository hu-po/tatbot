#include "trossen-arm-sys/src/lib.rs.h"
#include "libtrossen_arm/trossen_arm.hpp"
#include <arpa/inet.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tatbot::trossen {
namespace {
template<class T> void finite_seven(const T & values) {
  if (values.size() != 7) throw std::invalid_argument("exactly seven joint values required");
  for (double value : values) {
    if (!std::isfinite(value)) throw std::invalid_argument("joint values must be finite");
  }
}
rust::Vec<double> copy_values(const std::vector<double> & values) {
  finite_seven(values);
  rust::Vec<double> out;
  out.reserve(values.size());
  for (double value : values) out.push_back(value);
  return out;
}
}
void validate_command(rust::Slice<const double> q, rust::Slice<const double> v,
                      rust::Slice<const double> a, double goal_s) {
  finite_seven(q); finite_seven(v); finite_seven(a);
  // Zero selects direct targets for already-interpolated streamed samples.
  if (!std::isfinite(goal_s) || goal_s < 0.0 || goal_s > 60.0)
    throw std::invalid_argument("goal time must be finite and in [0,60] seconds");
}
Driver::Driver() : driver_(std::make_unique<trossen_arm::TrossenArmDriver>()) {}
Driver::~Driver() = default;
std::unique_ptr<Driver> make_driver() { return std::make_unique<Driver>(); }
void Driver::require_connected() {
  if (!configured_ || !driver_->get_is_configured()) throw std::runtime_error("driver is not configured");
}
void Driver::configure(rust::Str ip, bool follower, bool clear_error, double timeout_s) {
  const std::string address(ip);
  in_addr parsed{};
  if (address.find('\0') != std::string::npos || inet_pton(AF_INET, address.c_str(), &parsed) != 1)
    throw std::invalid_argument("explicit IPv4 controller address required");
  if (!std::isfinite(timeout_s) || timeout_s <= 0.0 || timeout_s > 20.0)
    throw std::invalid_argument("connection timeout must be in (0,20] seconds");
  if (configured_ || driver_->get_is_configured()) throw std::runtime_error("fresh driver required before reconnect");
  driver_->configure(trossen_arm::Model::wxai_v0,
      follower ? trossen_arm::StandardEndEffector::wxai_v0_follower
               : trossen_arm::StandardEndEffector::wxai_v0_leader,
      address, clear_error, timeout_s);
  follower_ = follower;
  configured_ = true;
  invalidate_status();
}
void Driver::invalidate_status() { status_valid_ = false; modes_.clear(); error_.clear(); }
Measurement Driver::measure(double max_status_age_s) {
  require_connected();
  if (!std::isfinite(max_status_age_s) || max_status_age_s < 0.0)
    throw std::invalid_argument("status age must be a non-negative finite number of seconds");
  Measurement out;
  out.positions = copy_values(driver_->get_all_positions());
  out.velocities = copy_values(driver_->get_all_velocities());
  out.accelerations = copy_values(driver_->get_all_accelerations());
  out.efforts = copy_values(driver_->get_all_efforts());
  out.external_efforts = copy_values(driver_->get_all_external_efforts());
  out.compensation_efforts = copy_values(driver_->get_all_compensation_efforts());
  // The two TCP status queries are staggered so a periodic control tick pays
  // for at most one of them; the first call after invalidation pays for both.
  const auto now = std::chrono::steady_clock::now();
  const std::chrono::duration<double> max_age(max_status_age_s);
  if (!status_valid_ || now - modes_at_ > max_age) {
    modes_ = driver_->get_modes();
    if (modes_.size() != 7) throw std::runtime_error("controller mode width is not seven");
    modes_at_ = now;
  }
  if (!status_valid_ || now - error_at_ > max_age) {
    error_ = driver_->get_error_information();
    error_at_ = status_valid_ ? now : now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(max_age / 2);
  }
  status_valid_ = true;
  for (auto mode : modes_) out.modes.push_back(static_cast<uint8_t>(mode));
  out.error = rust::String(error_);
  return out;
}
rust::Vec<JointLimit> Driver::limits() {
  require_connected();
  const auto limits = driver_->get_joint_limits();
  if (limits.size() != 7) throw std::runtime_error("controller limit width is not seven");
  rust::Vec<JointLimit> out;
  out.reserve(limits.size());
  for (const auto & limit : limits) {
    const std::vector<double> fields{limit.position_min, limit.position_max,
        limit.position_tolerance, limit.velocity_max, limit.velocity_tolerance,
        limit.effort_max, limit.effort_tolerance};
    finite_seven(fields);
    if (limit.position_min >= limit.position_max || limit.position_tolerance < 0.0 ||
        limit.velocity_max < 0.0 || limit.velocity_tolerance < 0.0 ||
        limit.effort_max < 0.0 || limit.effort_tolerance < 0.0)
      throw std::runtime_error("invalid controller joint limits");
    out.push_back(JointLimit{limit.position_min, limit.position_max,
        limit.position_tolerance, limit.velocity_max, limit.velocity_tolerance,
        limit.effort_max, limit.effort_tolerance});
  }
  return out;
}
void Driver::position_mode() {
  require_connected();
  invalidate_status();
  driver_->set_all_modes(trossen_arm::Mode::position);
}
void Driver::idle_mode() {
  require_connected();
  invalidate_status();
  driver_->set_all_modes(trossen_arm::Mode::idle);
}
void Driver::joint_modes(rust::Slice<const uint8_t> modes) {
  if (modes.size() != 7) throw std::invalid_argument("exactly seven joint modes required");
  std::vector<trossen_arm::Mode> vendor;
  vendor.reserve(7);
  for (uint8_t mode : modes) {
    switch (mode) {
      case 0: vendor.push_back(trossen_arm::Mode::idle); break;
      case 1: vendor.push_back(trossen_arm::Mode::position); break;
      case 3: vendor.push_back(trossen_arm::Mode::external_effort); break;
      default: throw std::invalid_argument("joint mode must be idle, position or external_effort");
    }
  }
  require_connected();
  invalidate_status();
  driver_->set_joint_modes(vendor);
}
void Driver::zero_arm_external_efforts() {
  require_connected();
  driver_->set_arm_external_efforts(std::vector<double>(6, 0.0), 0.0, false);
}
void Driver::joint_position(uint8_t index, double position, double goal_s) {
  if (index > 6) throw std::invalid_argument("joint index must be 0..6");
  if (!std::isfinite(position)) throw std::invalid_argument("joint position must be finite");
  if (!std::isfinite(goal_s) || goal_s < 0.0 || goal_s > 60.0)
    throw std::invalid_argument("goal time must be finite and in [0,60] seconds");
  require_connected();
  driver_->set_joint_position(index, position, goal_s, false);
}
EndEffectorReadback Driver::end_effector() {
  require_connected();
  const auto ee = driver_->get_end_effector();
  EndEffectorReadback out;
  out.palm_mass_kg = ee.palm.mass;
  out.finger_left_mass_kg = ee.finger_left.mass;
  out.finger_right_mass_kg = ee.finger_right.mass;
  for (int i = 0; i < 3; ++i) out.palm_origin_xyz_m.push_back(ee.palm.origin_xyz[i]);
  for (int i = 0; i < 9; ++i) out.palm_inertia.push_back(ee.palm.inertia[i]);
  out.offset_finger_left_m = ee.offset_finger_left;
  out.offset_finger_right_m = ee.offset_finger_right;
  out.pitch_circle_radius_m = ee.pitch_circle_radius;
  for (int i = 0; i < 6; ++i) out.t_flange_tool.push_back(ee.t_flange_tool[i]);
  return out;
}
void Driver::positions(rust::Slice<const double> q, rust::Slice<const double> v,
                       rust::Slice<const double> a, double goal_s) {
  validate_command(q, v, a, goal_s);
  require_connected();
  driver_->set_all_positions(std::vector<double>(q.begin(), q.end()), goal_s, false,
      std::vector<double>(v.begin(), v.end()), std::vector<double>(a.begin(), a.end()));
}
rust::Vec<double> Driver::hold() {
  require_connected();
  const auto q = driver_->get_all_positions();
  finite_seven(q);
  // Carried freeze semantics: immediate actual-position seed, never a previous target.
  driver_->set_all_positions(q, 0.0, false);
  return copy_values(q);
}
void Driver::load_config(rust::Str path) {
  require_connected();
  const std::string file(path);
  if (file.empty() || file.find('\0') != std::string::npos) throw std::invalid_argument("configuration path");
  invalidate_status();
  driver_->load_configs_from_file(file);
  // Match configure_arm: old arm files contain obsolete EE properties.
  driver_->set_end_effector(follower_ ? trossen_arm::StandardEndEffector::wxai_v0_follower
                                    : trossen_arm::StandardEndEffector::wxai_v0_leader);
}
void Driver::cleanup() { configured_ = false; invalidate_status(); driver_->cleanup(false); }
}
