#pragma once
#include "rust/cxx.h"
#include <chrono>
#include <memory>
#include <string>
#include <vector>
namespace trossen_arm { class TrossenArmDriver; enum class Mode : uint8_t; }
namespace tatbot::trossen {
struct Measurement;
struct JointLimit;
struct EndEffectorReadback;
class Driver {
 public:
  Driver();
  ~Driver();
  void configure(rust::Str ip, bool follower, bool clear_error, double timeout_s);
  // Joint feedback comes from the controller's UDP output the SDK already
  // holds (sub-microsecond). Modes and the error string are TCP queries of
  // ~1.5 ms each on the arm node, so they are re-read only when older than
  // max_status_age_s; 0 forces a fresh query. Any mode or configuration
  // command invalidates the cached status.
  Measurement measure(double max_status_age_s);
  rust::Vec<JointLimit> limits();
  void position_mode();
  void idle_mode();
  // Per-joint modes for hand guiding: 0 idle, 1 position, 3 external effort.
  void joint_modes(rust::Slice<const uint8_t> modes);
  // Zero commanded external effort on the six arm joints, immediately.
  void zero_arm_external_efforts();
  // One joint's position target, immediately (goal time 0) or interpolated.
  void joint_position(uint8_t index, double position, double goal_s);
  EndEffectorReadback end_effector();
  void positions(rust::Slice<const double> q, rust::Slice<const double> v,
                 rust::Slice<const double> a, double goal_s);
  rust::Vec<double> hold();
  void load_config(rust::Str path);
  void cleanup();
 private:
  void require_connected();
  void invalidate_status();
  std::unique_ptr<trossen_arm::TrossenArmDriver> driver_;
  bool configured_{false};
  bool follower_{true};
  bool status_valid_{false};
  std::vector<trossen_arm::Mode> modes_;
  std::string error_;
  std::chrono::steady_clock::time_point modes_at_{};
  std::chrono::steady_clock::time_point error_at_{};
};
std::unique_ptr<Driver> make_driver();
void validate_command(rust::Slice<const double> q, rust::Slice<const double> v,
                      rust::Slice<const double> a, double goal_s);
}
