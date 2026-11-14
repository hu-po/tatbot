#pragma once
// The xArm controller's report streams, read by our own parser: the C++ SDK reads neither port's frames this way.
// Port 30000 is the real-time report, about 250 frames a second of 784 bytes on firmware 2.7.0: commanded and
// measured joints, their speeds, motor currents and estimated torques. Port 30001 is the normal report, about 5
// frames a second of 145 bytes: state and mode, error and warning codes, motor brakes and enables, and the TCP offset,
// load and collision sensitivity the controller holds. Every frame leads with its own length (u32, big-endian), its
// header integers are big-endian and every float little-endian; xArm-Python-SDK's xarm/x3/report.py is the
// reference. Angles are in radians, the TCP offset in mm and rad, the load in kg and mm.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "tatbot_hardware/core.hpp"

namespace tatbot_hardware::xarm
{

inline constexpr int kRealtimePort = 30000;
inline constexpr int kNormalPort = 30001;

// The controller's state (the low nibble of a report's state byte): 1 moving, 2 sleeping, 3 paused, 4 stopped
// (an e-stop, an error or motors off), 5 reconfigured (a mode or setting changed: set the state to 0 to move again).
inline constexpr int kStateMoving = 1;
inline constexpr int kStateStopped = 4;
inline constexpr int kStateReconfigured = 5;
inline constexpr int kModeServo = 1;   // set_servo_angle_j executes only the latest command

struct Realtime
{
  uint64_t timestamp_us = 0;   // the controller's clock
  int state = 0, mode = 0, cmd_num = 0;
  Vec7 target_q{}, target_qd{}, q{}, qd{}, current{}, torque{};
};

struct Normal
{
  int state = 0, mode = 0, cmd_num = 0;
  Vec7 q{};
  uint8_t brakes = 0, enables = 0;   // bit i: motor i + 1 (its brake released; its motor enabled)
  int error_code = 0, warn_code = 0;
  std::array<double, 6> tcp_offset{};
  std::array<double, 4> tcp_load{};   // mass, then the centre of mass x, y, z
  int collision_sensitivity = 0;
};

// Splits a report stream into frames by their length field. A length outside [min_frame, max_frame] means the
// stream is out of step (it never is on a healthy socket): corrupt() is then set and the reader keeps nothing,
// so the caller reconnects.
class FrameReader
{
public:
  explicit FrameReader(size_t min_frame = 7, size_t max_frame = 4096);
  void feed(const uint8_t * data, size_t size);
  // The next whole frame, its length field included; nullopt when no whole frame is buffered.
  std::optional<std::vector<uint8_t>> next();
  bool corrupt() const {return corrupt_;}

private:
  std::vector<uint8_t> buffer_;
  size_t min_, max_;
  bool corrupt_ = false;
};

// nullopt when the frame is shorter than the fields read.
std::optional<Realtime> decode_realtime(const std::vector<uint8_t> & frame);
std::optional<Normal> decode_normal(const std::vector<uint8_t> & frame);

}  // namespace tatbot_hardware::xarm
