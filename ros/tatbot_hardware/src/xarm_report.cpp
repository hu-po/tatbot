#include "tatbot_hardware/xarm_report.hpp"

#include <cstring>

namespace tatbot_hardware::xarm
{
namespace
{

uint64_t big_endian(const std::vector<uint8_t> & frame, size_t at, size_t bytes)
{
  uint64_t value = 0;
  for (size_t i = 0; i < bytes; ++i) {value = (value << 8) | frame[at + i];}
  return value;
}

float little_endian_float(const std::vector<uint8_t> & frame, size_t at)
{
  const uint32_t bits = static_cast<uint32_t>(frame[at]) | static_cast<uint32_t>(frame[at + 1]) << 8 |
    static_cast<uint32_t>(frame[at + 2]) << 16 | static_cast<uint32_t>(frame[at + 3]) << 24;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template<size_t N>
std::array<double, N> floats(const std::vector<uint8_t> & frame, size_t at)
{
  std::array<double, N> out{};
  for (size_t i = 0; i < N; ++i) {out[i] = little_endian_float(frame, at + 4 * i);}
  return out;
}

// The real-time frame: size (4), timestamp (8), state and mode (1), command count (2), 17 reserved, then 7-float
// blocks of target angles, target speeds, target accelerations, angles, speeds, accelerations, currents, torques.
constexpr size_t kRtJoints = 32, kRtBlock = 7 * 4, kRtSize = kRtJoints + 8 * kRtBlock;
// The normal frame: size (4), state and mode (1), command count (2), angles (7 floats), TCP pose (6), torques (7),
// brakes, enables, error, warning (1 each), TCP offset (6 floats), TCP load (4), collision sensitivity (1), ...
constexpr size_t kNmAngles = 7, kNmBrakes = kNmAngles + 28 + 24 + 28, kNmOffset = kNmBrakes + 4, kNmLoad = kNmOffset + 24,
  kNmSensitivity = kNmLoad + 16, kNmSize = kNmSensitivity + 1;

}  // namespace

FrameReader::FrameReader(size_t min_frame, size_t max_frame)
: min_(min_frame), max_(max_frame) {}

void FrameReader::feed(const uint8_t * data, size_t size)
{
  if (!corrupt_) {buffer_.insert(buffer_.end(), data, data + size);}
}

std::optional<std::vector<uint8_t>> FrameReader::next()
{
  if (corrupt_ || buffer_.size() < 4) {return std::nullopt;}
  const size_t length = big_endian(buffer_, 0, 4);
  if (length < min_ || length > max_) {
    corrupt_ = true;
    buffer_.clear();
    return std::nullopt;
  }
  if (buffer_.size() < length) {return std::nullopt;}
  std::vector<uint8_t> frame(buffer_.begin(), buffer_.begin() + length);
  buffer_.erase(buffer_.begin(), buffer_.begin() + length);
  return frame;
}

std::optional<Realtime> decode_realtime(const std::vector<uint8_t> & frame)
{
  if (frame.size() < kRtSize) {return std::nullopt;}
  Realtime out;
  out.timestamp_us = big_endian(frame, 4, 8);
  out.state = frame[12] & 0x0F;
  out.mode = frame[12] >> 4;
  out.cmd_num = static_cast<int>(big_endian(frame, 13, 2));
  out.target_q = floats<kJoints>(frame, kRtJoints);
  out.target_qd = floats<kJoints>(frame, kRtJoints + kRtBlock);
  out.q = floats<kJoints>(frame, kRtJoints + 3 * kRtBlock);
  out.qd = floats<kJoints>(frame, kRtJoints + 4 * kRtBlock);
  out.current = floats<kJoints>(frame, kRtJoints + 6 * kRtBlock);
  out.torque = floats<kJoints>(frame, kRtJoints + 7 * kRtBlock);
  return out;
}

std::optional<Normal> decode_normal(const std::vector<uint8_t> & frame)
{
  if (frame.size() < kNmSize) {return std::nullopt;}
  Normal out;
  out.state = frame[4] & 0x0F;
  out.mode = frame[4] >> 4;
  out.cmd_num = static_cast<int>(big_endian(frame, 5, 2));
  out.q = floats<kJoints>(frame, kNmAngles);
  out.brakes = frame[kNmBrakes];
  out.enables = frame[kNmBrakes + 1];
  out.error_code = frame[kNmBrakes + 2];
  out.warn_code = frame[kNmBrakes + 3];
  out.tcp_offset = floats<6>(frame, kNmOffset);
  out.tcp_load = floats<4>(frame, kNmLoad);
  out.collision_sensitivity = frame[kNmSensitivity];
  return out;
}

}  // namespace tatbot_hardware::xarm
