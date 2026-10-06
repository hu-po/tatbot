#pragma once
// The <arm>_safety GPIO interfaces and their encodings. Mirrors tatbot_description/names.py
// (a test keeps them equal); ros/README.md "Names" is the table.
#include <array>
#include <string_view>

namespace tatbot_hardware::names
{

inline constexpr std::array<std::string_view, 21> kSafetyState = {
  "estop_source", "estop_ok", "estop_age_s", "probe_triggered", "latched", "latch_reason",
  "guard_mode", "guard_tripped", "trip_q0", "trip_q1", "trip_q2", "trip_q3", "trip_q4",
  "trip_q5", "trip_q6", "unlatch_ack", "land_ack", "landing", "landed", "controller_error",
  "rt_period_max_ms"};
inline constexpr std::array<std::string_view, 3> kSafetyCommand = {"guard_mode", "unlatch", "land"};

enum EstopSource : int { kEstopNone = 0, kEstopSerial = 1, kEstopUdp = 2 };
enum LatchReason : int {
  kLatchNone = 0,
  kLatchEstop = 1,
  kLatchEstopStale = 2,
  kLatchCarriageContact = 3,
  kLatchStall = 4,
  kLatchOverVelocity = 5,
  kLatchGuardTipLag = 6,
  kLatchGuardProbe = 7,
  kLatchControllerError = 8,
  kLatchDeactivated = 9,
  kLatchStepRefused = 10,
};
enum GuardMode : int { kGuardNone = 0, kGuardTipLag = 1, kGuardProbe = 2 };

}  // namespace tatbot_hardware::names
