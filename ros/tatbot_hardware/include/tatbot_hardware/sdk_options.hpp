#pragma once
// The Trossen SDK objects an SdkOptions resolves to (sdk_backend.cpp). Only with the SDK (TATBOT_HAVE_SDK).
#ifdef TATBOT_HAVE_SDK
#include <libtrossen_arm/trossen_arm.hpp>

#include <map>
#include <string>
#include <vector>

#include "tatbot_hardware/backend.hpp"

namespace tatbot_hardware
{
// The dated preset with the identified palm, if any. Throws std::invalid_argument on an unknown preset.
trossen_arm::EndEffector sdk_end_effector(const SdkOptions & options);
// Trossen's named position-mode gain set. Throws std::invalid_argument on an unknown name.
const std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> & sdk_motor_parameters(
  const std::string & name);
// What connect sets: the named set with the arm joints' idle velocity ki scaled by idle_velocity_ki_scale. Throws
// std::invalid_argument on an unknown name or a scale that is not finite and positive.
std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> sdk_gains(const SdkOptions & options);
}  // namespace tatbot_hardware
#endif
