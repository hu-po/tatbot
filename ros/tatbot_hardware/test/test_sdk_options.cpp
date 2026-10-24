// SdkOptions against the real Trossen SDK (v1.11): the dated preset with the identified palm, refused names,
// Trossen's named position-mode gain sets, and the idle velocity integrals scaled over them.
#include <gtest/gtest.h>

#include <stdexcept>

#include "tatbot_hardware/sdk_options.hpp"

using tatbot_hardware::SdkOptions;
using tatbot_hardware::sdk_end_effector;
using tatbot_hardware::sdk_gains;
using tatbot_hardware::sdk_motor_parameters;

TEST(SdkOptions, TheDatedPresetCarriesTheIdentifiedPalmAndNothingElse)
{
  const auto & preset = trossen_arm::StandardEndEffector::wxai_v0_follower_20250509;
  SdkOptions options;
  options.end_effector = "wxai_v0_follower_20250509";
  const auto stock = sdk_end_effector(options);
  EXPECT_DOUBLE_EQ(stock.palm.mass, preset.palm.mass);
  EXPECT_EQ(stock.palm.origin_xyz, preset.palm.origin_xyz);

  options.palm_mass_kg = 1.612;
  options.palm_com_m = {0.008, -0.001, 0.009};
  const auto fitted = sdk_end_effector(options);
  EXPECT_DOUBLE_EQ(fitted.palm.mass, 1.612);
  EXPECT_EQ(fitted.palm.origin_xyz, options.palm_com_m);
  EXPECT_EQ(fitted.palm.inertia, preset.palm.inertia);
  EXPECT_DOUBLE_EQ(fitted.finger_left.mass, preset.finger_left.mass);
  EXPECT_DOUBLE_EQ(fitted.pitch_circle_radius, preset.pitch_circle_radius);
  EXPECT_EQ(fitted.t_flange_tool, preset.t_flange_tool);
}

TEST(SdkOptions, UndatedAndUnknownNamesAreRefused)
{
  SdkOptions options;
  options.end_effector = "wxai_v0_follower";   // driver 1.11 moved it to the 2026-06-26 model
  EXPECT_THROW(sdk_end_effector(options), std::invalid_argument);
  options.end_effector = "wxai_v0_leader_20250509";
  EXPECT_NO_THROW(sdk_end_effector(options));
  EXPECT_THROW(sdk_motor_parameters("wxai_v0_default"), std::invalid_argument);
}

TEST(SdkOptions, TheNamedGainSetsAreTrossens)
{
  const auto old_gains = sdk_motor_parameters("wxai_v0_20250509");
  ASSERT_EQ(old_gains.size(), 7u);
  const auto shoulder = old_gains[1].at(trossen_arm::Mode::position);
  EXPECT_DOUBLE_EQ(shoulder.position.kp, 120.0);
  EXPECT_GT(shoulder.velocity.ki, 0.0);   // the velocity integral that holds the payload
  EXPECT_DOUBLE_EQ(sdk_motor_parameters("wxai_v0_20260317")[1].at(trossen_arm::Mode::position).position.kp, 16.0);
}

TEST(SdkOptions, OnlyTheArmJointsIdleVelocityIntegralsAreScaled)
{
  using trossen_arm::Mode;
  const auto named = sdk_motor_parameters("wxai_v0_20250509");
  SdkOptions options;
  EXPECT_DOUBLE_EQ(sdk_gains(options)[1].at(Mode::idle).velocity.ki, named[1].at(Mode::idle).velocity.ki);

  options.idle_velocity_ki_scale = {8, 8, 8, 3, 3, 3};
  const auto gains = sdk_gains(options);
  ASSERT_EQ(gains.size(), 7u);
  for (size_t j = 0; j < 7; ++j) {
    const double scale = j < 6 ? options.idle_velocity_ki_scale[j] : 1.0;   // the carriage keeps Trossen's
    EXPECT_DOUBLE_EQ(gains[j].at(Mode::idle).velocity.ki, named[j].at(Mode::idle).velocity.ki * scale);
    EXPECT_DOUBLE_EQ(gains[j].at(Mode::idle).velocity.kp, named[j].at(Mode::idle).velocity.kp);
    EXPECT_DOUBLE_EQ(gains[j].at(Mode::idle).velocity.imax, named[j].at(Mode::idle).velocity.imax);
    EXPECT_DOUBLE_EQ(gains[j].at(Mode::idle).position.kp, 0.0);
    EXPECT_DOUBLE_EQ(gains[j].at(Mode::position).velocity.ki, named[j].at(Mode::position).velocity.ki);
  }
  EXPECT_DOUBLE_EQ(gains[1].at(Mode::idle).velocity.ki, 0.32);

  options.idle_velocity_ki_scale[4] = 0.0;
  EXPECT_THROW(sdk_gains(options), std::invalid_argument);
}

TEST(SdkOptions, RosTuningScalesThePositionModeGainsOfTheArmJoints)
{
  using trossen_arm::Mode;
  const auto named = sdk_motor_parameters("wxai_v0_20250509");
  SdkOptions options;
  options.position_kp_scale = {1, 1, 1, 1.5, 2, 1};
  options.velocity_kp_scale = {1, 1, 1, 1, 3, 1};
  options.velocity_ki_scale = {1, 2, 1, 1, 1, 1};
  const auto gains = sdk_gains(options);
  EXPECT_DOUBLE_EQ(gains[3].at(Mode::position).position.kp, named[3].at(Mode::position).position.kp * 1.5);
  EXPECT_DOUBLE_EQ(gains[4].at(Mode::position).position.kp, named[4].at(Mode::position).position.kp * 2);
  EXPECT_DOUBLE_EQ(gains[4].at(Mode::position).velocity.kp, named[4].at(Mode::position).velocity.kp * 3);
  EXPECT_DOUBLE_EQ(gains[1].at(Mode::position).velocity.ki, named[1].at(Mode::position).velocity.ki * 2);
  EXPECT_DOUBLE_EQ(gains[6].at(Mode::position).position.kp, named[6].at(Mode::position).position.kp);
  EXPECT_DOUBLE_EQ(gains[4].at(Mode::idle).velocity.kp, named[4].at(Mode::idle).velocity.kp);   // idle untouched
  options.velocity_kp_scale[2] = -1.0;
  EXPECT_THROW(sdk_gains(options), std::invalid_argument);
}
