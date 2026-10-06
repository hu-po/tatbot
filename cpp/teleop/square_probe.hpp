#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "motion_constants.hpp"

namespace tatbot::square
{

using JointPose = std::array<double, 6>;
using FullJointPose = std::array<double, 7>;

// The carriage-IK drawing envelope, shared with the Python planner through
// config/motion_constants.json (motion_constants.hpp is generated from it).
inline constexpr double CARRIAGE_IK_BIAS_M = tatbot::motion_constants::CARRIAGE_IK_BIAS_M;
inline constexpr double CARRIAGE_IK_MIN_M = tatbot::motion_constants::CARRIAGE_IK_MIN_M;
inline constexpr double CARRIAGE_IK_MAX_M = tatbot::motion_constants::CARRIAGE_IK_MAX_M;
inline constexpr double CARRIAGE_IK_PEN_UP_MIN_M = tatbot::motion_constants::CARRIAGE_IK_PEN_UP_MIN_M;
inline constexpr double CARRIAGE_IK_PEN_UP_MAX_M = tatbot::motion_constants::CARRIAGE_IK_PEN_UP_MAX_M;

// Minimal WXAI FK used by the path planner. Its geometry mirrors
// urdf/tatbot.urdf right/joint_0..5 plus right/ee_gripper. A caller
// selecting another prefix must prove equivalent URDF/controller geometry.
// Nothing compares it with the vendor SDK's live FK any more: that gate left
// with wxai_teleop's square-probe mode, and wxai_teleop no longer
// links this library. The ROS CLIK parity test, which SKIPs until
// path_plan_check is built, holds it to the URDF instead.
std::array<double, 3> wxai_tcp_translation(const JointPose & joints);

// Ballpoint contact point through the measured right/tool_mount chain: the tip
// model of every seven-DOF path.
std::array<double, 3> wxai_ballpoint_tip_translation(
  const JointPose & joints, double carriage_m);

// The tool the seven-axis planner steers: its working point in link 6 at
// carriage zero and the unit axis the carriage moves it along. The follower's
// ballpoint is the compiled default (BALLPOINT_TIP_IN_LINK6, carriage +Y). A
// samples file may declare another configured WXAI arm: its tool mount must
// ride a +Y carriage in link 6, and its tip comes from the file because only
// the right follower has a compiled tool constant. The caller must establish
// that the arm has this WXAI chain before passing an expected prefix below.
struct ToolModel
{
  std::array<double, 3> tip_in_link6{};
  std::array<double, 3> carriage_axis_in_link6{};
};
ToolModel ballpoint_tool_model();
ToolModel declared_wxai_tool_model(const std::array<double, 3> & tip_in_link6);
// Working point of `tool` in the base frame at `joints` and `carriage_m`.
std::array<double, 3> wxai_tool_tip_translation(
  const JointPose & joints, double carriage_m, const ToolModel & tool);

struct CarriageJointPlan
{
  std::vector<FullJointPose> positions;
  std::vector<FullJointPose> velocities;
  std::vector<std::array<double, 3>> cartesian_references;
  // (tick index, capture index) for every sample that asks for a capture.
  std::vector<std::pair<size_t, size_t>> capture_ticks;
  std::vector<std::pair<size_t, size_t>> dip_ticks;   // (tick, k): where ink dip k bottoms out
  size_t endpoint_tick = 0;
  double max_joint_velocity_rad_s = 0.0;
  double max_joint_acceleration_rad_s2 = 0.0;   // planned velocity change per tick; reported, never a cap
  double max_carriage_velocity_m_s = 0.0;
  double max_carriage_acceleration_m_s2 = 0.0;
  double min_carriage_m = CARRIAGE_IK_BIAS_M;
  double max_carriage_m = CARRIAGE_IK_BIAS_M;
  double max_cartesian_velocity_m_s = 0.0;
  double path_length_m = 0.0;
  double max_model_error_mm = 0.0;
  double max_orientation_error_rad = 0.0;
};

using Rotation = std::array<std::array<double, 3>, 3>;

// Link-6 rotation in the arm base frame for the ballpoint tip model — the
// same frame plan_joint_path's per-sample rotation targets are expressed in.
Rotation wxai_link6_rotation(const JointPose & joints);

// One control tick of a Cartesian tip path: where the ballpoint tip must be,
// how fast it is moving, and the link-6 rotation to hold there. `capture` > 0
// marks the row that requests wrist-camera capture k (orbit files).
struct PathSample
{
  double t_s = 0.0;
  std::array<double, 3> position{};
  std::array<double, 3> velocity{};
  Rotation rotation{};
  bool pen = false;
  size_t capture = 0;
  size_t dip = 0;   // k > 0 on the row where ink dip k bottoms out (2026-09-03)
};

// A samples file (`orbit.csv` / `path.csv`, contract in docs/surface-formats.md) as
// parsed. `report` keeps the free-form header keys for printing.
// `constants_sha` is the planner's config/motion_constants.json digest: an
// orbit or path file must carry it and it must equal motion_constants::SHA.
struct PathFile
{
  std::string kind;
  std::string constants_sha;
  double period_s = 0.0;
  std::array<double, 3> tip_in_link6{};
  // header `arm` (right unless declared): the physical URDF prefix whose base
  // frame the samples are in and whose tool the planner steers.
  std::string arm = "right";
  ToolModel tool{};
  size_t capture_count = 0;
  size_t dip_count = 0;
  double start_tolerance_m = 0.001;
  bool carriage_ik = true;  // header `carriage_ik,0` keeps the carriage locked while drawing
  std::vector<std::pair<std::string, std::string>> report;
  std::vector<PathSample> samples;
};

// Parse and validate a samples file. With no expected prefix it accepts only
// the installed right/left WXAI prefixes; offline callers may
// supply a configured WXAI chain's physical URDF prefix explicitly. This is
// an identity assertion, not proof of chain equivalence or motion authority.
// Refuses an unknown schema, a frame other than `<arm>/base_link`, a period
// that differs from `period_s`, a right-arm tip model that differs from the
// ballpoint constant by more than 0.1 mm (other tips are declared and bounded), a
// non-orthonormal rotation, a non-finite value, a capture index out of
// sequence, or (kind orbit / path) a missing or different `constants_sha`:
// the samples writer and this parser must have been built from the same
// config/motion_constants.json.
// An offline caller may instead bind a compatible WXAI chain's tool tip
// independently of the CSV. Its explicit prefix is required, the working
// point remains bounded to 50..450 mm, and CSV parity is checked at 1 nm.
// This model declaration grants no calibration or hardware authority.
PathFile load_path_file(
  const std::string & path, double period_s, const std::string & expected_wxai_prefix = "",
  const std::optional<std::array<double, 3>> & expected_tip_in_link6 = std::nullopt);

// Seven-DOF ballpoint tip path plan, with the reference position, feedforward
// velocity and target rotation taken from the samples. Refuses unless the
// first tip sample is within `start_tolerance_m`; the path's orientation cap
// also applies at sample 0. Every cap (joint speed, model error, orientation
// error, carriage envelope, carriage speed and acceleration, joint limits)
// comes from motion_constants.hpp.
CarriageJointPlan plan_joint_path(
  const JointPose & start_joints,
  double start_carriage_m,
  const std::vector<PathSample> & samples,
  double period_s,
  double start_tolerance_m = 0.001,
  bool carriage_ik = true,
  const ToolModel & tool = ballpoint_tool_model());

}  // namespace tatbot::square
