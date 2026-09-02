#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tatbot::square
{

using Pose = std::array<double, 6>;
using JointPose = std::array<double, 6>;
using FullJointPose = std::array<double, 7>;

inline constexpr double CARRIAGE_IK_BIAS_M = 0.002;
inline constexpr double CARRIAGE_IK_MIN_M = 0.0005;
inline constexpr double CARRIAGE_IK_MAX_M = 0.0035;

// Four Cartesian targets which return to start. The first X and Y edges point
// toward the base-frame origin so a hand-guided start near the workspace edge
// does not ask the IK solver to extend farther outward. Orientation and Z
// remain exactly as the operator left them at the trigger point.
std::array<Pose, 4> targets(const Pose & start, double side_m);

double translation_error_mm(const Pose & measured, const Pose & target);

struct SegmentSample
{
  Pose position;
  Pose feedforward_velocity;
};

// Time-scaled quintic segment: position, velocity and acceleration are all
// continuous at the corners, with zero endpoint velocity and acceleration.
SegmentSample quintic_segment(
  const Pose & start, const Pose & target, double elapsed_s, double duration_s);

struct JointPlan
{
  std::vector<JointPose> positions;
  std::vector<JointPose> velocities;
  std::vector<std::array<double, 3>> cartesian_references;
  std::array<size_t, 4> edge_end_ticks{};
  double max_joint_velocity_rad_s = 0.0;
  double max_cartesian_velocity_m_s = 0.0;
  double path_length_m = 0.0;
  double max_model_error_mm = 0.0;
  double max_orientation_error_rad = 0.0;
};

// Minimal WXAI FK used only by the one-shot probe. Its geometry mirrors
// urdf/tatbot.urdf right/joint_0..5 plus right/ee_gripper. Before motion,
// wxai_teleop compares this result to the vendor SDK's live FK and refuses a
// mismatch, so a stale duplicated constant fails closed.
std::array<double, 3> wxai_tcp_translation(const JointPose & joints);

// Ballpoint contact point through the measured right/tool_mount chain. This is
// used only by the explicitly gated carriage-IK A/B mode; the ordinary square
// and spiral retain the vendor SDK TCP model above.
std::array<double, 3> wxai_ballpoint_tip_translation(
  const JointPose & joints, double carriage_m);

// Precompute the complete square as a joint-position trajectory with damped
// least-squares differential IK. Nothing is sent to the arm until every sample
// has passed model tracking, joint velocity and joint-limit checks.
JointPlan plan_joint_square(
  const JointPose & start_joints,
  const std::array<Pose, 4> & cartesian_targets,
  double edge_s,
  double period_s);

// Precompute one Archimedean spiral about the trigger point. Radius and angle
// follow an approximately constant arc-length speed, with a short quintic
// speed ease at each end so velocity and acceleration meet the stationary
// hold continuously. The tip stays at trigger Z/orientation; the final point
// is radius_m along +base-X from the center.
JointPlan plan_joint_spiral(
  const JointPose & start_joints,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s);

struct CarriageJointPlan
{
  std::vector<FullJointPose> positions;
  std::vector<FullJointPose> velocities;
  std::vector<std::array<double, 3>> cartesian_references;
  // (tick index, capture index) for every sample that asks for a capture.
  std::vector<std::pair<size_t, size_t>> capture_ticks;
  size_t endpoint_tick = 0;
  double max_joint_velocity_rad_s = 0.0;
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
// marks a row where the executor must hold still and request a wrist-camera
// capture of that index before advancing (draw orbit only).
struct PathSample
{
  double t_s = 0.0;
  std::array<double, 3> position{};
  std::array<double, 3> velocity{};
  Rotation rotation{};
  bool pen = false;
  size_t capture = 0;
};

// A samples file (`orbit.csv` / `path.csv`, contract in docs/draw.md) as the
// executor sees it. `report` keeps the free-form header keys for printing.
struct PathFile
{
  std::string kind;
  double period_s = 0.0;
  std::array<double, 3> tip_in_link6{};
  size_t capture_count = 0;
  double start_tolerance_m = 0.001;
  bool carriage_ik = true;  // header `carriage_ik,0` keeps the carriage locked while drawing
  std::vector<std::pair<std::string, std::string>> report;
  std::vector<PathSample> samples;
};

// Parse and validate a samples file. Refuses an unknown schema, a frame other
// than right/base_link, a period that differs from `period_s`, a tip model
// that differs from the ballpoint constant by more than 0.1 mm, a
// non-orthonormal rotation, a non-finite value, or a capture index out of
// sequence.
PathFile load_path_file(const std::string & path, double period_s);

// Seven-DOF ballpoint tip path plan: the carriage-IK spiral's planner with the
// reference position, feedforward velocity and target rotation taken from the
// samples instead of generated inline. Refuses unless the first sample is
// within `start_tolerance_m` / 0.02 rad of the start pose. Every existing cap
// (joint speed, model error, orientation error, carriage envelope, carriage
// speed and acceleration, joint limits) applies unchanged.
CarriageJointPlan plan_joint_path(
  const JointPose & start_joints,
  double start_carriage_m,
  const std::vector<PathSample> & samples,
  double period_s,
  double start_tolerance_m = 0.001,
  bool carriage_ik = true);

// Archimedean spiral about `center` at fixed `rotation`, as one control-tick
// sample per period: the exact reference the carriage-IK A/B streamed.
std::vector<PathSample> spiral_path_samples(
  const std::array<double, 3> & center,
  const Rotation & rotation,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s);

// Seven-DOF ballpoint-only A/B candidate. The carriage begins at a positive
// 2 mm bias established before the operator approaches the paper, remains in
// a tightly guarded 0.5..3.5 mm drawing envelope, and returns toward its bias
// through a null-space objective while the measured tool tip follows the same
// spiral reference as the six-joint plan. Since the draw executor landed this
// is spiral_path_samples + plan_joint_path; it is kept as the paper baseline.
CarriageJointPlan plan_joint_spiral_with_carriage(
  const JointPose & start_joints,
  double start_carriage_m,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s);

struct GuardTrip
{
  std::string code;
  size_t joint = 0;
  double value = 0.0;
  double limit = 0.0;
};

// Script-only measured-motion backstop. The normal carriage contact cap and
// hardware E-stop stay active in wxai_teleop; this adds the same 2.5 rad/s and
// rolling 9 Nm envelopes used by the LeRobot follower.
class MotionGuard
{
public:
  MotionGuard(
    double velocity_limit = 2.5,
    double overforce_limit = 9.0,
    double overforce_window_s = 0.5,
    double overforce_fraction = 0.5,
    size_t overforce_min_samples = 8);

  void reset();
  std::optional<GuardTrip> observe(
    double now_s,
    const std::vector<double> & arm_velocities,
    const std::vector<double> & arm_efforts);

private:
  double velocity_limit_;
  double overforce_limit_;
  double overforce_window_s_;
  double overforce_fraction_;
  size_t overforce_min_samples_;
  std::deque<std::pair<double, bool>> overforce_;
};

}  // namespace tatbot::square
