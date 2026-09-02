#include "square_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "square_probe_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main()
{
  using tatbot::square::MotionGuard;
  using tatbot::square::Pose;
  using tatbot::square::JointPose;
  using tatbot::square::plan_joint_square;
  using tatbot::square::plan_joint_spiral;
  using tatbot::square::plan_joint_spiral_with_carriage;
  using tatbot::square::wxai_link6_rotation;
  using tatbot::square::spiral_path_samples;
  using tatbot::square::plan_joint_path;
  using tatbot::square::load_path_file;
  using tatbot::square::quintic_segment;
  using tatbot::square::targets;
  using tatbot::square::wxai_tcp_translation;
  using tatbot::square::wxai_ballpoint_tip_translation;

  const Pose start{0.20, -0.10, 0.05, 0.1, 0.2, 0.3};
  const auto square = targets(start, 0.006);
  require(std::fabs(square[0][0] - 0.194) < 1e-12,
    "positive-X start did not choose inward -X edge");
  require(std::fabs(square[1][1] + 0.094) < 1e-12,
    "negative-Y start did not choose inward +Y edge");
  require(std::fabs(square[2][0] - start[0]) < 1e-12, "third edge did not return X");
  require(square[3] == start, "fourth edge did not close at the start pose");
  for (const auto & pose : square) {
    require(pose[2] == start[2], "square changed Z");
    require(pose[3] == start[3] && pose[4] == start[4] && pose[5] == start[5],
      "square changed orientation");
  }
  const Pose negative_start{-0.20, 0.10, 0.05, 0.1, 0.2, 0.3};
  const auto negative_square = targets(negative_start, 0.006);
  require(std::fabs(negative_square[0][0] + 0.194) < 1e-12,
    "negative-X start did not choose inward +X edge");
  require(std::fabs(negative_square[1][1] - 0.094) < 1e-12,
    "positive-Y start did not choose inward -Y edge");
  bool invalid_rejected = false;
  try {
    (void) targets(start, 0.0);
  } catch (const std::invalid_argument &) {
    invalid_rejected = true;
  }
  require(invalid_rejected, "non-positive side was accepted");

  const Pose segment_start{0.0, 0.0, 0.1, 1.0, 2.0, 3.0};
  const Pose segment_target{0.006, 0.0, 0.1, 1.0, 2.0, 3.0};
  const auto segment_begin = quintic_segment(segment_start, segment_target, 0.0, 12.0);
  const auto segment_middle = quintic_segment(segment_start, segment_target, 6.0, 12.0);
  const auto segment_end = quintic_segment(segment_start, segment_target, 12.0, 12.0);
  require(segment_begin.position == segment_start && segment_begin.feedforward_velocity[0] == 0.0,
    "quintic segment does not start at rest");
  require(std::fabs(segment_middle.position[0] - 0.003) < 1e-12,
    "quintic segment midpoint is not halfway");
  require(std::fabs(segment_middle.feedforward_velocity[0] - 0.0009375) < 1e-12,
    "quintic segment peak velocity changed");
  require(segment_end.position == segment_target &&
    std::fabs(segment_end.feedforward_velocity[0]) < 1e-12,
    "quintic segment does not end at rest");

  // A recorded hardware witness: the canonical model must
  // reproduce the vendor SDK's live TCP before it is allowed to plan motion.
  const JointPose witness_joints{
    0.140573740005, 1.457045793533, 0.664721131325,
    0.048256658018, -0.018501564860, 1.589036345482};
  const auto witness_tcp = wxai_tcp_translation(witness_joints);
  require(std::fabs(witness_tcp[0] - 0.386395695312) < 1e-12 &&
    std::fabs(witness_tcp[1] - 0.0581346411709) < 1e-12 &&
    std::fabs(witness_tcp[2] - 0.0623176194811) < 1e-12,
    "WXAI model does not reproduce the live SDK FK witness");
  Pose witness_pose{
    witness_tcp[0], witness_tcp[1], witness_tcp[2], 0.0, 0.0, 0.0};
  const auto witness_targets = targets(witness_pose, 0.006);
  const auto joint_plan = plan_joint_square(witness_joints, witness_targets, 12.0, 0.0025);
  require(joint_plan.positions.size() == 4 * 4800,
    "joint square plan has the wrong sample count");
  require(joint_plan.edge_end_ticks == std::array<size_t, 4>{4800, 9600, 14400, 19200},
    "joint square plan edge boundaries changed");
  require(joint_plan.max_joint_velocity_rad_s < 0.004,
    "joint square plan is faster than the hardware-qualified model witness");
  require(joint_plan.max_model_error_mm < 0.01,
    "joint square plan has excessive model tracking error");
  require(joint_plan.max_orientation_error_rad < 1e-5,
    "joint square plan does not hold its starting orientation");
  const auto planned_end = wxai_tcp_translation(joint_plan.positions.back());
  require(std::fabs(planned_end[0] - witness_tcp[0]) < 1e-6 &&
    std::fabs(planned_end[1] - witness_tcp[1]) < 1e-6 &&
    std::fabs(planned_end[2] - witness_tcp[2]) < 1e-6,
    "joint square plan did not close at the starting TCP");

  const auto spiral_plan = plan_joint_spiral(
    witness_joints, 0.006, 3.0, 180.0, 2.0, 0.0025);
  require(spiral_plan.positions.size() == 72000,
    "joint spiral plan has the wrong sample count");
  require(spiral_plan.edge_end_ticks[0] == 72000,
    "joint spiral plan endpoint changed");
  require(spiral_plan.cartesian_references.size() == spiral_plan.positions.size(),
    "joint spiral plan omitted Cartesian references");
  require(spiral_plan.max_cartesian_velocity_m_s > 0.00032 &&
    spiral_plan.max_cartesian_velocity_m_s < 0.00033,
    "joint spiral plan Cartesian speed changed");
  require(spiral_plan.path_length_m > 0.0572 && spiral_plan.path_length_m < 0.0573,
    "joint spiral plan path length changed");
  const auto ten_second_reference = spiral_plan.cartesian_references[3999];
  const double ten_second_radius = std::hypot(
    ten_second_reference[0] - witness_tcp[0],
    ten_second_reference[1] - witness_tcp[1]);
  require(ten_second_radius > 0.0012 && ten_second_radius < 0.0013,
    "constant-arc spiral still dwells near its center");
  require(spiral_plan.max_joint_velocity_rad_s < 0.01,
    "joint spiral plan is faster than the guarded model witness");
  require(spiral_plan.max_model_error_mm < 0.01,
    "joint spiral plan has excessive model tracking error");
  require(spiral_plan.max_orientation_error_rad < 1e-5,
    "joint spiral plan does not hold its starting orientation");
  const auto spiral_end = wxai_tcp_translation(spiral_plan.positions.back());
  require(std::fabs(spiral_end[0] - (witness_tcp[0] + 0.006)) < 1e-6 &&
    std::fabs(spiral_end[1] - witness_tcp[1]) < 1e-6 &&
    std::fabs(spiral_end[2] - witness_tcp[2]) < 1e-6,
    "joint spiral plan did not reach its final radius at constant Z");

  // Exact follower pose at the scripted handoff in a recorded 120-second
  // physical baseline. The A/B planner must qualify this
  // region with the ballpoint carriage already biased 2 mm off its hard stop.
  const JointPose carriage_witness_joints{
    0.173762112856, 1.544403791428, 0.826848268509,
    -0.061226826161, 0.121118485928, 1.642061471939};
  const auto carriage_witness_tip = wxai_ballpoint_tip_translation(
    carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M);
  const auto carriage_plan = plan_joint_spiral_with_carriage(
    carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M,
    0.006, 3.0, 120.0, 2.0, 0.0025);
  require(carriage_plan.positions.size() == 48000 && carriage_plan.endpoint_tick == 48000,
    "carriage-IK spiral plan has the wrong sample count");
  require(carriage_plan.cartesian_references.size() == carriage_plan.positions.size(),
    "carriage-IK spiral plan omitted Cartesian references");
  require(carriage_plan.min_carriage_m >= tatbot::square::CARRIAGE_IK_MIN_M &&
    carriage_plan.max_carriage_m <= tatbot::square::CARRIAGE_IK_MAX_M,
    "carriage-IK spiral left its drawing envelope");
  require(carriage_plan.max_carriage_m - carriage_plan.min_carriage_m > 0.00025,
    "carriage-IK spiral did not materially exercise the carriage");
  require(carriage_plan.max_carriage_velocity_m_s < 0.001 &&
    carriage_plan.max_carriage_acceleration_m_s2 < 0.02,
    "carriage-IK spiral exceeded its planned motion envelope");
  require(carriage_plan.max_joint_velocity_rad_s < 0.01,
    "carriage-IK spiral arm plan is faster than its model witness");
  require(carriage_plan.max_model_error_mm < 0.01 &&
    carriage_plan.max_orientation_error_rad < 1e-5,
    "carriage-IK spiral has excessive model tracking error");
  JointPose carriage_end_joints{};
  std::copy_n(carriage_plan.positions.back().begin(), 6, carriage_end_joints.begin());
  const auto carriage_end_tip = wxai_ballpoint_tip_translation(
    carriage_end_joints, carriage_plan.positions.back()[6]);
  require(std::fabs(carriage_end_tip[0] - (carriage_witness_tip[0] + 0.006)) < 1e-6 &&
    std::fabs(carriage_end_tip[1] - carriage_witness_tip[1]) < 1e-6 &&
    std::fabs(carriage_end_tip[2] - carriage_witness_tip[2]) < 1e-6,
    "carriage-IK spiral did not reach its final radius at constant modeled tip Z");

  // The draw executor's generic path planner must reproduce the carriage-IK
  // spiral exactly when handed the spiral's own samples, and must reject a
  // samples file that lies about its tip model or starts away from the arm.
  {
    const auto initial_rotation = wxai_link6_rotation(carriage_witness_joints);
    const auto samples = spiral_path_samples(
      carriage_witness_tip, initial_rotation, 0.006, 3.0, 120.0, 2.0, 0.0025);
    require(samples.size() == 48000, "spiral samples have the wrong count");
    const auto path_plan = plan_joint_path(
      carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M, samples, 0.0025);
    require(path_plan.positions.size() == carriage_plan.positions.size(),
      "path plan sample count differs from the spiral plan");
    for (size_t i = 0; i < path_plan.positions.size(); i += 997) {
      for (size_t j = 0; j < 7; ++j) {
        require(path_plan.positions[i][j] == carriage_plan.positions[i][j],
          "path plan diverged from the carriage-IK spiral plan");
      }
    }
    require(path_plan.capture_ticks.empty(), "spiral samples requested captures");

    const std::string path = "/tmp/square_probe_test_samples.csv";
    {
      std::ofstream out(path);
      out << std::setprecision(17);
      out << "schema,tatbot.draw-samples/1\nkind,path\nframe,right/base_link\n"
          << "period_s,0.0025\ntip_x_m,0.20550927\ntip_y_m,0.01083364\ntip_z_m,-0.00149001\n"
          << "sample_count," << samples.size() << "\ncapture_count,1\nstart_tolerance_m,0.001\n"
          << "lean_max_deg,0.0\n"
          << "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture\n";
      for (size_t i = 0; i < samples.size(); ++i) {
        const auto & sample = samples[i];
        out << sample.t_s;
        for (double v : sample.position) {out << ',' << v;}
        for (double v : sample.velocity) {out << ',' << v;}
        for (const auto & row : sample.rotation) {for (double v : row) {out << ',' << v;}}
        out << ",1," << (i == 4000 ? 1 : 0) << '\n';
      }
    }
    const auto loaded = load_path_file(path, 0.0025);
    require(loaded.kind == "path" && loaded.samples.size() == samples.size() &&
      loaded.capture_count == 1 && loaded.report.size() == 1 &&
      loaded.report[0].first == "lean_max_deg",
      "samples file did not round-trip its header");
    require(loaded.samples[4000].capture == 1 && loaded.samples[3999].capture == 0,
      "samples file lost its capture flag");
    const auto loaded_plan = plan_joint_path(
      carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M, loaded.samples, 0.0025);
    require(loaded_plan.capture_ticks.size() == 1 && loaded_plan.capture_ticks[0].first == 4000 &&
      loaded_plan.capture_ticks[0].second == 1, "path plan lost the capture tick");
    require(loaded_plan.max_model_error_mm < 0.01, "loaded path plan tracks poorly");

    bool refused = false;
    try {load_path_file(path, 0.002);} catch (const std::exception &) {refused = true;}
    require(refused, "samples file with the wrong period was accepted");
    {
      std::ofstream out(path);
      out << "schema,tatbot.draw-samples/1\nkind,path\nframe,right/base_link\nperiod_s,0.0025\n"
          << "tip_x_m,0.2\ntip_y_m,0.01083364\ntip_z_m,-0.00149001\nsample_count,1\n"
          << "capture_count,0\n"
          << "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture\n"
          << "0.0025,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0\n";
    }
    refused = false;
    try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
    require(refused, "samples file with a wrong tip model was accepted");
    auto far = samples;
    far.front().position[0] += 0.002;
    refused = false;
    try {
      plan_joint_path(carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M, far, 0.0025);
    } catch (const std::exception &) {refused = true;}
    require(refused, "path plan starting 2 mm away was accepted");
  }

  MotionGuard guard(2.5, 9.0, 0.5, 0.5, 8);
  require(!guard.observe(0.0, {0.0, 0.0}, {0.0, 0.0}), "quiet sample tripped");
  const auto velocity = guard.observe(0.01, {0.0, -2.6}, {0.0, 0.0});
  require(velocity && velocity->code == "measured_velocity" && velocity->joint == 1,
    "velocity trip missing");

  guard.reset();
  std::optional<tatbot::square::GuardTrip> effort;
  for (int i = 0; i <= 50; ++i) {
    effort = guard.observe(i * 0.01, {0.0, 0.0}, {9.5, 0.0});
    if (effort) {break;}
  }
  require(effort && effort->code == "rolling_overforce", "rolling effort trip missing");

  std::cout << "square_probe_test: ok" << std::endl;
  return 0;
}
