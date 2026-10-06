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

// Archimedean spiral about `center` at fixed `rotation`, one control-tick
// sample per period, eased in and out at constant arc-length speed: the
// seven-DOF path planner's paper-validated regression witness (the 2026-09
// carriage-IK A/B drew it).
std::vector<tatbot::square::PathSample> spiral_path_samples(
  const std::array<double, 3> & center,
  const tatbot::square::Rotation & rotation,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s)
{
  const size_t ticks = static_cast<size_t>(std::ceil(duration_s / period_s));
  constexpr double pi = 3.14159265358979323846;
  const double total_angle = 2.0 * pi * turns;
  const double spiral_scale = radius_m / total_angle;
  const double path_length = 0.5 * spiral_scale * (
    total_angle * std::sqrt(1.0 + total_angle * total_angle) +
    std::asinh(total_angle));
  const double cruise_speed = path_length / (duration_s - ease_s);

  std::vector<tatbot::square::PathSample> samples;
  samples.reserve(ticks);
  for (size_t tick = 1; tick <= ticks; ++tick) {
    const double elapsed_s = std::min(duration_s, static_cast<double>(tick) * period_s);
    double distance = 0.0;
    double path_speed = 0.0;
    if (elapsed_s < ease_s) {
      const double u = elapsed_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    } else if (elapsed_s <= duration_s - ease_s) {
      distance = 0.5 * cruise_speed * ease_s +
        cruise_speed * (elapsed_s - ease_s);
      path_speed = cruise_speed;
    } else {
      const double remaining_s = duration_s - elapsed_s;
      const double u = remaining_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = path_length - cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    }
    distance = std::clamp(distance, 0.0, path_length);
    double angle = total_angle * distance / path_length;
    for (size_t iteration = 0; iteration < 6; ++iteration) {
      const double root = std::sqrt(1.0 + angle * angle);
      const double integrated_length = 0.5 * spiral_scale * (
        angle * root + std::asinh(angle));
      angle -= (integrated_length - distance) / (spiral_scale * root);
      angle = std::clamp(angle, 0.0, total_angle);
    }
    const double radius = spiral_scale * angle;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    tatbot::square::PathSample sample;
    sample.t_s = elapsed_s;
    sample.position = {
      center[0] + radius * cosine,
      center[1] + radius * sine,
      center[2]};
    sample.velocity = {
      path_speed * (cosine - angle * sine) / std::sqrt(1.0 + angle * angle),
      path_speed * (sine + angle * cosine) / std::sqrt(1.0 + angle * angle),
      0.0};
    sample.rotation = rotation;
    sample.pen = true;
    samples.push_back(sample);
  }
  return samples;
}

}  // namespace

int main()
{
  using tatbot::square::JointPose;
  using tatbot::square::wxai_link6_rotation;
  using tatbot::square::plan_joint_path;
  using tatbot::square::load_path_file;
  using tatbot::square::wxai_tcp_translation;
  using tatbot::square::wxai_ballpoint_tip_translation;

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

  // Pen-up scans preserve measured rest, while any contact retains IK bounds.
  for (double rest : {0.0, -1.7546117305755615e-6, -0.0001}) {
    tatbot::square::PathSample sample{};
    sample.position = wxai_ballpoint_tip_translation(witness_joints, rest);
    sample.rotation = wxai_link6_rotation(witness_joints);
    sample.pen = false;
    auto scan_samples = std::vector<tatbot::square::PathSample>(3, sample);
    const auto scan = plan_joint_path(witness_joints, rest, scan_samples, 0.0025);
    for (const auto & position : scan.positions) {
      require(position[6] == rest, "scan changed measured resting carriage");
    }
    scan_samples.back().pen = true;
    bool refused = false;
    try {(void) plan_joint_path(witness_joints, rest, scan_samples, 0.0025);}
    catch (const std::invalid_argument &) {refused = true;}
    require(refused, "contact accepted a carriage outside drawing bounds");
  }

  // Exact follower pose at the scripted handoff in a recorded 120-second
  // physical baseline (the 2026-09 carriage-IK A/B). Handed the spiral's own
  // samples with the ballpoint carriage already biased 2 mm off its hard stop,
  // the path planner must hold the paper-validated result, and it must reject
  // a samples file that lies about its tip model or starts away from the arm.
  const JointPose carriage_witness_joints{
    0.173762112856, 1.544403791428, 0.826848268509,
    -0.061226826161, 0.121118485928, 1.642061471939};
  const auto carriage_witness_tip = wxai_ballpoint_tip_translation(
    carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M);
  {
    const auto initial_rotation = wxai_link6_rotation(carriage_witness_joints);
    const auto samples = spiral_path_samples(
      carriage_witness_tip, initial_rotation, 0.006, 3.0, 120.0, 2.0, 0.0025);
    require(samples.size() == 48000, "spiral samples have the wrong count");
    const auto path_plan = plan_joint_path(
      carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M, samples, 0.0025);
    require(path_plan.positions.size() == 48000 && path_plan.endpoint_tick == 48000,
      "carriage-IK spiral plan has the wrong sample count");
    require(path_plan.cartesian_references.size() == path_plan.positions.size(),
      "carriage-IK spiral plan omitted Cartesian references");
    require(path_plan.min_carriage_m >= tatbot::square::CARRIAGE_IK_MIN_M &&
      path_plan.max_carriage_m <= tatbot::square::CARRIAGE_IK_MAX_M,
      "carriage-IK spiral left its drawing envelope");
    require(path_plan.max_carriage_m - path_plan.min_carriage_m > 0.00025,
      "carriage-IK spiral did not materially exercise the carriage");
    require(path_plan.max_carriage_velocity_m_s < 0.001 &&
      path_plan.max_carriage_acceleration_m_s2 < 0.02,
      "carriage-IK spiral exceeded its planned motion envelope");
    require(path_plan.max_joint_velocity_rad_s < 0.01,
      "carriage-IK spiral arm plan is faster than its model witness");
    require(path_plan.max_model_error_mm < 0.01 &&
      path_plan.max_orientation_error_rad < 1e-5,
      "carriage-IK spiral has excessive model tracking error");
    JointPose carriage_end_joints{};
    std::copy_n(path_plan.positions.back().begin(), 6, carriage_end_joints.begin());
    const auto carriage_end_tip = wxai_ballpoint_tip_translation(
      carriage_end_joints, path_plan.positions.back()[6]);
    require(std::fabs(carriage_end_tip[0] - (carriage_witness_tip[0] + 0.006)) < 1e-6 &&
      std::fabs(carriage_end_tip[1] - carriage_witness_tip[1]) < 1e-6 &&
      std::fabs(carriage_end_tip[2] - carriage_witness_tip[2]) < 1e-6,
      "carriage-IK spiral did not reach its final radius at constant modeled tip Z");
    require(path_plan.capture_ticks.empty(), "spiral samples requested captures");

    const std::string path = "/tmp/square_probe_test_samples.csv";
    // the planner stamps every samples file with the config/motion_constants.json
    // digest it was compiled against; the executor refuses any other
    const std::string sha_line = std::string("constants_sha,") + tatbot::motion_constants::SHA + "\n";
    auto write_samples = [&](const std::string & sha_header) {
        std::ofstream out(path);
        out << std::setprecision(17);
        out << "schema,tatbot.draw-samples/1\nkind,path\nframe,right/base_link\n"
            << sha_header
            << "period_s,0.0025\ntip_x_m,0.20498692817078468\ntip_y_m,0.012312678895000949\ntip_z_m,-0.0005439999999999881\n"
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
      };
    write_samples(sha_line);
    const auto loaded = load_path_file(path, 0.0025);
    require(loaded.kind == "path" && loaded.samples.size() == samples.size() &&
      loaded.capture_count == 1 && loaded.report.size() == 1 &&
      loaded.report[0].first == "lean_max_deg",
      "samples file did not round-trip its header");
    require(loaded.constants_sha == tatbot::motion_constants::SHA,
      "samples file did not round-trip constants_sha");
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
    // constants_sha: required for kind orbit/path, and it must be this build's
    write_samples("");
    std::string refusal;
    try {load_path_file(path, 0.0025);} catch (const std::exception & e) {refusal = e.what();}
    require(refusal.find("constants_sha") != std::string::npos &&
      refusal.find(tatbot::motion_constants::SHA) != std::string::npos,
      "samples file without constants_sha was accepted, or the refusal did not name the executor's SHA");
    write_samples("constants_sha,000000000000\n");
    refusal.clear();
    try {load_path_file(path, 0.0025);} catch (const std::exception & e) {refusal = e.what();}
    require(refusal.find("000000000000") != std::string::npos &&
      refusal.find(tatbot::motion_constants::SHA) != std::string::npos,
      "samples file with another constants_sha was accepted, or the refusal did not name both SHAs");
    {
      // a kind that is neither orbit nor path may omit the digest (nothing writes one today)
      std::ofstream out(path);
      out << "schema,tatbot.draw-samples/1\nkind,witness\nframe,right/base_link\nperiod_s,0.0025\n"
          << "tip_x_m,0.20498692817078468\ntip_y_m,0.012312678895000949\ntip_z_m,-0.0005439999999999881\nsample_count,1\n"
          << "capture_count,0\n"
          << "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture\n"
          << "0.0025,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0\n";
    }
    require(load_path_file(path, 0.0025).constants_sha.empty(), "a non-orbit/path kind needs no constants_sha");
    {
      std::ofstream out(path);
      out << "schema,tatbot.draw-samples/1\nkind,path\nframe,right/base_link\nperiod_s,0.0025\n"
          << sha_line
          << "tip_x_m,0.2\ntip_y_m,0.01053147691319204\ntip_z_m,0.000082\nsample_count,1\n"
          << "capture_count,0\n"
          << "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture\n"
          << "0.0025,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0\n";
    }
    refused = false;
    try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
    require(refused, "samples file with a wrong tip model was accepted");
    {
      // The leader arm declares its tool per file: the same joint chain, the
      // mirrored carriage axis, and a tip the executor cannot know on its own.
      const std::array<double, 3> left_tip{0.2014, -0.0105, 0.000082};
      const auto left_tool = tatbot::square::declared_wxai_tool_model(left_tip);
      require(left_tool.carriage_axis_in_link6[1] == 1.0, "leader carriage axis is not +Y (left/carriage_left)");
      const auto left_rotation = wxai_link6_rotation(carriage_witness_joints);
      const double left_rest = 0.0;
      auto write_left = [&](const std::string & arm_line, const std::string & frame_line) {
          std::ofstream out(path);
          out << std::setprecision(17);
          const auto tip = tatbot::square::wxai_tool_tip_translation(
            carriage_witness_joints, left_rest, left_tool);
          out << "schema,tatbot.draw-samples/1\nkind,path\n" << arm_line << frame_line << sha_line
              << "period_s,0.0025\ntip_x_m," << left_tip[0] << "\ntip_y_m," << left_tip[1]
              << "\ntip_z_m," << left_tip[2] << "\nsample_count,2\ncapture_count,0\n"
              << "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture\n";
          for (int row = 0; row < 2; ++row) {
            out << 0.0025 * (row + 1);
            for (double v : tip) {out << ',' << v;}
            out << ",0,0,0";
            for (const auto & r : left_rotation) {for (double v : r) {out << ',' << v;}}
            out << ",0,0\n";
          }
        };
      write_left("arm,left\n", "frame,left/base_link\n");
      const auto left = load_path_file(path, 0.0025);
      require(left.arm == "left" && left.tool.carriage_axis_in_link6[1] == 1.0 &&
        left.tool.tip_in_link6 == left_tip, "leader samples file did not resolve the leader tool");
      const auto left_plan = plan_joint_path(
        carriage_witness_joints, left_rest, left.samples, 0.0025, left.start_tolerance_m,
        left.carriage_ik, left.tool);
      require(left_plan.positions.size() == 2 && left_plan.max_model_error_mm < 0.01,
        "leader hold did not plan at its own tip");
      require(left_plan.positions.back()[6] == left_rest, "leader pen-up hold moved the carriage");
      // A caller that has resolved a third WXAI chain may bind its physical
      // prefix explicitly. The runtime executor has no such binding and
      // continues to refuse an unfamiliar prefix by default.
      write_left("arm,third\n", "frame,third/base_link\n");
      refused = false;
      try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
      require(refused, "unbound third-arm samples were accepted");
      const auto third = load_path_file(path, 0.0025, "third");
      require(third.arm == "third" && third.tool.tip_in_link6 == left_tip,
        "explicit third-arm prefix did not bind its tool model");
      const auto third_plan = plan_joint_path(
        carriage_witness_joints, left_rest, third.samples, 0.0025,
        third.start_tolerance_m, third.carriage_ik, third.tool);
      require(third_plan.positions.size() == 2 && third_plan.max_model_error_mm < 0.01,
        "third-arm hold did not plan with the WXAI chain");
      refused = false;
      try {load_path_file(path, 0.0025, "left");} catch (const std::exception &) {refused = true;}
      require(refused, "a different expected prefix was accepted");
      refused = false;
      try {load_path_file(path, 0.0025, "third/unsafe");} catch (const std::exception &) {refused = true;}
      require(refused, "an invalid expected prefix was accepted");
      write_left("arm,third\n", "frame,right/base_link\n");
      refused = false;
      try {load_path_file(path, 0.0025, "third");} catch (const std::exception &) {refused = true;}
      require(refused, "third-arm samples in another base frame were accepted");
      write_left("arm,left\n", "frame,left/base_link\n");
      // the follower's tool model cannot reach the leader's tip: its cap refuses
      refused = false;
      try {
        (void) plan_joint_path(
          carriage_witness_joints, left_rest, left.samples, 0.0025, 0.1, left.carriage_ik);
      } catch (const std::exception &) {refused = true;}
      require(refused, "follower tool model accepted the leader's tip");
      write_left("arm,left\n", "frame,right/base_link\n");
      refused = false;
      try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
      require(refused, "leader file in the follower frame was accepted");
      write_left("", "frame,left/base_link\n");
      refused = false;
      try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
      require(refused, "left/base_link without arm,left was accepted");
      write_left("arm,right\n", "frame,right/base_link\n");
      refused = false;
      try {load_path_file(path, 0.0025);} catch (const std::exception &) {refused = true;}
      require(refused, "the follower accepted a tip that is not its ballpoint");
      refused = false;
      try {load_path_file(path, 0.0025, "right");} catch (const std::exception &) {refused = true;}
      require(refused, "explicit right prefix bypassed the compiled ballpoint constant");
    }
    auto far = samples;
    far.front().position[0] += 0.002;
    refused = false;
    try {
      plan_joint_path(carriage_witness_joints, tatbot::square::CARRIAGE_IK_BIAS_M, far, 0.0025);
    } catch (const std::exception &) {refused = true;}
    require(refused, "path plan starting 2 mm away was accepted");
  }

  for (const double rest : {-0.006, -0.004719, -0.002, 0.03056, 0.032, 0.034}) {
    tatbot::square::PathSample sample;
    sample.t_s = 0.0025;
    sample.position = wxai_ballpoint_tip_translation(carriage_witness_joints, rest);
    sample.rotation = wxai_link6_rotation(carriage_witness_joints);
    const auto plan = plan_joint_path(carriage_witness_joints, rest, {sample}, 0.0025);
    require(plan.positions.back()[6] == rest && plan.velocities.back()[6] == 0.0,
      "pen-up scan did not preserve the measured carriage");
    sample.pen = true;
    bool refused = false;
    try {plan_joint_path(carriage_witness_joints, rest, {sample}, 0.0025);}
    catch (const std::exception &) {refused = true;}
    require(refused, "contact accepted a carriage outside the drawing range");
  }
  {
    tatbot::square::PathSample sample;
    sample.t_s = 0.0025;
    sample.position = carriage_witness_tip;
    sample.rotation = wxai_link6_rotation(carriage_witness_joints);
    auto settled_joints = carriage_witness_joints;
    settled_joints[5] += 0.0023;
    const auto plan = plan_joint_path(
      settled_joints, tatbot::square::CARRIAGE_IK_BIAS_M, {sample}, 0.0025);
    require(plan.max_orientation_error_rad > 0.0015 &&
      plan.max_orientation_error_rad < 0.005,
      "small connected wrist sag did not pass the single path orientation cap");
  }
  // The plan reports its largest commanded joint velocity change per tick
  // (2026-09-14, never a cap): a rotation target that turns at a constant
  // rate and then stops steps the command by that rate in one tick.
  {
    const double rest = tatbot::square::CARRIAGE_IK_BIAS_M;
    const auto tip = wxai_ballpoint_tip_translation(carriage_witness_joints, rest);
    const auto base_rotation = wxai_link6_rotation(carriage_witness_joints);
    std::vector<tatbot::square::PathSample> turning;
    for (size_t tick = 0; tick < 400; ++tick) {
      const double angle = 0.05 * 0.0025 * static_cast<double>(std::min<size_t>(tick, 200));
      const double c = std::cos(angle);
      const double s_ = std::sin(angle);
      // rotate about base Z, applied on the left of the witness rotation
      tatbot::square::Rotation spin{{{c, -s_, 0.0}, {s_, c, 0.0}, {0.0, 0.0, 1.0}}};
      tatbot::square::PathSample sample;
      sample.t_s = 0.0025 * static_cast<double>(tick + 1);
      sample.position = tip;
      for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
          sample.rotation[row][col] = 0.0;
          for (size_t k = 0; k < 3; ++k) {
            sample.rotation[row][col] += spin[row][k] * base_rotation[k][col];
          }
        }
      }
      turning.push_back(sample);
    }
    const auto plan = plan_joint_path(carriage_witness_joints, rest, turning, 0.0025);
    double step = 0.0;
    for (size_t tick = 0; tick < plan.velocities.size(); ++tick) {
      for (size_t joint = 0; joint < 6; ++joint) {
        const double before = tick ? plan.velocities[tick - 1][joint] : 0.0;
        step = std::max(step, std::fabs(plan.velocities[tick][joint] - before));
      }
    }
    require(std::fabs(plan.max_joint_acceleration_rad_s2 - step / 0.0025) < 1e-9,
      "reported joint acceleration is not the largest commanded velocity step per tick");
    require(plan.max_joint_acceleration_rad_s2 > 5.0,
      "a one-tick rotation-rate step should read as a large planned joint acceleration");
  }

  for (const double outside : {-0.0061, 0.0341}) {
    tatbot::square::PathSample sample;
    sample.t_s = 0.0025;
    sample.position = wxai_ballpoint_tip_translation(carriage_witness_joints, outside);
    sample.rotation = wxai_link6_rotation(carriage_witness_joints);
    bool refused = false;
    try {plan_joint_path(carriage_witness_joints, outside, {sample}, 0.0025);}
    catch (const std::exception &) {refused = true;}
    require(refused, "pen-up scan accepted carriage beyond the stationary bounds");
  }

  std::cout << "square_probe_test: ok" << std::endl;
  return 0;
}
