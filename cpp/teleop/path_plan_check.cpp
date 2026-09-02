// Offline preflight for a draw samples file (docs/draw.md): load it with the
// executor's own parser and run the executor's own seven-DOF planner from a
// stated start pose. No arm, no SDK. `draw_stage.py plan` calls this so an
// offline plan is judged by the same code that will stream it.
//
//   path_plan_check <samples.csv> <period_s> j0 j1 j2 j3 j4 j5 carriage_m
//
// Exit 0 with a `key,value` report on stdout when the plan is accepted,
// exit 3 with the refusal on stderr when it is not, exit 2 on usage.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "square_probe.hpp"

int main(int argc, char ** argv)
{
  if (argc != 10) {
    std::cerr << "usage: path_plan_check <samples.csv> <period_s> j0 j1 j2 j3 j4 j5 carriage_m\n";
    return 2;
  }
  try {
    const std::string path = argv[1];
    const double period_s = std::stod(argv[2]);
    tatbot::square::JointPose joints{};
    for (size_t i = 0; i < 6; ++i) {joints[i] = std::stod(argv[3 + i]);}
    const double carriage_m = std::stod(argv[9]);
    const auto file = tatbot::square::load_path_file(path, period_s);
    const auto plan = tatbot::square::plan_joint_path(
      joints, carriage_m, file.samples, period_s, file.start_tolerance_m, file.carriage_ik);
    std::cout << std::setprecision(12)
              << "status,accepted\nkind," << file.kind
              << "\nsample_count," << plan.positions.size()
              << "\ncapture_count," << plan.capture_ticks.size()
              << "\ncarriage_ik," << (file.carriage_ik ? 1 : 0)
              << "\npath_length_mm," << plan.path_length_m * 1e3
              << "\nmodel_max_error_mm," << plan.max_model_error_mm
              << "\nmodel_max_orientation_error_rad," << plan.max_orientation_error_rad
              << "\nplan_max_joint_velocity_rad_s," << plan.max_joint_velocity_rad_s
              << "\nplan_max_cartesian_velocity_mm_s," << plan.max_cartesian_velocity_m_s * 1e3
              << "\nplan_min_carriage_mm," << plan.min_carriage_m * 1e3
              << "\nplan_max_carriage_mm," << plan.max_carriage_m * 1e3
              << "\nplan_max_carriage_velocity_mm_s," << plan.max_carriage_velocity_m_s * 1e3
              << "\nplan_max_carriage_acceleration_mm_s2,"
              << plan.max_carriage_acceleration_m_s2 * 1e3 << "\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "status,refused\nreason," << error.what() << "\n";
    return 3;
  }
}
