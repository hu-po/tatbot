// Offline preflight for a draw samples file (docs/surface-formats.md): load it with
// square_probe's parser and run its seven-DOF planner from a stated start pose.
// No arm, no SDK. Nothing streams these plans any more; the simulator's native
// reference and the ROS CLIK parity test call this to judge their ports
// against the planner they came from.
//
//   path_plan_check <samples.csv> <period_s> j0 j1 j2 j3 j4 j5 carriage_m
//     [--json] [--arm-prefix <configured WXAI URDF prefix>] [--tool-tip-in-link6 x y z]
//
// Exit 0 with a `key,value` report on stdout when the plan is accepted,
// exit 3 with the refusal on stderr when it is not, exit 2 on usage.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "square_probe.hpp"

int main(int argc, char ** argv)
{
  const char * usage =
    "usage: path_plan_check <samples.csv> <period_s> j0 j1 j2 j3 j4 j5 carriage_m "
    "[--json] [--arm-prefix <configured WXAI URDF prefix>] [--tool-tip-in-link6 x y z]\n";
  if (argc < 10) {
    std::cerr << usage;
    return 2;
  }
  bool json = false;
  bool prefix_seen = false;
  bool tip_seen = false;
  std::array<std::string, 3> tip_text{};
  std::string expected_wxai_prefix;
  for (int i = 10; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--json" && !json) {
      json = true;
    } else if (option == "--arm-prefix" && !prefix_seen && i + 1 < argc) {
      prefix_seen = true;
      expected_wxai_prefix = argv[++i];
    } else if (option == "--tool-tip-in-link6" && !tip_seen && i + 3 < argc) {
      tip_seen = true;
      for (auto & text : tip_text) {text = argv[++i];}
    } else {
      std::cerr << usage;
      return 2;
    }
  }
  if ((prefix_seen && expected_wxai_prefix.empty()) || (tip_seen && !prefix_seen)) {
    std::cerr << usage;
    return 2;
  }
  try {
    const std::string path = argv[1];
    const double period_s = std::stod(argv[2]);
    tatbot::square::JointPose joints{};
    for (size_t i = 0; i < 6; ++i) {joints[i] = std::stod(argv[3 + i]);}
    const double carriage_m = std::stod(argv[9]);
    if (!std::isfinite(period_s) || period_s <= 0.0 || !std::isfinite(carriage_m)) {
      throw std::runtime_error("non-finite seed or invalid period");
    }
    for (const auto joint : joints) {
      if (!std::isfinite(joint)) {throw std::runtime_error("non-finite seed joint");}
    }
    std::optional<std::array<double, 3>> expected_tip;
    if (tip_seen) {
      expected_tip.emplace();
      for (size_t i = 0; i < tip_text.size(); ++i) {
        size_t used = 0;
        (*expected_tip)[i] = std::stod(tip_text[i], &used);
        if (used != tip_text[i].size() || !std::isfinite((*expected_tip)[i])) {
          throw std::runtime_error("bound tool tip must contain finite numbers");
        }
      }
    }
    const auto file = tatbot::square::load_path_file(path, period_s, expected_wxai_prefix, expected_tip);
    const auto plan = tatbot::square::plan_joint_path(
      joints, carriage_m, file.samples, period_s, file.start_tolerance_m, file.carriage_ik,
      file.tool);
    if (json) {
      if (plan.positions.size() != file.samples.size() ||
        plan.velocities.size() != file.samples.size() ||
        plan.cartesian_references.size() != file.samples.size())
      {
        throw std::runtime_error("joint plan sample alignment mismatch");
      }
      // Machine-readable offline plan; this does not authorize an executor.
      std::cout << std::setprecision(17)
                << "{\"schema\":\"tatbot.joint-plan/1\",\"hardware_authority\":false"
                << ",\"constants_sha\":" << std::quoted(file.constants_sha)
                << ",\"period_s\":" << period_s
                << ",\"sample_count\":" << plan.positions.size()
                << ",\"max_model_error_mm\":" << plan.max_model_error_mm
                << ",\"max_orientation_error_rad\":" << plan.max_orientation_error_rad;
      std::cout << ",\"seed\":[";
      for (size_t i = 0; i < joints.size(); ++i) {
        if (i) {std::cout << ',';}
        std::cout << joints[i];
      }
      std::cout << ',' << carriage_m << ']';
      std::cout << ",\"tool_model\":{\"arm_prefix\":" << std::quoted(file.arm)
                << ",\"source\":" << std::quoted(tip_seen ? "bound-input" :
        (file.arm == "right" ? "compiled-default" : "samples-header"))
                << ",\"tip_in_link6\":[";
      for (size_t i = 0; i < 3; ++i) {
        if (i) {std::cout << ',';}
        std::cout << file.tool.tip_in_link6[i];
      }
      std::cout << "],\"carriage_axis_in_link6\":[0,1,0]}";
      const auto rows = [](const auto & values) {
          std::cout << '[';
          for (size_t row = 0; row < values.size(); ++row) {
            if (row) {std::cout << ',';}
            std::cout << '[';
            for (size_t col = 0; col < values[row].size(); ++col) {
              if (col) {std::cout << ',';}
              std::cout << values[row][col];
            }
            std::cout << ']';
          }
          std::cout << ']';
        };
      std::cout << ",\"positions\":"; rows(plan.positions);
      std::cout << ",\"velocities\":"; rows(plan.velocities);
      std::cout << ",\"cartesian_references\":"; rows(plan.cartesian_references);
      std::cout << ",\"pen\":[";
      for (size_t i = 0; i < file.samples.size(); ++i) {
        if (i) {std::cout << ',';}
        std::cout << (file.samples[i].pen ? "true" : "false");
      }
      const auto marks = [](const auto & values) {
          std::cout << '[';
          for (size_t i = 0; i < values.size(); ++i) {
            if (i) {std::cout << ',';}
            std::cout << '[' << values[i].first << ',' << values[i].second << ']';
          }
          std::cout << ']';
        };
      std::cout << "],\"capture_ticks\":"; marks(plan.capture_ticks);
      std::cout << ",\"dip_ticks\":"; marks(plan.dip_ticks);
      std::cout << "}\n";
      return 0;
    }
    std::cout << std::setprecision(12)
              << "status,accepted\nkind," << file.kind
              << "\nsample_count," << plan.positions.size()
              << "\ncapture_count," << plan.capture_ticks.size()
              << "\ncarriage_ik," << (file.carriage_ik ? 1 : 0)
              << "\npath_length_mm," << plan.path_length_m * 1e3
              << "\nmodel_max_error_mm," << plan.max_model_error_mm
              << "\nmodel_max_orientation_error_rad," << plan.max_orientation_error_rad
              << "\nplan_max_joint_velocity_rad_s," << plan.max_joint_velocity_rad_s
              << "\nplan_max_joint_acceleration_rad_s2," << plan.max_joint_acceleration_rad_s2
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
