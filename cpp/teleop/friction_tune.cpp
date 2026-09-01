// Interactive per-joint friction compensation tuning for a WXAI V0 arm — a
// parameterized version of the vendor's demos/cpp/joint_characteristics_finetune.cpp.
// This is the fix for a "sticky" feeling leader: the firmware cancels joint
// friction using per-joint JointCharacteristic values (EEPROM), and the
// factory defaults are conservative.
//
// Usage:
//   friction_tune <ip> <joint_index 0-6> [config.yaml]
//                 [--estop DEV | --no-estop]
//
// The selected joint floats in external_effort mode (all others idle — rest
// the arm in a pose where the joint can move, and support it as needed).
// Repeatedly enter increments to friction_constant_term until the joint moves
// freely without buzzing or running away, per the vendor's guidance:
//   1) start with a very small increment like 1e-3
//   2) double the INCREMENT until there's a noticeable behavioral change
//   3) use that increment size to dial the value in
// Too high a value makes the joint self-drive/oscillate — back off immediately.
//
// Enter 's' to save ALL current configs to <ip>_tuned.yaml (use as the arm's
// config file afterwards), 'q' or Ctrl+C to quit without saving a file.
// Changes made here are applied to the arm's EEPROM immediately either way.

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <poll.h>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "estop_monitor.hpp"
#include "libtrossen_arm/trossen_arm.hpp"

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "usage: friction_tune <ip> <joint_index 0-6> [config.yaml] "
              << "[--estop DEV | --no-estop]" << std::endl;
    return 1;
  }
  const std::string ip = argv[1];
  const int index = std::stoi(argv[2]);
  std::string config_file;
  // From the hardware profile (TATBOT_ESTOP_DEVICE) or --estop; an empty
  // device with the monitor enabled fails at open — never a silent skip.
  const char * estop_env = std::getenv("TATBOT_ESTOP_DEVICE");
  std::string estop_device = (estop_env && *estop_env) ? estop_env : "";
  bool estop_enabled = true;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--no-estop") {
      estop_enabled = false;
    } else if (arg == "--estop" && i + 1 < argc) {
      estop_device = argv[++i];
    } else if (config_file.empty()) {
      config_file = arg;
    } else {
      std::cerr << "unexpected argument: " << arg << std::endl;
      return 1;
    }
  }

  try {
    std::atomic<int> estop_state{tatbot::estop::disabled};
    std::unique_ptr<tatbot::estop::Monitor> estop;
    if (estop_enabled) {
      estop = std::make_unique<tatbot::estop::Monitor>(
        estop_device, true, estop_state);
      bool announced = false;
      while (estop_state.load() > tatbot::estop::ok) {
        if (!announced) {
          std::cout << "E-stop engaged or awaiting heartbeat — release/reconnect "
                    << "before friction tuning." << std::endl;
          announced = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }

    trossen_arm::TrossenArmDriver driver;
    driver.configure(
      trossen_arm::Model::wxai_v0,
      trossen_arm::StandardEndEffector::wxai_v0_base,
      ip,
      true  // clear_error
    );
    if (!config_file.empty()) {
      driver.load_configs_from_file(config_file);
    }
    if (index < 0 || index >= driver.get_num_joints()) {
      std::cerr << "joint_index out of range" << std::endl;
      return 1;
    }

    // Float only the joint under test.
    std::vector<trossen_arm::Mode> modes(driver.get_num_joints(), trossen_arm::Mode::idle);
    modes[index] = trossen_arm::Mode::external_effort;
    driver.set_joint_modes(modes);
    driver.set_joint_external_effort(index, 0.0, false);

    std::cout << "Joint " << index << " floating. Move it by hand and judge the drag.\n"
              << "Enter an increment to friction_constant_term (e.g. 1e-3), 's' to save\n"
              << "all configs to " << ip << "_tuned.yaml, 'q' to quit." << std::endl;

    bool estop_holding = false;
    while (true) {
      if (estop_enabled && estop_state.load() > tatbot::estop::ok) {
        if (!estop_holding) {
          driver.set_all_modes(trossen_arm::Mode::position);
          driver.set_all_positions(driver.get_all_positions(), 0.0, false);
          estop_holding = true;
          std::cout << "\nE-STOP: joint frozen; release/reconnect to continue."
                    << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      if (estop_holding) {
        driver.set_joint_modes(modes);
        driver.set_joint_external_effort(index, 0.0, false);
        estop_holding = false;
        std::cout << "E-stop clear: friction tuning resumed." << std::endl;
      }

      auto characteristics = driver.get_joint_characteristics();
      std::cout << "friction_constant_term[" << index << "] = "
                << characteristics[index].friction_constant_term << "\n> " << std::flush;
      pollfd stdin_poll = {STDIN_FILENO, POLLIN, 0};
      while (poll(&stdin_poll, 1, 50) == 0) {
        if (estop_enabled && estop_state.load() > tatbot::estop::ok) {break;}
      }
      if (estop_enabled && estop_state.load() > tatbot::estop::ok) {
        std::cout << std::endl;
        continue;
      }
      std::string input;
      if (!std::getline(std::cin, input) || input == "q") {
        break;
      }
      if (input == "s") {
        const std::string out = ip + "_tuned.yaml";
        driver.save_configs_to_file(out);
        std::cout << "Saved configs to " << out << std::endl;
        break;
      }
      if (input.empty()) {
        continue;
      }
      characteristics[index].friction_constant_term += std::stod(input);
      driver.set_joint_characteristics(characteristics);
    }

    driver.set_all_modes(trossen_arm::Mode::idle);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
