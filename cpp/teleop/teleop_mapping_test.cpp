#include "teleop_mapping.hpp"

#include <cstdlib>
#include <iostream>

namespace {
void check(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
}

int main()
{
  using tatbot::teleop::Alignment;
  using tatbot::teleop::map_joint;
  // Incident reproduction: both wrists were parked near +90 degrees. The
  // former absolute mirror walked the receiver toward -90 degrees over 13 s.
  const std::vector<double> parked_right{
    -0.0005722, 0.012014, 0.019642, 0.008962, 0.0001907, 1.567674, 0.0};
  const std::vector<double> parked_left{
    -0.0001907, -0.0001907, -0.0001907, -0.012398, -0.0005722, 1.567292, 0.02};
  auto wrist = Alignment::wrist_calibration(0.35);
  wrist.restart(parked_right, parked_left);
  check(!wrist.aligning() && wrist.duration() == 0.0, "wrist mode scheduled an alignment");
  for (int tick = 0; tick < 24000; ++tick) {
    wrist.advance(0.0025);
    for (size_t joint = 0; joint < 6; ++joint) {
      check(std::abs(wrist.position(parked_right[joint], joint) - parked_left[joint]) < 1e-12,
        "stationary leader caused wrist calibration motion");
    }
  }
  for (size_t joint = 0; joint < 6; ++joint) {
    const double direction = joint == 0 || joint == 4 || joint == 5 ? -1.0 : 1.0;
    check(std::abs(wrist.position(parked_right[joint] + 0.1, joint)
      - parked_left[joint] - direction * 0.1) < 1e-12, "increment mapping has wrong sign");
  }
  auto moved_left = parked_left;
  auto moved_right = parked_right;
  moved_left[5] = -0.7; moved_right[5] = 2.0;
  wrist.restart(moved_right, moved_left);
  wrist.advance(60.0);
  check(std::abs(wrist.position(2.0, 5) + 0.7) < 1e-12,
    "resume erased the new measured wrist offset");
  const std::vector<double> input{0.4, -0.5, 0.7, 0.2, -0.3, 0.1, 0.02};
  const std::vector<double> receiver{-0.1, -0.2, 0.5, 0.3, -0.4, 0.2, 0.0};
  const std::vector<double> signs{-1, 1, 1, 1, -1, -1};
  for (bool mirrored : {false, true}) {
    Alignment alignment(true, 0.35, mirrored);
    alignment.restart(input, receiver);
    for (size_t i = 0; i < 6; ++i) {
      check(std::abs(map_joint(input[i], i, mirrored) + alignment.offset(i) - receiver[i]) < 1e-12,
        "startup would step away from the measured receiver pose");
    }
    std::vector<double> previous(receiver);
    for (double t = 0; t < alignment.duration() + 0.01; t += 0.001) {
      alignment.advance(0.001);
      for (size_t i = 0; i < 6; ++i) {
        const double target = map_joint(input[i], i, mirrored) + alignment.residual() * alignment.offset(i);
        check(std::abs(target - previous[i]) / 0.001 <= 0.350001, "alignment exceeded rate");
        previous[i] = target;
      }
    }
    for (size_t i = 0; i < 6; ++i) {
      check(std::abs(previous[i] - (mirrored ? signs[i] : 1) * input[i]) < 1e-12,
        "rotary joint did not reach its intended mapped angle");
      check(map_joint(0.2, i, mirrored) == 0.2 * (mirrored ? signs[i] : 1),
        "velocity mapping differs from position mapping");
      check(map_joint(3.0, i, mirrored) * 0.2 == 3.0 * map_joint(0.2, i, mirrored),
        "effort and velocity mappings disagree");
    }
    auto displaced = receiver;
    displaced[0] = 0.9;
    alignment.restart(input, displaced);
    check(std::abs(map_joint(input[0], 0, mirrored) + alignment.offset(0) - 0.9) < 1e-12,
      "resume reused an old baseline");
  }
  // The mapping Jacobian must be identical for target velocity and reflected
  // effort, preserving virtual work while base and wrist directions reverse.
  check(map_joint(0.2, 0, true) == -0.2, "base velocity was not mirrored");
  check(map_joint(3.0, 0, true) * 0.2 == 3.0 * map_joint(0.2, 0, true),
    "effort and velocity mappings disagree");
  check(map_joint(0.03, 6, true) == 0.03, "carriage was mirrored");
  Alignment mirrored(true, 0.35, true);
  auto opposite = input;
  for (size_t i = 0; i < 6; ++i) {opposite[i] *= signs[i];}
  opposite[6] = 0.9;
  mirrored.restart(input, opposite);
  check(!mirrored.aligning(), "matching mirrored arms or carriage offset triggered alignment");
  for (size_t joint : {size_t(0), size_t(4), size_t(5)}) {
    auto displaced = opposite;
    displaced[joint] += 0.3;
    mirrored.restart(input, displaced);
    check(mirrored.largest_joint() == joint && mirrored.aligning(),
      "mirrored wrist mismatch did not size the startup ramp");
    check(std::abs(map_joint(input[joint], joint, true) + mirrored.offset(joint)
      - displaced[joint]) < 1e-12, "mirrored wrist resume would step the target");
  }
  std::cout << "rotary signs:";
  for (size_t i = 0; i < 6; ++i) {std::cout << ' ' << map_joint(1.0, i, true);}
  std::cout << '\n';
  std::cout << "mapping: startup, bounded alignment, resume, velocity, effort and carriage PASS\n";
}
