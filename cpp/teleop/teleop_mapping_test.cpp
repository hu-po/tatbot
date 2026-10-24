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
  const std::vector<double> input{0.4, -0.5, 0.7, 0.2, -0.3, 0.1, 0.02};
  const std::vector<double> receiver{-0.1, -0.2, 0.5, 0.3, -0.4, 0.2, 0.0};
  Alignment alignment(true, 0.35);
  alignment.restart(input, receiver);
  for (size_t i = 0; i < 6; ++i) {
    check(std::abs(input[i] + alignment.offset(i) - receiver[i]) < 1e-12,
      "startup would step away from the measured receiver pose");
  }
  std::vector<double> previous(receiver);
  for (double t = 0; t < alignment.duration() + 0.01; t += 0.001) {
    alignment.advance(0.001);
    for (size_t i = 0; i < 6; ++i) {
      const double target = input[i] + alignment.residual() * alignment.offset(i);
      check(std::abs(target - previous[i]) / 0.001 <= 0.350001, "alignment exceeded rate");
      previous[i] = target;
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    check(std::abs(previous[i] - input[i]) < 1e-12,
      "rotary joint did not reach its intended mapped angle");
    check(std::abs(alignment.position(input[i], i) - input[i]) < 1e-12,
      "aligned mapping is not absolute");
  }
  auto displaced = receiver;
  displaced[0] = 0.9;
  alignment.restart(input, displaced);
  check(std::abs(input[0] + alignment.offset(0) - 0.9) < 1e-12,
    "resume reused an old baseline");
  // The carriage never follows the leader, so its offset never sizes the ramp.
  auto carriage_only = input;
  carriage_only[6] = 0.9;
  alignment.restart(input, carriage_only);
  check(!alignment.aligning(), "a carriage offset triggered alignment");
  Alignment relative(false, 0.35);
  relative.restart(input, receiver);
  relative.advance(60.0);
  check(std::abs(relative.position(input[2], 2) - receiver[2]) < 1e-12,
    "relative mapping faded its startup offset");
  std::cout << "mapping: startup, bounded alignment, resume, relative and carriage PASS\n";
}
