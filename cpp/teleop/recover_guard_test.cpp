#include "recover_guard.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void check(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

struct Limit
{
  double position_min;
  double position_max;
  double position_tolerance;
};

constexpr double PI = 3.14159265358979323846;

// The goldens' limits: the vendor's, but for the follower carriage's -6 mm.
const std::vector<Limit> GOLDEN{
  {-PI, PI, 0.2}, {0.0, PI, 0.2}, {0.0, 2.356, 0.2}, {-PI / 2, PI / 2, 0.4},
  {-PI / 2, PI / 2, 0.4}, {-PI, PI, 0.4}, {-0.006, 0.04, 0.004}};

// A controller as the guard sees it: the limits it enforces now and the ones a
// golden push installs (none: no golden to push). It counts the pushes.
struct Controller
{
  std::vector<Limit> limits;
  std::vector<Limit> golden;
  int pushes = 0;
  std::ostringstream log{};

  std::vector<double> guard(const std::vector<double> & measured)
  {
    return tatbot::recover::guard_measured_pose(
      "leader@192.0.2.3", measured, [this] {return limits;},
      [this] {
        ++pushes;
        if (golden.empty()) {return false;}
        limits = golden;
        return true;
      }, log);
  }

  // The refusal's message, or "" when the guard returned a hold.
  std::string refusal(const std::vector<double> & measured)
  {
    try {
      guard(measured);
    } catch (const tatbot::recover::BeyondLimits & e) {
      return e.what();
    }
    return "";
  }
};

bool says(const std::string & text, const char * part) {return text.find(part) != std::string::npos;}
}  // namespace

int main()
{
  using tatbot::recover::LIMIT_MARGIN;
  // The blue controller power-cycled at the staged wrist roll: healthy, idle,
  // joint 5 counted a full turn low. Refused with no golden pushed; the guard
  // is what stands between that reading and the takeover.
  {
    Controller controller{GOLDEN, GOLDEN};
    const std::string why =
      controller.refusal({0.018, 0.002, 0.001, -0.008, -0.008, 1.487 - 2 * PI, 0.0});
    check(!why.empty(), "a wrist counted a full turn off was clamped instead of refused");
    check(controller.pushes == 0, "the refusal pushed the golden");
    check(says(why, "joint 5 at -4.796 rad (limits -3.142..+3.142, tolerance 0.400)"),
      "the refusal does not name the joint, its reading and its band");
    check(!says(why, "joint 0") && !says(why, "joints"), "the refusal named a joint inside its band");
    check(says(why, "Power the controller off, turn joint 5 by hand to near 0, power it on"),
      "the refusal does not say how to recover");
  }
  // Joints resting on their stops a few mrad past a limit stay inside the
  // feedback band: taken over, the hold kept just inside the limit as before.
  {
    Controller controller{GOLDEN, GOLDEN};
    const auto hold = controller.guard({0.0, -0.0055, -0.0017, 0.0, 0.0, PI / 2, 0.0});
    check(controller.pushes == 0, "a pose inside the feedback band pushed the golden");
    check(hold[1] == LIMIT_MARGIN && hold[2] == LIMIT_MARGIN,
      "a joint on its stop was not held just inside its limit");
    check(hold[5] == PI / 2 && hold[6] == 0.0, "a joint inside its limits was moved");
  }
  // The band's edge: 0.39 past pi is taken over, 0.41 is refused, on both sides.
  {
    Controller controller{GOLDEN, GOLDEN};
    for (double sign : {1.0, -1.0}) {
      std::vector<double> measured{0.0, 0.1, 0.1, 0.0, 0.0, sign * (PI + 0.39), 0.0};
      check(controller.refusal(measured).empty(), "a joint inside its band was refused");
      measured[5] = sign * (PI + 0.41);
      check(!controller.refusal(measured).empty(), "a joint past its band was not refused");
    }
    // Every joint past its band is named.
    const std::string why = controller.refusal({0.0, -0.21, 0.1, 0.0, 2.0, 0.0, 0.0});
    check(says(why, "joint 1 at -0.210 rad") && says(why, "; joint 4 at +2.000 rad") &&
      says(why, "turn joints 1, 4 by hand"), "a refusal of two joints does not name both");
  }
  // The carriage on its stop past its boot limit: the golden, then the hold at
  // the measured value; without a golden, the hold clamped into the boot
  // limit. Both as before.
  {
    auto boot = GOLDEN;
    boot[6] = {0.0, 0.04, 0.004};
    const std::vector<double> on_stop{0.0, 0.1, 0.1, 0.0, 0.0, PI / 2, -0.0047};
    Controller controller{boot, GOLDEN};
    const auto hold = controller.guard(on_stop);
    check(controller.pushes == 1, "the carriage past its boot limit did not push the golden");
    check(hold[6] == -0.0047, "the carriage hold moved although the golden admits it");
    Controller bare{boot, {}};
    check(bare.guard(on_stop)[6] == LIMIT_MARGIN && bare.pushes == 1,
      "the carriage hold was not clamped into the boot limit without a golden");
  }
  // A golden pushed for the carriage is judged too: a joint inside the boot
  // band but past the golden's is refused before any command.
  {
    auto boot = GOLDEN;
    boot[5] = {-4.0, 4.0, 0.4};
    boot[6] = {0.0, 0.04, 0.004};
    Controller controller{boot, GOLDEN};
    check(says(controller.refusal({0.0, 0.1, 0.1, 0.0, 0.0, -3.7, -0.0047}), "joint 5 at -3.700 rad"),
      "a joint past the golden's band was clamped");
    check(controller.pushes == 1, "the golden was not pushed for the carriage");
  }
  // Unreadable limits judge nothing and clamp nothing; a tolerance that is not
  // a positive finite number counts as none.
  {
    Controller unreadable{{}, {}};
    const std::vector<double> measured{0.0, 0.1, 0.1, 0.0, 0.0, -4.796, 0.0};
    check(unreadable.guard(measured) == measured && unreadable.pushes == 0,
      "the guard judged limits it could not read");
    auto odd = GOLDEN;
    odd[5].position_tolerance = std::nan("");
    odd[4].position_tolerance = -1.0;
    Controller controller{odd, odd};
    check(!controller.refusal({0.0, 0.1, 0.1, 0.0, 0.0, PI + 0.01, 0.0}).empty(),
      "a non-finite tolerance admitted a joint past its limit");
    check(!controller.refusal({0.0, 0.1, 0.1, 0.0, PI / 2 + 0.01, 0.0, 0.0}).empty(),
      "a negative tolerance admitted a joint past its limit");
  }
  return 0;
}
