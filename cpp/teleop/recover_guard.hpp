#pragma once

// arm_recover's guard on the measured pose, free of the vendor SDK so ctest can
// drive it. `Limit` is any type with position_min, position_max and
// position_tolerance: trossen_arm::JointLimit in arm_recover.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tatbot::recover
{
constexpr size_t CARRIAGE = 6;         // joints 0-5 rotate (rad); the carriage after them is linear (m)
constexpr double LIMIT_MARGIN = 1e-4;  // slack kept inside a clamped controller limit

inline std::string fmt(double value, int precision = 4)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

inline std::string fmt_signed(double value)
{
  std::ostringstream out;
  out << std::showpos << std::fixed << std::setprecision(3) << value;
  return out.str();
}

inline std::string joints(const std::vector<size_t> & indices)
{
  std::string out = "[";
  for (size_t i = 0; i < indices.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(indices[i]);
  }
  return out + "]";
}

// Measured axes outside the controller's admitted feedback range.
template<class Limit>
std::vector<size_t> outside_limits(
  const std::vector<double> & positions, const std::vector<Limit> & limits)
{
  std::vector<size_t> bad;
  for (size_t i = 0; i < positions.size() && i < limits.size(); ++i) {
    // Rotary encoders may sit in the controller's configured feedback band at
    // a hard stop. The carriage remains an exact datum: its boot-limit
    // overtravel is what triggers loading the session golden.
    const double raw_tolerance = i < CARRIAGE ? limits[i].position_tolerance : 0.0;
    const double tolerance = std::isfinite(raw_tolerance) && raw_tolerance > 0.0 ?
      raw_tolerance : 0.0;
    if (positions[i] < limits[i].position_min - tolerance ||
      positions[i] > limits[i].position_max + tolerance)
    {
      bad.push_back(i);
    }
  }
  return bad;
}

// An arm joint measured past its limits by more than the controller's own
// tolerance. The takeover would clamp it into its limits: after a power cycle
// at the staged wrist roll the blue controller counted joint 5 a full turn low
// (-4.796 rad where it had read +1.49), and that clamp is a 95 deg snap to -pi,
// after which staging turns the wrist and its camera cable one full extra turn.
// Thrown before anything is commanded, the golden included.
struct BeyondLimits : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

// Throw BeyondLimits for the arm joints outside their feedback band.
template<class Limit>
void refuse_beyond_limits(
  const std::string & name, const std::vector<double> & positions,
  const std::vector<Limit> & limits)
{
  std::vector<size_t> beyond;
  for (size_t i : outside_limits(positions, limits)) {
    if (i < CARRIAGE) {beyond.push_back(i);}
  }
  if (beyond.empty()) {return;}
  std::string measured;
  std::string which;
  for (size_t i : beyond) {
    measured += (measured.empty() ? "joint " : "; joint ") + std::to_string(i) + " at " +
      fmt_signed(positions[i]) + " rad (limits " + fmt_signed(limits[i].position_min) + ".." +
      fmt_signed(limits[i].position_max) + ", tolerance " + fmt(limits[i].position_tolerance, 3) + ")";
    which += (which.empty() ? "" : ", ") + std::to_string(i);
  }
  throw BeyondLimits(
          name + " landing refused, nothing commanded: " + measured + ". A power cycle can leave a "
          "joint counted a full turn off, and the takeover would snap it into its limits before "
          "staging turns it the long way round. Power the controller off, turn joint" +
          (beyond.size() > 1 ? "s " : " ") + which + " by hand to near 0, power it on, then run "
          "`tatbot arm recover` again.");
}

// Make the measured pose a legal position-mode hold before takeover, or refuse
// it.
//
// A power-cycled controller boots with its own limits and reports NO error
// while its motors idle, so the fault-only golden path never fires; the
// takeover then commands a hold past the boot limit and every following
// command fails with "modes different than configured modes". When the
// measured pose is outside the controller's live limits, push the golden
// and re-read; then clamp the hold target to whatever limits the controller
// actually enforces. An arm joint past its limits by more than the tolerance
// is not clamped but refused (BeyondLimits), before the golden and before any
// command, and again against the golden's limits once they are in.
//
// `read_limits()` returns the limits the controller enforces now, empty when
// they cannot be read; `push_golden()` loads the arm's golden and says whether
// it did.
template<class ReadLimits, class PushGolden>
std::vector<double> guard_measured_pose(
  const std::string & name, const std::vector<double> & positions,
  ReadLimits read_limits, PushGolden push_golden, std::ostream & log)
{
  auto limits = read_limits();
  refuse_beyond_limits(name, positions, limits);
  auto bad = outside_limits(positions, limits);
  if (!bad.empty()) {
    std::string detail;
    for (size_t i : bad) {
      detail += (detail.empty() ? "" : ", ") + fmt(positions[i]) + " vs [" +
        fmt(limits[i].position_min) + ", " + fmt(limits[i].position_max) + "]";
    }
    log << name << " measured pose outside the controller's feedback tolerance at joints "
        << joints(bad) << " (" << detail << ") — applying the golden before takeover" << std::endl;
    if (push_golden()) {
      limits = read_limits();
      refuse_beyond_limits(name, positions, limits);
      bad = outside_limits(positions, limits);
      if (!bad.empty()) {
        log << name << " still outside the golden feedback tolerance at joints " << joints(bad)
            << "; holding at the clamped target" << std::endl;
      }
    }
  }
  std::vector<double> hold = positions;
  std::vector<size_t> moved;
  for (size_t i = 0; i < hold.size() && i < limits.size(); ++i) {
    hold[i] = std::min(std::max(hold[i], limits[i].position_min + LIMIT_MARGIN),
        limits[i].position_max - LIMIT_MARGIN);
    if (hold[i] != positions[i]) {moved.push_back(i);}
  }
  std::vector<size_t> reported;
  for (size_t i : moved) {
    if (std::find(bad.begin(), bad.end(), i) != bad.end()) {reported.push_back(i);}
  }
  if (!reported.empty()) {
    std::string detail;
    for (size_t i : reported) {
      detail += (detail.empty() ? "" : ", ") + fmt(positions[i]) + "->" + fmt(hold[i]);
    }
    log << name << " hold target clamped into limits at joints " << joints(reported)
        << " (" << detail << ")" << std::endl;
  }
  return hold;
}
}  // namespace tatbot::recover
