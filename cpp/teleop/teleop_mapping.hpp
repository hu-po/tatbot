#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tatbot::teleop
{
// Apply the same Jacobian to positions, velocities and reflected efforts.
inline double map_joint(double value, size_t joint, bool mirror_arm)
{
  // Reflection across root y=0 reverses the axial rotations about x/z.
  // Shoulder, elbow and wrist pitch rotate about y and retain their signs.
  return mirror_arm && (joint == 0 || joint == 4 || joint == 5) ? -value : value;
}

// Startup alignment (see the header comment). The follower's target is the
// leader's angle plus an offset that starts at whatever mismatch the two arms
// powered on with and fades to zero over a speed-bounded smoothstep ramp;
// afterwards the mapping is absolute, with joints 0/4/5 negated in mirrored mode,
// independent of how either arm was parked before power-on.
// In --relative mode the offset never fades, which is the old delta mapping.
class Alignment
{
public:
  Alignment(bool absolute, double rate, bool mirror_arm = false)
  : absolute_(absolute), mirror_arm_(mirror_arm), rate_(rate) {}

  // Free-space calibration follows hand-guided increments. Never erase the
  // measured parking/mount offsets by moving toward mirrored encoder zero.
  static Alignment wrist_calibration(double rate) {return Alignment(false, rate, true);}

  // Begin a ramp from a fresh pair of baselines (startup, and every resume).
  void restart(
    const std::vector<double> & leader_start,
    const std::vector<double> & follower_start)
  {
    offset_.assign(leader_start.size(), 0.0);
    largest_ = 0.0;
    largest_joint_ = 0;
    for (size_t i = 0; i < offset_.size(); ++i) {
      offset_[i] = follower_start[i] - map_joint(leader_start[i], i, mirror_arm_);
      // The last joint is the follower's tool carriage, which never follows
      // the leader; its offset is never applied, so it must not size the
      // ramp either.
      if (i + 1 < offset_.size() && std::abs(offset_[i]) > largest_) {
        largest_ = std::abs(offset_[i]);
        largest_joint_ = i;
      }
    }
    elapsed_ = 0.0;
    // smoothstep's slope peaks at 1.5x its average, so stretch the ramp by the
    // same factor to keep the fastest instant under `rate`. Below a tenth of a
    // degree there is nothing to ramp and the offset is dropped outright.
    duration_ = (absolute_ && largest_ > already_aligned_rad) ?
      1.5 * largest_ / rate_ : 0.0;
    if (duration_ == 0.0 && absolute_) {
      std::fill(offset_.begin(), offset_.end(), 0.0);
    }
  }

  void advance(double dt) {elapsed_ += dt;}

  // Fraction of the startup offset still applied: 1 at the start of the ramp,
  // 0 once aligned. Always 1 in relative mode.
  double residual() const
  {
    if (!absolute_) {return 1.0;}
    if (elapsed_ >= duration_) {return 0.0;}
    const double u = elapsed_ / duration_;
    return 1.0 - u * u * (3.0 - 2.0 * u);
  }

  bool aligning() const {return absolute_ && elapsed_ < duration_;}
  double position(double leader, size_t joint) const
  {return map_joint(leader, joint, mirror_arm_) + residual() * offset_[joint];}
  double offset(size_t joint) const {return offset_[joint];}
  double duration() const {return duration_;}
  double largest_rad() const {return largest_;}
  size_t largest_joint() const {return largest_joint_;}

private:
  static constexpr double already_aligned_rad = 0.0017;  // 0.1 deg
  bool absolute_;
  bool mirror_arm_;
  double rate_;
  std::vector<double> offset_;
  double elapsed_ = 0.0;
  double duration_ = 0.0;
  double largest_ = 0.0;
  size_t largest_joint_ = 0;
};

}  // namespace tatbot::teleop
