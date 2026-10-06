#include "tatbot_hardware/backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace tatbot_hardware
{

FakeBackend::FakeBackend(FakeOptions options)
: o_(std::move(options))
{
  state_.q = o_.start;
}

void FakeBackend::connect()
{
  std::lock_guard<std::mutex> lock(mutex_);
  connected_ = true;
}

Feedback FakeBackend::read()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!connected_) {throw std::runtime_error("fake arm: not connected");}
  Feedback fb = state_;
  for (size_t i = 0; i < kJoints; ++i) {fb.qd[i] += velocity_offset_[i];}
  if (reads_++ < o_.zero_reads) {return Feedback{};}
  return fb;
}

void FakeBackend::command(const Vec7 & q, const Vec7 & qd)
{
  std::lock_guard<std::mutex> lock(mutex_);
  last_q_ = q;
  last_qd_ = qd;
  ++commands_;
  if (idle_ || frozen_ || !error_.empty()) {
    state_.qd.fill(0.0);
    return;
  }
  Vec7 next = state_.q;
  for (size_t i = 0; i < kJoints; ++i) {
    next[i] += o_.dt * (qd[i] + (q[i] - state_.q[i]) / o_.tau);
  }
  if (o_.page_stiffness_n_m > 0 && o_.fk) {
    // A sprung page: the tip goes through, and the joints feel J^T f (finite-difference Jacobian).
    const TipPose tip = o_.fk(next);
    const double depth = std::max(0.0, o_.page_z - tip.p[2]);
    for (size_t j = 0; j < kCarriage; ++j) {
      Vec7 dq = next;
      dq[j] += 1e-6;
      const TipPose pj = o_.fk(dq);
      double tau = 0;
      for (size_t i = 0; i < 3; ++i) {tau += (pj.p[i] - tip.p[i]) / 1e-6 * o_.page_stiffness_n_m * depth * tip.z[i];}
      state_.effort[j] = tau;
    }
  } else if (o_.fk && o_.fk(next).p[2] < o_.page_z && o_.fk(next).p[2] < o_.fk(state_.q).p[2]) {
    next = state_.q;   // the paper stops the tip; the commanded pose runs ahead of it
  }
  for (size_t i = 0; i < kJoints; ++i) {state_.qd[i] = (next[i] - state_.q[i]) / o_.dt;}
  state_.q = next;
}

void FakeBackend::set_position_mode()
{
  std::lock_guard<std::mutex> lock(mutex_);
  idle_ = false;
}

void FakeBackend::set_idle()
{
  std::lock_guard<std::mutex> lock(mutex_);
  idle_ = true;
}

std::string FakeBackend::error()
{
  std::lock_guard<std::mutex> lock(mutex_);
  return error_;
}

void FakeBackend::inject_error(const std::string & text)
{
  std::lock_guard<std::mutex> lock(mutex_);
  error_ = text;
}

void FakeBackend::set_effort(size_t joint, double value)
{
  std::lock_guard<std::mutex> lock(mutex_);
  state_.effort.at(joint) = value;
}

void FakeBackend::set_velocity_offset(size_t joint, double value)
{
  std::lock_guard<std::mutex> lock(mutex_);
  velocity_offset_.at(joint) = value;
}

void FakeBackend::set_frozen(bool frozen)
{
  std::lock_guard<std::mutex> lock(mutex_);
  frozen_ = frozen;
}

void FakeBackend::set_page_z(double z)
{
  std::lock_guard<std::mutex> lock(mutex_);
  o_.page_z = z;
}

Vec7 FakeBackend::last_q() const {std::lock_guard<std::mutex> lock(mutex_); return last_q_;}
Vec7 FakeBackend::last_qd() const {std::lock_guard<std::mutex> lock(mutex_); return last_qd_;}
bool FakeBackend::idle() const {std::lock_guard<std::mutex> lock(mutex_); return idle_;}
int FakeBackend::commands() const {std::lock_guard<std::mutex> lock(mutex_); return commands_;}

}  // namespace tatbot_hardware
