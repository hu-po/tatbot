// The README section 8 interlocks for one arm. Ported from cpp/teleop (arm_recover landing,
// the no-step start) and rust/tatbot-arm (contact.rs carriage cap, tracking_watchdog.rs stall).
#include "tatbot_hardware/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tatbot_hardware
{
namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kSettleS = 0.15;       // arm_recover PHASE_SETTLE_S: after a move, before judging it
constexpr double kLimitMargin = 1e-4;   // arm_recover LIMIT_MARGIN

bool all_finite(const Vec7 & v)
{
  return std::all_of(v.begin(), v.end(), [](double x) {return std::isfinite(x);});
}

// Largest arm-joint difference, and the carriage difference.
void step_size(const Vec7 & a, const Vec7 & b, double & arm, double & carriage)
{
  arm = 0;
  for (size_t i = 0; i < kCarriage; ++i) {arm = std::max(arm, std::fabs(a[i] - b[i]));}
  carriage = std::fabs(a[kCarriage] - b[kCarriage]);
}

std::string fmt(const Vec7 & v)
{
  std::ostringstream out;
  out.precision(4);
  out << '[';
  for (size_t i = 0; i < kJoints; ++i) {out << (i ? ", " : "") << v[i];}
  return out.str() + ']';
}
}  // namespace

void min_jerk(const Vec7 & a, const Vec7 & b, double t, double duration, Vec7 & q, Vec7 & qd)
{
  const double s = duration > 0 ? std::clamp(t / duration, 0.0, 1.0) : 1.0;
  const double p = s * s * s * (10 - 15 * s + 6 * s * s);
  const double v = duration > 0 && s < 1 ? 30 * s * s * (1 - s) * (1 - s) / duration : 0.0;
  for (size_t i = 0; i < kJoints; ++i) {
    q[i] = a[i] + (b[i] - a[i]) * p;
    qd[i] = (b[i] - a[i]) * v;
  }
}

void ContactCap::configure(const Config & c)
{
  cap_ = c.carriage_contact_cap_n;
  deflect_ = c.carriage_contact_deflect_m;
  judge_below_ = c.carriage_judge_below_rad_s;
  need_ = std::max(1, c.carriage_baseline_samples);
  trip_ticks_ = std::max(1, c.carriage_trip_ticks);
  settle_ticks_ = std::max(0, c.carriage_settle_ticks);
  rebaseline_need_ = std::max(1, c.carriage_rebaseline_samples);
  effort_ = c.carriage_qualified;
  rebasing_ = false;
  last_target_ = std::numeric_limits<double>::quiet_NaN();
  since_moved_ = settle_ticks_;
  samples_.clear();
  samples_.reserve(static_cast<size_t>(need_));
  armed_ = false;
  over_ = 0;
}

bool ContactCap::observe(
  double effort_n, double position_m, double target_m, double arm_speed, bool aligning)
{
  if (!std::isfinite(effort_n) || !std::isfinite(position_m) || !std::isfinite(target_m)) {
    over_ = 0;
    return false;
  }
  // A commanded carriage move changes the effort that holds it: from its floor (-13 N) to the 2 mm
  // travel bias (+13-15 N, steady) with the arm still (bench 2026-09-27), so a rest baseline taken on
  // the floor reads the hold as 27 N of contact. The force is not judged while the target moves nor
  // settle_ticks after; then a fresh rest median at the held target becomes the baseline. The
  // deflection check stays live throughout.
  // Any change of the commanded target is a move: a lift's quintic start creeps under 1 um a tick while
  // its effort rises 24 N (bench 2026-09-27); a held target repeats bit for bit. A move of more than
  // kRebaseMoveM from the target the baseline was taken at re-takes it; the sub-um splices at a
  // stroke's start only pause the judging.
  constexpr double kMoveM = 1e-9, kRebaseMoveM = 5e-6;
  const bool moving = std::isfinite(last_target_) && std::fabs(target_m - last_target_) > kMoveM;
  last_target_ = target_m;
  since_moved_ = moving ? 0 : std::min(since_moved_ + 1, settle_ticks_);
  if (moving && armed_ && std::fabs(target_m - baseline_target_) > kRebaseMoveM) {
    armed_ = false;
    rebasing_ = true;
  }
  // A rest median must be contiguous and exclude commanded carriage motion.
  if (!armed_) {
    if (aligning || since_moved_ < settle_ticks_) {
      samples_.clear();
    } else {
      samples_.push_back(effort_n);
      if (static_cast<int>(samples_.size()) >= (rebasing_ ? rebaseline_need_ : need_)) {
        std::nth_element(samples_.begin(), samples_.begin() + samples_.size() / 2, samples_.end());
        baseline_ = samples_[samples_.size() / 2];
        baseline_target_ = target_m;
        armed_ = true;
        rebasing_ = false;
        samples_.clear();
      }
    }
  }
  double contact = std::fabs(effort_n - (armed_ ? baseline_ : 0.0));
  const bool judged = armed_ && arm_speed < judge_below_ && since_moved_ >= settle_ticks_;
  if (judged && contact < cap_ * 0.25) {
    baseline_ += (effort_n - baseline_) * (0.0025 / 10.0);   // slow drift, contact.rs
    contact = std::fabs(effort_n - baseline_);
  }
  // An unqualified carriage (the leader's) is screened by its deflection alone.
  const bool pushed = (effort_ && judged && contact > cap_) || position_m - target_m > deflect_;
  over_ = pushed ? over_ + 1 : 0;
  return over_ >= trip_ticks_;
}

ArmCore::ArmCore(Config config, TipFk fk)
: cfg_(std::move(config)), fk_(std::move(fk))
{
  cap_.configure(cfg_);
  safety_.fill(0.0);
  safety_[kOk] = 1.0;
  safety_[kAge] = -1.0;
  for (size_t i = 0; i < kJoints; ++i) {safety_[kTripQ0 + i] = kNaN;}
}

std::string ArmCore::take_event()
{
  std::string out;
  out.swap(event_);
  return out;
}

Vec7 ArmCore::clamp(Vec7 q) const
{
  for (size_t i = 0; i < kJoints; ++i) {
    if (cfg_.lower[i] < cfg_.upper[i]) {
      q[i] = std::clamp(q[i], cfg_.lower[i] + kLimitMargin, cfg_.upper[i] - kLimitMargin);
    }
  }
  return q;
}

void ArmCore::observe_feedback(double now, const Feedback & fb)
{
  // configure() returns before the SDK daemon has any robot output: an all-zero pose is the
  // SDK's default, not a measurement (arm_recover fresh_measurement).
  const bool nonzero = all_finite(fb.q) &&
    std::any_of(fb.q.begin(), fb.q.end(), [](double x) {return x != 0.0;});
  if (nonzero && nonzero_since_ < 0) {nonzero_since_ = now;}
}

void ArmCore::activate(double now, const Feedback & fb)
{
  if (!all_finite(fb.q)) {throw std::runtime_error("activation refused: joint feedback is not finite");}
  const Vec7 held = clamp(fb.q);
  double arm_step, carriage_step;
  step_size(held, fb.q, arm_step, carriage_step);
  if (arm_step > cfg_.step_limit_rad || carriage_step > cfg_.step_limit_m) {
    throw std::runtime_error("activation refused: clamping the measured pose would exceed the first-command "
            "step limit; power off and reposition the out-of-range joint before restarting");
  }
  observe_feedback(now, fb);
  if (phase_ == Phase::kLanded) {
    // Woken: a lifecycle re-activation of this arm's hardware alone (`client wake`) takes it back where
    // it rests, as the stack's start does, so one arm comes back without a stack restart taking the other
    // arm down with it (2026-09-30). Its first command must still start at the measured pose.
    safety_[kLanded] = 0;
    idle_sent_ = false;
    phase_ = Phase::kRunning;
  }
  // A latch from before the deactivate (anything but the deactivate hold itself) survives
  // re-activation: hold the measured pose until a Decide.
  const bool keep = (phase_ == Phase::kLatched || phase_ == Phase::kRetracting) &&
    safety_[kReason] != names::kLatchDeactivated;
  phase_ = keep ? Phase::kLatched : Phase::kRunning;
  first_pending_ = true;
  hold_ = last_target_ = held;
  if (!keep) {
    safety_[kLatched] = 0;
    safety_[kReason] = names::kLatchNone;
  }
  stall_since_ = lag_since_ = -1;
  cap_.configure(cfg_);
}

void ArmCore::latch(double now, const Feedback & fb, int reason)
{
  if (phase_ == Phase::kLanded || phase_ == Phase::kInactive || phase_ == Phase::kLatched) {
    return;   // the hold pose and the first reason are kept until unlatch
  }
  if (phase_ != Phase::kRetracting) {   // a retract is already latched with reason 3
    safety_[kLatched] = 1;
    safety_[kReason] = reason;
  }
  safety_[kLanding] = 0;
  phase_ = Phase::kLatched;
  hold_ = clamp(fb.q);
  stall_since_ = lag_since_ = -1;
  std::ostringstream text;
  text << "latched (reason " << reason << ") at " << fmt(hold_);
  set_event(text.str());
  (void)now;
}

Output ArmCore::hold_now(double now, const Feedback & fb, int reason)
{
  if (phase_ == Phase::kLanded) {return Output{};}
  if (phase_ == Phase::kInactive) {   // never activated: hold where it is
    phase_ = Phase::kRunning;
    hold_ = clamp(fb.q);
  }
  latch(now, fb, reason);
  return hold();
}

Output ArmCore::hold() const
{
  Output out;
  out.send = true;
  out.q = hold_;
  out.qd.fill(0.0);
  return out;
}

Output ArmCore::track(const Vec7 & q, const Vec7 & qd)
{
  Output out;
  out.send = true;
  out.q = last_target_ = clamp(q);
  for (size_t i = 0; i < kJoints; ++i) {out.qd[i] = std::isfinite(qd[i]) ? qd[i] : 0.0;}
  return out;
}

bool ArmCore::stalled(double now, const Vec7 & target, const Vec7 & q)
{
  double error = 0, unused = 0;
  step_size(target, q, error, unused);
  if (error < cfg_.stall_error_rad) {
    stall_since_ = -1;
    return false;
  }
  if (stall_since_ < 0) {
    stall_since_ = now;
    stall_anchor_ = q;
    return false;
  }
  if (now - stall_since_ <= cfg_.stall_time_s) {return false;}
  double progress = 0;
  step_size(q, stall_anchor_, progress, unused);
  if (progress >= cfg_.stall_progress_rad) {
    stall_since_ = now;
    stall_anchor_ = q;
    return false;
  }
  return true;
}

bool ArmCore::guard_tripped(double now, const Vec7 & target, const Feedback & fb)
{
  if (guard_level_ != names::kGuardTipLag || !fk_ ||
    now - guard_armed_at_ < cfg_.tip_lag_arm_after_s)
  {
    lag_since_ = contact_since_ = -1;
    force_sum_ = 0;
    force_n_ = 0;
    return false;
  }
  // Contact: the arm's joints, links and EE mount give ~20 mm at ~1.5 N/mm before the joints lag
  // (bench 2026-09-26), so the tip lag alone trips ~25 mm deep; the joint torques see the first
  // millimetre. The force is judged against its mean over the first contact_baseline_s after
  // arming, which carries the pose's gravity-compensation and friction bias.
  if (cfg_.contact_force_n > 0) {
    const double force = tip_force_along_tool(fk_, fb.q, fb.effort);
    if (std::isfinite(force)) {
      if (now - guard_armed_at_ < cfg_.tip_lag_arm_after_s + cfg_.contact_baseline_s) {
        force_sum_ += force;
        ++force_n_;
      } else if (force_n_ > 0 && force - force_sum_ / force_n_ > cfg_.contact_force_n) {
        if (contact_since_ < 0) {contact_since_ = now;}
        if (now - contact_since_ >= cfg_.contact_hold_s) {
          std::ostringstream text;
          text << "touch guard: contact force " << force - force_sum_ / force_n_ << " N";
          set_event(text.str());
          return true;
        }
      } else {
        contact_since_ = -1;
      }
    }
  }
  // Commanded-minus-measured tip along the commanded tool axis: positive only when the tip is
  // held back short of the command toward the paper (the paper's outward normal is -z_tcp). Lateral tracking error and sag away from the paper never count.
  const TipPose a = fk_(target), b = fk_(fb.q);
  double lag = 0;
  for (size_t i = 0; i < 3; ++i) {lag += (a.p[i] - b.p[i]) * a.z[i];}
  if (lag <= cfg_.tip_lag_trip_m) {
    lag_since_ = -1;
    return false;
  }
  if (lag_since_ < 0) {lag_since_ = now;}
  if (now - lag_since_ < cfg_.tip_lag_hold_s) {return false;}
  set_event("touch guard: tip lag");
  return true;
}

Vec7 ArmCore::joints_at(double t) const
{
  Vec7 q;
  q.fill(kNaN);
  if (hist_size_ == 0) {return q;}
  auto at = [this](size_t back) {return (hist_head_ + kHistory - 1 - back) % kHistory;};
  size_t newer = at(0);
  if (t >= hist_t_[newer]) {return hist_q_[newer];}
  for (size_t back = 1; back < hist_size_; ++back) {
    const size_t older = at(back);
    if (hist_t_[older] <= t) {
      const double span = hist_t_[newer] - hist_t_[older];
      const double s = span > 0 ? (t - hist_t_[older]) / span : 1.0;
      for (size_t i = 0; i < kJoints; ++i) {
        q[i] = hist_q_[older][i] + (hist_q_[newer][i] - hist_q_[older][i]) * s;
      }
      return q;
    }
    newer = older;
  }
  return hist_q_[newer];   // older than the history: its oldest sample
}

bool ArmCore::probe_stop(double now, const ProbeStatus & probe, Vec7 & q_edge, bool & fault)
{
  // Moving toward the ball without a working probe could push its stylus past its overtravel.
  if (!probe.enabled || !probe.fresh) {
    fault = true;
    set_event(probe.enabled ? "probe guard: no fresh probe frame; holding" :
      "probe guard: this stack reads no probe; holding");
    return true;
  }
  if (!probe.triggered) {
    probe_clear_ = true;
    return false;
  }
  if (!probe_clear_) {
    fault = true;
    set_event("probe guard: the probe already read triggered when armed (still touching, or a broken wire)");
    return true;
  }
  // The touch happened at the relay's kernel stamp of the rising edge: the latched joints are those of
  // that instant, not of this later tick (the frame's transport, a tick). A stamp from before the
  // arming is not this touch's; then this tick stands.
  double edge = now;
  if (std::isfinite(probe.rise_age_s) && now - probe.rise_age_s >= guard_armed_at_) {edge = now - probe.rise_age_s;}
  q_edge = joints_at(edge);
  std::ostringstream text;
  text << "probe guard: touch " << (now - edge) * 1e3 << " ms before this tick, at " << fmt(q_edge);
  set_event(text.str());
  return true;
}

double tip_force_along_tool(const TipFk & fk, const Vec7 & q, const Vec7 & tau)
{
  if (!fk) {return kNaN;}
  constexpr double h = 1e-6;
  const TipPose p0 = fk(q);
  double jac[3][6];
  for (size_t j = 0; j < 6; ++j) {
    Vec7 dq = q;
    dq[j] += h;
    const TipPose pj = fk(dq);
    for (size_t i = 0; i < 3; ++i) {jac[i][j] = (pj.p[i] - p0.p[i]) / h;}
  }
  double a[3][3] = {}, b[3] = {};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 6; ++j) {b[i] += jac[i][j] * tau[j];}
    for (size_t k = 0; k < 3; ++k) {
      for (size_t j = 0; j < 6; ++j) {a[i][k] += jac[i][j] * jac[k][j];}
    }
    a[i][i] += 1e-9;
  }
  const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  if (!(std::fabs(det) > 1e-18)) {return kNaN;}
  double f[3];
  for (size_t col = 0; col < 3; ++col) {   // Cramer's rule
    double m[3][3];
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {m[i][k] = k == col ? b[i] : a[i][k];}
    }
    f[col] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
  }
  return f[0] * p0.z[0] + f[1] * p0.z[1] + f[2] * p0.z[2];
}

void ArmCore::start_landing(double now, const Feedback & fb)
{
  phase_ = Phase::kLanding;
  motion_t0_ = now;
  from_ = clamp(fb.q);
  // arm_recover: the staged sweep keeps the measured carriage; the sleep pose is every joint at
  // zero except the wrist roll, with the carriage at its configured rest.
  staged_ = clamp(cfg_.staged_positions);
  staged_[kCarriage] = from_[kCarriage];
  sleep_.fill(0.0);
  sleep_[kWristRoll] = cfg_.staged_positions[kWristRoll];
  sleep_[kCarriage] = cfg_.staged_positions[kCarriage];
  sleep_ = clamp(sleep_);
  safety_[kLatched] = 0;
  safety_[kReason] = names::kLatchNone;
  safety_[kTripped] = 0;
  safety_[kLanding] = 1;
  stall_since_ = -1;
  set_event("landing from " + fmt(from_));
}

Output ArmCore::landing(double now, const Feedback & fb)
{
  const double t = now - motion_t0_;
  const double t1 = cfg_.landing_takeover_s, t2 = t1 + cfg_.landing_staged_s;
  const double t3 = t2 + cfg_.landing_sleep_s;
  Output out;
  out.send = true;
  if (t < t1) {
    out.q = from_;
  } else if (t < t2) {
    min_jerk(from_, staged_, t - t1, cfg_.landing_staged_s, out.q, out.qd);
  } else {
    min_jerk(staged_, sleep_, t - t2, cfg_.landing_sleep_s, out.q, out.qd);
  }
  if (t >= t3 + kSettleS) {
    double arm = 0, carriage = 0;
    step_size(fb.q, sleep_, arm, carriage);
    if (arm <= cfg_.landing_verify_rad && carriage <= cfg_.landing_verify_carriage_m) {
      phase_ = Phase::kLanded;
      safety_[kLanding] = 0;
      safety_[kLanded] = 1;
      std::ostringstream text;
      text << "landed and idle: worst joint " << arm << " rad, carriage " << carriage * 1e3 << " mm";
      set_event(text.str());
      Output idle;
      idle.idle_request = !idle_sent_;
      idle_sent_ = true;
      return idle;
    }
  }
  if (t > cfg_.landing_budget_s) {
    latch(now, fb, names::kLatchStall);
    set_event("landing did not verify within its budget; holding");
    return hold();
  }
  last_target_ = out.q;
  return out;
}

Output ArmCore::tick(
  double now, const Feedback & fb, const Commands & cmd, const EstopStatus & estop,
  bool controller_error, const ProbeStatus & probe)
{
  observe_feedback(now, fb);
  if (all_finite(fb.q)) {
    hist_t_[hist_head_] = now;
    hist_q_[hist_head_] = fb.q;
    hist_head_ = (hist_head_ + 1) % kHistory;
    hist_size_ = std::min(hist_size_ + 1, kHistory);
  }
  safety_[kSource] = estop.source;
  safety_[kOk] = estop.ok ? 1 : 0;
  safety_[kAge] = estop.age_s;
  safety_[kProbe] = probe.triggered ? 1 : 0;
  safety_[kCtrlError] = controller_error ? 1 : 0;
  const int level = std::isfinite(cmd.guard_mode) ? static_cast<int>(std::lround(cmd.guard_mode)) : 0;
  const int guard = level >= 0 && level <= 2 ? level : 0;
  if (guard != guard_level_) {
    guard_level_ = guard;
    guard_armed_at_ = now;
    lag_since_ = contact_since_ = -1;
    force_sum_ = 0;
    force_n_ = 0;
    probe_clear_ = false;
  }
  safety_[kGuard] = guard;
  const bool unlatch_req = std::isfinite(cmd.unlatch) && cmd.unlatch != last_unlatch_;
  const bool land_req = std::isfinite(cmd.land) && cmd.land != last_land_;
  if (unlatch_req) {last_unlatch_ = cmd.unlatch;}
  if (land_req) {last_land_ = cmd.land;}

  if (phase_ == Phase::kInactive) {return Output{};}
  if (phase_ == Phase::kLanded) {
    if (unlatch_req) {safety_[kUnlatchAck] = cmd.unlatch;}
    if (land_req) {safety_[kLandAck] = cmd.land;}
    return Output{};
  }

  // Conditions that hold the arm from any moving phase, checked every tick.
  const int estop_reason = estop.pressed ? names::kLatchEstop : names::kLatchEstopStale;
  if (!estop.ok) {
    latch(now, fb, estop_reason);
  } else if (controller_error) {
    latch(now, fb, names::kLatchControllerError);
  } else if (phase_ == Phase::kRunning || phase_ == Phase::kLanding) {
    bool fast = false;
    for (size_t i = 0; i < kJoints; ++i) {
      const double limit = i == kCarriage ? cfg_.over_velocity_m_s : cfg_.over_velocity_rad_s;
      fast = fast || !(std::fabs(fb.qd[i]) <= limit);
    }
    if (fast) {latch(now, fb, names::kLatchOverVelocity);}
  }

  Output out;
  if (land_req) {
    // Acked either way; a refused land leaves landing 0 and says why in the log.
    safety_[kLandAck] = cmd.land;
    if (!estop.ok) {
      set_event("land refused: e-stop not released");
    } else if (controller_error) {
      set_event("land refused: controller error (restart the stack, then land)");
    } else if (phase_ == Phase::kRetracting) {
      set_event("land refused: carriage retracting");
    } else if (phase_ == Phase::kLatched || phase_ == Phase::kRunning) {
      start_landing(now, fb);
    }
  }
  if (unlatch_req) {
    safety_[kUnlatchAck] = cmd.unlatch;
    if (phase_ == Phase::kRetracting) {set_event("unlatch refused: carriage retracting");}
    if (phase_ == Phase::kLatched) {
      double arm = 0, carriage = 0;
      step_size(cmd.q, hold_, arm, carriage);
      if (!estop.ok) {
        safety_[kReason] = estop_reason;
      } else if (controller_error) {
        safety_[kReason] = names::kLatchControllerError;
      } else if (!all_finite(cmd.q) || !(arm <= cfg_.step_limit_rad) ||
        !(carriage <= cfg_.step_limit_m))
      {
        safety_[kReason] = names::kLatchStepRefused;
        set_event("unlatch refused: command steps from the held pose");
      } else {
        phase_ = Phase::kRunning;
        first_pending_ = false;
        safety_[kLatched] = 0;
        safety_[kReason] = names::kLatchNone;
        safety_[kTripped] = 0;
        stall_since_ = lag_since_ = -1;
        last_target_ = hold_;
        set_event("unlatched");
        out = hold();   // this tick holds; commands are tracked from the next
        out.reset_commands = true;
        return out;
      }
    }
  }

  switch (phase_) {
    case Phase::kLatched:
      return hold();
    case Phase::kRetracting: {
        const double t = now - motion_t0_;
        min_jerk(from_, hold_, t, cfg_.carriage_retract_s, out.q, out.qd);
        out.send = true;
        if (t >= cfg_.carriage_retract_s + kSettleS) {
          phase_ = Phase::kLatched;
          std::ostringstream text;
          text << "carriage retract: measured " << fb.q[kCarriage] * 1e3 << " mm (target "
               << hold_[kCarriage] * 1e3 << " mm)";
          set_event(text.str());
        }
        return out;
      }
    case Phase::kLanding:
      if (stalled(now, last_target_, fb.q)) {
        latch(now, fb, names::kLatchStall);
        return hold();
      }
      return landing(now, fb);
    default:
      break;
  }

  // Running.
  if (first_pending_) {
    if (!all_finite(cmd.q)) {return hold();}
    double arm = 0, carriage = 0;
    step_size(cmd.q, fb.q, arm, carriage);
    const bool warm = nonzero_since_ >= 0 && now - nonzero_since_ >= cfg_.feedback_warmup_s;
    if (!warm || !(arm <= cfg_.step_limit_rad) || !(carriage <= cfg_.step_limit_m)) {
      latch(now, fb, names::kLatchStepRefused);
      return hold();
    }
    first_pending_ = false;
  }
  const Vec7 target = all_finite(cmd.q) ? clamp(cmd.q) : last_target_;
  const Vec7 ff = all_finite(cmd.q) ? cmd.qd : Vec7{};
  if (stalled(now, target, fb.q)) {
    latch(now, fb, names::kLatchStall);
    return hold();
  }
  if (guard_level_ == names::kGuardProbe) {
    Vec7 q_edge{};
    bool fault = false;
    if (probe_stop(now, probe, q_edge, fault)) {
      latch(now, fb, names::kLatchGuardProbe);
      if (!fault) {   // a touch: tripped, with the joints at its edge; a fault is a hold without a trip
        safety_[kTripped] = 1;
        for (size_t i = 0; i < kJoints; ++i) {safety_[kTripQ0 + i] = q_edge[i];}
      }
      return hold();
    }
  }
  if (guard_tripped(now, target, fb)) {
    latch(now, fb, names::kLatchGuardTipLag);
    safety_[kTripped] = 1;
    for (size_t i = 0; i < kJoints; ++i) {safety_[kTripQ0 + i] = fb.q[i];}
    return hold();
  }
  double arm_speed = 0;
  for (size_t i = 0; i < kCarriage; ++i) {arm_speed = std::max(arm_speed, std::fabs(fb.qd[i]));}
  const bool aligning = std::isfinite(ff[kCarriage]) && std::fabs(ff[kCarriage]) > 1e-6;
  if (cap_.observe(fb.effort[kCarriage], fb.q[kCarriage], target[kCarriage], arm_speed, aligning)) {
    latch(now, fb, names::kLatchCarriageContact);
    if (!cfg_.carriage_qualified) {   // no retract has been exercised on this carriage: hold
      set_event("carriage deflected over its screen: holding (an unqualified carriage never retracts)");
      return hold();
    }
    phase_ = Phase::kRetracting;
    motion_t0_ = now;
    from_ = hold_;
    hold_[kCarriage] = std::min(cfg_.carriage_retract_m, cfg_.upper[kCarriage] - kLimitMargin);
    if (!(cfg_.lower[kCarriage] < cfg_.upper[kCarriage])) {hold_[kCarriage] = cfg_.carriage_retract_m;}
    set_event("carriage contact over its cap: retracting");
    min_jerk(from_, hold_, 0.0, cfg_.carriage_retract_s, out.q, out.qd);
    out.send = true;
    return out;
  }
  return track(target, ff);
}

}  // namespace tatbot_hardware
