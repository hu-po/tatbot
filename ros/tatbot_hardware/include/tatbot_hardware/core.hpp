#pragma once
// The framework-free arm core: every README section 8 interlock for one arm, one tick at a time.
// No ROS, no SDK, no clock of its own: the adapter (tatbot_arm.cpp) or a test feeds it measured
// feedback, the incoming commands and the e-stop status, and sends what it returns.
#include <array>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "tatbot_hardware/names.hpp"

namespace tatbot_hardware
{

inline constexpr size_t kJoints = 7;    // joint_0..joint_5 (rad), then the carriage (m)
inline constexpr size_t kCarriage = 6;
inline constexpr size_t kWristRoll = 5;
using Vec7 = std::array<double, kJoints>;
using Vec3 = std::array<double, 3>;
// base_frame -> tcp_frame for a joint vector (KDL in the adapter, anything in tests): the tip
// position and the tcp +z axis (along the tool, toward the paper), both in the base frame.
struct TipPose
{
  Vec3 p{}, z{};
};
using TipFk = std::function<TipPose(const Vec7 &)>;

struct Feedback
{
  Vec7 q{}, qd{}, effort{};
};

struct Config
{
  // Safety numbers: stack.yaml safety.* (README section 8); defaults are the stack.yaml values.
  double step_limit_rad = 0.05, step_limit_m = 0.005, feedback_warmup_s = 0.10;
  double stall_error_rad = 0.35, stall_time_s = 2.0, stall_progress_rad = 0.05;
  double over_velocity_rad_s = 3.0, over_velocity_m_s = 0.25;
  int carriage_baseline_samples = 800, carriage_trip_ticks = 40, carriage_settle_ticks = 120,
    carriage_rebaseline_samples = 200;
  double carriage_judge_below_rad_s = 0.3, carriage_retract_s = 0.6;
  double carriage_contact_cap_n = 20.0, carriage_contact_deflect_m = 0.002;
  double carriage_retract_m = 0.032;
  // config/trossen/tatbot.yaml <role>.carriage_qualified. false (the leader's carriage: its effort
  // swings 14-20 N with posture and no retract has been exercised on it): effort is not judged, only
  // deflection, and a trip holds instead of retracting.
  bool carriage_qualified = true;
  double tip_lag_trip_m = 0.0008, tip_lag_hold_s = 0.15, tip_lag_arm_after_s = 0.5;
  // The touch guard's contact trigger: the tip force along the tool axis from the joint torques,
  // over its mean in the first contact_baseline_s after arming (<= 0 turns it off).
  double contact_force_n = 2.5, contact_hold_s = 0.05, contact_baseline_s = 0.3;
  double landing_takeover_s = 0.5, landing_staged_s = 4.0, landing_sleep_s = 3.0;
  double landing_verify_rad = 0.2, landing_verify_carriage_m = 0.0005, landing_budget_s = 45.0;
  Vec7 staged_positions{0, 0, 0, 0, 0, 1.5707963267948966, 0};
  // Joint limits from the URDF; every sent position is clamped into them.
  Vec7 lower{}, upper{};
};

struct Commands
{
  Vec7 q{}, qd{};                 // JTC position and velocity commands (NaN = none yet)
  double guard_mode = 0, unlatch = 0, land = 0;
};

struct EstopStatus
{
  int source = names::kEstopNone;
  bool ok = true;       // released and fresh; always true for none
  bool pressed = false; // fresh frames say pressed (else a not-ok status is stale)
  double age_s = -1;    // age of the last valid frame; -1 for none or before the first
};

// The station probe (estop.hpp ProbeReader), fed every tick; the default is no probe.
struct ProbeStatus
{
  bool enabled = false;     // a probe reader is configured
  bool fresh = false;       // a valid frame within its timeout
  bool triggered = false;   // the latest frame says touched (or a broken wire)
  // Seconds since the latest rising edge on this clock: the relay's kernel stamp, mapped. NaN: none.
  double rise_age_s = std::numeric_limits<double>::quiet_NaN();
};

enum class Phase { kInactive, kRunning, kLatched, kRetracting, kLanding, kLanded };

struct Output
{
  bool send = false;            // false: send nothing (inactive, or landed and idle)
  bool idle_request = false;    // landed and verified: switch the motors to idle, once
  bool reset_commands = false;  // set the command interfaces to the measured pose (unlatch)
  Vec7 q{}, qd{};               // position target and velocity feed-forward
};

// rust/tatbot-arm/src/contact.rs: 800-sample median rest baseline (not collected while the
// carriage is commanded to move), slow drift below 25% of the cap, effort judged only below
// 0.3 rad/s arm speed, deflection judged always, trip after 40 consecutive ticks.
class ContactCap
{
public:
  void configure(const Config & c);
  bool observe(double effort_n, double position_m, double target_m, double arm_speed, bool aligning);
  bool armed() const {return armed_;}
  double baseline() const {return baseline_;}

private:
  std::vector<double> samples_;
  bool armed_ = false, rebasing_ = false, effort_ = true;
  double baseline_ = 0, cap_ = 20, deflect_ = 0.002, judge_below_ = 0.3;
  double last_target_ = std::numeric_limits<double>::quiet_NaN(), baseline_target_ = 0;
  int need_ = 800, trip_ticks_ = 40, over_ = 0, settle_ticks_ = 120, since_moved_ = 0,
    rebaseline_need_ = 200;
};

class ArmCore
{
public:
  explicit ArmCore(Config config, TipFk fk = {});

  // Enter control at the measured pose: hold it until a first command passes the no-step check.
  // A latch from before a deactivate (other than the deactivate hold, reason 9) is kept. A landed
  // arm is taken back where it rests (woken by a re-activation of its hardware alone).
  void activate(double now, const Feedback & fb);
  // Deactivate, error or shutdown: hold the measured pose (reason 9 or 8), never idle.
  Output hold_now(double now, const Feedback & fb, int reason);
  Output tick(
    double now, const Feedback & fb, const Commands & cmd, const EstopStatus & estop,
    bool controller_error, const ProbeStatus & probe = {});

  // Feed every measurement, from configure on: the first command needs feedback_warmup_s of it.
  void observe_feedback(double now, const Feedback & fb);
  // The measured joints at time t, interpolated from the last kHistory ticks (clamped to their ends).
  Vec7 joints_at(double t) const;
  Phase phase() const {return phase_;}
  // The GPIO state interfaces in names::kSafetyState order (rt_period_max_ms is the adapter's).
  const std::array<double, names::kSafetyState.size()> & safety() const {return safety_;}
  const Vec7 & hold_pose() const {return hold_;}
  const Config & config() const {return cfg_;}
  // One line for the log when something latched, unlatched or landed; "" otherwise.
  std::string take_event();

private:
  enum S : size_t {
    kSource, kOk, kAge, kProbe, kLatched, kReason, kGuard, kTripped, kTripQ0, kUnlatchAck = 15,
    kLandAck, kLanding, kLanded, kCtrlError, kRtPeriod
  };
  void latch(double now, const Feedback & fb, int reason);
  void start_landing(double now, const Feedback & fb);
  bool stalled(double now, const Vec7 & target, const Vec7 & q);
  bool guard_tripped(double now, const Vec7 & target, const Feedback & fb);
  // The armed probe guard: true to stop. `fault` when it cannot protect (no fresh probe, or the probe
  // already triggered when armed); else a touch, with the joints at its edge in q_edge.
  bool probe_stop(double now, const ProbeStatus & probe, Vec7 & q_edge, bool & fault);
  Output track(const Vec7 & q, const Vec7 & qd);
  Output hold() const;
  Output landing(double now, const Feedback & fb);
  Vec7 clamp(Vec7 q) const;
  // Events within one tick join into its one line (a probe stop's edge age, then the latch it caused).
  void set_event(const std::string & text) {event_ = event_.empty() ? text : event_ + "; " + text;}

  Config cfg_;
  TipFk fk_;
  ContactCap cap_;
  Phase phase_ = Phase::kInactive;
  std::array<double, names::kSafetyState.size()> safety_{};
  Vec7 hold_{}, last_target_{};
  bool first_pending_ = true;
  double nonzero_since_ = -1;
  double last_unlatch_ = 0, last_land_ = 0;
  // stall watchdog (rust/tatbot-arm tracking_watchdog.rs)
  double stall_since_ = -1;
  Vec7 stall_anchor_{};
  // touch guard: tip lag and contact force
  int guard_level_ = 0;
  double guard_armed_at_ = -1, lag_since_ = -1, contact_since_ = -1, force_sum_ = 0;
  int force_n_ = 0;
  // probe guard: a touch needs the probe seen at rest since arming
  bool probe_clear_ = false;
  // the measured joints of the last kHistory ticks (2.5 s at 400 Hz), for the probe's edge time
  static constexpr size_t kHistory = 1024;
  std::array<double, kHistory> hist_t_{};
  std::array<Vec7, kHistory> hist_q_{};
  size_t hist_head_ = 0, hist_size_ = 0;
  // carriage retract and landing profiles
  double motion_t0_ = 0;
  Vec7 from_{}, staged_{}, sleep_{};
  bool idle_sent_ = false;
  std::string event_;
};

// The tip force along the tcp +z axis (toward the paper) that the arm joints' external torques
// imply for a point force at the tcp: tau = Jv^T f, so f = (Jv Jv^T)^-1 Jv tau, with Jv the
// translational Jacobian by finite differences of fk. NaN without fk.
double tip_force_along_tool(const TipFk & fk, const Vec7 & q, const Vec7 & tau);

// Minimum-jerk blend from a to b: position and velocity at time t of a move lasting duration.
void min_jerk(const Vec7 & a, const Vec7 & b, double t, double duration, Vec7 & q, Vec7 & qd);

}  // namespace tatbot_hardware
