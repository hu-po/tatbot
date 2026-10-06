// Every README section 8 row, plus deactivate/error hold and the velocity feed-forward, on
// ArmCore driving the fake SDK backend at 400 Hz.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "tatbot_hardware/backend.hpp"
#include "tatbot_hardware/core.hpp"

using namespace tatbot_hardware;

namespace
{
constexpr double kDt = 0.0025;

size_t idx(std::string_view name)
{
  for (size_t i = 0; i < names::kSafetyState.size(); ++i) {
    if (names::kSafetyState[i] == name) {return i;}
  }
  throw std::out_of_range(std::string(name));
}

// Tip z falls as joint_1 rises: 1 mm of tip per 10 mrad; joint_0 moves the tip sideways (y).
// The tool points straight down (tcp +z = -base z), onto a page below.
TipPose toy_fk(const Vec7 & q) {return {{0.3, 0.1 * q[0], 0.2 - 0.1 * q[1]}, {0.0, 0.0, -1.0}};}

struct Rig
{
  Config cfg;
  FakeBackend fake;
  ArmCore core;
  Feedback fb;
  Commands cmd;
  EstopStatus estop;
  ProbeStatus probe;
  Output out;
  double t = 0;

  explicit Rig(Config c = {}, FakeOptions o = {}, TipFk fk = {})
  : cfg(c), fake(o), core(c, std::move(fk))
  {
    fake.connect();
    fb = fake.read();
    core.observe_feedback(t, fb);
  }
  void activate()
  {
    t += 0.2;   // past the feedback warm-up
    fake.set_position_mode();
    fb = fake.read();
    core.activate(t, fb);
    cmd.q = fb.q;
    cmd.qd = Vec7{};
  }
  void step(int n = 1)
  {
    for (int i = 0; i < n; ++i) {
      t += kDt;
      fb = fake.read();
      out = core.tick(t, fb, cmd, estop, !fake.error().empty(), probe);
      if (out.reset_commands) {cmd.q = fb.q; cmd.qd = Vec7{};}
      if (out.send) {fake.command(out.q, out.qd);}
      if (out.idle_request) {fake.set_idle();}
    }
  }
  double s(std::string_view name) const {return core.safety()[idx(name)];}
  void press() {estop.ok = false; estop.pressed = true; estop.source = names::kEstopSerial; estop.age_s = 0.01;}
  void release() {estop.ok = true; estop.pressed = false;}
  void unlatch(double id) {cmd.q = fb.q; cmd.qd = Vec7{}; cmd.unlatch = id; step();}
};

void expect_near(const Vec7 & a, const Vec7 & b, double tol)
{
  for (size_t i = 0; i < kJoints; ++i) {EXPECT_NEAR(a[i], b[i], tol) << "joint " << i;}
}
}  // namespace

TEST(HwCore, FirstCommandAtMeasuredIsTrackedWithVelocityFeedForward)
{
  Rig r;
  r.activate();
  r.step(5);
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.qd[0] = 0.1;
  r.cmd.qd[kCarriage] = 0.002;
  for (int i = 0; i < 400; ++i) {
    r.cmd.q[0] += 0.1 * kDt;
    r.cmd.q[kCarriage] += 0.002 * kDt;
    r.step();
  }
  // The JTC velocity reaches the SDK unchanged as feed-forward.
  EXPECT_DOUBLE_EQ(r.fake.last_qd()[0], 0.1);
  EXPECT_DOUBLE_EQ(r.fake.last_qd()[kCarriage], 0.002);
  EXPECT_NEAR(r.fb.q[0], r.cmd.q[0], 1e-3);
  EXPECT_EQ(r.s("latched"), 0);
}

TEST(HwCore, ActivationRefusesAnEncoderTurnOutsideLimitsBeforeChangingPhase)
{
  Config c;
  c.lower[5] = -M_PI;
  c.upper[5] = M_PI;
  ArmCore core(c);
  Feedback fb;
  fb.q[5] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(core.activate(0.2, fb), std::runtime_error);
  EXPECT_EQ(core.phase(), Phase::kInactive);
  fb.q[5] = -4.615;
  EXPECT_THROW(core.activate(0.2, fb), std::runtime_error);
  EXPECT_EQ(core.phase(), Phase::kInactive);
  fb.q[5] += 2 * M_PI;
  EXPECT_NO_THROW(core.activate(0.2, fb));
  EXPECT_EQ(core.phase(), Phase::kRunning);
  EXPECT_DOUBLE_EQ(core.hold_pose()[5], fb.q[5]);
}

TEST(HwCore, FirstCommandSteppingAwayIsRefusedAndHeld)
{
  Rig r;
  r.activate();
  const Vec7 measured = r.fb.q;
  r.cmd.q[2] += 0.06;   // > 0.05 rad
  r.step(10);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchStepRefused);
  expect_near(r.out.q, measured, 1e-9);
  expect_near(r.fb.q, measured, 1e-6);
}

TEST(HwCore, FirstCommandBeforeFeedbackWarmupIsRefused)
{
  FakeOptions o;
  o.zero_reads = 1;   // the SDK's all-zero default before its first robot output
  Rig r({}, o);
  r.fake.set_position_mode();
  r.t += 0.05;
  r.fb = r.fake.read();          // first non-zero feedback at t = 0.05
  r.core.observe_feedback(r.t, r.fb);
  r.core.activate(r.t, r.fb);
  r.cmd.q = r.fb.q;
  r.step();                      // 2.5 ms of feedback < 0.10 s
  EXPECT_EQ(r.s("latch_reason"), names::kLatchStepRefused);
}

TEST(HwCore, EstopPressHoldsThePoseAtTheLatchInstantAndIgnoresCommands)
{
  Rig r;
  r.activate();
  r.cmd.qd[0] = 0.2;
  for (int i = 0; i < 200; ++i) {r.cmd.q[0] += 0.2 * kDt; r.step();}
  r.press();
  r.t += kDt;
  r.fb = r.fake.read();
  const Vec7 at_latch = r.fb.q;
  r.out = r.core.tick(r.t, r.fb, r.cmd, r.estop, false);
  r.fake.command(r.out.q, r.out.qd);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);
  EXPECT_EQ(r.s("estop_ok"), 0);
  expect_near(r.out.q, at_latch, 1e-12);
  for (double v : r.out.qd) {EXPECT_EQ(v, 0.0);}
  for (int i = 0; i < 400; ++i) {r.cmd.q[0] += 0.2 * kDt; r.step();}
  expect_near(r.out.q, at_latch, 1e-12);   // commands ignored
  EXPECT_FALSE(r.fake.idle());             // motors powered, holding
  // Release alone does not unlatch.
  r.release();
  r.step(100);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("estop_ok"), 1);
}

TEST(HwCore, StaleHeartbeatHoldsWithItsOwnReason)
{
  Rig r;
  r.activate();
  r.estop.source = names::kEstopUdp;
  r.estop.ok = false;
  r.estop.age_s = 0.2;
  r.step();
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstopStale);
  EXPECT_EQ(r.s("estop_source"), names::kEstopUdp);
}

TEST(HwCore, UnlatchFollowsTheNoStepProtocol)
{
  Rig r;
  r.activate();
  r.press();
  r.step(4);
  // Still pressed: acked, stays latched.
  r.unlatch(1);
  EXPECT_EQ(r.s("unlatch_ack"), 1);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);
  r.release();
  r.step(4);
  // A command that steps away: acked, stays latched with reason 10.
  r.cmd.q = r.core.hold_pose();
  r.cmd.q[3] += 0.08;
  r.cmd.unlatch = 2;
  r.step();
  EXPECT_EQ(r.s("unlatch_ack"), 2);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchStepRefused);
  // The carriage has its own 5 mm limit.
  r.cmd.q = r.core.hold_pose();
  r.cmd.q[kCarriage] += 0.006;
  r.cmd.unlatch = 3;
  r.step();
  EXPECT_EQ(r.s("latched"), 1);
  // The same id again is not a new request.
  r.cmd.q = r.core.hold_pose();
  r.step();
  EXPECT_EQ(r.s("latched"), 1);
  // A hold command at the latched pose is accepted, and commands are tracked again.
  r.unlatch(4);
  EXPECT_EQ(r.s("unlatch_ack"), 4);
  EXPECT_EQ(r.s("latched"), 0);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchNone);
  r.cmd.q[0] += 0.01;
  r.step(200);
  EXPECT_NEAR(r.fb.q[0], r.cmd.q[0], 1e-4);
}

TEST(HwCore, CarriageContactOverCapRetractsThenHolds)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, 3.0);
  r.step(800);   // the rest baseline
  const Vec7 arm = r.fb.q;
  r.fake.set_effort(kCarriage, 24.0);   // 21 N over the baseline
  r.step(39);
  EXPECT_EQ(r.s("latched"), 0);
  r.step();
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
  EXPECT_EQ(r.core.phase(), Phase::kRetracting);
  r.step(static_cast<int>((0.6 + 0.2) / kDt));
  EXPECT_EQ(r.core.phase(), Phase::kLatched);
  EXPECT_NEAR(r.fb.q[kCarriage], 0.032, 5e-4);
  for (size_t i = 0; i < kCarriage; ++i) {EXPECT_NEAR(r.fb.q[i], arm[i], 1e-6);}
  // Commands stay ignored after the retract.
  r.cmd.q[kCarriage] = 0.0;
  r.step(40);
  EXPECT_NEAR(r.out.q[kCarriage], 0.032, 1e-9);
}

TEST(HwCore, CarriageEffortIsNotJudgedWhileTheArmMovesFast)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, 3.0);
  r.step(800);
  r.fake.set_velocity_offset(1, 0.5);   // > 0.3 rad/s
  r.fake.set_effort(kCarriage, 40.0);
  r.step(200);
  EXPECT_EQ(r.s("latched"), 0);
}

TEST(HwCore, HoldingTheTravelBiasIsNotContactButContactThereIs)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, -13.0);   // a carriage resting on its floor
  r.step(800);                           // the rest baseline, arm still
  r.fake.set_effort(kCarriage, 15.0);    // the lift's effort
  for (int i = 0; i < 400; ++i) {r.cmd.q[kCarriage] += 0.002 * kDt; r.step();}   // 2 mm at 2 mm/s
  r.fake.set_effort(kCarriage, 14.0);    // holding at the bias: 27 N off the floor baseline, steady
  r.step(2000);
  EXPECT_EQ(r.s("latched"), 0);
  r.fake.set_effort(kCarriage, 39.0);    // 25 N over the hold with the target still: contact
  r.step(39);
  EXPECT_EQ(r.s("latched"), 0);
  r.step();
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, TheNewBaselineIsTakenOnlyOnceTheTargetHasSettled)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, -13.0);
  r.step(800);
  r.fake.set_effort(kCarriage, 14.0);
  for (int i = 0; i < 400; ++i) {r.cmd.q[kCarriage] += 0.002 * kDt; r.step();}
  r.step(120 + 199);                     // settle, then one sample short of the new median
  r.fake.set_effort(kCarriage, 60.0);
  r.step(60);                            // its last sample is 60 N; the median is still the hold
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, ALiftsSlowQuinticStartIsAMoveNotContact)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, -5.6);    // resting under the travel bias, arm still
  r.step(800);
  const double q0 = r.cmd.q[kCarriage];
  const int n = 800;                     // 2 mm in 2 s, quintic: the first ~120 ticks move under 1 um each
  for (int i = 1; i <= n; ++i) {
    const double s = static_cast<double>(i) / n;
    r.cmd.q[kCarriage] = q0 + 0.002 * s * s * s * (10 - 15 * s + 6 * s * s);
    if (i == 60) {r.fake.set_effort(kCarriage, 18.7);}   // the lift's effort, 24 N over the rest
    r.step();
  }
  r.fake.set_effort(kCarriage, 14.0);    // holding the bias
  r.step(2000);
  EXPECT_EQ(r.s("latched"), 0);
  r.fake.set_effort(kCarriage, 39.0);    // 25 N over the hold with the target still: contact
  r.step(40);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, ASubMicronTargetSpliceDoesNotRetakeTheBaseline)
{
  Rig r;
  r.activate();
  r.fake.set_effort(kCarriage, 3.0);
  r.step(800);
  for (int i = 0; i < 5; ++i) {r.cmd.q[kCarriage] += 1e-7; r.step();}   // a stroke-start splice: 0.5 um
  r.fake.set_effort(kCarriage, 24.0);    // pressed 21 N over the baseline from before the splice
  r.step(120 + 40);                      // the settle, then the trip ticks
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, CarriageDeflectionTripsDuringACommandedMove)
{
  Rig r;
  r.activate();
  r.cmd.q[kCarriage] = 0.003;
  r.step(200);                  // the carriage opens to 3 mm
  r.fake.set_frozen(true);      // pushed and held there while the target keeps closing it
  int ticks = 0;
  while (r.s("latched") == 0 && ticks < 800) {r.cmd.q[kCarriage] -= 0.005 * kDt; r.step(); ++ticks;}
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_NEAR(ticks, (0.002 / 0.005) / kDt + 40, 3);   // 2 mm behind, then the 40 trip ticks
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, CarriageDeflectionTripsWithoutABaseline)
{
  Rig r;
  r.activate();
  r.cmd.q[kCarriage] = 0.003;
  r.step(200);                  // the carriage opens to 3 mm
  r.fake.set_frozen(true);      // pushed and held there
  r.cmd.q[kCarriage] = 0.0;     // 3 mm of deflection > 2 mm
  r.step(39);
  EXPECT_EQ(r.s("latched"), 0);
  r.step();
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
}

TEST(HwCore, TrackingStallHolds)
{
  Rig r;
  r.activate();
  r.step();
  r.fake.set_frozen(true);
  r.cmd.q[1] += 0.3;            // under 0.35 rad: never a stall
  r.step(static_cast<int>(3.0 / kDt));
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.q[1] += 0.1;            // 0.4 rad of error, no progress
  r.step(static_cast<int>(1.9 / kDt));
  EXPECT_EQ(r.s("latched"), 0);
  r.step(static_cast<int>(0.2 / kDt));
  EXPECT_EQ(r.s("latch_reason"), names::kLatchStall);
}

TEST(HwCore, MeasuredOverVelocityHolds)
{
  Rig r;
  r.activate();
  r.step(4);
  r.fake.set_velocity_offset(4, 3.1);
  r.step();
  EXPECT_EQ(r.s("latch_reason"), names::kLatchOverVelocity);
  Rig c;
  c.activate();
  c.step(4);
  c.fake.set_velocity_offset(kCarriage, 0.26);
  c.step();
  EXPECT_EQ(c.s("latch_reason"), names::kLatchOverVelocity);
}

TEST(HwCore, TipLagGuardTripsOnThePageAndReportsTheTripPose)
{
  FakeOptions o;
  o.fk = toy_fk;
  o.page_z = 0.195;   // 5 mm below the start tip at z = 0.2
  Rig r({}, o, toy_fk);
  r.activate();
  r.cmd.guard_mode = names::kGuardTipLag;
  // Descend at 5 mm/s (50 mrad/s on joint_1).
  bool tripped = false;
  double trip_t = 0;
  for (int i = 0; i < static_cast<int>(3.0 / kDt) && !tripped; ++i) {
    r.cmd.q[1] += 0.05 * kDt;
    r.cmd.qd[1] = 0.05;
    r.step();
    tripped = r.s("latched") == 1;
    trip_t = r.t;
  }
  ASSERT_TRUE(tripped);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardTipLag);
  EXPECT_EQ(r.s("guard_tripped"), 1);
  EXPECT_EQ(r.s("guard_mode"), names::kGuardTipLag);
  // Contact at ~1.0 s (5 mm at 5 mm/s); 0.8 mm of lag takes 0.16 s, then 0.15 s held.
  EXPECT_GT(trip_t, 0.2 + 1.2);
  EXPECT_LT(trip_t, 0.2 + 1.5);
  EXPECT_NEAR(r.s("trip_q1"), r.fb.q[1], 1e-6);
  EXPECT_NEAR(toy_fk(r.fb.q).p[2], 0.195, 1e-4);
  expect_near(r.out.q, r.core.hold_pose(), 0);
  // The session disarms, unlatches at the measured pose, lifts.
  r.cmd.guard_mode = 0;
  r.cmd.qd = Vec7{};
  r.unlatch(1);
  EXPECT_EQ(r.s("latched"), 0);
  EXPECT_EQ(r.s("guard_tripped"), 0);
  EXPECT_FALSE(std::isnan(r.s("trip_q0")));
}

TEST(HwCore, TipLagGuardWaitsForItsArmingTime)
{
  FakeOptions o;
  o.fk = toy_fk;
  o.page_z = 0.2;     // already on the page
  Rig r({}, o, toy_fk);
  r.activate();
  r.cmd.q[1] += 0.02;   // 2 mm of lag at once
  r.step();
  r.cmd.guard_mode = names::kGuardTipLag;
  r.step(static_cast<int>(0.49 / kDt));
  EXPECT_EQ(r.s("latched"), 0);
  r.step(static_cast<int>(0.2 / kDt));
  EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardTipLag);
}

TEST(HwCore, TipLagGuardIgnoresLateralAndUpwardError)
{
  // Held off the paper by more than the trip distance, but not along the tool toward the paper:
  // a frozen arm with the command 2 mm to the side, then 2 mm above the measured tip.
  FakeOptions o;
  o.fk = toy_fk;
  Rig r({}, o, toy_fk);
  r.activate();
  r.cmd.guard_mode = names::kGuardTipLag;
  r.step(static_cast<int>(0.6 / kDt));   // past the arming time
  r.fake.set_frozen(true);
  r.cmd.q[0] += 0.02;                    // 2 mm sideways
  r.step(static_cast<int>(0.4 / kDt));
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.q[0] -= 0.02;
  r.cmd.q[1] -= 0.02;                    // 2 mm up, away from the paper
  r.step(static_cast<int>(0.4 / kDt));
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.q[1] += 0.04;                    // 2 mm down, into the paper: trips
  r.step(static_cast<int>(0.4 / kDt));
  EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardTipLag);
}

TEST(HwCore, DeactivateAndErrorHoldNeverIdle)
{
  Rig r;
  r.activate();
  r.cmd.q[0] += 0.01;
  r.step(40);
  const Output held = r.core.hold_now(r.t, r.fb, names::kLatchDeactivated);
  EXPECT_TRUE(held.send);
  EXPECT_FALSE(held.idle_request);
  expect_near(held.q, r.fb.q, 1e-12);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchDeactivated);
  EXPECT_FALSE(r.fake.idle());

  Rig e;
  e.activate();
  e.step(4);
  e.fake.inject_error("joint 3 overcurrent");
  e.step();
  EXPECT_EQ(e.s("latch_reason"), names::kLatchControllerError);
  EXPECT_EQ(e.s("controller_error"), 1);
  e.unlatch(1);   // refused while the controller reports the error
  EXPECT_EQ(e.s("latched"), 1);
  EXPECT_FALSE(e.fake.idle());
}

TEST(HwCore, ReactivationKeepsALatchFromBeforeTheDeactivate)
{
  Rig r;
  r.activate();
  r.step(4);
  r.press();
  r.step();
  r.release();
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);
  (void)r.core.hold_now(r.t, r.fb, names::kLatchDeactivated);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);   // first reason kept
  r.activate();   // deactivate -> activate never unlatches
  r.step(10);
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);
  EXPECT_EQ(r.core.phase(), Phase::kLatched);
  r.unlatch(1);   // a Decide does
  EXPECT_EQ(r.s("latched"), 0);

  // A latch that is only the deactivate hold clears on re-activation.
  Rig d;
  d.activate();
  d.step(4);
  (void)d.core.hold_now(d.t, d.fb, names::kLatchDeactivated);
  EXPECT_EQ(d.s("latched"), 1);
  d.activate();
  d.step(10);
  EXPECT_EQ(d.s("latched"), 0);
  EXPECT_EQ(d.core.phase(), Phase::kRunning);
}

TEST(HwCore, LandingRunsStagedSleepVerifyThenIdles)
{
  Rig r;
  r.activate();
  for (int i = 0; i < 400; ++i) {r.cmd.q[1] += 0.25 * kDt; r.cmd.q[kCarriage] += 0.01 * kDt; r.step();}
  r.press();
  r.step();
  r.release();
  r.step();
  r.cmd.land = 1;
  r.step();
  EXPECT_EQ(r.s("land_ack"), 1);
  EXPECT_EQ(r.s("landing"), 1);
  EXPECT_EQ(r.s("latched"), 0);
  r.step(static_cast<int>(7.0 / kDt));
  EXPECT_EQ(r.s("landed"), 0);
  r.step(static_cast<int>(1.0 / kDt));
  EXPECT_EQ(r.s("landed"), 1);
  EXPECT_EQ(r.s("landing"), 0);
  EXPECT_TRUE(r.fake.idle());
  Vec7 sleep{};
  sleep[kWristRoll] = 1.5707963267948966;
  expect_near(r.fb.q, sleep, 1e-3);
  // Landed: idle until it is woken; commands and unlatch do nothing.
  const int sent = r.fake.commands();
  r.unlatch(9);
  r.step(10);
  EXPECT_EQ(r.fake.commands(), sent);
  EXPECT_EQ(r.s("unlatch_ack"), 9);
}

TEST(HwCore, ALandedArmWokenByAReactivationHoldsItsRestUntilACommandStartsThere)
{
  Rig r;
  r.activate();
  r.cmd.land = 1;
  r.step(static_cast<int>(8.5 / kDt));
  ASSERT_EQ(r.s("landed"), 1);
  ASSERT_TRUE(r.fake.idle());
  const Vec7 rest = r.fb.q;
  // The stack's own restart path, for one arm: its hardware deactivated (a landed arm sends nothing)
  // and activated again.
  r.core.hold_now(r.t, r.fb, names::kLatchDeactivated);
  EXPECT_EQ(r.core.phase(), Phase::kLanded);
  Vec7 stale = rest;
  stale[1] += 0.3;             // the trajectory controller's command from before the landing
  r.activate();
  r.cmd.q = stale;
  EXPECT_EQ(r.core.phase(), Phase::kRunning);
  EXPECT_EQ(r.s("landed"), 0);
  EXPECT_EQ(r.s("latched"), 0);
  EXPECT_FALSE(r.fake.idle());
  r.step();
  EXPECT_EQ(r.s("latched"), 1);           // a first command away from the rest is refused and held
  expect_near(r.core.hold_pose(), rest, 1e-6);
  r.unlatch(2);                            // from the rest it tracks again
  r.cmd.q[1] += 0.001;
  r.step(5);
  EXPECT_EQ(r.s("latched"), 0);
  EXPECT_EQ(r.core.phase(), Phase::kRunning);
  r.cmd.land = 3;                          // and lands again
  r.step(static_cast<int>(8.5 / kDt));
  EXPECT_EQ(r.s("landed"), 1);
  EXPECT_TRUE(r.fake.idle());
}

TEST(HwCore, EstopDuringLandingHolds)
{
  Rig r;
  r.activate();
  r.cmd.land = 1;
  r.step(static_cast<int>(2.0 / kDt));
  EXPECT_EQ(r.s("landing"), 1);
  r.press();
  r.step();
  EXPECT_EQ(r.s("landing"), 0);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchEstop);
  const Vec7 held = r.out.q;
  r.step(400);
  expect_near(r.out.q, held, 1e-12);
  EXPECT_FALSE(r.fake.idle());
  // Pressed: a land request is acked, refused, and the log says why.
  (void)r.core.take_event();
  r.cmd.land = 2;
  r.step();
  EXPECT_EQ(r.s("land_ack"), 2);
  EXPECT_EQ(r.s("landing"), 0);
  EXPECT_EQ(r.core.take_event(), "land refused: e-stop not released");
}

TEST(HwCore, JointLimitsClampEverySentPosition)
{
  Config c;
  c.lower.fill(-2.0);
  c.upper.fill(2.0);
  c.lower[kCarriage] = -0.006;
  c.upper[kCarriage] = 0.040;
  Rig r(c);
  r.activate();
  r.cmd.q[0] = 0.04;
  r.step();
  r.cmd.q[kCarriage] = 0.05;
  r.step();
  EXPECT_LE(r.out.q[kCarriage], 0.040);
}

TEST(HwCore, TickCost)
{
  Rig r({}, {}, toy_fk);
  r.activate();
  r.cmd.guard_mode = names::kGuardTipLag;
  const int n = 40000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {r.step();}
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  std::cout << "core tick + fake backend: " << us << " us per tick" << std::endl;
  EXPECT_LT(us, 250.0);
}

TEST(HwCore, TipForceAlongToolRecoversAPointForce)
{
  // A linear arm: tip = A q (joints 0..5), tool along (0, 0.6, -0.8). tau = A^T f recovers f.
  const double a[3][6] = {{0.3, -0.1, 0.05, 0.0, 0.02, 0.0}, {0.1, 0.25, 0.0, 0.04, 0.0, 0.01},
    {0.0, -0.2, -0.15, 0.0, 0.03, 0.0}};
  const TipFk fk = [&](const Vec7 & q) {
      TipPose p;
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 6; ++j) {p.p[i] += a[i][j] * q[j];}
      }
      p.z = {0.0, 0.6, -0.8};
      return p;
    };
  const double f[3] = {1.0, -2.0, 3.0};
  Vec7 tau{};
  for (size_t j = 0; j < 6; ++j) {
    for (size_t i = 0; i < 3; ++i) {tau[j] += a[i][j] * f[i];}
  }
  EXPECT_NEAR(tip_force_along_tool(fk, Vec7{0.1, 0.2, 0.3, 0, 0, 0, 0}, tau), -2.0 * 0.6 - 3.0 * 0.8, 1e-6);
  EXPECT_TRUE(std::isnan(tip_force_along_tool({}, Vec7{}, tau)));
}

TEST(HwCore, ContactGuardTripsOnFirstForceLongBeforeTheTipLags)
{
  // A sprung page 5 mm below the tip, 1.5 N/mm: the tip passes, the joints feel the force. At
  // 5 mm/s 2.5 N comes 1.67 mm past first contact, ~0.33 s after it, well before any tip lag.
  FakeOptions o;
  o.fk = toy_fk;
  o.page_z = 0.195;
  o.page_stiffness_n_m = 1500;
  Rig r({}, o, toy_fk);
  r.activate();
  r.cmd.guard_mode = names::kGuardTipLag;
  bool tripped = false;
  for (int i = 0; i < static_cast<int>(3.0 / kDt) && !tripped; ++i) {
    r.cmd.q[1] += 0.05 * kDt;
    r.cmd.qd[1] = 0.05;
    r.step();
    tripped = r.s("latched") == 1;
  }
  ASSERT_TRUE(tripped);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardTipLag);
  const double depth = 0.195 - toy_fk(r.fb.q).p[2];
  EXPECT_GT(depth, 0.0015);
  EXPECT_LT(depth, 0.0025);
  EXPECT_NEAR(toy_fk(r.core.hold_pose()).p[2], toy_fk(r.fb.q).p[2], 1e-4);
}

TEST(HwCore, ContactGuardJudgesTheRiseOverItsArmingBaseline)
{
  // A constant 6 N bias along the tool (gravity-compensation error) is the baseline; a 2 N rise
  // stays under the 2.5 N trip; a 3 N rise held 0.05 s trips. Contact force 0 turns it off.
  for (const double trip_n : {2.5, 0.0}) {
    Config c;
    c.contact_force_n = trip_n;
    FakeOptions o;
    o.fk = toy_fk;
    Rig r(c, o, toy_fk);
    r.activate();
    // toy_fk: tcp +z = -base z and tip z = 0.2 - 0.1 q1, so tau_1 = 0.1 * F along the tool.
    r.fake.set_effort(1, 0.1 * 6.0);
    r.cmd.guard_mode = names::kGuardTipLag;
    r.step(static_cast<int>(1.0 / kDt));
    EXPECT_EQ(r.s("latched"), 0);
    r.fake.set_effort(1, 0.1 * 8.0);
    r.step(static_cast<int>(0.5 / kDt));
    EXPECT_EQ(r.s("latched"), 0);
    r.fake.set_effort(1, 0.1 * 9.0);
    r.step(static_cast<int>(0.04 / kDt));
    EXPECT_EQ(r.s("latched"), 0);
    r.step(static_cast<int>(0.03 / kDt));
    EXPECT_EQ(r.s("latched"), trip_n > 0 ? 1 : 0) << "contact_force_n " << trip_n;
  }
}

// --- the station probe ------------------------------------------------------------------------------
namespace
{
ProbeStatus probe_at_rest()
{
  ProbeStatus p;
  p.enabled = true;
  p.fresh = true;
  return p;
}
}  // namespace

TEST(HwCore, ProbeGuardLatchesTheJointsOfTheKernelEdgeNotOfTheLaterTick)
{
  Rig r;
  r.activate();
  r.probe = probe_at_rest();
  r.cmd.guard_mode = names::kGuardProbe;
  std::vector<std::pair<double, double>> q1;   // (t, measured joint_1)
  for (int i = 0; i < 200; ++i) {   // a slow approach: 50 mrad/s on joint_1
    r.cmd.q[1] += 0.05 * kDt;
    r.cmd.qd[1] = 0.05;
    r.step();
    q1.emplace_back(r.t, r.fb.q[1]);
    ASSERT_EQ(r.s("latched"), 0);
  }
  // The frame reports a rising edge 12 ms before this tick (relay stamp, network and a tick of delay).
  r.probe.triggered = true;
  r.probe.rise_age_s = 0.012;
  r.cmd.q[1] += 0.05 * kDt;
  r.step();
  EXPECT_EQ(r.s("latched"), 1);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardProbe);
  EXPECT_EQ(r.s("guard_tripped"), 1);
  EXPECT_EQ(r.s("probe_triggered"), 1);
  // joint_1 as measured at t - 12 ms, interpolated between the ticks around it.
  const double edge = r.t - 0.012;
  double expected = NAN;
  for (size_t k = 1; k < q1.size(); ++k) {
    if (q1[k - 1].first <= edge && edge <= q1[k].first) {
      const double s = (edge - q1[k - 1].first) / (q1[k].first - q1[k - 1].first);
      expected = q1[k - 1].second + (q1[k].second - q1[k - 1].second) * s;
    }
  }
  ASSERT_FALSE(std::isnan(expected));
  EXPECT_NEAR(r.s("trip_q1"), expected, 1e-9);
  EXPECT_GT(r.fb.q[1] - r.s("trip_q1"), 0.0003);   // the arm moved on after the edge
  expect_near(r.out.q, r.core.hold_pose(), 0);
  const std::string event = r.core.take_event();   // the edge's age survives the latch line
  EXPECT_NE(event.find("probe guard: touch 12 ms before this tick"), std::string::npos) << event;
  EXPECT_NE(event.find("latched (reason 7)"), std::string::npos) << event;
}

TEST(HwCore, ProbeGuardHoldsWhenItCannotSeeTheProbe)
{
  for (int kind = 0; kind < 3; ++kind) {
    Rig r;
    r.activate();
    r.probe = probe_at_rest();
    if (kind == 0) {r.probe.fresh = false;}     // no frame within its timeout
    if (kind == 1) {r.probe = ProbeStatus{};}   // this stack reads no probe
    if (kind == 2) {r.probe.triggered = true;}  // triggered when armed: still touching, or a broken wire
    r.cmd.guard_mode = names::kGuardProbe;
    r.step();
    EXPECT_EQ(r.s("latched"), 1) << kind;
    EXPECT_EQ(r.s("latch_reason"), names::kLatchGuardProbe) << kind;
    EXPECT_EQ(r.s("guard_tripped"), 0) << "a fault is a hold, not a touch: " << kind;
    EXPECT_TRUE(std::isnan(r.s("trip_q0"))) << kind;
  }
}

TEST(HwCore, ProbeStateIsReportedAndStopsNothingWithoutItsGuard)
{
  Rig r;
  r.activate();
  r.probe = probe_at_rest();
  r.probe.triggered = true;
  r.cmd.q[1] += 0.001;
  r.step(40);
  EXPECT_EQ(r.s("probe_triggered"), 1);
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.guard_mode = names::kGuardTipLag;   // the page touch's guard ignores the probe
  r.step(40);
  EXPECT_EQ(r.s("latched"), 0);
}

TEST(HwCore, AProbeStampFromBeforeTheArmingIsNotThisTouchs)
{
  Rig r;
  r.activate();
  r.probe = probe_at_rest();
  r.cmd.guard_mode = names::kGuardProbe;
  for (int i = 0; i < 40; ++i) {r.cmd.q[1] += 0.05 * kDt; r.step();}
  r.probe.triggered = true;
  r.probe.rise_age_s = 5.0;   // older than the arming 0.1 s ago: this tick's joints stand
  r.cmd.q[1] += 0.05 * kDt;
  r.step();
  EXPECT_EQ(r.s("guard_tripped"), 1);
  EXPECT_NEAR(r.s("trip_q1"), r.fb.q[1], 1e-12);
}

TEST(HwCore, JointsAtInterpolatesAndClampsToItsHistory)
{
  Rig r;
  r.activate();
  EXPECT_TRUE(std::isnan(ArmCore(Config{}).joints_at(1.0)[0]));
  const double t0 = r.t;
  for (int i = 0; i < 100; ++i) {r.cmd.q[0] += 0.001; r.step();}
  const double mid = (r.t + t0) / 2;
  const Vec7 a = r.core.joints_at(mid), late = r.core.joints_at(r.t + 10), early = r.core.joints_at(t0 - 10);
  EXPECT_NEAR(late[0], r.fb.q[0], 1e-12);
  EXPECT_LT(early[0], a[0]);
  EXPECT_LT(a[0], late[0]);
}

TEST(HwCore, AnUnqualifiedCarriageIsScreenedByDeflectionAloneAndNeverRetracts)
{
  Config c;
  c.carriage_qualified = false;   // the leader's carriage (config/trossen/tatbot.yaml leader)
  Rig r(c);
  r.activate();
  r.fake.set_effort(kCarriage, 3.0);
  r.step(800);
  r.fake.set_effort(kCarriage, 40.0);   // posture swings its effort: not judged
  r.step(200);
  EXPECT_EQ(r.s("latched"), 0);
  r.cmd.q[kCarriage] = 0.003;
  r.step(200);
  r.fake.set_frozen(true);
  r.cmd.q[kCarriage] = 0.0;             // 3 mm of deflection
  r.step(40);
  EXPECT_EQ(r.s("latch_reason"), names::kLatchCarriageContact);
  EXPECT_EQ(r.core.phase(), Phase::kLatched);   // held where it is: no retract
  EXPECT_NEAR(r.out.q[kCarriage], r.core.hold_pose()[kCarriage], 0);
  EXPECT_LT(r.out.q[kCarriage], 0.01);
}
