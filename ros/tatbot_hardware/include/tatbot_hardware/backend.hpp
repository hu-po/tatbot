#pragma once
// What the adapter needs from an arm: the Trossen SDK (sdk=real), the built-in fake (sdk=fake) or, where it is
// built, the simulator's lockstep backend (sdk=sim).
// read() and command() run on the control thread (UDP on the SDK); set_position_mode(),
// disconnect() and error() are TCP round trips (~1.5 ms) and run only on the 10 Hz aux thread or
// during lifecycle transitions, serialized by the adapter. connect() may follow disconnect().
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "tatbot_hardware/core.hpp"

namespace tatbot_hardware
{

class Backend
{
public:
  virtual ~Backend() = default;
  virtual void connect() = 0;                                // may throw; on the real-time helper thread
  virtual Feedback read() = 0;
  virtual void command(const Vec7 & q, const Vec7 & qd) = 0;  // position mode, velocity feed-forward
  virtual void set_position_mode() = 0;
  virtual void disconnect() = 0;                             // close the session: the controller idles
  virtual std::string error() = 0;                           // "" when healthy or not connected
  // Set the controller's carriage position limits (volatile, reset on power-up) and return the
  // limits it held before; the fake keeps no limits.
  virtual std::pair<double, double> set_carriage_limits(double, double) {return {0.0, 0.0};}
  // What connect() set up (model, palm, gains, carriage offset), for the log; empty for the fake.
  virtual std::string describe() const {return "";}
  // What the controller holds after connect(), read back as one JSON object (versions, gains per mode, joint
  // characteristics, joint limits) for the run; empty for the fake. A TCP round trip per getter.
  virtual std::string snapshot() {return "";}
  // Once per control cycle, at the end of write(): the trajectory controller's command, what the core sent and its
  // state. A simulator steps its world here; the fake and the SDK do nothing.
  virtual void tick(const Commands &, const Output &, Phase, bool /*latched*/) {}
  // The backend's own clock (s), when it owns time (a simulator); NaN: the adapter's steady clock.
  virtual double clock_s() const {return std::numeric_limits<double>::quiet_NaN();}
};

constexpr std::array<double, 6> nan6()
{
  constexpr double n = std::numeric_limits<double>::quiet_NaN();
  return {n, n, n, n, n, n};
}

// The SDK session's model and gains, from the hardware parameters (tatbot_description.hardware_params): the
// arm's dated Trossen end effector (config/arms.json; the undated names moved to newer models in driver 1.11),
// the palm identified with its fitted tool (config/workspace.yaml <arm>.payload; NaN mass keeps the preset's),
// Trossen's named position-mode gains (config/trossen/tatbot.yaml; the firmware boots with others) with the arm
// joints' idle velocity integrals scaled (the idle a lost connection leaves the arm in), and the carriage's position
// offset from the arm's golden (NaN leaves the controller's).
struct SdkOptions
{
  std::string end_effector = "wxai_v0_follower_20250509";
  double palm_mass_kg = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 3> palm_com_m{};
  std::string motor_parameters = "wxai_v0_20250509";
  std::array<double, 6> idle_velocity_ki_scale{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double carriage_position_offset_m = std::numeric_limits<double>::quiet_NaN();
  // Position-mode tuning of the arm joints (config/trossen/tatbot.yaml ros_tuning): factors on the named set's
  // position kp and velocity kp/ki, and friction terms written into the controller's joint characteristics
  // (persistent) when they differ; NaN keeps the controller's own.
  std::array<double, 6> position_kp_scale{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::array<double, 6> velocity_kp_scale{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::array<double, 6> velocity_ki_scale{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::array<double, 6> friction_transition_velocity{nan6()};
  std::array<double, 6> friction_constant_term{nan6()};
};

// nullptr when built without the SDK (TATBOT_ARM_SDK off or no fetch possible). Throws on an unknown preset or gain
// set name.
std::unique_ptr<Backend> make_sdk_backend(const std::string & ip, const SdkOptions & options);

// The simulator's lockstep backend (sim_wire.hpp): its server's endpoint (tcp://host:port), the arm, and the wall
// times that bound it: keepalive_s without a tick before it steps on its own, reply_timeout_s for one STATE, and
// connect_timeout_s for the server to accept. loop_now_ns, when set, reads the control loop's own clock (the
// controller manager's): a step returns once that clock has the time it published, so the loop's sleep until
// the next cycle never has to be woken by /clock.
struct SimOptions
{
  std::string endpoint, arm;
  double keepalive_s = 0.1, reply_timeout_s = 30, connect_timeout_s = 30;
  std::function<int64_t()> loop_now_ns;
};

// nullptr when built without the sim backend (src/sim_backend.cpp absent). Throws when another sim arm in this
// process already owns the clock.
std::unique_ptr<Backend> make_sim_backend(const SimOptions & options);

struct FakeOptions
{
  double dt = 0.0025;           // one command() advances the model by dt (the 400 Hz tick)
  double tau = 0.02;            // first-order tracking time constant
  Vec7 start{0, 0, 0, 0, 0, 1.5707963267948966, 0};
  int zero_reads = 0;           // report an all-zero pose for this many reads (SDK start-up)
  TipFk fk;                     // with page_z: the tip may not go below page_z (base frame)
  double page_z = -std::numeric_limits<double>::infinity();
  // > 0: the page gives like the arm and EE mount under load (~1500 N/m) instead of stopping the tip: the tip passes
  // below page_z and the joints report the external torques J^T f of f = k * depth along tcp +z.
  double page_stiffness_n_m = 0;
};

// First-order tracking of the position target plus the velocity feed-forward; a page plane that
// blocks the tip; error, effort, velocity and freeze injection for the interlock tests.
class FakeBackend : public Backend
{
public:
  explicit FakeBackend(FakeOptions options = {});
  void connect() override;
  Feedback read() override;
  void command(const Vec7 & q, const Vec7 & qd) override;
  void set_position_mode() override;
  void disconnect() override;
  std::string error() override;

  void set_idle();
  void inject_error(const std::string & text);
  void refuse_connect(bool refuse);   // connect() throws, as at a controller that does not answer
  void set_effort(size_t joint, double value);
  void set_velocity_offset(size_t joint, double value);
  void set_frozen(bool frozen);
  // What the last command() received, and whether the motors are idle.
  Vec7 last_q() const;
  Vec7 last_qd() const;
  bool idle() const;
  bool connected() const;
  int commands() const;
  int connects() const;

private:
  mutable std::mutex mutex_;
  FakeOptions o_;
  Feedback state_;
  Vec7 last_q_{}, last_qd_{}, velocity_offset_{};
  std::string error_;
  bool idle_ = true, frozen_ = false, connected_ = false, refuse_connect_ = false;
  int reads_ = 0, commands_ = 0, connects_ = 0;
};

}  // namespace tatbot_hardware
