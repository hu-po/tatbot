#pragma once
// What the adapter needs from an arm: the Trossen SDK (sdk=real) or the built-in fake (sdk=fake).
// read() and command() run on the control thread (UDP on the SDK); set_position_mode(),
// set_idle() and error() are TCP round trips (~1.5 ms) and run only on the 10 Hz aux thread or
// during lifecycle transitions, serialized by the adapter.
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
  virtual void set_idle() = 0;
  virtual std::string error() = 0;                           // "" when healthy
  // Set the controller's carriage position limits (volatile, reset on power-up) and return the
  // limits it held before; the fake keeps no limits.
  virtual std::pair<double, double> set_carriage_limits(double, double) {return {0.0, 0.0};}
};

// nullptr when built without the SDK (TATBOT_ARM_SDK off or no fetch possible).
std::unique_ptr<Backend> make_sdk_backend(const std::string & ip, const std::string & end_effector);
bool sdk_available();

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
  void set_idle() override;
  std::string error() override;

  void inject_error(const std::string & text);
  void set_effort(size_t joint, double value);
  void set_velocity_offset(size_t joint, double value);
  void set_frozen(bool frozen);
  void set_page_z(double z);
  // What the last command() received, and whether the motors are idle.
  Vec7 last_q() const;
  Vec7 last_qd() const;
  bool idle() const;
  int commands() const;

private:
  mutable std::mutex mutex_;
  FakeOptions o_;
  Feedback state_;
  Vec7 last_q_{}, last_qd_{}, velocity_offset_{};
  std::string error_;
  bool idle_ = true, frozen_ = false, connected_ = false;
  int reads_ = 0, commands_ = 0;
};

}  // namespace tatbot_hardware
