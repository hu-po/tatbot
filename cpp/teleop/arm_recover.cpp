// arm_recover — land one arm after a controller fault, on the vendor SDK.
//
//   arm_recover <ip> <leader|follower> --staged a,b,c,d,e,f,g --estop DEV
//               [--golden PATH] [--attempts N]
//
// This is the native landing routine behind `tatbot arm recover`. It is the
// same ritual the LeRobot plugins run on a failed disconnect (recovery.py's
// land_arm), ported to the C++ arm stack so a recovery needs no Python
// environment: a FRESH driver session with clear_error, the golden pushed
// when the controller boots faulted or outside its own limits, a soft
// position-mode takeover at the measured pose, then staged -> sleep -> idle,
// verified rather than assumed. An arm joint measured past its limits by more
// than the controller's tolerance is refused instead, with nothing commanded
// (recover_guard.hpp). The carriage stays where it was measured during the
// staged sweep, then returns to the configured rest position as the arm
// settles into sleep.
//
// Every phase is non-blocking and heartbeat-monitored: if the e-stop engages
// mid-interpolation the arm is frozen at its measured pose, and the phase
// resumes from there once the button is released. Ctrl+C is shielded while
// the arm moves (a second press mid-sweep would leave it holding a stale
// target); a press more than HANG_ESCAPE_S after the first is honoured so a
// hung landing stays interruptible. The hardware e-stop is always live.
//
// Each attempt runs in its own forked process. The vendor driver is not
// designed to be retried inside one process: its header states that after an
// exception "the program is expected to terminate", and its destructor calls
// cleanup(), which throws on an already-closed socket — an in-process retry
// against an unreachable controller aborted with a core dump. A crashed
// attempt is now just a failed attempt; the parent never touches the SDK.
//
// Exit codes follow the CLI contract: 0 landed and idle, 1 landing failed or
// interrupted (arm state unknown), 2 usage, 3 e-stop engaged before any
// command, 5 the controller never answered (nothing was commanded), 6 driver
// ownership could not be obtained after stopping existing owners, 7 an arm
// joint measured past its limits (nothing was commanded: power the controller
// off, turn that joint by hand to near 0 and power it on).

#include <libtrossen_arm/trossen_arm.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "driver_lease.hpp"
#include "estop_monitor.hpp"
#include "recover_guard.hpp"

namespace
{

// recovery.py constants, kept identical so both landing paths behave alike.
constexpr double TAKEOVER_S = 0.5;
constexpr double STAGED_POSE_S = 4.0;
constexpr double SLEEP_POSE_S = 3.0;
constexpr double RETRY_DELAY_S = 2.0;
constexpr double CONFIGURE_TIMEOUT_S = 5.0;  // the driver default (20 s) is too long to hang a landing
constexpr double LANDING_DEADLINE_S = 45.0;
constexpr double LANDED_TOLERANCE_RAD = 0.20;  // "did it actually reach the sleep pose?"
constexpr double CARRIAGE_LANDED_TOLERANCE_M = 0.0005;
constexpr double PHASE_SETTLE_S = 0.15;
constexpr double ESTOP_INITIAL_WAIT_S = 0.5;
constexpr double MEASUREMENT_SETTLE_S = 0.1;  // let the daemon deliver real robot output first
constexpr double MEASUREMENT_WAIT_S = 1.0;
constexpr double HANG_ESCAPE_S = 10.0;  // Ctrl+C is honoured again after this long
constexpr std::chrono::milliseconds POLL{20};
constexpr size_t NUM_JOINTS = 7;
constexpr size_t WRIST_ROLL = 5;
using tatbot::recover::CARRIAGE;
using tatbot::recover::fmt;

std::atomic<int> g_estop{tatbot::estop::fault};
std::atomic<int> g_sigints{0};
std::atomic<long long> g_first_sigint_ns{0};

long long monotonic_ns()
{
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// Async-signal-safe: one atomic bump and one clock read. The shield's message
// is printed from the main thread when it notices the count moved.
void handle_sigint(int)
{
  if (g_sigints.fetch_add(1) == 0) {
    g_first_sigint_ns.store(monotonic_ns());
  }
}

struct Interrupted : std::exception
{
  const char * what() const noexcept override {return "landing interrupted";}
};

// The shield: swallow repeats while landing, let a press through once the
// landing looks hung. Called from every polling loop.
void check_interrupt(const std::string & name)
{
  static int announced = 0;
  const int count = g_sigints.load();
  if (count == 0) {return;}
  if (announced == 0) {
    announced = 1;
    std::cerr << name << " landing in progress — Ctrl+C ignored (the arm idles in a few "
              << "seconds; press again after " << HANG_ESCAPE_S << " s if it is stuck)" << std::endl;
  }
  if (count > 1 && monotonic_ns() - g_first_sigint_ns.load() > HANG_ESCAPE_S * 1e9) {
    std::cerr << name << " landing appears hung — honouring Ctrl+C; recover with "
              << "`tatbot arm recover`" << std::endl;
    throw Interrupted{};
  }
}

bool estop_engaged() {return g_estop.load() > tatbot::estop::ok;}

const char * estop_word()
{
  return g_estop.load() == tatbot::estop::pressed ? "button latched" : "no heartbeat";
}

// The controller's error string, or "" when healthy. Only these known healthy
// strings count as healthy; anything else is reported verbatim.
std::string controller_error(trossen_arm::TrossenArmDriver & driver)
{
  std::string raw;
  try {
    if (!driver.get_is_configured()) {return "";}
    raw = driver.get_error_information();
  } catch (const std::exception &) {
    return "";
  }
  const auto begin = raw.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {return "";}
  const auto end = raw.find_last_not_of(" \t\r\n");
  const std::string value = raw.substr(begin, end - begin + 1);
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  if (lower == "no error" || lower == "none" || lower == "error state: none") {return "";}
  return value;
}

struct Options
{
  std::string ip;
  std::string role;
  std::vector<double> staged;
  std::string golden;
  std::string estop_device;
  int attempts = 3;
};

int usage(const char * error)
{
  if (error != nullptr) {std::cerr << "arm_recover: " << error << "\n";}
  std::cerr << "usage: arm_recover <ip> <leader|follower> --staged a,b,c,d,e,f,g --estop DEV"
               " [--golden PATH] [--attempts N]\n";
  return 2;
}

bool parse_csv(const std::string & text, std::vector<double> & out)
{
  out.clear();
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, ',')) {
    try {
      size_t used = 0;
      const double value = std::stod(field, &used);
      if (used != field.size() || !std::isfinite(value)) {return false;}
      out.push_back(value);
    } catch (const std::exception &) {
      return false;
    }
  }
  return out.size() == NUM_JOINTS;
}

trossen_arm::EndEffector end_effector_for(const std::string & role)
{
  return role == "leader" ? trossen_arm::StandardEndEffector::wxai_v0_leader :
         trossen_arm::StandardEndEffector::wxai_v0_follower;
}

// Push the arm's golden (config/trossen/<role>.yaml) into the controller.
//
// A power-cycled controller boots with its own limits, and the empty follower
// carriage rests on its stop at ~-4.6 mm, past the boot -4 mm limit: the
// controller idled motor 6 at connect and every landing attempt failed with
// "Robot input with modes different than configured modes". The golden
// carries the -6 mm limit the sessions run with. This is the same load the
// teleop executor performs at every connect (configure_arm in
// wxai_teleop.cpp): the SDK's file loader, then the standard end effector
// re-applied because the per-arm YAMLs predate it.
bool apply_golden(
  trossen_arm::TrossenArmDriver & driver, const Options & opt, const std::string & name)
{
  if (opt.golden.empty()) {
    std::cerr << name << " landing: no golden given; controller keeps its boot limits" << std::endl;
    return false;
  }
  if (!std::filesystem::is_regular_file(opt.golden)) {
    std::cerr << name << " landing: golden " << opt.golden
              << " missing; controller keeps its boot limits" << std::endl;
    return false;
  }
  try {  // a landing must never die on its own config
    driver.load_configs_from_file(opt.golden);
    driver.set_end_effector(end_effector_for(opt.role));
    return true;
  } catch (const std::exception & e) {
    std::cerr << name << " golden not applied at landing: " << e.what() << std::endl;
    return false;
  }
}

std::vector<trossen_arm::JointLimit> controller_limits(trossen_arm::TrossenArmDriver & driver)
{
  try {
    auto limits = driver.get_joint_limits();
    if (limits.size() != NUM_JOINTS) {limits.clear();}
    return limits;
  } catch (const std::exception &) {
    return {};
  }
}

void sleep_poll() {std::this_thread::sleep_for(POLL);}

// Override any interpolation with a measured-pose position hold, then wait
// for the button. A press mid-sweep leaves the arm exactly where it is.
void freeze_until_released(trossen_arm::TrossenArmDriver & driver, const std::string & name)
{
  std::cerr << "E-STOP engaged (" << estop_word() << "): lifecycle motion frozen" << std::endl;
  driver.set_all_modes(trossen_arm::Mode::position);
  driver.set_all_positions(driver.get_all_positions(), 0.0, false);
  while (estop_engaged()) {
    check_interrupt(name);
    sleep_poll();
  }
  std::cerr << "E-stop released: resuming lifecycle motion" << std::endl;
}

// One interpolation phase, pausing at the measured pose on e-stop.
void run_monitored_phase(
  trossen_arm::TrossenArmDriver & driver, const std::string & name,
  const std::vector<double> & target, double seconds)
{
  using clock = std::chrono::steady_clock;
  while (true) {
    check_interrupt(name);
    if (estop_engaged()) {freeze_until_released(driver, name);}
    driver.set_all_positions(target, seconds, false);
    const auto started = clock::now();
    bool tripped = false;
    while (std::chrono::duration<double>(clock::now() - started).count() < seconds + PHASE_SETTLE_S) {
      check_interrupt(name);
      if (estop_engaged()) {tripped = true; break;}
      sleep_poll();
    }
    // Close the edge between the final poll and returning to the caller,
    // which may idle the arm after the last phase.
    if (!tripped && !estop_engaged()) {return;}
  }
}

// Child-attempt outcomes. Only the parent maps these onto the CLI contract.
enum Outcome : int {
  landed = 0,
  failed = 1,         // connected, did not land: retry
  estop_refused = 3,  // nothing commanded; do not retry
  unreachable = 5,    // no session opened; retry
  beyond_limits = 7,  // an arm joint past its limits, nothing commanded; do not retry
  interrupted = 8,    // do not retry; reported as 1
};

// The SDK's cleanup() throws on an already-closed socket, and its destructor
// calls cleanup() again. A driver whose cleanup failed is therefore leaked on
// purpose: this process ends in a moment anyway, and a throwing destructor
// would abort it before the outcome is reported.
void quiet_cleanup(std::unique_ptr<trossen_arm::TrossenArmDriver> & driver)
{
  if (!driver) {return;}
  try {
    driver->cleanup();
    driver.reset();
  } catch (const std::exception & e) {
    std::cerr << "controller cleanup failed (" << e.what() << "); leaving the session object"
              << std::endl;
    (void)driver.release();
  }
}

// Fresh session: clear_error is what clears controller fault state. A failed
// configure leaves the SDK object half torn down; it is leaked, never cleaned.
std::unique_ptr<trossen_arm::TrossenArmDriver> fresh_session(const Options & opt)
{
  auto driver = std::make_unique<trossen_arm::TrossenArmDriver>();
  try {
    driver->configure(
      trossen_arm::Model::wxai_v0, end_effector_for(opt.role), opt.ip, true, CONFIGURE_TIMEOUT_S);
  } catch (...) {
    (void)driver.release();
    throw;
  }
  return driver;
}

// configure() returns before the daemon has received any robot output, so
// the first get_all_positions() can be the SDK's zero-initialised default: a
// carriage on its stop at -4.7 mm read as 0.0000, the limit guard stayed
// quiet and the takeover idled motor 6. Wait for a measurement that is not
// the default before trusting it.
std::vector<double> fresh_measurement(trossen_arm::TrossenArmDriver & driver, const std::string & name)
{
  using clock = std::chrono::steady_clock;
  const auto started = clock::now();
  const auto settle = started + std::chrono::duration<double>(MEASUREMENT_SETTLE_S);
  const auto deadline = started + std::chrono::duration<double>(MEASUREMENT_WAIT_S);
  std::vector<double> positions;
  while (true) {
    check_interrupt(name);
    positions = driver.get_all_positions();
    const bool real = positions.size() == NUM_JOINTS &&
      std::any_of(positions.begin(), positions.end(), [](double q) {return q != 0.0;});
    const auto now = clock::now();
    if (real && now >= settle) {return positions;}
    if (now >= deadline) {
      std::cerr << name << " no live measurement within " << MEASUREMENT_WAIT_S
                << " s of connecting; proceeding with the controller's report" << std::endl;
      return positions;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// One landing attempt, in a child process: own e-stop monitor, own session.
int attempt_once(const Options & opt, int attempt)
{
  const std::string name = opt.role + "@" + opt.ip;
  // Monitor first: a latched or silent e-stop refuses the landing before any
  // controller command. FAULT is retained when no healthy frame arrives in
  // time — silence and an undecided startup state are both stop conditions.
  std::unique_ptr<tatbot::estop::Monitor> estop;
  try {
    estop = std::make_unique<tatbot::estop::Monitor>(opt.estop_device, true, g_estop);
  } catch (const std::exception & e) {
    std::cerr << "arm_recover: " << e.what() << std::endl;
    return estop_refused;
  }
  {
    using clock = std::chrono::steady_clock;
    const auto until = clock::now() + std::chrono::duration<double>(ESTOP_INITIAL_WAIT_S);
    while (g_estop.load() == tatbot::estop::fault && clock::now() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  if (estop_engaged()) {
    std::cerr << name << " landing refused: e-stop is engaged (" << estop_word()
              << ") — release it first, the arm must never be driven while latched" << std::endl;
    return estop_refused;
  }

  std::unique_ptr<trossen_arm::TrossenArmDriver> driver;
  try {
    std::cerr << name << " landing: attempt " << attempt << "/" << opt.attempts
              << " over a fresh driver session" << std::endl;
    try {
      driver = fresh_session(opt);
    } catch (const std::exception & e) {
      std::cerr << name << " controller did not answer: " << e.what() << std::endl;
      return unreachable;
    }
    std::string err = controller_error(*driver);
    // The golden matters only when the controller boots faulted against
    // its boot limits. In the ordinary handover the session that just
    // ended already loaded the golden, and the seconds the load takes sit
    // between the fresh session and the position-mode takeover — the arm
    // sagged visibly there. Take over first; heal only on a fault.
    if (!err.empty()) {
      std::cerr << name << " firmware error at landing: " << err << std::endl;
      if (apply_golden(*driver, opt, name)) {
        std::cerr << name << " landing: golden applied" << std::endl;
        // The fault was judged against the boot limits. The SDK clears
        // errors only at configure, so reconnect once now that the golden
        // limits are in the controller.
        quiet_cleanup(driver);
        driver = fresh_session(opt);
        err = controller_error(*driver);
        if (err.empty()) {
          std::cerr << name << " fault cleared after applying the golden limits" << std::endl;
        } else {
          std::cerr << name << " fault persists after the golden: " << err << std::endl;
        }
      }
    }

    const std::vector<double> positions = tatbot::recover::guard_measured_pose(
      name, fresh_measurement(*driver, name),
      [&] {return controller_limits(*driver);},
      [&] {return apply_golden(*driver, opt, name);}, std::cerr);
    if (positions.size() != NUM_JOINTS) {
      throw std::runtime_error("controller reported " + std::to_string(positions.size()) +
              " joints, expected " + std::to_string(NUM_JOINTS));
    }
    // Sleep = every joint at zero except the wrist roll (cube up) and the
    // configured carriage rest position. Keep the measured carriage only
    // through the staged sweep; the sleep phase closes it at the safe pose.
    std::vector<double> staged = opt.staged;
    std::vector<double> sleep(NUM_JOINTS, 0.0);
    sleep[WRIST_ROLL] = staged[WRIST_ROLL];
    sleep[CARRIAGE] = staged[CARRIAGE];
    staged[CARRIAGE] = positions[CARRIAGE];

    check_interrupt(name);
    if (estop_engaged()) {freeze_until_released(*driver, name);}
    driver->set_all_modes(trossen_arm::Mode::position);
    run_monitored_phase(*driver, name, positions, TAKEOVER_S);
    run_monitored_phase(*driver, name, staged, STAGED_POSE_S);
    run_monitored_phase(*driver, name, sleep, SLEEP_POSE_S);
    driver->set_all_modes(trossen_arm::Mode::idle);

    const std::vector<double> final_pose = driver->get_all_positions();
    double worst = 0.0;
    for (size_t i = 0; i < CARRIAGE && i < final_pose.size(); ++i) {
      worst = std::max(worst, std::fabs(final_pose[i] - sleep[i]));
    }
    const double carriage_error = final_pose.size() == NUM_JOINTS ?
      std::fabs(final_pose[CARRIAGE] - sleep[CARRIAGE]) :
      std::numeric_limits<double>::infinity();
    const bool ok = final_pose.size() == NUM_JOINTS && worst <= LANDED_TOLERANCE_RAD &&
      carriage_error <= CARRIAGE_LANDED_TOLERANCE_M;
    quiet_cleanup(driver);
    if (ok) {
      std::cerr << name << " landing complete: sleep pose, motors idle (carriage at rest "
                << fmt(final_pose[CARRIAGE]) << ")" << std::endl;
      return landed;
    }
    std::cerr << name << " landing did NOT reach the sleep pose (worst joint off by "
              << fmt(worst, 2) << " rad, carriage off rest by "
              << fmt(carriage_error * 1e3, 2) << " mm) — the controller accepted the commands "
              << "but did not execute them" << std::endl;
    return failed;
  } catch (const tatbot::recover::BeyondLimits & e) {
    std::cerr << e.what() << std::endl;
    quiet_cleanup(driver);
    return beyond_limits;
  } catch (const Interrupted &) {
    std::cerr << name << " landing interrupted — arm state unknown" << std::endl;
    quiet_cleanup(driver);
    return interrupted;
  } catch (const std::exception & e) {
    std::cerr << name << " landing attempt " << attempt << " failed: " << e.what() << std::endl;
    quiet_cleanup(driver);
    return failed;
  }
}

// Parent: fork one child per attempt, never touch the SDK, map outcomes.
int land_arm(const Options & opt)
{
  using clock = std::chrono::steady_clock;
  const std::string name = opt.role + "@" + opt.ip;
  const auto deadline = clock::now() + std::chrono::duration<double>(LANDING_DEADLINE_S);
  bool ever_connected = false;
  for (int attempt = 1; attempt <= opt.attempts; ++attempt) {
    std::cerr.flush();
    const pid_t child = fork();
    if (child < 0) {
      std::cerr << name << " cannot start a landing attempt: " << std::strerror(errno) << std::endl;
      return 1;
    }
    if (child == 0) {
      // The SDK's connect can block well past its own timeout. When the
      // launcher's outer `timeout` terminates the parent, the attempt must
      // not live on as an orphan still able to open a session.
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      if (getppid() == 1) {std::_Exit(failed);}
      const int code = attempt_once(opt, attempt);
      std::cerr.flush();
      std::_Exit(code);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
      if (errno != EINTR) {
        std::cerr << name << " lost track of the landing attempt: " << std::strerror(errno) << std::endl;
        return 1;
      }
    }
    int outcome = failed;
    if (WIFEXITED(status)) {
      outcome = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      std::cerr << name << " landing attempt " << attempt << " crashed (signal "
                << WTERMSIG(status) << ")" << std::endl;
      ever_connected = true;  // it may have commanded the arm before dying
    }
    switch (outcome) {
      case landed: return 0;
      case estop_refused: return 3;
      case beyond_limits: return 7;
      case interrupted: return 1;
      case unreachable: break;
      default: ever_connected = true; break;
    }
    if (g_sigints.load() > 0) {
      std::cerr << name << " interrupted between attempts — arm state unknown" << std::endl;
      return 1;
    }
    if (clock::now() > deadline) {
      std::cerr << name << " landing exceeded its " << LANDING_DEADLINE_S << " s budget" << std::endl;
      break;
    }
    if (attempt < opt.attempts) {
      const auto until = clock::now() + std::chrono::duration<double>(RETRY_DELAY_S);
      while (clock::now() < until) {
        if (g_sigints.load() > 0) {
          std::cerr << name << " interrupted between attempts — arm state unknown" << std::endl;
          return 1;
        }
        sleep_poll();
      }
    }
  }
  if (!ever_connected) {
    std::cerr << name << " controller never answered — is the arm powered on? (arms take ~20 s "
              << "to boot). Nothing was commanded." << std::endl;
    return 5;
  }
  std::cerr << name << " landing FAILED — the arm may still be holding position. "
            << "Power-cycle it and run `tatbot arm recover`" << std::endl;
  return 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  Options opt;
  std::vector<std::string> args(argv + 1, argv + argc);
  std::vector<std::string> positional;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string & arg = args[i];
    auto value = [&]() -> std::string {
        if (i + 1 >= args.size()) {throw std::runtime_error(arg + " needs a value");}
        return args[++i];
      };
    try {
      if (arg == "--staged") {
        if (!parse_csv(value(), opt.staged)) {
          return usage("--staged needs seven finite comma-separated values");
        }
      } else if (arg == "--estop") {opt.estop_device = value();} else if (arg == "--golden") {
        opt.golden = value();
      } else if (arg == "--attempts") {
        opt.attempts = std::stoi(value());
        if (opt.attempts < 1) {return usage("--attempts must be at least 1");}
      } else if (arg == "--help" || arg == "-h") {
        return usage(nullptr);
      } else if (!arg.empty() && arg[0] == '-') {
        return usage(("unknown option " + arg).c_str());
      } else {
        positional.push_back(arg);
      }
    } catch (const std::exception & e) {
      return usage(e.what());
    }
  }
  if (positional.size() != 2) {return usage("expected <ip> <leader|follower>");}
  opt.ip = positional[0];
  opt.role = positional[1];
  if (opt.role != "leader" && opt.role != "follower") {return usage("role must be leader or follower");}
  if (opt.staged.empty()) {return usage("--staged is required (config/trossen/tatbot.yaml staged_positions)");}
  // Production fails closed: the device is mandatory, there is no opt-out.
  if (opt.estop_device.empty()) {return usage("--estop DEV is required");}

  // Stop current driver owners, then retain the SAME inode's exclusive flock
  // through every attempt child. Only standalone recovery may take over;
  // ordinary teleop and session acquisition still refuse when busy.
  std::unique_ptr<tatbot::DriverLease> lease;
  try {
    lease = std::make_unique<tatbot::DriverLease>(
      "/tmp/tatbot-arm-driver.lock", tatbot::DriverLease::Mode::recover);
  } catch (const std::exception & e) {
    std::cerr << "arm_recover: " << e.what() << std::endl;
    return 6;
  }

  // Installed before the fork so parent and child share the shield: the
  // terminal delivers Ctrl+C to the whole foreground process group.
  struct sigaction action {};
  action.sa_handler = handle_sigint;
  action.sa_flags = SA_RESTART;  // a swallowed signal must not EINTR-abort the SDK's socket reads
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, nullptr);

  return land_arm(opt);
}
