// Minimal leader→follower joint-space teleoperation for two Trossen WidowX AI
// (WXAI V0) arms, built on the official trossen_arm SDK and modeled on its
// demos/cpp/teleoperation.cpp and gravity_compensation.cpp examples.
//
// The leader arm runs in external_effort mode so it can be hand-guided freely,
// including its gripper trigger (which is read but drives nothing). With
// --ff-gain 0 the leader is in pure gravity/friction compensation; with a
// positive gain the follower's external efforts are reflected back to the
// leader (force feedback, vendor uses 0.1).
//
// Two optional anti-stiction terms target what the firmware's velocity-gated
// friction model cannot (its compensation is zero at standstill, so no
// joint_characteristics value lowers breakaway force):
//   --damping <Nm per rad/s>  opposes raw leader joint velocity (capped) —
//                             absorbs the post-breakaway lurch. Try 0.5-1.5.
//   --assist <gain>           pushes with the operator's estimated hand torque
//                             (deadbanded, low-passed, capped) — lowers the
//                             breakaway force itself. Try 0.2-0.5.
// Both default off and act on the base joints (J0-J2) only — that is where
// the stiction is, and the light wrist joints oscillate under these gains. A
// runaway guard drops both terms (until resume) past 6 rad/s on any joint.
//
// The follower runs in position mode and its ARM joints track the leader's
// joint angles ABSOLUTELY: joint i ends up at joint i, whatever pose either
// arm was parked in before power-on (the WXAI encoders are absolute and their
// home calibration survives a power cycle). Whatever offset the arms start
// with is ramped out by an announced, speed-bounded alignment move
// (--align-rate); --relative keeps the old delta mapping, where that offset
// persists all session.
//
// The follower's last joint is NOT a gripper any more (since 2026-08-30):
// the right finger is removed and
// the tool sits in a bore on a printed mount bolted to the left finger
// carriage, so that carriage is a tool-axis DOF — 0.0 m (closed hard stop) is
// the pen at rest/extended, opening retracts the pen along its own axis.
// Nothing is gripped. The carriage is owned by the safety layer, never by the
// leader: it is seated at rest on startup, its external effort is read as the
// contact force, and a debounced contact above --contact-cap, an e-stop, or a
// driver fault retracts the pen before the arm freezes. The leader's trigger
// is never mirrored onto it.
//
// The leader signal is low-pass filtered (first-order, --tau) and the follower
// is given a short interpolation horizon per command (--goal-time) instead of
// jump-to-target, which together remove encoder/scheduling jitter at the cost
// of ~tau of imperceptible lag.
//
// Every tick is offered to a bounded, asynchronous binary flight-recorder log
// (timing stamps plus full leader/follower state) for offline analysis of
// hiccups and tracking quality — see analyze_log.py. A slow disk drops whole
// records instead of blocking control. Disable with --no-log.
//
// Usage:
//   wxai_teleop [leader_ip] [follower_ip] [leader_config.yaml] [follower_config.yaml]
//               [--ff-gain G] [--contact-cap N] [--carriage-retract M]
//               [--tau S] [--goal-time S]
//               [--period-us U] [--log PATH] [--no-log]
//               [--relative] [--align-rate R] [--align-confirm-deg D]
//               [--square-probe-mm M] [--square-edge-s S]
//               [--spiral-radius-mm M] [--spiral-turns N] [--spiral-duration-s S]
//               [--spiral-ease-s S] [--spiral-carriage-ik]
//               [--draw-dir DIR]
//               [--no-rt] [--rt-priority P]
//               [--telemetry-udp HOST:PORT] [--telemetry-fps HZ]
//               [--estop DEV] [--no-estop]
//               --ee-tool ID [--tool-uncalibrated] | --no-tool
//
// Arm addresses come from TATBOT_LEADER_IP / TATBOT_FOLLOWER_IP (exported by
// the tatbot CLI from the hardware profile) or the two positionals; leader =
// arm with the standard Trossen leader handle, follower = arm with the fixed
// tool mount on its left finger carriage. The optional YAML configs are the
// per-arm files saved by src/tatbot/bot/trossen_config.py (joint
// characteristics etc.); their stale tatbot 1.0 end-effector section is
// overridden with the standard EE model after loading.
//
// The loop pins itself to the machine's fastest cores and asks for SCHED_FIFO
// before the drivers connect, so the SDK's UDP daemon threads inherit both.
// On a hybrid CPU an unpinned loop is migrated onto an E-core under load,
// where the same tick takes ~3x longer and the follower is fed late, jerky
// targets — which the operator feels as shaking. --no-rt opts out; see the
// README's "Loop health" section.
//
// Ctrl+C stops teleoperation: both arms lock in place in position mode, then
// go idle only after you confirm with Enter (support the arms first — idle
// releases the joints).
//
// Hardware e-stop (firmware/estop_pico/, README "Hardware e-stop"): pressing
// the mushroom button — or losing its 100 Hz heartbeat — freezes both arms
// in position mode. Twist-releasing the latch (or restoring the heartbeat)
// re-reads both held poses and automatically resumes tracking with zero
// initial step in ordinary human teleop; a square probe is terminal and never
// resumes scripted motion. Neither arm goes limp. The default
// /dev/tatbot-estop device is mandatory. --estop DEV selects another mandatory
// device; --no-estop is an explicit hardware-free bench opt-out rejected by
// production launchers.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sched.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "libtrossen_arm/trossen_arm.hpp"
#include "estop_monitor.hpp"
#include "flight_recorder.hpp"
#include "realtime.hpp"
#include "square_probe.hpp"
#include "telemetry_udp.hpp"

namespace
{

// Stop hierarchy (see README "Stopping / e-stop behavior"): the first
// SIGINT/SIGTERM requests a controlled stop — arms hold at their actual
// positions and the carriage holds where it is. A second signal is an
// emergency release: idle immediately without waiting for the operator.
std::atomic<int> g_stop_signals{0};

void handle_signal(int)
{
  g_stop_signals.fetch_add(1);
}

// ---------------------------------------------------------------------------
// Hardware e-stop (see README "Hardware e-stop"): a latching NC mushroom
// button wired to a Pico that streams "EST1 <seq> <0|1>\n" heartbeat frames
// over USB CDC at 100 Hz (firmware/estop_pico/). The stream itself is the
// safety signal: a press, an unplugged cable, wedged firmware, or a dead
// board all stop producing state-1 frames and read as ENGAGED. The 400 Hz
// loop only ever does one relaxed atomic load; parsing happens on the
// monitor thread.

constexpr int estop_disabled = tatbot::estop::disabled;
constexpr int estop_ok = tatbot::estop::ok;
constexpr int estop_pressed = tatbot::estop::pressed;
constexpr int estop_fault = tatbot::estop::fault;

std::atomic<int> g_estop{estop_disabled};

enum class StopChoice { release, emergency, resume, estop };

bool request_probe_landing()
{
  const char * raw_path = std::getenv("TATBOT_PROBE_LAND_SENTINEL");
  if (!raw_path || !*raw_path) {return false;}
  const std::filesystem::path path(raw_path);
  if (path.parent_path() != "/tmp" ||
    path.filename().string().rfind("tatbot-probe-land-", 0) != 0)
  {
    return false;
  }
  std::ofstream sentinel(path);
  sentinel << "operator-release\n";
  return static_cast<bool>(sentinel);
}

// Make a single key available immediately (no Enter) while preserving signal
// generation, so Ctrl+C still follows the normal controlled-stop path. This is
// enabled only after startup alignment has consumed any Enter confirmation.
class SingleKeyInput
{
public:
  explicit SingleKeyInput(bool enable)
  {
    if (!enable || !isatty(STDIN_FILENO)) {return;}
    if (tcgetattr(STDIN_FILENO, &saved_) != 0) {return;}
    termios one_key = saved_;
    one_key.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    one_key.c_cc[VMIN] = 0;
    one_key.c_cc[VTIME] = 0;
    active_ = tcsetattr(STDIN_FILENO, TCSANOW, &one_key) == 0;
  }

  ~SingleKeyInput()
  {
    if (active_) {tcsetattr(STDIN_FILENO, TCSANOW, &saved_);}
  }

  bool active() const {return active_;}

private:
  termios saved_{};
  bool active_ = false;
};

// Wait at the controlled-stop prompt: Enter releases to idle, a line
// containing 'r' resumes teleoperation, a further stop signal is an
// emergency release, an e-stop engaging hands control to the e-stop flow.
// Stdin EOF (scripted runs) releases to idle.
StopChoice wait_for_choice()
{
  const int signals_at_hold = g_stop_signals.load();
  bool resume_requested = false;
  while (true) {
    if (g_stop_signals.load() > signals_at_hold) {
      return StopChoice::emergency;
    }
    if (g_estop.load() > estop_ok) {
      return StopChoice::estop;
    }
    struct pollfd stdin_poll = {STDIN_FILENO, POLLIN, 0};
    const int ready = poll(&stdin_poll, 1, 100);
    if (ready > 0 && (stdin_poll.revents & (POLLIN | POLLHUP)) != 0) {
      char input = '\0';
      const ssize_t count = read(STDIN_FILENO, &input, 1);
      if (count <= 0 || input == '\n') {
        return resume_requested ? StopChoice::resume : StopChoice::release;
      }
      if (input == 'r' || input == 'R') {
        resume_requested = true;
      }
    }
  }
}

// Carriage (tool-axis) constants. The follower's last joint is the left
// finger carriage carrying the fixed tool mount; it runs in position mode
// like the arm joints and is commanded only by the safety layer here.
//   rest    = the closed hard stop, pen extended for work.
//   retract = how far the pen is pulled back along its own axis on a trip
//             (e-stop, contact cap, driver fault). The firmware carriage limit
//             is 0.040 m (config/trossen/follower.yaml position_max), so the
//             full stroke is the ceiling.
//   contact cap = the carriage external effort (N, along the tool axis) that
//             counts as "pushing too hard"; sustained for the debounce window
//             it trips a retract-and-hold. Defaults are conservative for a
//             ballpoint on paper; --contact-cap overrides per session.
// Largest per-joint step the very first streamed command may take from the
// follower's measured pose. A correct ramp starts at zero; 3 deg is noise.
constexpr double FIRST_STEP_MAX_RAD = 0.05;
constexpr double CARRIAGE_REST_M = 0.0;          // m, closed hard stop = pen at rest
constexpr double CARRIAGE_RETRACT_M = 0.040;     // m, trip retract (= firmware limit)
constexpr double CARRIAGE_CONTACT_CAP_N = 20.0;  // N, default contact cap
// WHAT THE CONTACT SIGNAL IS, AND IS NOT (bench, 2026-08-30, two sessions):
// the firmware's carriage external-effort estimate is the only force signal
// the axis has, and it is coarse: with the arm STILL it wanders -12..+10 N
// (p1/p99) about a small orientation-dependent median, and during fast free
// moves it swings to +-18 N from the EE chain's inertial load; a deliberate
// hand push on the pen read +10.9 N. The position hold, meanwhile, is so
// stiff that the same push deflected the carriage 0.02 mm — deflection is
// not measurable. So the cap is on the effort's departure from its rest
// baseline (taken right after seating), only while the arm is moving at
// drawing speeds (below CONTACT_STILL_RAD_S) so inertial swings cannot count,
// sustained for CONTACT_CAP_DEBOUNCE_TICKS. It catches a HARD press, not
// light contact; a compliant carriage (lower position kp on joint 6) would
// make deflection readable and is the upgrade path if finer sensing is needed.
constexpr double CONTACT_STILL_RAD_S = 0.3;   // arm joint speed under which contact is assessed
// Deflection OPEN past this also trips. Bench 2026-08-30: the carriage's lead
// screw is self-locking — a hard hand push moved it 0.19 mm and it stayed —
// so neither this nor the effort cap will see ordinary contact; they catch a
// gross event only. The human on the leader (and the e-stop) is the contact
// guard in teleop; the LeRobot follower's joint-torque overforce guard is it
// for policies, and retracts the carriage when it trips.
constexpr double CARRIAGE_CONTACT_DEFLECT_M = 0.002;
constexpr int CONTACT_CAP_DEBOUNCE_TICKS = 40;   // consecutive ticks (100 ms at 400 Hz)
constexpr double CARRIAGE_TRIP_GOAL_S = 0.15;    // s, retract goal time on a trip
constexpr double CARRIAGE_RESUME_GOAL_S = 0.5;   // s, return-to-rest goal time on resume

// One-shot Cartesian capability probe. The operator still hand-guides to the
// start point, but autonomous motion cannot begin until both arms have been
// nearly still briefly and the operator taps SPACE. Readiness latches so the
// act of reaching for the keyboard does not invalidate a good hold; only
// actual follower motion above the normal contact-assessment speed resets it.
constexpr double SQUARE_SETTLED_RAD_S = 0.10;
constexpr double SQUARE_SETTLED_S = 0.20;
constexpr double SQUARE_READY_RESET_RAD_S = CONTACT_STILL_RAD_S;
constexpr double SQUARE_VELOCITY_ABORT_RAD_S = 2.5;
constexpr double SQUARE_OVERFORCE_ABORT_NM = 9.0;
constexpr double SQUARE_OVERFORCE_WINDOW_S = 0.5;
constexpr double SQUARE_OVERFORCE_FRACTION = 0.5;
constexpr size_t SQUARE_OVERFORCE_MIN_SAMPLES = 8;
constexpr double SQUARE_MODEL_FK_TOLERANCE_M = 0.00025;
constexpr double SQUARE_COMMAND_LEAD_ABORT_RAD = 0.05;
constexpr double CARRIAGE_IK_COMMAND_LEAD_ABORT_M = 0.0005;
constexpr double CARRIAGE_IK_PREFLIGHT_ENDPOINT_TOLERANCE_M = 0.00015;
constexpr double SQUARE_ENDPOINT_TOLERANCE_M = 0.00025;
constexpr double SQUARE_ENDPOINT_SETTLE_MAX_S = 3.0;
// Draw orbit: a capture is requested only after the measured arm has been
// this still for this long (bounded), so the depth frames are not taken on
// the settling bounce.
constexpr double DRAW_CAPTURE_SETTLED_RAD_S = 0.05;  // the measured floor at rest is ~0.007 rad/s; 0.02 never latched
constexpr double DRAW_CAPTURE_SETTLED_S = 0.3;
constexpr double DRAW_CAPTURE_SETTLE_MAX_S = 3.0;

// Anti-stiction terms (--damping / --assist). Command-sign convention, proven
// on hardware twice (828a6e8's DAMPING_SIGN test; the -ff_gain reflection):
// a commanded POSITIVE external effort produces actuator torque in the
// NEGATIVE joint direction. Hence damping that opposes velocity is
// +damping*vel, and assist that pushes WITH the operator's estimated torque
// is -assist*effort. If a term feels backwards on hardware (more lurch, more
// drag under push), the estimate's sign convention differs — flip that term's
// sign here, not the gain.
constexpr double DAMPING_CAP_NM = 2.0;      // per joint, same cap the cockpit used
constexpr double ASSIST_CAP_NM = 1.5;       // per joint, keeps a runaway gentle
// Both terms act on the base joints only: that is where the stiction lives
// (J0-J2; five 2026-08-31 flight logs show zero wrist breakaways), and the
// wrist joints are light enough that these gains destabilize them — damping
// 1.0 through the 20 ms vel_filt lag oscillated motors 4/5 into the firmware
// velocity limit within 2 s (2026-08-31 16:37 run). Damping must use the RAW
// velocity for the same reason: the cockpit's hardware-validated formula did,
// and the filter's phase lag turns damping into excitation.
constexpr size_t ANTISTICTION_JOINTS = 3;
// Soft runaway guard, mirroring the cockpit's watchdog: past this speed the
// anti-stiction terms switch off (latched until resume) instead of letting
// the firmware's velocity limit fault the whole session.
constexpr double ANTISTICTION_RUNAWAY_RAD_S = 6.0;
constexpr double ASSIST_DEADBAND_NM = 0.4;  // ignore model-error bias at rest —
                                            // gravity-model residuals of 1-3 Nm were
                                            // measured on gravity-loaded poses, but
                                            // they vary slowly; the deadband only has
                                            // to hide the noise floor, the low-pass
                                            // plus cap bound the rest
constexpr double ASSIST_TAU_S = 0.05;       // s, low-pass on the assist term

// --- fitted tool ------------------------------------------------------------
// --ee-tool names the datasheet (config/tools/<tool_id>.yaml) for whatever is
// seated in the mount: tip geometry, prompt, ink. Teleop itself reads only
// the `mount:` key — a tool with no mount cannot be fitted — and cross-checks
// the id against the live touch-off calibration so two tools' constants are
// never mixed.
//
// The datasheets in config/tools/ are deliberately flat and comment-heavy so a
// simple scraper can read them (scripts/lib/tool_spec.py does the same on the
// Python side); pulling in a YAML dependency for two scalars is not worth it.
namespace tool_registry
{

// Trim whitespace and a trailing `# comment` from a scraped value.
std::string clean_value(std::string text)
{
  const auto hash = text.find('#');
  if (hash != std::string::npos) {text = text.substr(0, hash);}
  const auto first = text.find_first_not_of(" \t\r\n\"'");
  if (first == std::string::npos) {return {};}
  const auto last = text.find_last_not_of(" \t\r\n\"'");
  return text.substr(first, last - first + 1);
}

// `key: value` at zero indentation. Datasheet scalars live at the top level.
std::string scrape_top_level(const std::filesystem::path & path, const std::string & key)
{
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (line.rfind(key + ":", 0) == 0) {return clean_value(line.substr(key.size() + 1));}
  }
  return {};
}

// `tool_id:` inside the `<arm>:` block of config/workspace.yaml.
std::string scrape_workspace_tool(const std::filesystem::path & path, const std::string & arm)
{
  std::ifstream file(path);
  std::string line;
  bool in_arm = false;
  while (std::getline(file, line)) {
    if (line.empty()) {continue;}
    const bool indented = (line[0] == ' ' || line[0] == '\t');
    if (!indented) {in_arm = (line.rfind(arm + ":", 0) == 0); continue;}
    if (!in_arm) {continue;}
    const auto trimmed = line.substr(line.find_first_not_of(" \t"));
    if (trimmed.rfind("tool_id:", 0) == 0) {return clean_value(trimmed.substr(8));}
  }
  return {};
}

// Repo root: walk up from the executable until config/tools/ appears. Keeps
// the binary runnable from anywhere, which matters because the operator
// launches it by absolute path over ssh.
std::filesystem::path repo_root()
{
  if (const char * override_path = std::getenv("TATBOT_REPO")) {
    return std::filesystem::path(override_path);
  }
  std::error_code ec;
  auto dir = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) {return {};}
  for (dir = dir.parent_path(); !dir.empty() && dir != dir.root_path(); dir = dir.parent_path()) {
    if (std::filesystem::exists(dir / "config" / "tools", ec)) {return dir;}
  }
  return {};
}

std::string known_tools(const std::filesystem::path & repo)
{
  std::string names;
  std::error_code ec;
  for (const auto & entry : std::filesystem::directory_iterator(repo / "config" / "tools", ec)) {
    if (entry.path().extension() == ".yaml") {
      if (!names.empty()) {names += ", ";}
      names += entry.path().stem().string();
    }
  }
  return names.empty() ? "none" : names;
}

// Validate the stated tool, refusing every way of being wrong: unstated,
// unknown, unmountable, or contradicting the live calibration.
//
// `uncalibrated` is the bootstrap: teleop reads no constants from
// workspace.yaml, but a tool whose touch-off has not been measured yet is
// refused here because the calibration names another tool — and the touch-off
// session that would fix that needs this teleop running. So the caller that
// IS the touch-off says so, and the mismatch is announced instead of refused.
// Nothing else should pass it.
void resolve_tool(const std::string & tool_id, bool uncalibrated = false)
{
  const auto repo = repo_root();
  if (repo.empty()) {
    throw std::runtime_error(
            "cannot locate the tatbot repo from the executable path, so the "
            "fitted tool's datasheet cannot be read. Set TATBOT_REPO.");
  }
  const auto datasheet = repo / "config" / "tools" / (tool_id + ".yaml");
  if (!std::filesystem::exists(datasheet)) {
    throw std::runtime_error(
            "unknown --ee-tool '" + tool_id + "': no " + datasheet.string() +
            " (known tools: " + known_tools(repo) + ")");
  }
  // The calibration constants under `right:` belong to whatever tool the
  // touch-off measured. Running a different one against them mixes two tools.
  const auto calibrated =
    scrape_workspace_tool(repo / "config" / "workspace.yaml", "right");
  if (!calibrated.empty() && calibrated != tool_id) {
    if (!uncalibrated) {
      throw std::runtime_error(
              "--ee-tool '" + tool_id + "' is fitted but config/workspace.yaml was "
              "measured with '" + calibrated + "'. Re-run the touch-off for the "
              "fitted tool (this teleop is what the tip phase drives):\n"
              "  tatbot --ee-tool " + tool_id + " teleop start --touchoff   (on the teleop host)\n"
              "  tatbot --ee-tool " + tool_id + " vision calib sweep --phases tip");
    }
    std::cerr << "note: --tool-uncalibrated: config/workspace.yaml was measured with '"
              << calibrated << "', not '" << tool_id << "'. Every workspace constant "
              << "stays " << calibrated << "'s until the touch-off for " << tool_id
              << " is written." << std::endl;
  }
  // The tool has to physically seat in the mount on the carriage. A missing
  // key is the default mount; an explicit `none` is a tool that has no way
  // onto this arm at all.
  auto mount = scrape_top_level(datasheet, "mount");
  if (mount.empty()) {mount = "tool_mount";}
  if (mount == "none") {
    throw std::runtime_error(
            "config/tools/" + tool_id + ".yaml has mount: none — this tool has no "
            "mount on the arm and cannot be fitted");
  }
}

}  // namespace tool_registry

// --- surface-first draw session (docs/draw.md) ------------------------------
// The executor writes the arm's pose for the Python stages and runs those
// stages as subprocesses while both arms hold. No JSON library: the two
// records are flat and written by hand; the samples files come back as CSV.

bool write_draw_pose(
  const std::filesystem::path & path,
  const tatbot::square::JointPose & joints,
  double carriage_m,
  const std::array<double, 3> & tip,
  const tatbot::square::Rotation & rotation,
  const std::string & tool,
  double period_s)
{
  std::ofstream out(path);
  if (!out) {return false;}
  out << std::setprecision(12);
  auto array = [&out](const double * values, size_t count) {
      out << '[';
      for (size_t i = 0; i < count; ++i) {out << (i ? ", " : "") << values[i];}
      out << ']';
    };
  const double t_wall = std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  out << "{\"schema\": \"tatbot.draw-pose/1\", \"frame\": \"right/base_link\", \"period_s\": "
      << period_s << ", \"joints\": ";
  array(joints.data(), joints.size());
  out << ", \"carriage_m\": " << carriage_m << ", \"tip\": ";
  array(tip.data(), tip.size());
  out << ", \"rotation\": [";
  for (size_t row = 0; row < 3; ++row) {
    if (row) {out << ", ";}
    array(rotation[row].data(), 3);
  }
  out << "], \"tool\": \"" << tool << "\", \"t_wall\": " << t_wall << "}\n";
  return static_cast<bool>(out);
}

// Run `draw_stage.py <stage> <dir>` to completion. Returns the stage's exit
// code; 124 on timeout, 125 when a stop signal or the e-stop interrupted it,
// 126 when it could not be started. The child inherits the terminal so its
// report is visible; the arms are holding in position mode throughout.
int run_draw_stage(const std::string & draw_dir, const std::string & stage, double timeout_s)
{
  const char * python = std::getenv("TATBOT_DRAW_PYTHON");
  if (!python || !*python) {return 126;}
  const auto repo = tool_registry::repo_root();
  if (repo.empty()) {return 126;}
  const std::string script = (repo / "scripts" / "draw_stage.py").string();
  const pid_t pid = fork();
  if (pid < 0) {return 126;}
  if (pid == 0) {
    // The child inherits this process's SCHED_FIFO priority and CPU pinning.
    // Left that way, NumPy's threads run at real-time priority on the same
    // cores as the SDK daemon that keeps each controller's session alive:
    // the controllers' UDP state streams were lost during an 8 s map stage,
    // the arms froze holding, and both TCP links broke (2026-09-01, runs
    // 20260901T234248Z and 20260902T000845Z). Back to a normal, niced,
    // unpinned process before exec.
    sched_param normal{};
    normal.sched_priority = 0;
    sched_setscheduler(0, SCHED_OTHER, &normal);
    setpriority(PRIO_PROCESS, 0, 10);
    cpu_set_t all_cpus;
    CPU_ZERO(&all_cpus);
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    for (long cpu = 0; cpu < online && cpu < CPU_SETSIZE; ++cpu) {CPU_SET(cpu, &all_cpus);}
    sched_setaffinity(0, sizeof(all_cpus), &all_cpus);
    setenv("OMP_NUM_THREADS", "2", 0);
    setenv("OPENBLAS_NUM_THREADS", "2", 0);
    setenv("MKL_NUM_THREADS", "2", 0);
    execl(python, python, script.c_str(), stage.c_str(), draw_dir.c_str(), static_cast<char *>(nullptr));
    _exit(126);
  }
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(timeout_s));
  const int signals_at_start = g_stop_signals.load();
  bool killed = false;
  int reason = 0;
  while (true) {
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      if (killed) {return reason;}
      return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }
    if (done < 0) {return 126;}
    if (!killed) {
      if (g_stop_signals.load() != signals_at_start || g_estop.load() > estop_ok) {
        kill(pid, SIGTERM);
        killed = true;
        reason = 125;
      } else if (std::chrono::steady_clock::now() > deadline) {
        kill(pid, SIGTERM);
        killed = true;
        reason = 124;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

bool write_capture_request(
  const std::filesystem::path & path, size_t k, const std::vector<double> & follower_pos)
{
  std::ofstream out(path);
  if (!out) {return false;}
  out << std::setprecision(12);
  const double t_wall = std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  out << "{\"schema\": \"tatbot.draw-capture-request/1\", \"k\": " << k << ", \"joints\": [";
  for (size_t i = 0; i + 1 < follower_pos.size(); ++i) {out << (i ? ", " : "") << follower_pos[i];}
  out << "], \"carriage_m\": " << follower_pos.back() << ", \"t_wall\": " << t_wall << "}\n";
  return static_cast<bool>(out);
}

// Arm addresses come from the hardware profile (plan Phase 2): the tatbot
// CLI and launchers export TATBOT_LEADER_IP / TATBOT_FOLLOWER_IP from the
// resolved profile; positionals still override. No baked-in addresses.
inline std::string env_or(const char * name, const char * fallback)
{
  const char * v = std::getenv(name);
  return (v && *v) ? std::string(v) : std::string(fallback);
}

struct Options
{
  std::string leader_ip = env_or("TATBOT_LEADER_IP", "");
  std::string follower_ip = env_or("TATBOT_FOLLOWER_IP", "");
  std::string leader_config;
  std::string follower_config;
  double ff_gain = 0.1;        // force feedback gain (vendor default), 0 = off
  // Anti-stiction experiments (2026-08-31), both default OFF so runs can A/B
  // them. Firmware friction compensation is velocity-gated — zero at
  // standstill — so no joint_characteristics value can lower breakaway force;
  // these two act where the firmware cannot.
  double damping = 0.0;        // --damping: Nm per rad/s opposing leader joint
                               // velocity — swallows the post-breakaway lurch
  double assist = 0.0;         // --assist: gain on the leader's own external-
                               // effort estimate, pushing WITH the operator —
                               // lowers the force needed to break away at v=0
  double contact_cap_n = CARRIAGE_CONTACT_CAP_N;
  double contact_deflect_m = CARRIAGE_CONTACT_DEFLECT_M;    // --contact-cap: trip threshold, N
  double carriage_retract_m = CARRIAGE_RETRACT_M;   // --carriage-retract: trip travel, m
  std::string ee_tool;         // required: which tool is seated in the mount
  bool tool_required = true;   // --no-tool is the explicit bench opt-out
  bool tool_uncalibrated = false;  // --tool-uncalibrated: this session is the touch-off
  double tau = 0.020;          // leader low-pass time constant, s
  double goal_time = 0.005;    // follower interpolation horizon, s
  int64_t period_us = 2500;    // loop period, us (2500 = 400 Hz)
  bool absolute = true;        // --relative: keep the old delta mapping
  double align_rate = 0.35;    // peak joint speed of the startup alignment, rad/s
  double align_confirm_deg = 15.0;  // ask before an alignment move larger than this
  bool square_probe = false;
  double square_probe_m = 0.0;  // one-shot Cartesian square after SPACE handoff
  double square_edge_s = 12.0;  // 0.5 mm/s for the default 6 mm edge
  bool spiral_probe = false;
  double spiral_radius_m = 0.006;
  double spiral_turns = 3.0;
  double spiral_duration_s = 180.0;
  double spiral_ease_s = 2.0;
  bool spiral_carriage_ik = false;
  std::string draw_dir;        // --draw-dir: surface-first draw session (docs/draw.md)
  bool draw_mode = false;
  bool realtime = true;        // --no-rt: leave scheduling to the kernel
  int rt_priority = 80;        // SCHED_FIFO priority for the control loop
  std::string log_path;        // empty = auto under teleop_logs/
  bool log_enabled = true;
  std::string telemetry_udp;  // empty disables optional visualization telemetry
  double telemetry_fps = 30.0;
  std::string estop_dev = "/dev/tatbot-estop";  // udev symlink for the Pico box
  bool estop_required = true;   // production fails closed; --no-estop is explicit bench opt-out
  bool estop_enabled = true;    // --no-estop: opt out of the hardware e-stop
};

Options parse_args(int argc, char ** argv)
{
  Options opt;
  if (const char * endpoint = std::getenv("TATBOT_TELEMETRY_UDP")) {
    opt.telemetry_udp = endpoint;
  }
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for " + arg);
        }
        return argv[++i];
      };
    if (arg == "--ff-gain") {opt.ff_gain = std::stod(value());} else
    if (arg == "--damping") {opt.damping = std::clamp(std::stod(value()), 0.0, 5.0);} else
    if (arg == "--assist") {opt.assist = std::clamp(std::stod(value()), 0.0, 1.0);} else
    if (arg == "--ee-tool") {opt.ee_tool = value();} else
    if (arg == "--no-tool") {opt.tool_required = false;} else
    if (arg == "--tool-uncalibrated") {opt.tool_uncalibrated = true;} else
    if (arg == "--contact-cap") {
      opt.contact_cap_n = std::clamp(std::stod(value()), 2.0, 40.0);
    } else
    if (arg == "--contact-deflect") {
      opt.contact_deflect_m = std::clamp(std::stod(value()), 0.0003, 0.02);
    } else
    if (arg == "--carriage-retract") {
      opt.carriage_retract_m = std::clamp(std::stod(value()), 0.005, CARRIAGE_RETRACT_M);
    } else
    if (arg == "--tau") {opt.tau = std::stod(value());} else
    if (arg == "--goal-time") {opt.goal_time = std::stod(value());} else
    if (arg == "--period-us") {opt.period_us = std::stoll(value());} else
    if (arg == "--relative") {opt.absolute = false;} else
    if (arg == "--align-rate") {
      opt.align_rate = std::clamp(std::stod(value()), 0.01, 1.5);
    } else
    if (arg == "--align-confirm-deg") {opt.align_confirm_deg = std::stod(value());} else
    if (arg == "--square-probe-mm") {
      opt.square_probe = true;
      opt.square_probe_m = std::stod(value()) * 1e-3;
    } else
    if (arg == "--square-edge-s") {opt.square_edge_s = std::stod(value());} else
    if (arg == "--spiral-radius-mm") {
      opt.spiral_probe = true;
      opt.spiral_radius_m = std::stod(value()) * 1e-3;
    } else
    if (arg == "--spiral-turns") {opt.spiral_turns = std::stod(value());} else
    if (arg == "--spiral-duration-s") {opt.spiral_duration_s = std::stod(value());} else
    if (arg == "--spiral-ease-s") {opt.spiral_ease_s = std::stod(value());} else
    if (arg == "--spiral-carriage-ik") {opt.spiral_carriage_ik = true;} else
    if (arg == "--draw-dir") {opt.draw_dir = value(); opt.draw_mode = true;} else
    if (arg == "--no-rt") {opt.realtime = false;} else
    if (arg == "--rt-priority") {
      opt.rt_priority = static_cast<int>(std::clamp(std::stod(value()), 1.0, 99.0));
    } else
    if (arg == "--log") {opt.log_path = value();} else
    if (arg == "--no-log") {opt.log_enabled = false;} else
    if (arg == "--telemetry-udp") {opt.telemetry_udp = value();} else
    if (arg == "--telemetry-fps") {opt.telemetry_fps = std::stod(value());} else
    if (arg == "--estop") {opt.estop_dev = value(); opt.estop_required = true;} else
    if (arg == "--no-estop") {opt.estop_enabled = false;} else
    if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {positional.push_back(arg);}
  }
  if (positional.size() > 0) {opt.leader_ip = positional[0];}
  if (positional.size() > 1) {opt.follower_ip = positional[1];}
  if (positional.size() > 2) {opt.leader_config = positional[2];}
  if (positional.size() > 3) {opt.follower_config = positional[3];}

  if (opt.period_us <= 0) {
    throw std::runtime_error("--period-us must be positive");
  }
  if (opt.square_probe && (!std::isfinite(opt.square_probe_m) ||
    opt.square_probe_m < 0.001 || opt.square_probe_m > 0.010))
  {
    throw std::runtime_error("--square-probe-mm must be between 1 and 10 mm");
  }
  if (opt.square_probe &&
    (!std::isfinite(opt.square_edge_s) || opt.square_edge_s < 2.0 || opt.square_edge_s > 30.0))
  {
    throw std::runtime_error("--square-edge-s must be between 2 and 30 seconds");
  }
  if (opt.square_probe && !opt.absolute) {
    throw std::runtime_error("the square probe requires absolute leader/follower mapping");
  }
  if (opt.square_probe && opt.spiral_probe) {
    throw std::runtime_error("square and spiral probes are mutually exclusive");
  }
  if (opt.spiral_probe && (!std::isfinite(opt.spiral_radius_m) ||
    opt.spiral_radius_m < 0.002 || opt.spiral_radius_m > 0.012))
  {
    throw std::runtime_error("--spiral-radius-mm must be between 2 and 12 mm");
  }
  if (opt.spiral_probe && (!std::isfinite(opt.spiral_turns) ||
    opt.spiral_turns < 1.0 || opt.spiral_turns > 6.0))
  {
    throw std::runtime_error("--spiral-turns must be between 1 and 6");
  }
  if (opt.spiral_probe && (!std::isfinite(opt.spiral_duration_s) ||
    opt.spiral_duration_s < 30.0 || opt.spiral_duration_s > 600.0))
  {
    throw std::runtime_error("--spiral-duration-s must be between 30 and 600 seconds");
  }
  if (opt.spiral_probe && (!std::isfinite(opt.spiral_ease_s) ||
    opt.spiral_ease_s < 0.5 || opt.spiral_ease_s > 10.0))
  {
    throw std::runtime_error("--spiral-ease-s must be between 0.5 and 10 seconds");
  }
  if (opt.spiral_probe && !opt.absolute) {
    throw std::runtime_error("the spiral probe requires absolute leader/follower mapping");
  }
  if (opt.spiral_carriage_ik && !opt.spiral_probe) {
    throw std::runtime_error("--spiral-carriage-ik is valid only with the spiral probe");
  }
  if (opt.spiral_carriage_ik && opt.ee_tool != "lutin-ballpoint-dot") {
    throw std::runtime_error(
            "carriage IK is qualified only for --ee-tool lutin-ballpoint-dot");
  }
  if (opt.spiral_carriage_ik) {
    const char * armed = std::getenv("TATBOT_CARRIAGE_IK_ARMED");
    if (!armed || std::string(armed) != "1") {
      throw std::runtime_error(
              "carriage IK was not armed by scripts/teleop_spiral.sh; "
              "use `tatbot --ee-tool lutin-ballpoint-dot teleop spiral --carriage-ik "
              "--nonce <fresh-literal>`");
    }
  }
  if (opt.draw_mode) {
    // Surface-first draw session (docs/draw.md): the wrapper arms it, the
    // seven-DOF ballpoint tip model is the only executor it has, and the
    // Python stages it shells out to are named by the wrapper too.
    if (opt.square_probe || opt.spiral_probe) {
      throw std::runtime_error("--draw-dir is exclusive with the square and spiral probes");
    }
    if (!opt.absolute) {
      throw std::runtime_error("the draw session requires absolute leader/follower mapping");
    }
    if (opt.ee_tool != "lutin-ballpoint-dot") {
      throw std::runtime_error(
              "the draw session is qualified only for --ee-tool lutin-ballpoint-dot (its tip model)");
    }
    const char * armed = std::getenv("TATBOT_DRAW_ARMED");
    const char * carriage_armed = std::getenv("TATBOT_CARRIAGE_IK_ARMED");
    if (!armed || std::string(armed) != "1" || !carriage_armed || std::string(carriage_armed) != "1") {
      throw std::runtime_error(
              "the draw session was not armed by scripts/draw_run.sh; "
              "use `tatbot --ee-tool lutin-ballpoint-dot draw run --nonce <fresh-literal>`");
    }
    const char * python = std::getenv("TATBOT_DRAW_PYTHON");
    if (!python || !*python || !std::filesystem::exists(python)) {
      throw std::runtime_error("TATBOT_DRAW_PYTHON must name the interpreter for scripts/draw_stage.py");
    }
    if (!std::filesystem::is_directory(opt.draw_dir) ||
      !std::filesystem::is_directory(std::filesystem::path(opt.draw_dir) / "capture"))
    {
      throw std::runtime_error("--draw-dir must be the wrapper's draw directory with its capture/ subdir");
    }
    if (!isatty(STDIN_FILENO)) {
      throw std::runtime_error("the draw session needs an interactive terminal for the SPACE triggers");
    }
  }
  if (opt.square_probe) {
    const char * armed = std::getenv("TATBOT_SQUARE_ARMED");
    if (!armed || std::string(armed) != "1") {
      throw std::runtime_error(
              "Cartesian square mode was not armed by scripts/teleop_square.sh; "
              "use `tatbot --ee-tool <id> teleop square --nonce <fresh-literal>`");
    }
    if (!isatty(STDIN_FILENO)) {
      throw std::runtime_error("Cartesian square mode needs an interactive terminal for the SPACE trigger");
    }
  }
  if (opt.spiral_probe) {
    const char * armed = std::getenv("TATBOT_SPIRAL_ARMED");
    if (!armed || std::string(armed) != "1") {
      throw std::runtime_error(
              "Cartesian spiral mode was not armed by scripts/teleop_spiral.sh; "
              "use `tatbot --ee-tool <id> teleop spiral --nonce <fresh-literal>`");
    }
    if (!isatty(STDIN_FILENO)) {
      throw std::runtime_error("Cartesian spiral mode needs an interactive terminal for the SPACE trigger");
    }
  }

  // Fail before touching an arm. The tool in the mount decides tip geometry,
  // prompt and ink downstream, and the previous tool's identity applied to
  // this one is a silent wrong answer rather than a loud one — so state it,
  // or say explicitly that nothing is fitted. Same shape as
  // --estop/--no-estop: production fails closed.
  if (opt.ee_tool.empty() && opt.tool_required) {
    const auto repo = tool_registry::repo_root();
    throw std::runtime_error(
            "--ee-tool <id> is required: name the tool seated in the mount so "
            "its datasheet is the one that is used.\n"
            "  known tools: " +
            (repo.empty() ? std::string("(repo not found)") : tool_registry::known_tools(repo)) +
            "\n  bench work with an empty mount: --no-tool");
  }
  if (!opt.ee_tool.empty()) {
    tool_registry::resolve_tool(opt.ee_tool, opt.tool_uncalibrated);
  }
  return opt;
}

// Fast, interruptible reachability probe for an arm controller. The driver's
// own connect blocks for ~20 s and cannot be interrupted by Ctrl+C; this
// checks the TCP port in 100 ms slices, watching the stop flag, and lets us
// print a friendly "is it powered on?" instead of a retry wall.
bool arm_reachable(const std::string & ip, int timeout_ms = 3000)
{
  const int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd < 0) {return true;}  // cannot probe; let the driver try
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(50001);
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    return true;
  }
  connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  bool ok = false;
  for (int waited = 0; waited < timeout_ms; waited += 100) {
    if (g_stop_signals.load() > 0) {break;}
    struct pollfd pfd = {fd, POLLOUT, 0};
    if (poll(&pfd, 1, 100) > 0 && (pfd.revents & POLLOUT) != 0) {
      int error = 0;
      socklen_t len = sizeof(error);
      getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len);
      ok = (error == 0);
      break;
    }
  }
  close(fd);
  return ok;
}

void configure_arm(
  trossen_arm::TrossenArmDriver & driver,
  const std::string & ip,
  const std::string & config_file,
  const std::string & label,
  const trossen_arm::EndEffector & end_effector)
{
  std::cout << "Connecting to " << label << " arm at " << ip << "..." << std::endl;
  driver.configure(
    trossen_arm::Model::wxai_v0,
    end_effector,
    ip,
    true  // clear_error
  );
  if (!config_file.empty()) {
    std::cout << "Loading " << label << " config from " << config_file << std::endl;
    driver.load_configs_from_file(config_file);
    // The per-arm YAMLs predate the standard end effectors (they carry the
    // tatbot 1.0 custom EE mass properties), so re-apply the correct standard
    // model after the load — gravity compensation depends on it.
    driver.set_end_effector(end_effector);
  }
}

// Startup alignment (see the header comment). The follower's target is the
// leader's angle plus an offset that starts at whatever mismatch the two arms
// powered on with and fades to zero over a speed-bounded smoothstep ramp;
// afterwards the mapping is exactly absolute, so joint i of the follower sits
// at joint i of the leader no matter how either arm was parked before power-on.
// In --relative mode the offset never fades, which is the old delta mapping.
class Alignment
{
public:
  Alignment(bool absolute, double rate)
  : absolute_(absolute), rate_(rate) {}

  // Begin a ramp from a fresh pair of baselines (startup, and every resume).
  void restart(
    const std::vector<double> & leader_start,
    const std::vector<double> & follower_start)
  {
    offset_.assign(leader_start.size(), 0.0);
    largest_ = 0.0;
    largest_joint_ = 0;
    for (size_t i = 0; i < offset_.size(); ++i) {
      offset_[i] = follower_start[i] - leader_start[i];
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
  double offset(size_t joint) const {return offset_[joint];}
  double duration() const {return duration_;}
  double largest_rad() const {return largest_;}
  size_t largest_joint() const {return largest_joint_;}

private:
  static constexpr double already_aligned_rad = 0.0017;  // 0.1 deg
  bool absolute_;
  double rate_;
  std::vector<double> offset_;
  double elapsed_ = 0.0;
  double duration_ = 0.0;
  double largest_ = 0.0;
  size_t largest_joint_ = 0;
};

constexpr double rad_to_deg = 57.29577951308232;

// Loop-health watchdog. A 400 Hz teleop loop that misses its deadline hands the
// follower targets that are both late and unevenly spaced, and the operator
// feels that as shaking — so say so out loud while it is happening instead of
// leaving it to be discovered in the flight log afterwards.
class LoopHealth
{
public:
  explicit LoopHealth(double period_s)
  : period_(period_s), window_(static_cast<size_t>(1.0 / period_s)) {}

  // Returns a warning to print, or an empty string.
  std::string tick(double busy_s, double lateness_s)
  {
    ++ticks_;
    ++session_ticks_;
    worst_busy_ = std::max(worst_busy_, busy_s);
    worst_lateness_ = std::max(worst_lateness_, lateness_s);
    session_worst_busy_ = std::max(session_worst_busy_, busy_s);
    if (busy_s > period_) {
      ++overruns_;
      ++session_overruns_;
    }
    if (ticks_ < window_) {return {};}
    const double rate = static_cast<double>(overruns_) / static_cast<double>(ticks_);
    const double worst_busy = worst_busy_;
    const double worst_lateness = worst_lateness_;
    const size_t overruns = overruns_;
    const size_t ticks = ticks_;
    ticks_ = 0;
    overruns_ = 0;
    worst_busy_ = 0.0;
    worst_lateness_ = 0.0;
    if (rate <= warn_fraction) {
      quiet_windows_ = std::min(quiet_windows_ + 1, rearm_windows);
      return {};
    }
    // One warning per degraded spell, and a spell only ends after two healthy
    // seconds — a load that flickers must not warn every other second.
    if (quiet_windows_ < rearm_windows) {return {};}
    quiet_windows_ = 0;
    std::ostringstream warning;
    warning << "WARNING: control loop is late — " << overruns << "/" << ticks
            << " ticks (" << static_cast<int>(rate * 100.0 + 0.5)
            << "%) overran " << period_ * 1e3 << " ms; worst tick "
            << worst_busy * 1e3 << " ms, worst wake-up "
            << worst_lateness * 1e3 << " ms late.\n"
            << "         The follower will feel rough. Something else on this "
               "machine is competing for CPU.";
    return warning.str();
  }

  std::string summary() const
  {
    if (session_ticks_ == 0) {return {};}
    std::ostringstream out;
    out << "Loop health: " << session_overruns_ << "/" << session_ticks_
        << " ticks overran " << period_ * 1e3 << " ms ("
        << static_cast<int>(
      100.0 * static_cast<double>(session_overruns_) /
      static_cast<double>(session_ticks_) + 0.5)
        << "%), worst tick " << session_worst_busy_ * 1e3 << " ms, skipped "
        << session_skipped_deadlines_ << " catch-up deadlines";
    return out.str();
  }

  void note_skipped_deadlines(uint64_t count)
  {
    session_skipped_deadlines_ += count;
  }

private:
  static constexpr double warn_fraction = 0.02;
  static constexpr size_t rearm_windows = 2;
  double period_;
  size_t window_;
  size_t ticks_ = 0;
  size_t overruns_ = 0;
  double worst_busy_ = 0.0;
  double worst_lateness_ = 0.0;
  size_t quiet_windows_ = rearm_windows;
  size_t session_ticks_ = 0;
  size_t session_overruns_ = 0;
  double session_worst_busy_ = 0.0;
  uint64_t session_skipped_deadlines_ = 0;
};

// Describe the alignment move and, when it is large enough to be startling,
// wait for the operator. Returns false if the operator interrupted instead.
bool announce_alignment(const Alignment & alignment, double confirm_deg, const char * when)
{
  if (!alignment.aligning()) {
    std::cout << when << ": follower already matches the leader (within 0.1 "
                 "deg); tracking immediately." << std::endl;
    return true;
  }
  const double largest_deg = alignment.largest_rad() * rad_to_deg;
  std::cout << when << ": follower is off the leader by " << largest_deg
            << " deg on joint " << alignment.largest_joint()
            << "; aligning over " << alignment.duration() << " s, then tracking"
            << " the leader's absolute joint angles." << std::endl;
  if (largest_deg < confirm_deg) {
    return true;
  }
  std::cout << "  The follower is about to MOVE " << largest_deg
            << " deg to meet the leader. Clear the workspace.\n"
            << "  Enter = start aligning, Ctrl+C = stop" << std::endl;
  while (true) {
    if (g_stop_signals.load() > 0) {return false;}
    struct pollfd stdin_poll = {STDIN_FILENO, POLLIN, 0};
    const int ready = poll(&stdin_poll, 1, 100);
    if (ready > 0 && (stdin_poll.revents & (POLLIN | POLLHUP)) != 0) {
      char input = '\0';
      const ssize_t count = read(STDIN_FILENO, &input, 1);
      if (count <= 0) {
        // Scripted run with no console: the move is speed-bounded and was
        // just announced, so proceed rather than stranding the launcher.
        std::cout << "  (no console attached; aligning)" << std::endl;
        return true;
      }
      if (input == '\n') {return true;}
    }
  }
}

// Flight-recorder binary format, consumed by analyze_log.py. All header
// fields are 8 bytes so the layout is padding-free. Each record is
// 5 + 6*num_joints doubles:
//   t_sched, t_wake, t_leader_read, t_follower_read, t_cmd  (s since start)
//   leader_pos[n], leader_vel[n],
//   follower_pos[n], follower_vel[n], follower_eff[n], target[n]
struct LogHeader
{
  char magic[8];  // "WXTLOG1\0"
  uint64_t num_joints;
  double period_s;
  double tau_s;
  double goal_time_s;
  double ff_gain;
  uint64_t abs_gripper;
  int64_t wall_start_ns;  // system_clock at loop start, for humans
};
static_assert(sizeof(LogHeader) == 64);

// Flight logs live outside the repo tree, on local disk. Records cross a
// bounded nonblocking pipe to a normal-priority writer; the 400 Hz loop never
// performs filesystem I/O.
std::string default_log_path()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
  // TATBOT_LOG_ROOT > XDG state dir; launchers export the resolved root
  // (scripts/lib/paths.sh), so the rig keeps ~/tatbot-logs. No CWD fallback:
  // an unset HOME lands in /tmp, visibly, not in a relative directory.
  const char * log_root = std::getenv("TATBOT_LOG_ROOT");
  const char * xdg = std::getenv("XDG_STATE_HOME");
  const char * home = std::getenv("HOME");
  std::string dir;
  if (log_root && *log_root) dir = std::string(log_root) + "/teleop/";
  else if (xdg && *xdg) dir = std::string(xdg) + "/tatbot/logs/teleop/";
  else if (home && *home) dir = std::string(home) + "/.local/state/tatbot/logs/teleop/";
  else dir = "/tmp/tatbot-teleop-logs/";
  return dir + "teleop_" + stamp + ".wxtl";
}

}  // namespace

int run(int argc, char ** argv)
{
  const Options opt = parse_args(argc, argv);
  if (opt.leader_ip.empty() || opt.follower_ip.empty()) {
    std::cerr << "wxai_teleop: no arm addresses. Pass them as positionals "
                 "(wxai_teleop <leader_ip> <follower_ip>) or run through the "
                 "tatbot CLI so the hardware profile exports TATBOT_LEADER_IP/"
                 "TATBOT_FOLLOWER_IP." << std::endl;
    return 2;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  // Before anything spawns a thread: the SDK's UDP daemon threads inherit both
  // the affinity mask and the scheduling policy from whoever creates them, so
  // the whole driver stack has to be placed here or not at all.
  if (opt.realtime) {
    const auto setup = tatbot::realtime::apply(opt.rt_priority);
    if (setup.affinity_applied) {
      std::cout << "Control loop pinned to CPUs "
                << tatbot::realtime::format_cpus(setup.cpus)
                << " (the machine's fastest cores)" << std::endl;
    } else {
      std::cerr << "WARNING: could not pin the control loop: "
                << setup.affinity_error << std::endl;
    }
    if (setup.fifo_applied) {
      std::cout << "Control loop scheduling: SCHED_FIFO priority "
                << opt.rt_priority << std::endl;
    } else {
      std::cerr << "ERROR: no real-time scheduling (" << setup.fifo_error
                << "). Refusing before either arm driver is constructed: a busy"
                   " machine can make the follower shake.\n"
                << "       Fix: install config/limits/99-tatbot-realtime.conf"
                   " and log in again. --no-rt is only an explicit bench opt-out."
                << std::endl;
      return 3;
    }
  }

  // Hardware e-stop monitor; started before the arms energize so a latched
  // button is caught at the startup gate below.
  std::unique_ptr<tatbot::estop::Monitor> estop;
  if (opt.estop_enabled) {
    estop = std::make_unique<tatbot::estop::Monitor>(
      opt.estop_dev, opt.estop_required, g_estop);
  }

  for (const auto & [label, ip] : {
      std::pair<std::string, std::string>{"leader", opt.leader_ip},
      std::pair<std::string, std::string>{"follower", opt.follower_ip}})
  {
    if (g_stop_signals.load() > 0) {
      std::cout << "Interrupted." << std::endl;
      return 0;
    }
    if (!arm_reachable(ip)) {
      std::cerr << "The " << label << " arm at " << ip
                << " is not reachable — is it powered on? "
                << "(arms take ~20 s to boot after power-on)" << std::endl;
      return 1;
    }
  }

  trossen_arm::TrossenArmDriver leader;
  trossen_arm::TrossenArmDriver follower;
  configure_arm(
    leader, opt.leader_ip, opt.leader_config, "leader",
    trossen_arm::StandardEndEffector::wxai_v0_leader);
  configure_arm(
    follower, opt.follower_ip, opt.follower_config, "follower",
    trossen_arm::StandardEndEffector::wxai_v0_follower);
  if (g_stop_signals.load() > 0) {
    std::cout << "Interrupted during startup; both arms left idle." << std::endl;
    return 0;
  }

  // Startup gate: a latched or faulted e-stop must be cleared before teleop
  // energizes the arms — twist-release the button (or fix the heartbeat).
  if (g_estop.load() > estop_ok) {
    std::cout << "E-stop engaged ("
              << (g_estop.load() == estop_pressed ? "button latched" : "no heartbeat")
              << ") — twist-release / reconnect to start, Ctrl+C to quit." << std::endl;
    while (g_estop.load() > estop_ok) {
      if (g_stop_signals.load() > 0) {
        std::cout << "Interrupted; both arms left idle." << std::endl;
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "E-stop clear." << std::endl;
  }

  const size_t num_joints = leader.get_num_joints();
  const size_t gripper = num_joints - 1;  // tool carriage (ex-gripper) is the last joint

  // Baselines for the alignment ramp: the follower starts at its own pose and
  // is walked onto the leader's absolute joint angles (see Alignment).
  // Recomputed on every resume, so a stop can never leave the two arms
  // permanently offset. Follower targets are clamped to its joint limits (band
  // widened to include its start pose) so the driver never faults when the
  // leader is guided somewhere the follower cannot reach.
  std::vector<double> leader_start;
  std::vector<double> follower_start;
  std::vector<double> pos_min(num_joints);
  std::vector<double> pos_max(num_joints);
  const auto joint_limits = follower.get_joint_limits();
  auto take_baselines = [&]() {
      leader_start = leader.get_all_positions();
      follower_start = follower.get_all_positions();
      for (size_t i = 0; i < num_joints; ++i) {
        pos_min[i] = std::min(joint_limits.at(i).position_min, follower_start[i]);
        pos_max[i] = std::max(joint_limits.at(i).position_max, follower_start[i]);
      }
    };
  take_baselines();

  Alignment alignment(opt.absolute, opt.align_rate);
  alignment.restart(leader_start, follower_start);

  // Flight recorder.
  std::unique_ptr<tatbot::flight::Recorder> log_file;
  if (opt.log_enabled) {
    std::string path = opt.log_path.empty() ? default_log_path() : opt.log_path;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    log_file = std::make_unique<tatbot::flight::Recorder>(path);
    std::cout << "Recording to " << path << std::endl;
  }

  std::unique_ptr<tatbot::telemetry::UdpPublisher> telemetry;
  if (!opt.telemetry_udp.empty()) {
    try {
      telemetry = std::make_unique<tatbot::telemetry::UdpPublisher>(
        opt.telemetry_udp, opt.telemetry_fps);
      std::cout << "Visualization telemetry -> " << telemetry->endpoint()
                << " at up to " << opt.telemetry_fps << " Hz (nonblocking)" << std::endl;
    } catch (const std::exception & error) {
      std::cerr << "WARNING: visualization telemetry disabled: " << error.what() << std::endl;
    }
  }

  const double dt = static_cast<double>(opt.period_us) * 1e-6;
  LoopHealth health(dt);
  bool emergency = false;
  bool handoff_holding = false;  // release in a probe/draw session: exit still holding
  const bool square_enabled = opt.square_probe || opt.spiral_probe || opt.draw_mode;
  const bool spiral_enabled = opt.spiral_probe;
  const bool draw_enabled = opt.draw_mode;
  // Seven-DOF ballpoint tip model: the spiral A/B opt-in, and every draw session.
  const bool carriage_ik = opt.spiral_carriage_ik || draw_enabled;
  const std::string probe_name = draw_enabled ? "draw" : (spiral_enabled ? "spiral" : "square");
  const size_t probe_segment_count = (spiral_enabled || draw_enabled) ? 1 : 4;
  bool square_finished = false;
  double carriage_preflight_worst_endpoint_error_m = 0.0;
  try {  // any driver throw below is caught to attempt a hold before idling

  // Leader: external_effort mode on all joints — zero effort is pure gravity
  // compensation; the follower's external efforts (contacts) are reflected
  // back scaled by -ff_gain. Follower: position mode on EVERY joint, the tool
  // carriage included — the carriage is a position axis owned by the safety
  // layer. Ordinary runs seat it at the rest stop before tracking begins; the
  // explicitly gated carriage-IK A/B run qualifies a small off-paper reversal
  // and leaves it at a 2 mm bias before the operator approaches the paper.
  std::vector<double> leader_efforts(num_joints, 0.0);
  leader.set_all_modes(trossen_arm::Mode::external_effort);
  leader.set_all_external_efforts(leader_efforts, 0.0, false);
  follower.set_all_modes(trossen_arm::Mode::position);

  // Carriage command helper: non-blocking, only ever called when the target
  // changes, so the 400 Hz stream to the arm joints is never joined by a
  // per-tick carriage command.
  double carriage_target = CARRIAGE_REST_M;
  auto command_carriage = [&](double metres, double goal_time_s) {
      carriage_target = metres;
      follower.set_joint_position(
        static_cast<uint8_t>(gripper), metres, goal_time_s, false);
    };
  // Seat the carriage at rest so every session starts from a known extension.
  // The carriage-IK candidate then performs a small, slow, off-paper reversal
  // witness before hand-guiding is enabled. Each endpoint must be measured
  // within 0.15 mm; a failed or stuck carriage refuses the paper trial.
  follower.set_joint_position(static_cast<uint8_t>(gripper), CARRIAGE_REST_M, 1.0, true);
  carriage_target = CARRIAGE_REST_M;
  if (carriage_ik) {
    std::cout << "CARRIAGE-IK OFF-PAPER PREFLIGHT: keep the pen clear. "
                 "Testing 2.0 -> 1.5 -> 2.5 -> 2.0 mm before hand-guiding."
              << std::endl;
    const std::array<std::pair<double, double>, 4> carriage_witness{{
      {tatbot::square::CARRIAGE_IK_BIAS_M, 2.0},
      {0.0015, 1.0},
      {0.0025, 1.0},
      {tatbot::square::CARRIAGE_IK_BIAS_M, 1.0}}};
    for (const auto & waypoint : carriage_witness) {
      follower.set_joint_position(
        static_cast<uint8_t>(gripper), waypoint.first, waypoint.second, true);
      const double measured_m = follower.get_all_positions().at(gripper);
      const double error_m = std::fabs(measured_m - waypoint.first);
      carriage_preflight_worst_endpoint_error_m = std::max(
        carriage_preflight_worst_endpoint_error_m, error_m);
      if (!std::isfinite(measured_m) ||
        error_m > CARRIAGE_IK_PREFLIGHT_ENDPOINT_TOLERANCE_M)
      {
        throw std::runtime_error(
                "carriage-IK off-paper preflight endpoint error is " +
                std::to_string(error_m * 1e3) + " mm (limit " +
                std::to_string(CARRIAGE_IK_PREFLIGHT_ENDPOINT_TOLERANCE_M * 1e3) +
                " mm); refusing hand-guiding and paper motion");
      }
    }
    carriage_target = tatbot::square::CARRIAGE_IK_BIAS_M;
    std::cout << "CARRIAGE-IK OFF-PAPER PREFLIGHT PASS: worst endpoint error "
              << carriage_preflight_worst_endpoint_error_m * 1e3
              << " mm; carriage holding at "
              << carriage_target * 1e3 << " mm. Hand-guide only after this line."
              << std::endl;
  }
  double contact_baseline = 0.0;
  {
    std::vector<double> samples;
    for (int i = 0; i < 40; ++i) {
      samples.push_back(follower.get_all_external_efforts()[gripper]);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::sort(samples.begin(), samples.end());
    contact_baseline = samples[samples.size() / 2];
  }

  std::cout << "Teleoperation running: hand-guide the leader, Ctrl+C to stop.\n"
            << "Fitted tool: "
            << (opt.ee_tool.empty() ? "none (--no-tool)" : opt.ee_tool)
            << (opt.tool_uncalibrated ? " (uncalibrated: this session is its touch-off)" : "") << "\n"
            << "Carriage: "
            << (carriage_ik ? "carriage-IK bias " : "rest ")
            << carriage_target * 1e3 << " mm, contact cap "
            << opt.contact_cap_n << " N -> retract " << opt.carriage_retract_m * 1e3
            << " mm on a trip" << std::endl;
  if (opt.damping > 0.0 || opt.assist > 0.0) {
    std::cout << "Anti-stiction: damping " << opt.damping
              << " Nm/(rad/s), assist " << opt.assist
              << " (deadband " << ASSIST_DEADBAND_NM << " Nm)" << std::endl;
  }
  if (opt.square_probe) {
    std::cout << "\nCARTESIAN SQUARE PROBE (one shot):\n"
              << "  1. Hand-guide the pen tip to light contact at the desired start point.\n"
              << "  2. Hold briefly until READY is printed.\n"
              << "  3. Tap SPACE once — no Enter.\n"
              << "The follower will trace a base-X/Y square with a preflighted smooth "
                 "joint-position trajectory; its first X and Y edges point toward the arm base: "
              << opt.square_probe_m * 1e3 << " mm per edge over "
              << opt.square_edge_s << " s per edge ("
              << opt.square_probe_m * 1e3 / opt.square_edge_s << " mm/s), then retract the pen.\n"
              << "Any E-stop, contact cap, measured-velocity or rolling-effort trip terminates the probe; "
                 "it never resumes scripted motion."
              << std::endl;
  }
  if (opt.spiral_probe) {
    std::cout << "\nCARTESIAN EXPANDING-SPIRAL PROBE (one shot):\n"
              << "  1. Hand-guide the pen tip to light contact at the desired spiral CENTER.\n"
              << "  2. Leave at least " << opt.spiral_radius_m * 1e3
              << " mm of clear paper in every base-X/Y direction.\n"
              << "  3. Hold briefly until READY is printed.\n"
              << "  4. Tap SPACE once — no Enter.\n"
              << "The follower will trace a continuous " << opt.spiral_turns
              << "-turn Archimedean spiral to " << opt.spiral_radius_m * 1e3
              << " mm radius over " << opt.spiral_duration_s
              << " s at approximately constant arc-length speed, with "
              << opt.spiral_ease_s << " s quintic speed eases at each end; trigger Z and "
                 "orientation remain fixed"
              << (carriage_ik ?
                "; the arm and carriage coordinate through the measured ballpoint-tip model "
                "inside a 0.5..3.5 mm carriage envelope" : "")
              << ", then the pen retracts.\n"
              << "Any E-stop, contact cap, measured-velocity or rolling-effort trip terminates the probe; "
                 "it never resumes scripted motion."
              << std::endl;
  }
  if (draw_enabled) {
    std::cout << "\nSURFACE-FIRST DRAW SESSION (docs/draw.md), dir " << opt.draw_dir << ":\n"
              << "  1. Hand-guide the pen tip to LIGHT contact at the design centre.\n"
              << "  2. Hold briefly until READY is printed, then tap SPACE once.\n"
              << "     The follower lifts to standoff and orbits the wrist cameras over the\n"
              << "     patch (autonomous, preflighted), holding still at each capture, then\n"
              << "     holds while the map and the path are compiled and shadowed in Rerun.\n"
              << "  3. Inspect the shadow. When READY prints again, tap SPACE once to draw.\n"
              << "Any refusal, timeout, E-stop, contact cap, measured-velocity or rolling-effort\n"
              << "trip retracts the pen and ends scripted motion for this process."
              << std::endl;
  }
  // The operator may move the leader while the announcement waits for Enter —
  // orienting it to the follower's rolled idle is the natural thing to do —
  // and an offset taken before that move would then be added to the leader's
  // NEW pose (2026-08-30: +103 deg on joint 5 became a target of pi, clamped,
  // and a 1.4 rad step tripped the velocity limit). So the baselines are
  // re-taken AFTER the confirmation, and if the announced move changed by more
  // than a degree the announcement repeats with the real number.
  auto confirm_and_rebaseline = [&](double confirm_deg, const char * when) -> bool {
      for (int attempt = 0; attempt < 5; ++attempt) {
        take_baselines();
        alignment.restart(leader_start, follower_start);
        const double announced = alignment.largest_rad();
        // On an interrupt here the loop below is skipped (a signal has
        // arrived) and the normal controlled-stop path holds both arms.
        if (!announce_alignment(alignment, confirm_deg, when)) {return false;}
        take_baselines();
        alignment.restart(leader_start, follower_start);
        if (std::abs(alignment.largest_rad() - announced) * rad_to_deg <= 1.0) {return true;}
        std::cout << "  The leader moved while waiting (offset now "
                  << alignment.largest_rad() * rad_to_deg << " deg); re-announcing." << std::endl;
      }
      return false;
    };
  if (opt.absolute) {
    if (!confirm_and_rebaseline(opt.align_confirm_deg, "Startup alignment")) {
      g_stop_signals.fetch_add(g_stop_signals.load() == 0 ? 1 : 0);
    }
  } else {
    std::cout << "Relative mapping (--relative): the follower keeps its "
              << alignment.largest_rad() * rad_to_deg
              << " deg startup offset from the leader all session." << std::endl;
  }

  const auto loop_period = std::chrono::microseconds(opt.period_us);
  const double alpha = dt / (opt.tau + dt);

  // Low-pass filter state, seeded with the current leader state.
  std::vector<double> pos_filt = leader.get_all_positions();
  std::vector<double> vel_filt(num_joints, 0.0);
  // Anti-stiction state: the leader's external-effort read (refreshed per tick
  // when --assist is on) and the low-passed, deadbanded assist term.
  std::vector<double> leader_eff(num_joints, 0.0);
  std::vector<double> assist_filt(num_joints, 0.0);
  const double assist_alpha = dt / (ASSIST_TAU_S + dt);

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  // t_wake below is measured from t0, so its wall-clock origin must be sampled
  // here too.  In particular, do not capture it before announce_alignment():
  // operator-confirmed startup alignment can take seconds, which used to shift
  // every flight-log sample earlier than camera observations by that duration.
  const int64_t wall_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  if (log_file) {
    LogHeader header{};
    std::memcpy(header.magic, "WXTLOG1", 8);
    header.num_joints = num_joints;
    header.period_s = static_cast<double>(opt.period_us) * 1e-6;
    header.tau_s = opt.tau;
    header.goal_time_s = opt.goal_time;
    header.ff_gain = opt.ff_gain;
    header.abs_gripper = 1;  // the carriage channel is absolute (safety-owned, never mirrored)
    header.wall_start_ns = wall_start_ns;
    if (!log_file->write_header(&header, sizeof(header))) {
      throw std::runtime_error("cannot queue the flight-log header");
    }
  }
  auto since_start = [&t0]() {
      return std::chrono::duration<double>(clock::now() - t0).count();
    };

  std::vector<double> record(5 + 6 * num_joints);
  // Snapshot into fixed-capacity storage. The SDK getters return references to
  // driver-owned vectors; assigning those references to a new vector allocated
  // five times on every tick.
  std::vector<double> leader_pos(num_joints);
  std::vector<double> leader_vel(num_joints);
  std::vector<double> follower_pos(num_joints);
  std::vector<double> follower_vel(num_joints);
  std::vector<double> follower_eff(num_joints);
  auto snapshot = [num_joints](
      const std::vector<double> & source, std::vector<double> & destination,
      const char * label) {
      if (source.size() != num_joints) {
        throw std::runtime_error(
                std::string(label) + " returned " + std::to_string(source.size()) +
                " joints; expected " + std::to_string(num_joints));
      }
      std::copy(source.begin(), source.end(), destination.begin());
    };
  std::vector<double> target(num_joints);
  std::vector<double> arm_target(num_joints - 1);
  std::vector<double> arm_vel(num_joints - 1);
  std::vector<double> full_vel(num_joints);
  std::vector<double> square_guard_vel(num_joints - 1);
  std::vector<double> square_guard_eff(num_joints - 1);
  uint64_t telemetry_sequence = 0;
  int stop_baseline = g_stop_signals.load();
  // Contact-cap debounce: consecutive ticks with |carriage effort| over the
  // cap. A trip retracts the pen and drops into the controlled-stop path.
  int contact_over_ticks = 0;
  bool contact_trip = false;
  // First streamed command of each (re)start must not step (see the loop).
  bool first_tick = true;
  bool stale_baseline = false;
  // Runaway latch for --damping/--assist; clears on each (re)start.
  bool antistiction_off = false;
  bool square_started = false;
  bool square_guard_trip = false;
  size_t square_edge = 0;
  size_t square_settled_ticks = 0;
  bool square_ready = false;
  tatbot::square::Pose square_start{};
  std::array<tatbot::square::Pose, 4> square_targets{};
  std::array<std::string, 4> square_directions{};
  std::vector<tatbot::square::Pose> square_measured;
  std::vector<double> square_errors_mm;
  struct SpiralTraceSample
  {
    double elapsed_s = 0.0;
    std::array<double, 3> reference{};
    std::array<double, 3> measured{};
    double target_carriage_m = 0.0;
    double measured_carriage_m = 0.0;
  };
  std::vector<SpiralTraceSample> spiral_trace;
  tatbot::square::JointPlan square_joint_plan;
  tatbot::square::CarriageJointPlan carriage_joint_plan;
  size_t square_plan_tick = 0;
  bool square_settling = false;
  double square_model_fk_error_mm = 0.0;
  tatbot::square::MotionGuard square_guard(
    SQUARE_VELOCITY_ABORT_RAD_S,
    SQUARE_OVERFORCE_ABORT_NM,
    SQUARE_OVERFORCE_WINDOW_S,
    SQUARE_OVERFORCE_FRACTION,
    SQUARE_OVERFORCE_MIN_SAMPLES);
  clock::time_point square_settle_started_at{};
  bool square_tracking_trip = false;
  enum class DrawStage { none, orbit, ready_draw, drawing };
  DrawStage draw_stage = DrawStage::none;
  size_t draw_capture_index = 0;
  bool draw_capture_pending = false;
  size_t draw_capture_settled_ticks = 0;
  size_t draw_capture_wait_ticks = 0;
  clock::time_point draw_capture_deadline{};
  tatbot::square::FullJointPose draw_hold_positions{};
  std::vector<std::pair<std::string, std::string>> draw_report;
  bool draw_refused = false;
  std::string draw_refusal;
  auto print_draw_report = [&]() {
      if (draw_report.empty()) {return;}
      std::cout << "  stage report:";
      for (const auto & [key, value] : draw_report) {std::cout << ' ' << key << '=' << value;}
      std::cout << std::endl;
    };
  SingleKeyInput square_key_input(square_enabled);
  if (square_enabled && !square_key_input.active()) {
    throw std::runtime_error("could not enable single-key SPACE input on this terminal");
  }
  while (true) {  // session loop: teleop until a stop, optionally resume
  auto next_tick = clock::now();
  contact_trip = false;
  contact_over_ticks = 0;
  first_tick = true;
  stale_baseline = false;
  antistiction_off = false;
  square_settled_ticks = 0;
  square_ready = false;
  while (g_stop_signals.load() == stop_baseline &&
    g_estop.load(std::memory_order_relaxed) <= estop_ok)
  {
    const double t_sched = std::chrono::duration<double>(next_tick - t0).count();
    const double t_wake = since_start();

    snapshot(leader.get_all_positions(), leader_pos, "leader positions");
    snapshot(leader.get_all_velocities(), leader_vel, "leader velocities");
    if (opt.assist > 0.0) {
      snapshot(leader.get_all_external_efforts(), leader_eff, "leader efforts");
    }
    const double t_leader_read = since_start();

    snapshot(follower.get_all_positions(), follower_pos, "follower positions");
    snapshot(follower.get_all_velocities(), follower_vel, "follower velocities");
    snapshot(follower.get_all_external_efforts(), follower_eff, "follower efforts");
    const double t_follower_read = since_start();

    // Keep the teleop filter warm while the operator positions the start. Once
    // the SPACE handoff switches to the native Cartesian controller, the
    // follower is no longer sent any leader-derived target.
    for (size_t i = 0; i < num_joints; ++i) {
      pos_filt[i] += alpha * (leader_pos[i] - pos_filt[i]);
      vel_filt[i] += alpha * (leader_vel[i] - vel_filt[i]);
    }
    if (!square_started) {
      // Absolute mapping: the follower arm goes where the leader's joints are,
      // plus whatever is left of the startup offset (zero once the ramp
      // finishes). The carriage is never derived from the leader trigger.
      const double residual = alignment.residual();
      for (size_t i = 0; i < gripper; ++i) {
        target[i] = std::clamp(
          pos_filt[i] + residual * alignment.offset(i),
          pos_min[i],
          pos_max[i]);
      }
      target[gripper] = carriage_target;
      // The first command after a (re)start must be the follower's own pose.
      if (first_tick) {
        first_tick = false;
        double worst = 0.0; size_t worst_joint = 0;
        for (size_t i = 0; i < gripper; ++i) {
          const double step = std::abs(target[i] - follower_pos[i]);
          if (step > worst) {worst = step; worst_joint = i;}
        }
        if (worst > FIRST_STEP_MAX_RAD) {
          std::cout << "\nREFUSED first command: joint " << worst_joint << " would step "
                    << worst * rad_to_deg << " deg at once (baselines stale — the leader "
                    << "moved after alignment was confirmed). Holding; press r to re-align."
                    << std::endl;
          stale_baseline = true;
          break;
        }
      }
      const bool was_aligning = alignment.aligning();
      alignment.advance(dt);
      if (was_aligning && !alignment.aligning()) {
        std::cout << "Aligned: follower now tracks the leader's absolute joint "
                     "angles." << std::endl;
      }
    } else {
      // The binary flight format only has a joint target field. Native
      // Cartesian goals have no truthful joint-space target to put there, so
      // log the measured joints; square_probe.csv records the actual Cartesian
      // targets and endpoints.
      target = follower_pos;
      target[gripper] = carriage_target;
    }
    // Contact force: the carriage effort's departure from its rest baseline,
    // assessed only at drawing speeds (see CONTACT_STILL_RAD_S) and debounced,
    // so neither noise nor an inertial swing can trip; a sustained hard press
    // retracts the pen and hands control to the stop path.
    const double contact = std::fabs(follower_eff[gripper] - contact_baseline);
    double arm_speed = 0.0;
    for (size_t i = 0; i < gripper; ++i) {arm_speed = std::max(arm_speed, std::fabs(follower_vel[i]));}
    const bool assessable = arm_speed < CONTACT_STILL_RAD_S;
    const double deflect = follower_pos[gripper] - carriage_target;  // + = pushed open
    const bool pushed = (assessable && contact > opt.contact_cap_n) || deflect > opt.contact_deflect_m;
    contact_over_ticks = pushed ? contact_over_ticks + 1 : 0;
    if (contact_over_ticks >= CONTACT_CAP_DEBOUNCE_TICKS) {
      command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
      std::cout << "\nCONTACT CAP: " << contact << " N over the rest baseline ("
                << contact_baseline << " N; raw " << follower_eff[gripper] << " N; cap "
                << opt.contact_cap_n << " N), carriage deflected " << deflect * 1e3
                << " mm (limit " << opt.contact_deflect_m * 1e3 << " mm) for "
                << CONTACT_CAP_DEBOUNCE_TICKS * dt * 1e3 << " ms — pen retracted "
                << opt.carriage_retract_m * 1e3 << " mm, arms holding" << std::endl;
      contact_trip = true;
      break;
    }

    if (square_enabled && !square_started) {
      double leader_speed = 0.0;
      double follower_speed = 0.0;
      for (size_t i = 0; i < gripper; ++i) {
        leader_speed = std::max(leader_speed, std::fabs(leader_vel[i]));
        follower_speed = std::max(follower_speed, std::fabs(follower_vel[i]));
      }
      const bool settled = !alignment.aligning() &&
        leader_speed <= SQUARE_SETTLED_RAD_S && follower_speed <= SQUARE_SETTLED_RAD_S;
      if (!square_ready) {
        square_settled_ticks = settled ? square_settled_ticks + 1 : 0;
        if (square_settled_ticks * dt >= SQUARE_SETTLED_S) {
          square_ready = true;
          std::cout << "\nREADY: tap SPACE once to start the " << probe_name << " (no Enter)."
                    << std::endl;
        }
      } else if (follower_speed > SQUARE_READY_RESET_RAD_S) {
        square_ready = false;
        square_settled_ticks = 0;
        std::cout << "\nReadiness reset: follower speed " << follower_speed
                  << " rad/s exceeded " << SQUARE_READY_RESET_RAD_S
                  << ". Hold briefly for READY again." << std::endl;
      }

      struct pollfd trigger_poll = {STDIN_FILENO, POLLIN, 0};
      if (poll(&trigger_poll, 1, 0) > 0 && (trigger_poll.revents & POLLIN) != 0) {
        char key = '\0';
        const ssize_t count = read(STDIN_FILENO, &key, 1);
        if (count <= 0) {
          std::cout << "\n" << probe_name
                    << " probe console closed; stopping without scripted motion."
                    << std::endl;
          break;
        }
        if (key != ' ') {
          std::cout << "Ignored key; wait for READY, then tap SPACE once." << std::endl;
        } else if (!square_ready) {
          std::cout << "REFUSED SPACE: not ready yet (leader " << leader_speed
                    << " rad/s, follower " << follower_speed << " rad/s; need both <= "
                    << SQUARE_SETTLED_RAD_S << " for " << SQUARE_SETTLED_S
                    << " s). Hold briefly for READY, then tap SPACE." << std::endl;
        } else {
          const auto controller_start = follower.get_cartesian_positions();
          snapshot(follower.get_all_positions(), follower_pos, "follower trigger positions");
          tatbot::square::JointPose start_joints{};
          std::copy(follower_pos.begin(), follower_pos.end() - 1, start_joints.begin());
          const auto model_start = tatbot::square::wxai_tcp_translation(start_joints);
          double model_error_squared = 0.0;
          for (size_t axis = 0; axis < 3; ++axis) {
            const double error = model_start[axis] - controller_start[axis];
            model_error_squared += error * error;
          }
          square_model_fk_error_mm = std::sqrt(model_error_squared) * 1000.0;
          if (square_model_fk_error_mm > SQUARE_MODEL_FK_TOLERANCE_M * 1e3) {
            throw std::runtime_error(
                    probe_name + " model/live FK mismatch is " +
                    std::to_string(square_model_fk_error_mm) + " mm (limit " +
                    std::to_string(SQUARE_MODEL_FK_TOLERANCE_M * 1e3) +
                    " mm); refusing scripted motion");
          }
          square_start = controller_start;
          if (carriage_ik) {
            if (std::fabs(follower_pos[gripper] - tatbot::square::CARRIAGE_IK_BIAS_M) >
              CARRIAGE_IK_PREFLIGHT_ENDPOINT_TOLERANCE_M)
            {
              throw std::runtime_error(
                      "carriage left its 2 mm bias before the trigger; refusing scripted motion");
            }
            const auto ballpoint_start = tatbot::square::wxai_ballpoint_tip_translation(
              start_joints, follower_pos[gripper]);
            for (size_t axis = 0; axis < 3; ++axis) {
              square_start[axis] = ballpoint_start[axis];
            }
          }
          if (spiral_enabled) {
            square_targets.fill(square_start);
            square_targets[0][0] += opt.spiral_radius_m;
            square_directions[0] = "expanding about the trigger center";
          } else if (draw_enabled) {
            square_targets.fill(square_start);
            square_directions[0] = "standoff orbit for the wrist cameras";
          } else {
            square_targets = tatbot::square::targets(square_start, opt.square_probe_m);
            const char x_sign = square_targets[0][0] < square_start[0] ? '-' : '+';
            const char y_sign = square_targets[1][1] < square_targets[0][1] ? '-' : '+';
            square_directions = {
              std::string(1, x_sign) + "base-X",
              std::string(1, y_sign) + "base-Y",
              std::string(1, x_sign == '+' ? '-' : '+') + "base-X",
              std::string(1, y_sign == '+' ? '-' : '+') + "base-Y"};
          }
          leader.set_all_modes(trossen_arm::Mode::position);
          leader.set_all_positions(leader_pos, 0.0, false);
          follower.set_arm_modes(trossen_arm::Mode::position);
          if (carriage_ik) {
            target = follower_pos;
            std::vector<double> full_velocity(num_joints, 0.0);
            follower.set_all_positions(target, 0.0, false, full_velocity);
            carriage_target = follower_pos[gripper];
            if (draw_enabled) {
              // Stage 1: record the contact pose, let the Python side plan the
              // camera orbit from it, and preflight that orbit here. Every
              // failure retracts and ends the session before any motion.
              const auto draw_dir = std::filesystem::path(opt.draw_dir);
              const auto trigger_rotation = tatbot::square::wxai_link6_rotation(start_joints);
              const std::array<double, 3> trigger_tip{
                square_start[0], square_start[1], square_start[2]};
              if (!write_draw_pose(
                  draw_dir / "trigger.json", start_joints, follower_pos[gripper], trigger_tip,
                  trigger_rotation, opt.ee_tool, dt))
              {
                draw_refused = true;
                draw_refusal = "could not write trigger.json";
              }
              if (!draw_refused) {
                std::cout << "\nDRAW STAGE orbit: planning the camera orbit (draw_stage.py); arms holding."
                          << std::endl;
                const int rc = run_draw_stage(opt.draw_dir, "orbit", 60.0);
                if (rc != 0) {
                  draw_refused = true;
                  draw_refusal = "orbit stage exit " + std::to_string(rc);
                }
              }
              if (!draw_refused) {
                try {
                  const auto orbit = tatbot::square::load_path_file(
                    (draw_dir / "orbit.csv").string(), dt);
                  carriage_joint_plan = tatbot::square::plan_joint_path(
                    start_joints, follower_pos[gripper], orbit.samples, dt,
                    orbit.start_tolerance_m, orbit.carriage_ik);
                  draw_report = orbit.report;
                  for (size_t axis = 0; axis < 3; ++axis) {
                    square_targets[0][axis] = orbit.samples.back().position[axis];
                  }
                  draw_stage = DrawStage::orbit;
                  draw_capture_index = 0;
                  draw_capture_pending = false;
                } catch (const std::exception & error) {
                  draw_refused = true;
                  draw_refusal = error.what();
                }
              }
              if (draw_refused) {
                command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
                std::cout << "\nDRAW REFUSED before motion: " << draw_refusal
                          << " — pen retracted, arms holding." << std::endl;
                break;
              }
              print_draw_report();
            } else {
              carriage_joint_plan = tatbot::square::plan_joint_spiral_with_carriage(
                start_joints, follower_pos[gripper], opt.spiral_radius_m,
                opt.spiral_turns, opt.spiral_duration_s, opt.spiral_ease_s, dt);
            }
          } else {
            std::copy(start_joints.begin(), start_joints.end(), arm_target.begin());
            std::fill(arm_vel.begin(), arm_vel.end(), 0.0);
            follower.set_arm_positions(arm_target, 0.0, false, arm_vel);
          }
          if (spiral_enabled) {
            if (!carriage_ik) {
              square_joint_plan = tatbot::square::plan_joint_spiral(
                start_joints, opt.spiral_radius_m, opt.spiral_turns,
                opt.spiral_duration_s, opt.spiral_ease_s, dt);
            }
          } else if (!draw_enabled) {
            square_joint_plan = tatbot::square::plan_joint_square(
              start_joints, square_targets, opt.square_edge_s, dt);
          }
          square_edge = 0;
          square_plan_tick = 0;
          square_settling = false;
          square_guard.reset();
          square_started = true;
          std::cout << "\nSCRIPTED MOTION START: " << probe_name << " center xyz=["
                    << square_start[0] << ", " << square_start[1] << ", " << square_start[2]
                    << "] m; ";
          if (draw_enabled) {
            std::cout << "camera orbit, " << carriage_joint_plan.positions.size()
                      << " ticks with " << carriage_joint_plan.capture_ticks.size()
                      << " captures; ";
          } else if (spiral_enabled) {
            std::cout << opt.spiral_turns << " turns to "
                      << opt.spiral_radius_m * 1e3 << " mm radius over "
                      << opt.spiral_duration_s << " s, constant arc-length speed with "
                      << opt.spiral_ease_s << " s endpoint eases; ";
          } else {
            std::cout << "inward sequence " << square_directions[0] << ", "
                      << square_directions[1] << ", " << square_directions[2] << ", "
                      << square_directions[3] << "; ";
          }
          std::cout << "joint plan preflight: live/model FK "
                    << square_model_fk_error_mm << " mm, peak joint speed ";
          if (carriage_ik) {
            std::cout << carriage_joint_plan.max_joint_velocity_rad_s
                      << " rad/s, carriage "
                      << carriage_joint_plan.min_carriage_m * 1e3 << ".."
                      << carriage_joint_plan.max_carriage_m * 1e3
                      << " mm, peak carriage speed "
                      << carriage_joint_plan.max_carriage_velocity_m_s * 1e3
                      << " mm/s, peak carriage acceleration "
                      << carriage_joint_plan.max_carriage_acceleration_m_s2 * 1e3
                      << " mm/s^2, peak tip speed "
                      << carriage_joint_plan.max_cartesian_velocity_m_s * 1e3
                      << " mm/s, model error "
                      << carriage_joint_plan.max_model_error_mm
                      << " mm, orientation error "
                      << carriage_joint_plan.max_orientation_error_rad;
          } else {
            std::cout << square_joint_plan.max_joint_velocity_rad_s
                      << " rad/s, peak TCP speed "
                      << square_joint_plan.max_cartesian_velocity_m_s * 1e3
                      << " mm/s, model error "
                      << square_joint_plan.max_model_error_mm << " mm, orientation error "
                      << square_joint_plan.max_orientation_error_rad;
          }
          std::cout << " rad; "
                    << (draw_enabled ? "orbit" : (spiral_enabled ? "continuous trace" : "edge 1/4"))
                    << " starting. "
                    << "E-stop operator: stay ready."
                    << std::endl;
        }
      }
    }

    double t_cmd = since_start();
    if (square_started) {
      std::copy(follower_vel.begin(), follower_vel.end() - 1, square_guard_vel.begin());
      std::copy(follower_eff.begin(), follower_eff.end() - 1, square_guard_eff.begin());
      if (const auto trip = square_guard.observe(t_wake, square_guard_vel, square_guard_eff)) {
        command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
        std::cout << "\n" << probe_name << " SAFETY ABORT [" << trip->code << "]: joint "
                  << trip->joint << " value " << trip->value << " exceeded/violated "
                  << trip->limit << " — pen retracted, arms holding; scripted motion is terminal"
                  << std::endl;
        square_guard_trip = true;
        break;
      }

      auto command_square_joints = [&](
        const tatbot::square::JointPose & positions,
        const tatbot::square::JointPose & velocities) -> bool {
          double worst_lead = 0.0;
          size_t worst_joint = 0;
          for (size_t joint = 0; joint < positions.size(); ++joint) {
            const double lead = std::fabs(positions[joint] - follower_pos[joint]);
            if (lead > worst_lead) {worst_lead = lead; worst_joint = joint;}
          }
          if (worst_lead > SQUARE_COMMAND_LEAD_ABORT_RAD) {
            command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
            square_tracking_trip = true;
            std::cout << "\n" << probe_name << " TRACKING ABORT: joint " << worst_joint
                      << " command lead " << worst_lead << " rad exceeded "
                      << SQUARE_COMMAND_LEAD_ABORT_RAD
                      << " rad — pen retracted, arms holding; scripted motion is terminal"
                      << std::endl;
            return false;
          }
          std::copy(positions.begin(), positions.end(), arm_target.begin());
          std::copy(velocities.begin(), velocities.end(), arm_vel.begin());
          std::copy(positions.begin(), positions.end(), target.begin());
          target[gripper] = carriage_target;
          follower.set_arm_positions(arm_target, 0.0, false, arm_vel);
          return true;
        };
      auto command_carriage_joints = [&](
        const tatbot::square::FullJointPose & positions,
        const tatbot::square::FullJointPose & velocities) -> bool {
          double worst_arm_lead = 0.0;
          size_t worst_joint = 0;
          for (size_t joint = 0; joint < gripper; ++joint) {
            const double lead = std::fabs(positions[joint] - follower_pos[joint]);
            if (lead > worst_arm_lead) {worst_arm_lead = lead; worst_joint = joint;}
          }
          const double carriage_lead = std::fabs(positions[gripper] - follower_pos[gripper]);
          if (worst_arm_lead > SQUARE_COMMAND_LEAD_ABORT_RAD ||
            carriage_lead > CARRIAGE_IK_COMMAND_LEAD_ABORT_M)
          {
            command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
            square_tracking_trip = true;
            if (carriage_lead > CARRIAGE_IK_COMMAND_LEAD_ABORT_M) {
              std::cout << "\nspiral TRACKING ABORT: carriage command lead "
                        << carriage_lead * 1e3 << " mm exceeded "
                        << CARRIAGE_IK_COMMAND_LEAD_ABORT_M * 1e3 << " mm";
            } else {
              std::cout << "\nspiral TRACKING ABORT: joint " << worst_joint
                        << " command lead " << worst_arm_lead << " rad exceeded "
                        << SQUARE_COMMAND_LEAD_ABORT_RAD << " rad";
            }
            std::cout << " — pen retracted, arms holding; scripted motion is terminal"
                      << std::endl;
            return false;
          }
          std::copy(positions.begin(), positions.end(), target.begin());
          std::copy(velocities.begin(), velocities.end(), full_vel.begin());
          carriage_target = positions[gripper];
          follower.set_all_positions(target, 0.0, false, full_vel);
          return true;
        };

      bool draw_waiting = false;
      if (draw_enabled && draw_stage == DrawStage::ready_draw) {
        // Between the map and the draw: hold the last orbit pose, latch
        // readiness the same way the trigger did, and wait for SPACE 2.
        draw_waiting = true;
        if (!command_carriage_joints(draw_hold_positions, tatbot::square::FullJointPose{})) {break;}
        double follower_speed = 0.0;
        for (size_t joint = 0; joint < gripper; ++joint) {
          follower_speed = std::max(follower_speed, std::fabs(follower_vel[joint]));
        }
        if (!square_ready) {
          square_settled_ticks = follower_speed <= SQUARE_SETTLED_RAD_S ? square_settled_ticks + 1 : 0;
          if (square_settled_ticks * dt >= SQUARE_SETTLED_S) {
            square_ready = true;
            std::cout << "\nREADY: inspect the shadow, then tap SPACE once to draw (no Enter); "
                         "Ctrl+C ends here with the pen retracted."
                      << std::endl;
          }
        } else if (follower_speed > SQUARE_READY_RESET_RAD_S) {
          square_ready = false;
          square_settled_ticks = 0;
        }
        struct pollfd draw_poll = {STDIN_FILENO, POLLIN, 0};
        if (poll(&draw_poll, 1, 0) > 0 && (draw_poll.revents & POLLIN) != 0) {
          char key = '\0';
          const ssize_t count = read(STDIN_FILENO, &key, 1);
          if (count <= 0) {
            std::cout << "\ndraw console closed; stopping before the path." << std::endl;
            break;
          }
          if (key != ' ') {
            std::cout << "Ignored key; wait for READY, then tap SPACE once to draw." << std::endl;
          } else if (!square_ready) {
            std::cout << "REFUSED SPACE: follower not settled yet; hold for READY." << std::endl;
          } else {
            square_plan_tick = 0;
            square_settling = false;
            square_guard.reset();
            draw_stage = DrawStage::drawing;
            std::cout << "\nSCRIPTED MOTION START: draw path, "
                      << carriage_joint_plan.positions.size() << " ticks ("
                      << carriage_joint_plan.positions.size() * dt << " s), peak tip speed "
                      << carriage_joint_plan.max_cartesian_velocity_m_s * 1e3
                      << " mm/s, carriage " << carriage_joint_plan.min_carriage_m * 1e3 << ".."
                      << carriage_joint_plan.max_carriage_m * 1e3
                      << " mm. E-stop operator: stay ready." << std::endl;
          }
        }
      }
      if (draw_waiting) {
        // holding for SPACE 2; nothing else to command this tick
      } else if (square_settling) {
        auto measured = follower.get_cartesian_positions();
        if (carriage_ik) {
          tatbot::square::JointPose measured_joints{};
          std::copy(follower_pos.begin(), follower_pos.end() - 1, measured_joints.begin());
          const auto measured_tip = tatbot::square::wxai_ballpoint_tip_translation(
            measured_joints, follower_pos[gripper]);
          for (size_t axis = 0; axis < 3; ++axis) {measured[axis] = measured_tip[axis];}
        }
        const double error_mm = tatbot::square::translation_error_mm(
          measured, square_targets[square_edge]);
        double follower_speed = 0.0;
        for (size_t joint = 0; joint < gripper; ++joint) {
          follower_speed = std::max(follower_speed, std::fabs(follower_vel[joint]));
        }
        if (error_mm <= SQUARE_ENDPOINT_TOLERANCE_M * 1e3 &&
          follower_speed <= SQUARE_SETTLED_RAD_S)
        {
          square_measured.push_back(measured);
          square_errors_mm.push_back(error_mm);
          std::cout << (draw_enabled ?
            (draw_stage == DrawStage::orbit ? std::string("Orbit endpoint") : std::string("Draw endpoint")) :
            spiral_enabled ? std::string("Spiral endpoint") :
            "Square edge " + std::to_string(square_edge + 1) + "/4 endpoint")
                    << " FK error: "
                    << error_mm << " mm (controller-reported, not an independent ink measurement)"
                    << std::endl;
          ++square_edge;
          square_settling = false;
          if (draw_enabled && draw_stage == DrawStage::orbit) {
            // Stage 2: the orbit is complete and the arm is settled at its
            // last pose. Record it, let the Python side fuse the captures,
            // anchor, compile and preflight, then load the path and hold for
            // SPACE 2. Every failure retracts and ends the session.
            const auto draw_dir = std::filesystem::path(opt.draw_dir);
            tatbot::square::JointPose hold_joints{};
            std::copy(follower_pos.begin(), follower_pos.end() - 1, hold_joints.begin());
            std::copy(follower_pos.begin(), follower_pos.end(), draw_hold_positions.begin());
            const auto hold_tip = tatbot::square::wxai_ballpoint_tip_translation(
              hold_joints, follower_pos[gripper]);
            const auto hold_rotation = tatbot::square::wxai_link6_rotation(hold_joints);
            if (!write_draw_pose(
                draw_dir / "hold.json", hold_joints, follower_pos[gripper], hold_tip,
                hold_rotation, opt.ee_tool, dt))
            {
              draw_refused = true;
              draw_refusal = "could not write hold.json";
            }
            if (!draw_refused) {
              std::cout << "\nDRAW STAGE map: fusing the captures, anchoring, compiling and "
                           "preflighting the path (draw_stage.py); arms holding."
                        << std::endl;
              const int rc = run_draw_stage(opt.draw_dir, "map", 300.0);
              if (rc == 3) {
                draw_refused = true;
                draw_refusal = "preflight refused the path (see preflight.json in the draw dir)";
              } else if (rc != 0) {
                draw_refused = true;
                draw_refusal = "map stage exit " + std::to_string(rc);
              }
            }
            if (!draw_refused && !std::filesystem::exists(draw_dir / "path.csv")) {
              // scan-only session: the map is the deliverable.
              command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
              square_finished = true;
              std::cout << "draw SCAN COMPLETE: surface mapped and shadowed, no path requested; "
                           "pen retracted, arms holding." << std::endl;
              break;
            }
            if (!draw_refused) {
              try {
                const auto path = tatbot::square::load_path_file(
                  (draw_dir / "path.csv").string(), dt);
                carriage_joint_plan = tatbot::square::plan_joint_path(
                  hold_joints, follower_pos[gripper], path.samples, dt, path.start_tolerance_m,
                  path.carriage_ik);
                draw_report = path.report;
                for (size_t axis = 0; axis < 3; ++axis) {
                  square_targets[0][axis] = path.samples.back().position[axis];
                }
              } catch (const std::exception & error) {
                draw_refused = true;
                draw_refusal = error.what();
              }
            }
            if (draw_refused) {
              command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
              std::cout << "\nDRAW REFUSED after the orbit: " << draw_refusal
                        << " — pen retracted, arms holding." << std::endl;
              break;
            }
            square_edge = 0;
            square_ready = false;
            square_settled_ticks = 0;
            draw_stage = DrawStage::ready_draw;
            std::cout << "\nPATH PREFLIGHT PASS: " << carriage_joint_plan.positions.size()
                      << " ticks, model error " << carriage_joint_plan.max_model_error_mm
                      << " mm, orientation error " << carriage_joint_plan.max_orientation_error_rad
                      << " rad, peak joint speed " << carriage_joint_plan.max_joint_velocity_rad_s
                      << " rad/s, carriage " << carriage_joint_plan.min_carriage_m * 1e3 << ".."
                      << carriage_joint_plan.max_carriage_m * 1e3 << " mm." << std::endl;
            print_draw_report();
            std::cout << "Holding at the standoff. Inspect the shadow in Rerun; SPACE draws, "
                         "Ctrl+C ends here." << std::endl;
          } else if (square_edge == probe_segment_count) {
            command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
            square_finished = true;
            std::cout << probe_name << " COMPLETE: pen retracted "
                      << opt.carriage_retract_m * 1e3
                      << " mm; arms holding. Measure the ink on paper for physical accuracy."
                      << std::endl;
            break;
          }
          if (!draw_enabled) {
            std::cout << "Starting edge " << square_edge + 1 << "/4 -> "
                      << square_directions[square_edge] << std::endl;
          }
        } else if (std::chrono::duration<double>(
          clock::now() - square_settle_started_at).count() >= SQUARE_ENDPOINT_SETTLE_MAX_S)
        {
          square_measured.push_back(measured);
          square_errors_mm.push_back(error_mm);
          command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
          square_tracking_trip = true;
          std::cout << "\n" << probe_name << " TRACKING ABORT: "
                    << (spiral_enabled ? "endpoint" :
                      "edge " + std::to_string(square_edge + 1))
                    << " remained " << error_mm << " mm from its endpoint after "
                    << SQUARE_ENDPOINT_SETTLE_MAX_S << " s of settling (limit "
                    << SQUARE_ENDPOINT_TOLERANCE_M * 1e3
                    << " mm) — pen retracted, arms holding; scripted motion is terminal"
                    << std::endl;
          break;
        } else {
          if (carriage_ik) {
            const auto & endpoint = carriage_joint_plan.positions[
              carriage_joint_plan.endpoint_tick - 1];
            if (!command_carriage_joints(endpoint, tatbot::square::FullJointPose{})) {break;}
          } else {
            const auto & endpoint = square_joint_plan.positions[
              square_joint_plan.edge_end_ticks[square_edge] - 1];
            if (!command_square_joints(endpoint, tatbot::square::JointPose{})) {break;}
          }
        }
      } else {
        const size_t plan_size = carriage_ik ?
          carriage_joint_plan.positions.size() : square_joint_plan.positions.size();
        if (square_plan_tick >= plan_size) {
          throw std::runtime_error("square joint plan exhausted before completion");
        }
        bool draw_capture_hold = false;
        if (draw_enabled && draw_stage == DrawStage::orbit &&
          draw_capture_index < carriage_joint_plan.capture_ticks.size() &&
          carriage_joint_plan.capture_ticks[draw_capture_index].first == square_plan_tick)
        {
          // A capture row: hold this sample until the wrist-camera capture
          // lands (or the deadline passes), then advance as normal.
          const size_t k = carriage_joint_plan.capture_ticks[draw_capture_index].second;
          const auto capture_dir = std::filesystem::path(opt.draw_dir) / "capture";
          const auto done = capture_dir / ("capture-" + std::to_string(k) + ".done");
          if (draw_capture_pending && std::filesystem::exists(done)) {
            draw_capture_pending = false;
            ++draw_capture_index;
            std::cout << "capture " << k << "/" << carriage_joint_plan.capture_ticks.size()
                      << " landed; orbit continues." << std::endl;
          } else {
            if (!command_carriage_joints(
                carriage_joint_plan.positions[square_plan_tick], tatbot::square::FullJointPose{}))
            {
              break;
            }
            // The reference stopped a hold ago, but the arm settles with a
            // visible bounce (operator, first session): request the capture
            // only once the measured joints have been still for a while, with
            // a bounded wait so a noisy encoder cannot stall the orbit.
            if (!draw_capture_pending) {
              draw_capture_settled_ticks =
                arm_speed <= DRAW_CAPTURE_SETTLED_RAD_S ? draw_capture_settled_ticks + 1 : 0;
              ++draw_capture_wait_ticks;
              const bool settled = draw_capture_settled_ticks * dt >= DRAW_CAPTURE_SETTLED_S;
              const bool waited_out = draw_capture_wait_ticks * dt >= DRAW_CAPTURE_SETTLE_MAX_S;
              if (!settled && !waited_out) {
                draw_capture_hold = true;
              } else {
                if (waited_out && !settled) {
                  std::cout << "capture " << k << ": arm still moving " << arm_speed
                            << " rad/s after " << DRAW_CAPTURE_SETTLE_MAX_S
                            << " s; capturing anyway." << std::endl;
                }
                draw_capture_settled_ticks = 0;
                draw_capture_wait_ticks = 0;
              }
            }
            if (draw_capture_hold) {
              // still settling; the plan does not advance this tick
            } else if (!draw_capture_pending) {
              if (!write_capture_request(
                  capture_dir / ("request-" + std::to_string(k) + ".json"), k, follower_pos))
              {
                command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
                square_tracking_trip = true;
                std::cout << "\ndraw ABORT: could not write the capture request — pen retracted, "
                             "arms holding; scripted motion is terminal" << std::endl;
                break;
              }
              draw_capture_pending = true;
              draw_capture_deadline = clock::now() + std::chrono::seconds(15);
              std::cout << "capture " << k << "/" << carriage_joint_plan.capture_ticks.size()
                        << " requested; holding still." << std::endl;
            } else if (clock::now() > draw_capture_deadline) {
              command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
              square_tracking_trip = true;
              std::cout << "\ndraw ABORT: capture " << k << " did not land within 15 s — pen "
                           "retracted, arms holding; scripted motion is terminal" << std::endl;
              break;
            }
            draw_capture_hold = true;
          }
        }
        if (draw_capture_hold) {
          // holding for the capture; the plan does not advance this tick
        } else {
        if (carriage_ik) {
          if (!command_carriage_joints(
            carriage_joint_plan.positions[square_plan_tick],
            carriage_joint_plan.velocities[square_plan_tick]))
          {
            break;
          }
        } else {
          if (!command_square_joints(
            square_joint_plan.positions[square_plan_tick],
            square_joint_plan.velocities[square_plan_tick]))
          {
            break;
          }
        }
        if ((spiral_enabled || (draw_enabled && draw_stage == DrawStage::drawing)) &&
          square_plan_tick % 40 == 0)
        {
          tatbot::square::JointPose measured_joints{};
          std::copy(follower_pos.begin(), follower_pos.end() - 1, measured_joints.begin());
          const auto measured_tip = carriage_ik ?
            tatbot::square::wxai_ballpoint_tip_translation(
              measured_joints, follower_pos[gripper]) :
            tatbot::square::wxai_tcp_translation(measured_joints);
          const auto & reference = carriage_ik ?
            carriage_joint_plan.cartesian_references[square_plan_tick] :
            square_joint_plan.cartesian_references[square_plan_tick];
          spiral_trace.push_back(SpiralTraceSample{
            (static_cast<double>(square_plan_tick) + 1.0) * dt,
            reference,
            measured_tip,
            carriage_target,
            follower_pos[gripper]});
        }
        ++square_plan_tick;
        const size_t endpoint_tick = carriage_ik ?
          carriage_joint_plan.endpoint_tick : square_joint_plan.edge_end_ticks[square_edge];
        if (square_plan_tick == endpoint_tick) {
          square_settling = true;
          square_settle_started_at = clock::now();
        }
        }  // not holding for a capture
      }
      t_cmd = since_start();
    } else {
      // Stream the arm target with the filtered leader velocity as feedforward
      // and a short interpolation horizon. The carriage remains safety-owned.
      std::copy(target.begin(), target.end() - 1, arm_target.begin());
      std::copy(vel_filt.begin(), vel_filt.end() - 1, arm_vel.begin());
      follower.set_arm_positions(arm_target, opt.goal_time, false, arm_vel);

      if (opt.ff_gain > 0.0 || opt.damping > 0.0 || opt.assist > 0.0) {
        if (!antistiction_off && (opt.damping > 0.0 || opt.assist > 0.0)) {
          double worst = 0.0;
          for (size_t i = 0; i < gripper; ++i) {
            worst = std::max(worst, std::fabs(leader_vel[i]));
          }
          if (worst > ANTISTICTION_RUNAWAY_RAD_S) {
            antistiction_off = true;
            std::cout << "\nANTI-STICTION OFF: a leader joint hit " << worst
                      << " rad/s — damping/assist disabled until resume." << std::endl;
          }
        }
        for (size_t i = 0; i < num_joints; ++i) {
          leader_efforts[i] = -opt.ff_gain * follower_eff[i];
          if (antistiction_off || i >= ANTISTICTION_JOINTS) {continue;}
          if (opt.damping > 0.0) {
            leader_efforts[i] += std::clamp(
              opt.damping * leader_vel[i], -DAMPING_CAP_NM, DAMPING_CAP_NM);
          }
          if (opt.assist > 0.0) {
            const double e = leader_eff[i];
            const double db = (std::fabs(e) > ASSIST_DEADBAND_NM) ?
              e - std::copysign(ASSIST_DEADBAND_NM, e) : 0.0;
            assist_filt[i] += assist_alpha * (db - assist_filt[i]);
            leader_efforts[i] += std::clamp(
              -opt.assist * assist_filt[i], -ASSIST_CAP_NM, ASSIST_CAP_NM);
          }
        }
      }
      leader.set_all_external_efforts(leader_efforts, 0.0, false);
      t_cmd = since_start();
    }

    if (telemetry) {
      telemetry->publish(
        wall_start_ns + static_cast<int64_t>(t_wake * 1e9),
        telemetry_sequence++, leader_pos, follower_pos, target, follower_eff);
    }

    if (log_file) {
      double * r = record.data();
      *r++ = t_sched; *r++ = t_wake; *r++ = t_leader_read;
      *r++ = t_follower_read; *r++ = t_cmd;
      const std::vector<double> * const parts[] =
      {&leader_pos, &leader_vel, &follower_pos, &follower_vel, &follower_eff, &target};
      for (const std::vector<double> * v : parts) {
        r = std::copy(v->begin(), v->end(), r);
      }
      log_file->append_record(record.data(), record.size() * sizeof(double));
    }

    const double t_tick_done = since_start();
    const std::string warning = health.tick(t_tick_done - t_wake, t_wake - t_sched);
    if (!warning.empty()) {
      std::cerr << warning << std::endl;
    }

    const uint64_t skipped = tatbot::realtime::advance_deadline(
      next_tick, loop_period, clock::now());
    health.note_skipped_deadlines(skipped);
    std::this_thread::sleep_until(next_tick);
  }

  // Stop requested (Ctrl+C, e-stop, or a contact trip). On an e-stop the pen
  // is retracted FIRST, then everything freezes at the ACTUAL positions
  // (snapping targets to measured zeroes any residual position error). A
  // plain controlled stop leaves the carriage where it is; a contact trip or
  // any square-probe exit retracts it. The carriage holds its commanded
  // position through any pause; only idling releases it.
  const bool estop_triggered = g_estop.load() > estop_ok;
  if ((estop_triggered || square_enabled) && carriage_target != opt.carriage_retract_m) {
    command_carriage(opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S);
  }
  leader.set_all_modes(trossen_arm::Mode::position);
  leader.set_all_positions(leader.get_all_positions(), 0.0, false);
  auto freeze_follower_arm = [&]() {
      follower.set_arm_modes(trossen_arm::Mode::position);
      const std::vector<double> follower_now = follower.get_all_positions();
      std::copy(follower_now.begin(), follower_now.end() - 1, arm_target.begin());
      follower.set_arm_positions(arm_target, 0.0, false);
  };
  freeze_follower_arm();

  if (square_enabled) {
    if (const char * run_dir = std::getenv("TATBOT_RUN_DIR"); run_dir && *run_dir) {
      const std::filesystem::path report_path =
        std::filesystem::path(run_dir) /
        (draw_enabled ? "draw_probe.csv" : (spiral_enabled ? "spiral_probe.csv" : "square_probe.csv"));
      std::ofstream report(report_path);
      if (report) {
        report << std::setprecision(12);
        const double report_max_model_error_mm = carriage_ik ?
          carriage_joint_plan.max_model_error_mm : square_joint_plan.max_model_error_mm;
        const double report_max_orientation_error_rad = carriage_ik ?
          carriage_joint_plan.max_orientation_error_rad :
          square_joint_plan.max_orientation_error_rad;
        const double report_max_joint_velocity_rad_s = carriage_ik ?
          carriage_joint_plan.max_joint_velocity_rad_s :
          square_joint_plan.max_joint_velocity_rad_s;
        const double report_max_cartesian_velocity_m_s = carriage_ik ?
          carriage_joint_plan.max_cartesian_velocity_m_s :
          square_joint_plan.max_cartesian_velocity_m_s;
        report << "status,"
               << (square_finished ? "complete" : (square_started ? "aborted" : "not_started"))
               << "\ncontroller,"
               << (carriage_ik ?
          "preflighted_seven_joint_ballpoint_dls" : "preflighted_joint_position_dls")
               << "\nmodel_live_fk_error_mm," << square_model_fk_error_mm
               << "\nmodel_max_error_mm," << report_max_model_error_mm
               << "\nmodel_max_orientation_error_rad,"
               << report_max_orientation_error_rad
               << "\nplan_max_joint_velocity_rad_s,"
               << report_max_joint_velocity_rad_s
               << "\nplan_max_cartesian_velocity_mm_s,"
               << report_max_cartesian_velocity_m_s * 1e3
               << "\ncommand_lead_limit_rad," << SQUARE_COMMAND_LEAD_ABORT_RAD
               << "\nendpoint_tolerance_mm," << SQUARE_ENDPOINT_TOLERANCE_M * 1e3 << "\n";
        if (spiral_enabled || draw_enabled) {
          if (draw_enabled) {
            report << "draw_dir," << opt.draw_dir << "\nfinal_stage,"
                   << (draw_stage == DrawStage::drawing ? "drawing" :
              draw_stage == DrawStage::ready_draw ? "ready_draw" :
              draw_stage == DrawStage::orbit ? "orbit" : "none")
                   << "\ncaptures_landed," << draw_capture_index << "\n";
            for (const auto & [key, value] : draw_report) {
              report << "path_" << key << ',' << value << "\n";
            }
          }
          if (spiral_enabled) {
          report << "radius_mm," << opt.spiral_radius_m * 1e3
                 << "\nturns," << opt.spiral_turns
                 << "\nduration_s," << opt.spiral_duration_s
                 << "\nease_s," << opt.spiral_ease_s
                 << "\npath_length_mm,"
                 << (carriage_ik ? carriage_joint_plan.path_length_m :
            square_joint_plan.path_length_m) * 1e3 << "\n"
                 << "carriage_ik," << (carriage_ik ? 1 : 0) << "\n";
          }
          if (carriage_ik) {
            report << "plan_min_carriage_mm," << carriage_joint_plan.min_carriage_m * 1e3
                   << "\nplan_max_carriage_mm," << carriage_joint_plan.max_carriage_m * 1e3
                   << "\nplan_max_carriage_velocity_mm_s,"
                   << carriage_joint_plan.max_carriage_velocity_m_s * 1e3
                   << "\nplan_max_carriage_acceleration_mm_s2,"
                   << carriage_joint_plan.max_carriage_acceleration_m_s2 * 1e3
                   << "\ncarriage_command_lead_limit_mm,"
                   << CARRIAGE_IK_COMMAND_LEAD_ABORT_M * 1e3
                   << "\noffpaper_preflight_worst_endpoint_error_mm,"
                   << carriage_preflight_worst_endpoint_error_m * 1e3 << "\n";
          }
          report << "elapsed_s,target_x_m,target_y_m,target_z_m,measured_x_m,measured_y_m,measured_z_m,error_x_mm,error_y_mm,error_z_mm,target_radius_mm,measured_radius_mm,target_carriage_mm,measured_carriage_mm,carriage_error_mm\n";
          for (const auto & sample : spiral_trace) {
            report << sample.elapsed_s;
            for (double value : sample.reference) {report << ',' << value;}
            for (double value : sample.measured) {report << ',' << value;}
            for (size_t axis = 0; axis < 3; ++axis) {
              report << ',' << (sample.measured[axis] - sample.reference[axis]) * 1e3;
            }
            const double target_dx = sample.reference[0] - square_start[0];
            const double target_dy = sample.reference[1] - square_start[1];
            const double measured_dx = sample.measured[0] - square_start[0];
            const double measured_dy = sample.measured[1] - square_start[1];
            report << ',' << std::hypot(target_dx, target_dy) * 1e3
                   << ',' << std::hypot(measured_dx, measured_dy) * 1e3
                   << ',' << sample.target_carriage_m * 1e3
                   << ',' << sample.measured_carriage_m * 1e3
                   << ',' << (sample.measured_carriage_m - sample.target_carriage_m) * 1e3
                   << '\n';
          }
          if (!square_errors_mm.empty()) {
            report << "endpoint_fk_error_mm," << square_errors_mm.back() << '\n';
          }
        } else {
          report << "side_mm," << opt.square_probe_m * 1e3
                 << "\nedge_s," << opt.square_edge_s << "\n";
          report << "edge,target_x_m,target_y_m,target_z_m,measured_x_m,measured_y_m,measured_z_m,fk_error_mm\n";
          for (size_t i = 0; i < square_measured.size(); ++i) {
            report << i + 1;
            for (size_t axis = 0; axis < 3; ++axis) {report << ',' << square_targets[i][axis];}
            for (size_t axis = 0; axis < 3; ++axis) {report << ',' << square_measured[i][axis];}
            report << ',' << square_errors_mm[i] << '\n';
          }
        }
        std::cout << (draw_enabled ? "Draw" : spiral_enabled ? "Spiral" : "Square")
                  << " probe report: " << report_path
                  << " (FK/encoder evidence only; measure the physical ink separately)"
                  << std::endl;
      } else {
        std::cerr << "WARNING: could not write " << probe_name
                  << "_probe.csv under " << run_dir << std::endl;
      }
    }
  }

  // E-stop flow: HOLD while the button is latched or its heartbeat is absent,
  // then automatically re-baseline and resume when the input is healthy.
  auto run_estop_flow = [&](bool resume_allowed) -> StopChoice {
      std::cout << "\nE-STOP: "
                << (g_estop.load() == estop_fault ?
        "heartbeat lost (device unplugged or dead?)" : "button pressed")
                << " — pen retracted, arms holding.\n"
                << (resume_allowed ?
        "  twist-release / reconnect = automatically resume tracking\n" :
        "  scripted probe is TERMINATED; clear the E-stop leaves both arms holding\n")
                << "  Ctrl+C = EMERGENCY RELEASE, idle immediately (arms fall)"
                << std::endl;
      const int signals_at_hold = g_stop_signals.load();
      const auto result = tatbot::estop::wait_for_clear(
        g_estop, g_stop_signals, signals_at_hold);
      if (result == tatbot::estop::WaitResult::emergency) {
        return StopChoice::emergency;
      }
      std::cout << (resume_allowed ?
        "\nE-stop clear — automatically resuming from held poses." :
        "\nE-stop clear — scripted probe remains terminated; arms still holding.")
                << std::endl;
      return StopChoice::resume;
    };

  auto wait_square_release = [&]() -> StopChoice {
      while (true) {
        std::cout << "  Enter  = release this hold, then land follower and leader to sleep/idle\n"
                  << "           (keep both landing paths clear)\n"
                  << "  Ctrl+C = EMERGENCY RELEASE, idle immediately; automatic landing is skipped\n"
                  << "  Scripted motion cannot resume in this process."
                  << std::endl;
        StopChoice terminal = wait_for_choice();
        if (terminal == StopChoice::estop) {
          const StopChoice cleared = run_estop_flow(false);
          if (cleared == StopChoice::emergency) {return cleared;}
          continue;
        }
        // wait_for_choice recognizes r, but the one-shot probe deliberately
        // treats r+Enter as Enter after the operator has supported the arms.
        return terminal == StopChoice::resume ? StopChoice::release : terminal;
      }
    };

  StopChoice choice;
  bool released_from_estop = estop_triggered;
  if (estop_triggered) {
    choice = run_estop_flow(!square_enabled);
    if (square_enabled && choice != StopChoice::emergency) {
      choice = wait_square_release();
    }
  } else {
    if (square_enabled) {
      if (square_finished) {
        std::cout << "\n" << probe_name << " probe complete: pen retracted, arms holding."
                  << std::endl;
      } else if (contact_trip) {
        std::cout << "\n" << probe_name
                  << " probe terminated by contact cap: pen retracted, arms holding."
                  << std::endl;
      } else if (square_guard_trip) {
        std::cout << "\n" << probe_name
                  << " probe terminated by measured-motion guard: pen retracted, arms holding."
                  << std::endl;
      } else if (square_tracking_trip) {
        std::cout << "\n" << probe_name
                  << " probe terminated by endpoint tracking guard: pen retracted, arms holding."
                  << std::endl;
      } else if (draw_refused) {
        std::cout << "\ndraw session refused: " << draw_refusal
                  << " — pen retracted, arms holding." << std::endl;
      } else if (square_started) {
        std::cout << "\n" << probe_name << " probe interrupted: pen retracted, arms holding."
                  << std::endl;
      } else {
        std::cout << "\n" << probe_name << " probe stopped before the trigger; arms holding."
                  << std::endl;
      }
      choice = wait_square_release();
    } else if (contact_trip) {
      std::cout << "\nContact trip: pen retracted, arms holding." << std::endl;
    } else if (stale_baseline) {
      std::cout << "\nStale alignment: nothing was sent, arms holding. r re-aligns from the current poses." << std::endl;
    } else {
      std::cout << "\nControlled stop: arms holding, carriage holds." << std::endl;
    }
    if (!square_enabled) {
      std::cout << "  Enter     = release arms and carriage to idle (support the arms first)\n"
                << "  r + Enter = resume teleoperation"
                << (carriage_target != CARRIAGE_REST_M ? " (pen returns to rest)" : "") << "\n"
                << "  Ctrl+C    = EMERGENCY RELEASE, idle immediately (arms fall)"
                << std::endl;
      choice = wait_for_choice();
      if (choice == StopChoice::estop) {
        choice = run_estop_flow(true);  // e-stop pressed at the prompt: same flow
        released_from_estop = true;
      }
    }
  }
  // Release in a probe/draw session hands the arms to the wrapper's landing
  // routine STILL HOLDING: idling here dropped the follower ~2 cm under gravity
  // in the seconds before il_recover_arm.sh reconnected (first draw session,
  // 2026-09-01). The landing takes control softly at the current pose
  // (recovery.py), so the process exits without idling and without running
  // the driver destructors, which would idle too. The e-stop stays live.
  if (square_enabled && choice == StopChoice::release) {
    if (request_probe_landing()) {
      handoff_holding = true;
    } else {
      std::cerr << "WARNING: could not record the operator's automatic-landing request; "
                   "the wrapper will leave the arms idle instead of moving them."
                << std::endl;
    }
  }
  if (choice == StopChoice::resume) {
    // Re-baseline at the held poses so teleop restarts with zero step, then
    // ramp that fresh offset out again — otherwise a stop would be a way to
    // silently reintroduce the mismatch absolute mapping exists to remove.
    // An e-stop release used to produce no motion at all, so there the
    // confirmation bar is lowered to anything past a couple of degrees.
    take_baselines();
    alignment.restart(leader_start, follower_start);
    if (opt.absolute) {
      const double confirm_deg = released_from_estop ?
        std::min(opt.align_confirm_deg, 2.0) : opt.align_confirm_deg;
      if (!confirm_and_rebaseline(confirm_deg, "Resume alignment")) {
        std::cout << "Interrupted before aligning; back to the hold prompt."
                  << std::endl;
        continue;  // arms stay held; stop_baseline is deliberately not advanced
      }
    }
    pos_filt = leader.get_all_positions();
    std::fill(vel_filt.begin(), vel_filt.end(), 0.0);
    std::fill(leader_efforts.begin(), leader_efforts.end(), 0.0);
    std::fill(assist_filt.begin(), assist_filt.end(), 0.0);
    // Whatever retracted the pen (e-stop, contact trip), tracking resumes
    // with it back at rest; the debounce restarts from zero.
    if (carriage_target != CARRIAGE_REST_M) {
      command_carriage(CARRIAGE_REST_M, CARRIAGE_RESUME_GOAL_S);
    }
    contact_over_ticks = 0;
    stop_baseline = g_stop_signals.load();
    leader.set_all_modes(trossen_arm::Mode::external_effort);
    leader.set_all_external_efforts(leader_efforts, 0.0, false);
    std::cout << "Teleoperation resumed: hand-guide the leader, Ctrl+C to stop." << std::endl;
    continue;
  }
  emergency = (choice == StopChoice::emergency);
  break;
  }  // session loop

  } catch (const std::exception & e) {
    // The driver throws on connection loss, limit violations, etc., and its
    // DESTRUCTOR idles the arm (it sags). Before unwinding, try to leave
    // each arm holding position instead — independently, so one arm's fault
    // doesn't drop the other — and keep holding until the operator confirms
    // the arms are supported.
    std::cerr << "\nDriver fault: " << e.what() << std::endl;
    bool holding = false;
    try {
      leader.set_all_modes(trossen_arm::Mode::position);
      leader.set_all_positions(leader.get_all_positions(), 0.0, false);
      holding = true;
    } catch (...) {}
    try {
      // Pen off the surface first, then hold the arm.
      follower.set_joint_position(
        static_cast<uint8_t>(gripper), opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S, false);
    } catch (...) {}
    try {
      follower.set_arm_modes(trossen_arm::Mode::position);
      const std::vector<double> follower_now = follower.get_all_positions();
      follower.set_arm_positions(
        std::vector<double>(follower_now.begin(), follower_now.end() - 1), 0.0, false);
      holding = true;
    } catch (...) {}
    if (holding) {
      std::cout << "Arms holding after fault — Enter continues to automatic landing; "
                   "Ctrl+C emergency-releases and skips landing."
                << std::endl;
      // Keep holding through an engaged e-stop (it can't do more than the
      // hold already does); only Enter / Ctrl+C release to idle.
      StopChoice fault_choice;
      while ((fault_choice = wait_for_choice()) == StopChoice::estop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (fault_choice == StopChoice::emergency) {
        try {leader.set_all_modes(trossen_arm::Mode::idle);} catch (...) {}
        try {follower.set_all_modes(trossen_arm::Mode::idle);} catch (...) {}
        std::cout << "EMERGENCY RELEASE after fault: automatic landing will be skipped."
                  << std::endl;
        return 130;
      }
      if (!request_probe_landing()) {
        std::cerr << "WARNING: could not record the operator's automatic-landing request; "
                     "the wrapper will not move the arms after this fault."
                  << std::endl;
      }
    }
    throw;
  }

  if (emergency) {
    std::cout << "EMERGENCY RELEASE: idling both arms now." << std::endl;
  }
  if (!(handoff_holding && !emergency)) {
    leader.set_all_modes(trossen_arm::Mode::idle);
    follower.set_all_modes(trossen_arm::Mode::idle);
  }

  const std::string health_summary = health.summary();
  if (!health_summary.empty()) {
    std::cout << health_summary << std::endl;
  }

  if (telemetry) {
    const auto & stats = telemetry->stats();
    std::cout << "Visualization telemetry: sent=" << stats.sent
              << " rate_limited=" << stats.rate_limited
              << " send_errors=" << stats.send_errors << std::endl;
  }

  if (log_file) {
    const auto stats = log_file->finish();
    std::cout << "Flight recorder: enqueued=" << stats.records_enqueued
              << " dropped=" << stats.records_dropped
              << " write_errors=" << stats.write_errors << std::endl;
  }

  if (emergency) {return 130;}
  if (handoff_holding) {
    std::cout << "Arms handed to the landing routine still holding (not idled); "
                 "the wrapper lands follower, then leader." << std::endl;
    std::cout.flush();
    std::cerr.flush();
    // Skip the driver destructors: their cleanup idles the arms.
    std::_Exit(square_enabled && !square_finished ? 3 : 0);
  }
  return square_enabled && !square_finished ? 3 : 0;
}

int main(int argc, char ** argv)
{
  // The driver throws on connection loss, limit violations, etc.; its
  // destructor idles the arm. Surface the error instead of terminating.
  try {
    return run(argc, argv);
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
