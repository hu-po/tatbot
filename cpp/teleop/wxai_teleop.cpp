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
// initial step. Neither arm goes limp. The default
// /dev/tatbot-estop device is mandatory. --estop DEV selects another mandatory
// device; --no-estop is an explicit hardware-free bench opt-out rejected by
// production launchers.

#include "driver_lease.hpp"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
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
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "libtrossen_arm/trossen_arm.hpp"
#include "motion_constants.hpp"
#include "estop_monitor.hpp"
#include "flight_recorder.hpp"
#include "realtime.hpp"
#include "telemetry_udp.hpp"
#include "teleop_mapping.hpp"
#include "teleop_startup.hpp"

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

using tatbot::teleop::StopChoice;

bool request_probe_landing(const char * tag = "operator-release")
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
  sentinel << tag << "\n";
  return static_cast<bool>(sentinel);
}

// Wait at the controlled-stop prompt: Enter releases to idle, a line
// containing 'r' resumes teleoperation, a further stop signal is an
// emergency release, an e-stop engaging hands control to the e-stop flow.
// Stdin EOF keeps holding; it is never an implicit Enter.
StopChoice wait_for_choice(int accepted_signals)
{
  return tatbot::teleop::wait_for_hold_choice(
    STDIN_FILENO, g_stop_signals, accepted_signals, g_estop, estop_ok);
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
// Trip / completion retract. The firmware position limit is 0.040 m and the
// carriage velocity limit 0.25 m/s (config/trossen/follower.yaml): a target ON
// the limit reached in 0.15 s (0.27 m/s) was silently ignored -- the flight
// log of the 2026-09-02 draws shows the carriage at 2.00 mm from first tick to
// last while the console said "pen retracted 40 mm". 36 mm over 0.4 s is
// 0.09 m/s, inside both limits; the retract now reads the carriage back and
// prints it so the claim can be checked.
// 32 mm: 30 mm of travel from the 2 mm drawing bias, 4 mm (the position
// tolerance) under the 40 mm limit. Operator 2026-09-02: a retract is 30 mm
// at least; a 2 mm one is pointless.
constexpr double CARRIAGE_RETRACT_M = 0.032;
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
// Rest baseline: a 2 s median (800 ticks). The carriage's effort readout dithers
// ~10 N peak-to-peak at ~1 Hz with the arm motionless (flight logs 2026-09-04),
// so a 400 ms median sampled one phase of that dither and sat up to 10 N off
// the true rest; the touch calibration then tripped the cap on a hand at the
// pen switch (three sessions in a row). The baseline also tracks slow drift:
// while the arm is still and the effort sits within CONTACT_BASELINE_TRACK of
// it, it relaxes toward the measurement with CONTACT_BASELINE_TAU_S. A press
// rises far faster than that and is never absorbed.
constexpr int CONTACT_BASELINE_TICKS = 800;      // ticks of tracking that define the rest baseline (2 s)
constexpr double CONTACT_BASELINE_TAU_S = 10.0;  // slow drift tracking of the rest baseline
constexpr double CONTACT_BASELINE_TRACK = 0.25;  // ...only while within this fraction of the cap
constexpr double CARRIAGE_TRIP_GOAL_S = 0.6;     // s, retract goal time (30 mm -> 0.05 m/s, gentle on the trajectory limits)
// In-session landing (recovery.py's STAGED_POSE_S / LANDED_TOLERANCE_RAD).
constexpr double LAND_STAGED_S = 4.0;
constexpr double LAND_SETTLE_S = 0.5;
constexpr double LAND_TOLERANCE_RAD = 0.20;
constexpr double LAND_CARRIAGE_TOLERANCE_M = 0.0005;
constexpr double CARRIAGE_RESUME_GOAL_S = 0.5;   // s, return-to-rest goal time on resume

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
void resolve_tool(const std::string & tool_id, bool uncalibrated = false,
  const std::string & physical_arm = "right")
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
    scrape_workspace_tool(repo / "config" / "workspace.yaml", physical_arm);
  if (!calibrated.empty() && calibrated != tool_id) {
    if (!uncalibrated) {
      throw std::runtime_error(
              "--ee-tool '" + tool_id + "' is fitted but config/workspace.yaml was "
              "measured with '" + calibrated + "'. Re-run the touch-off for the "
              "fitted tool:\n"
              "  tatbot --ee-tool " + tool_id + " teleop start --touchoff   (on the teleop host)\n"
              "  or record it under `" + physical_arm + ":` with its datasheet's nominal tip, deploy "
              "the ROS stack, and run tatbot ros calib run --arm " + physical_arm +
              "   (on the station probe)");
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
  // loop period, us (2500 = 400 Hz): config/motion_constants.json period_s
  int64_t period_us = static_cast<int64_t>(std::llround(tatbot::motion_constants::PERIOD_S * 1e6));
  bool supervised_start = false;  // conductor must release a gate before any arm connection
  bool wrist_calibration = false;  // right input, left receiver, mirror 0/4/5; free-space only
  bool absolute = true;        // --relative: keep the old delta mapping
  double align_rate = 0.35;    // peak joint speed of the startup alignment, rad/s
  double align_confirm_deg = 15.0;  // ask before an alignment move larger than this
  // --staged-positions a,b,...: the staged (= sleep) pose from config/trossen/tatbot.yaml.
  // When given, a release lands both arms INSIDE this session (position mode,
  // live) instead of handing them, holding, to a fresh driver session: that
  // handover sagged the follower at Enter (2026-09-02). Enter at the stop
  // prompt lands.
  std::vector<double> staged_positions;
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
    if (arg == "--supervised-start") {opt.supervised_start = true;} else
    if (arg == "--wrist-calibration") {opt.wrist_calibration = true;} else
    if (arg == "--relative") {opt.absolute = false;} else
    if (arg == "--align-rate") {
      opt.align_rate = std::clamp(std::stod(value()), 0.01, 1.5);
    } else
    if (arg == "--align-confirm-deg") {opt.align_confirm_deg = std::stod(value());} else
    if (arg == "--staged-positions") {
      std::stringstream ss(value());
      std::string item;
      opt.staged_positions.clear();
      while (std::getline(ss, item, ',')) {opt.staged_positions.push_back(std::stod(item));}
    } else
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

  if (opt.wrist_calibration && opt.ee_tool.empty()) {
    throw std::runtime_error("--wrist-calibration is free-space mirrored teleop only; no reference, contact or scripted motion");
  }
  // Calibration mirrors increments from the measured poses, never encoder zero.
  if (opt.wrist_calibration) {opt.absolute = false;}
  if (opt.period_us <= 0) {
    throw std::runtime_error("--period-us must be positive");
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
    if (opt.wrist_calibration) {
      const auto left_tool = tool_registry::scrape_workspace_tool(
        tool_registry::repo_root() / "config/workspace.yaml", "left");
      if (left_tool.empty()) {throw std::runtime_error("wrist calibration needs the left arm tool identity");}
      tool_registry::resolve_tool(left_tool, false, "left");
    }
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

using tatbot::teleop::Alignment;
using tatbot::teleop::map_joint;

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
bool announce_alignment(
  const Alignment & alignment, double confirm_deg, const char * when, int accepted_signals)
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
            << " the configured absolute joint mapping." << std::endl;
  if (largest_deg < confirm_deg) {
    return true;
  }
  std::cout << "  The follower is about to MOVE " << largest_deg
            << " deg to meet the leader. Clear the workspace.\n"
            << "  Enter = start aligning, Ctrl+C = stop" << std::endl;
  const bool accepted = tatbot::teleop::wait_for_alignment_confirmation(
    STDIN_FILENO, g_stop_signals, accepted_signals, g_estop, estop_ok);
  if (!accepted) {
    std::cout << "Alignment cancelled: stop, E-stop or console closed; holding." << std::endl;
  }
  return accepted;
}

// Flight-recorder binary format, consumed by analyze_log.py. All header
// fields are 8 bytes so the layout is padding-free. Each record is
// 5 + 6*num_joints doubles:
//   t_sched, t_wake, t_leader_read, t_follower_read, t_cmd  (s since start)
//   leader_pos[n], leader_vel[n],
//   follower_pos[n], follower_vel[n], follower_eff[n], target[n]
struct LogHeader
{
  char magic[8];  // WXTLOG1, or WXTLOG2 for the explicit mirrored calibration roles
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
  std::cout << "WXAI_TELEOP_PID=" << getpid() << std::endl;
  if (opt.supervised_start) {
    // The conductor learns this PID before permitting any arm connection.
    // If it is interrupted first, a late-starting child remains inert here.
    std::cout << "SUPERVISED_START_READY: no arm connection opened" << std::endl;
    if (!tatbot::teleop::wait_for_alignment_confirmation(
        STDIN_FILENO, g_stop_signals, 0, g_estop, estop_ok)) {
      std::cout << "Supervised startup cancelled before arm connection." << std::endl;
      return 130;
    }
  }

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
      // What failed, why it matters, what was NOT started, then one useful next
      // step. The priority comes from the option this launch selected, and the
      // limit is the one measured a moment ago, so neither number is a guess.
      std::cerr << "Teleop could not start: this login session cannot use real-time"
                   " scheduling.\n"
                << "       The control loop requires priority " << opt.rt_priority << "; ";
      if (setup.rtprio_soft >= 0) {
        std::cerr << "this session's limit is " << setup.rtprio_soft << ".\n";
      } else {
        std::cerr << "this session's limit could not be read (" << setup.fifo_error << ").\n";
      }
      std::cerr << "       Neither arm was connected: a loop the kernel can preempt makes the"
                   " follower shake.\n";
      if (setup.limits_installed) {
        // Already installed. "Install it again" would not be a diagnosis, and
        // an operator who follows it learns nothing.
        std::cerr << "       The limits file is installed at " << setup.limits_path
                  << ", but its policy is not effective in this session.\n";
      } else {
        std::cerr << "       No real-time limits policy is installed at " << setup.limits_path << ".\n"
                  << "       Install it: sudo cp config/limits/99-tatbot-realtime.conf "
                  << setup.limits_path << "\n"
                  << "       then open a NEW login session; limits are applied at login.\n";
      }
      std::cerr << "       Next: tatbot teleop check\n"
                << "       (--no-rt is a bench opt-out for a hardware-free run, not a way to"
                   " drive the arms.)"
                << std::endl;
      return 3;
    }
  }

  std::unique_ptr<tatbot::DriverLease> driver_lease;
  try {
    driver_lease = std::make_unique<tatbot::DriverLease>();
  } catch (const std::exception & error) {
    std::cerr << error.what() << std::endl;
    return 6;
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

  // Controllers, profiles and SDK end-effectors stay with their physical arms.
  trossen_arm::TrossenArmDriver left_arm;
  trossen_arm::TrossenArmDriver right_arm;
  configure_arm(
    left_arm, opt.leader_ip, opt.leader_config, "physical left",
    trossen_arm::StandardEndEffector::wxai_v0_leader);
  configure_arm(
    right_arm, opt.follower_ip, opt.follower_config, "physical right",
    trossen_arm::StandardEndEffector::wxai_v0_follower);
  auto & leader = opt.wrist_calibration ? right_arm : left_arm;
  auto & follower = opt.wrist_calibration ? left_arm : right_arm;
  if (opt.wrist_calibration) {
    std::cout << "Wrist calibration: pink/right ballpoint is LEADER; left laser is FOLLOWER.\n"
              << "Joints 0, 4 and 5 mirror movement from the measured starting poses; carriage is independent.\n"
              << "Free-space capture: keep both tools clear of the table and laser emission off."
              << std::endl;
  }
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

  // Ordinary teleop fades its startup offset; wrist calibration preserves it.
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

  Alignment alignment = opt.wrist_calibration ? Alignment::wrist_calibration(opt.align_rate) :
    Alignment(opt.absolute, opt.align_rate);
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
        opt.telemetry_udp, opt.telemetry_fps, opt.wrist_calibration);
      std::cout << "Visualization telemetry -> " << telemetry->endpoint()
                << " at up to " << opt.telemetry_fps << " Hz (nonblocking)" << std::endl;
    } catch (const std::exception & error) {
      std::cerr << "WARNING: visualization telemetry disabled: " << error.what() << std::endl;
    }
  }

  const double dt = static_cast<double>(opt.period_us) * 1e-6;
  LoopHealth health(dt);
  bool emergency = false;
  bool handoff_holding = false;  // an in-session landing that fell back: exit still holding
  try {  // any driver throw below is caught to attempt a hold before idling

  // Leader: external_effort mode on all joints — zero effort is pure gravity
  // compensation; the follower's external efforts (contacts) are reflected
  // back scaled by -ff_gain. Follower: position mode on EVERY joint, the tool
  // carriage included — the carriage is a position axis owned by the safety
  // layer. It is seated at the rest stop before tracking begins.
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
  // The retract: BLOCKING, then read back and printed. Every "pen retracted"
  // line before 2026-09-02 was the command, not a measurement, and the flight
  // log ends with the loop, so nothing ever showed whether the carriage moved.
  // Every site that calls this ends the tracking loop, so the 0.6 s stall of
  // the control loop is acceptable; the arms hold their targets.
  auto retract_pen = [&]() {
      carriage_target = opt.carriage_retract_m;
      try {
        follower.set_joint_position(
          static_cast<uint8_t>(gripper), opt.carriage_retract_m, CARRIAGE_TRIP_GOAL_S, true);
        const std::vector<double> now = follower.get_all_positions();
        std::cout << "pen retract: carriage measured " << now[gripper] * 1e3 << " mm (target "
                  << opt.carriage_retract_m * 1e3 << " mm)"
                  << (std::fabs(now[gripper] - opt.carriage_retract_m) > 0.002 ? " -- PEN NOT RETRACTED" : "")
                  << std::endl;
      } catch (const std::exception & e) {
        std::cout << "pen retract FAILED: " << e.what() << std::endl;
      }
    };
  // Seat the carriage at rest so every session starts from a known extension.
  follower.set_joint_position(static_cast<uint8_t>(gripper), CARRIAGE_REST_M, 1.0, true);
  carriage_target = CARRIAGE_REST_M;
  // The carriage's external effort at rest is a property of the arm's STATE,
  // not of the tool: the reading before the session (motors idle) sits far
  // from the holding reading with the tool's weight along the carriage axis
  // (two 2026-09-04 sessions: -9.0 N idle, +15..17 N tracking, and the cap
  // tripped on the difference 100 ms after alignment). So the baseline is
  // taken in the loop, over CONTACT_BASELINE_TICKS once the follower is
  // tracking and the alignment is over; until then the effort cap is not
  // judged (the deflection trip stays armed throughout).
  double contact_baseline = 0.0;
  bool contact_baseline_set = false;
  std::vector<double> contact_baseline_samples;
  contact_baseline_samples.reserve(CONTACT_BASELINE_TICKS);

  std::cout << "Teleoperation running: hand-guide the leader, Ctrl+C to stop.\n"
            << "Fitted tool: "
            << (opt.ee_tool.empty() ? "none (--no-tool)" : opt.ee_tool)
            << (opt.tool_uncalibrated ? " (uncalibrated: this session is its touch-off)" : "") << "\n"
            << "Carriage: rest "
            << carriage_target * 1e3 << " mm, contact cap "
            << opt.contact_cap_n << " N -> retract " << opt.carriage_retract_m * 1e3
            << " mm on a trip" << std::endl;
  if (opt.damping > 0.0 || opt.assist > 0.0) {
    std::cout << "Anti-stiction: damping " << opt.damping
              << " Nm/(rad/s), assist " << opt.assist
              << " (deadband " << ASSIST_DEADBAND_NM << " Nm)" << std::endl;
  }
  // The operator may move the leader while the announcement waits for Enter —
  // orienting it to the follower's rolled idle is the natural thing to do —
  // and an offset taken before that move would then be added to the leader's
  // NEW pose (2026-08-30: +103 deg on joint 5 became a target of pi, clamped,
  // and a 1.4 rad step tripped the velocity limit). So the baselines are
  // re-taken AFTER the confirmation, and if the announced move changed by more
  // than a degree the announcement repeats with the real number.
  auto confirm_and_rebaseline = [&](double confirm_deg, const char * when, int accepted_signals) -> bool {
      for (int attempt = 0; attempt < 5; ++attempt) {
        if (g_stop_signals.load() != accepted_signals || g_estop.load() > estop_ok) {return false;}
        take_baselines();
        alignment.restart(leader_start, follower_start);
        const double announced = alignment.largest_rad();
        // On an interrupt here the loop below is skipped (a signal has
        // arrived) and the normal controlled-stop path holds both arms.
        if (!announce_alignment(alignment, confirm_deg, when, accepted_signals)) {return false;}
        if (g_stop_signals.load() != accepted_signals || g_estop.load() > estop_ok) {return false;}
        take_baselines();
        alignment.restart(leader_start, follower_start);
        if (std::abs(alignment.largest_rad() - announced) * rad_to_deg <= 1.0) {return true;}
        std::cout << "  The leader moved while waiting (offset now "
                  << alignment.largest_rad() * rad_to_deg << " deg); re-announcing." << std::endl;
      }
      return false;
    };
  if (opt.absolute) {
    if (!confirm_and_rebaseline(opt.align_confirm_deg, "Startup alignment", 0)) {
      g_stop_signals.fetch_add(g_stop_signals.load() == 0 ? 1 : 0);
    }
  } else if (opt.wrist_calibration) {
    take_baselines();
    alignment.restart(leader_start, follower_start);
    std::cout << "Wrist mapping anchored at current poses; no startup alignment; tracking immediately."
              << std::endl;
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
    std::memcpy(header.magic, opt.wrist_calibration ? "WXTLOG2" : "WXTLOG1", 8);
    header.num_joints = num_joints;
    header.period_s = static_cast<double>(opt.period_us) * 1e-6;
    header.tau_s = opt.tau;
    header.goal_time_s = opt.goal_time;
    header.ff_gain = opt.ff_gain;
    // v2 bits 0..5: absolute carriage, right leader, mirror 0/4/5, start-pose anchoring.
    header.abs_gripper = opt.wrist_calibration ? 63 : 1;
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
  uint64_t telemetry_sequence = 0;
  // Every startup stop remains pending; never consume a signal from the prompt.
  int stop_baseline = 0;
  // Contact-cap debounce: consecutive ticks with |carriage effort| over the
  // cap. A trip retracts the pen and drops into the controlled-stop path.
  int contact_over_ticks = 0;
  bool contact_trip = false;
  // First streamed command of each (re)start must not step (see the loop).
  bool first_tick = true;
  bool stale_baseline = false;
  // Runaway latch for --damping/--assist; clears on each (re)start.
  bool antistiction_off = false;
  while (true) {  // session loop: teleop until a stop, optionally resume
  auto next_tick = clock::now();
  contact_trip = false;
  contact_over_ticks = 0;
  contact_baseline_set = false;          // (re)taken while tracking, after the alignment
  contact_baseline_samples.clear();
  first_tick = true;
  stale_baseline = false;
  antistiction_off = false;
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

    // Low-pass the leader state (see --tau).
    for (size_t i = 0; i < num_joints; ++i) {
      pos_filt[i] += alpha * (leader_pos[i] - pos_filt[i]);
      vel_filt[i] += alpha * (leader_vel[i] - vel_filt[i]);
    }
    // Absolute mapping: the follower arm goes where the leader's joints are,
    // plus whatever is left of the startup offset (zero once the ramp
    // finishes). The carriage is never derived from the leader trigger.
    for (size_t i = 0; i < gripper; ++i) {
      target[i] = std::clamp(
        alignment.position(pos_filt[i], i),
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
    // Contact force: the carriage effort's departure from its rest baseline,
    // assessed only at drawing speeds (see CONTACT_STILL_RAD_S) and debounced,
    // so neither noise nor an inertial swing can trip; a sustained hard press
    // retracts the pen and hands control to the stop path.
    if (!contact_baseline_set && !alignment.aligning()) {
      contact_baseline_samples.push_back(follower_eff[gripper]);
      if (static_cast<int>(contact_baseline_samples.size()) >= CONTACT_BASELINE_TICKS) {
        std::sort(contact_baseline_samples.begin(), contact_baseline_samples.end());
        contact_baseline = contact_baseline_samples[contact_baseline_samples.size() / 2];
        contact_baseline_set = true;
        std::cout << "Contact cap armed: carriage rest baseline " << contact_baseline << " N (2 s median while tracking; cap "
                  << opt.contact_cap_n << " N over it, baseline tracks drift within "
                  << opt.contact_cap_n * CONTACT_BASELINE_TRACK << " N)." << std::endl;
      }
    }
    double contact = std::fabs(follower_eff[gripper] - contact_baseline);
    double arm_speed = 0.0;
    for (size_t i = 0; i < gripper; ++i) {arm_speed = std::max(arm_speed, std::fabs(follower_vel[i]));}
    const bool assessable = contact_baseline_set && arm_speed < CONTACT_STILL_RAD_S;
    if (assessable && contact < opt.contact_cap_n * CONTACT_BASELINE_TRACK) {
      contact_baseline += (follower_eff[gripper] - contact_baseline) * (dt / CONTACT_BASELINE_TAU_S);
      contact = std::fabs(follower_eff[gripper] - contact_baseline);
    }
    const double deflect = follower_pos[gripper] - carriage_target;  // + = pushed open
    const bool pushed = (assessable && contact > opt.contact_cap_n) || deflect > opt.contact_deflect_m;
    contact_over_ticks = pushed ? contact_over_ticks + 1 : 0;
    if (contact_over_ticks >= CONTACT_CAP_DEBOUNCE_TICKS) {
      retract_pen();
      std::cout << "\nCONTACT CAP: " << contact << " N over the rest baseline ("
                << contact_baseline << " N; raw " << follower_eff[gripper] << " N; cap "
                << opt.contact_cap_n << " N), carriage deflected " << deflect * 1e3
                << " mm (limit " << opt.contact_deflect_m * 1e3 << " mm) for "
                << CONTACT_CAP_DEBOUNCE_TICKS * dt * 1e3 << " ms — pen retracted "
                << opt.carriage_retract_m * 1e3 << " mm, arms holding" << std::endl;
      contact_trip = true;
      break;
    }

    // Stream the arm target with the filtered leader velocity as feedforward
    // and a short interpolation horizon. The carriage remains safety-owned.
    std::copy(target.begin(), target.end() - 1, arm_target.begin());
    for (size_t i = 0; i < gripper; ++i) {
      arm_vel[i] = map_joint(vel_filt[i], i, opt.wrist_calibration);
    }
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
        leader_efforts[i] = -opt.ff_gain * map_joint(follower_eff[i], i, opt.wrist_calibration);
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
    const double t_cmd = since_start();

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
  // plain controlled stop leaves the carriage where it is; a contact trip
  // retracts it. The carriage holds its commanded position through any pause;
  // only idling releases it.
  const bool estop_triggered = g_estop.load() > estop_ok;
  if (estop_triggered && carriage_target != opt.carriage_retract_m) {
    retract_pen();
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

  // E-stop flow: HOLD while the button is latched or its heartbeat is absent,
  // then automatically re-baseline and resume when the input is healthy.
  auto run_estop_flow = [&]() -> StopChoice {
      std::cout << "\nE-STOP: "
                << (g_estop.load() == estop_fault ?
        "heartbeat lost (device unplugged or dead?)" : "button pressed")
                << " — pen retracted, arms holding.\n"
                << "  twist-release / reconnect = automatically resume tracking\n"
                << "  Ctrl+C = EMERGENCY RELEASE, idle immediately (arms fall)"
                << std::endl;
      const int signals_at_hold = g_stop_signals.load();
      const auto result = tatbot::estop::wait_for_clear(
        g_estop, g_stop_signals, signals_at_hold);
      if (result == tatbot::estop::WaitResult::emergency) {
        return StopChoice::emergency;
      }
      std::cout << "\nE-stop clear — automatically resuming from held poses."
                << std::endl;
      return StopChoice::resume;
    };

  // Preserve a second signal that arrived while the first stop was holding the arms.
  const int accepted_hold_signals = std::min(g_stop_signals.load(), stop_baseline + 1);
  StopChoice choice;
  bool released_from_estop = estop_triggered;
  if (estop_triggered) {
    choice = run_estop_flow();
  } else {
    if (contact_trip) {
      std::cout << "\nContact trip: pen retracted, arms holding." << std::endl;
    } else if (stale_baseline) {
      std::cout << "\nStale alignment: nothing was sent, arms holding. r re-aligns from the current poses." << std::endl;
    } else {
      std::cout << "\nControlled stop: arms holding, carriage holds." << std::endl;
    }
    if (opt.staged_positions.size() == num_joints) {
      std::cout << "  Enter     = land both arms in this session (staged -> sleep -> idle); keep the paths clear\n";
    } else {
      std::cout << "  Enter     = release arms and carriage to idle (support the arms first)\n";
    }
    std::cout << "  r + Enter = resume teleoperation"
              << (carriage_target != CARRIAGE_REST_M ? " (pen returns to rest)" : "") << "\n"
              << "  Ctrl+C    = EMERGENCY RELEASE, idle immediately (arms fall)"
              << std::endl;
    choice = wait_for_choice(accepted_hold_signals);
    if (choice == StopChoice::estop) {
      choice = run_estop_flow();  // e-stop pressed at the prompt: same flow
      released_from_estop = true;
    }
  }
  // A release that cannot land in-session hands the arms to the wrapper's
  // landing routine STILL HOLDING: idling here dropped the follower ~2 cm under
  // gravity in the seconds before il_recover_arm.sh reconnected (2026-09-01).
  // The landing takes control softly at the current pose (recovery.py), so the
  // process exits without idling and without running the driver destructors,
  // which would idle too. The e-stop stays live.
  bool landed_in_session = false;
  const bool lands_here = opt.staged_positions.size() == num_joints;
  if (lands_here && choice == StopChoice::release) {
    // Land here, in the live position-mode session: measured pose -> staged (= sleep)
    // pose over LAND_STAGED_S, verify, idle. Follower first, then leader. A stop
    // signal or the e-stop during the move leaves that arm holding where it is and
    // falls through to the holding handover below.
    auto land_arm = [&](trossen_arm::TrossenArmDriver & arm, const char * name) -> bool {
        const int signals_at_land = g_stop_signals.load();
        std::vector<double> goal(opt.staged_positions.begin(), opt.staged_positions.end());
        try {
          // The configured staged pose owns the carriage rest target too: a
          // retract left by a trip must be closed before the arm idles.
          arm.set_all_positions(goal, LAND_STAGED_S, false);
          const auto started = std::chrono::steady_clock::now();
          while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() <
            LAND_STAGED_S + LAND_SETTLE_S)
          {
            if (g_stop_signals.load() > signals_at_land || g_estop.load() > estop_ok) {
              arm.set_all_positions(arm.get_all_positions(), 0.0, false);
              std::cout << name << " landing interrupted; holding where it is." << std::endl;
              return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
          }
          const std::vector<double> now = arm.get_all_positions();
          double worst = 0.0;
          for (size_t joint = 0; joint < gripper; ++joint) {
            worst = std::max(worst, std::fabs(now[joint] - goal[joint]));
          }
          const double carriage_error = std::fabs(now[gripper] - goal[gripper]);
          if (worst > LAND_TOLERANCE_RAD || carriage_error > LAND_CARRIAGE_TOLERANCE_M) {
            std::cout << name << " did not reach the sleep pose (worst joint off by " << worst
                      << " rad, carriage off rest by " << carriage_error * 1e3
                      << " mm); holding." << std::endl;
            return false;
          }
          arm.set_all_modes(trossen_arm::Mode::idle);
          std::cout << name << " landed in-session: sleep pose, motors idle (carriage "
                    << now[gripper] * 1e3 << " mm)." << std::endl;
          return true;
        } catch (const std::exception & e) {
          std::cout << name << " landing failed: " << e.what() << "; holding." << std::endl;
          return false;
        }
      };
    std::cout << "\nLanding both arms in this session (staged -> sleep -> idle, "
              << LAND_STAGED_S << " s each). Keep both landing paths clear." << std::endl;
    const bool follower_landed = land_arm(follower, "follower");
    const bool leader_landed = follower_landed && land_arm(leader, "leader");
    landed_in_session = follower_landed && leader_landed;
    if (landed_in_session) {
      if (!request_probe_landing("landed-in-session")) {
        std::cerr << "WARNING: could not record the in-session landing for the wrapper." << std::endl;
      }
    } else {
      std::cout << "Falling back to the holding handover; the wrapper lands what is not landed." << std::endl;
    }
  }
  if (lands_here && choice == StopChoice::release && !landed_in_session) {
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
    const int resume_signals = g_stop_signals.load();
    take_baselines();
    alignment.restart(leader_start, follower_start);
    if (opt.absolute) {
      const double confirm_deg = released_from_estop ?
        std::min(opt.align_confirm_deg, 2.0) : opt.align_confirm_deg;
      if (!confirm_and_rebaseline(confirm_deg, "Resume alignment", resume_signals)) {
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
    if (g_stop_signals.load() != resume_signals || g_estop.load() > estop_ok) {continue;}
    stop_baseline = resume_signals;
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
      const int fault_hold_signals = g_stop_signals.load();
      while ((fault_choice = wait_for_choice(fault_hold_signals)) == StopChoice::estop) {
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
    std::_Exit(0);
  }
  return 0;
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
