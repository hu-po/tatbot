// The ros2_control adapter. Lifecycle, parameters and interface export follow trossen_arm_ros's
// TrossenArmHardwareInterface (see sdk_backend.cpp for its licence); everything that decides
// motion is in core.cpp.
#include "tatbot_hardware/tatbot_arm.hpp"

#include <sched.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"
#include "tatbot_hardware/names.hpp"

namespace tatbot_hardware
{
namespace
{
using hardware_interface::return_type;
constexpr double kFeedbackWaitS = 1.0;   // arm_recover MEASUREMENT_WAIT_S

double now_s()
{
  static const auto epoch = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

std::vector<double> numbers(std::string text)
{
  for (char & c : text) {
    if (c == '[' || c == ']' || c == ',') {c = ' ';}
  }
  std::istringstream in(text);
  std::vector<double> out;
  double v;
  while (in >> v) {out.push_back(v);}
  return out;
}

// Serializes SDK connects across both arms (README section 5).
std::mutex & connect_mutex()
{
  static std::mutex mutex;
  return mutex;
}

// One flight-log record per control tick (little-endian, packed; README section 13).
#pragma pack(push, 1)
struct FlightRecord
{
  double t;
  float q[kJoints], qd[kJoints], effort[kJoints], cmd_q[kJoints], sent_q[kJoints], sent_qd[kJoints];
  float estop_age_s, period_ms;
  uint8_t estop_ok, latched, reason, phase;
};
#pragma pack(pop)
}  // namespace

TipFk make_tip_fk(
  const std::string & urdf, const std::string & base_frame, const std::string & tcp_frame,
  const std::vector<std::string> & joint_names, std::string & why)
{
  struct Chain
  {
    KDL::Chain chain;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> solver;
    KDL::JntArray q;
    std::vector<size_t> index;
  };
  KDL::Tree tree;
  auto s = std::make_shared<Chain>();
  if (!kdl_parser::treeFromString(urdf, tree) || !tree.getChain(base_frame, tcp_frame, s->chain)) {
    why = "no chain " + base_frame + " -> " + tcp_frame + " in the robot description";
    return {};
  }
  for (const auto & segment : s->chain.segments) {
    if (segment.getJoint().getType() == KDL::Joint::None) {continue;}
    const auto it = std::find(joint_names.begin(), joint_names.end(), segment.getJoint().getName());
    if (it == joint_names.end()) {
      why = "chain joint " + segment.getJoint().getName() + " is not one of the arm's joints";
      return {};
    }
    s->index.push_back(static_cast<size_t>(it - joint_names.begin()));
  }
  s->solver = std::make_unique<KDL::ChainFkSolverPos_recursive>(s->chain);
  s->q.resize(s->chain.getNrOfJoints());
  return [s](const Vec7 & q) {
           for (size_t k = 0; k < s->index.size(); ++k) {s->q(k) = q[s->index[k]];}
           KDL::Frame frame;
           s->solver->JntToCart(s->q, frame);
           const KDL::Vector z = frame.M.UnitZ();
           return TipPose{{frame.p.x(), frame.p.y(), frame.p.z()}, {z.x(), z.y(), z.z()}};
         };
}

TatbotArm::~TatbotArm() {release();}

TatbotArm::CallbackReturn TatbotArm::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {return CallbackReturn::ERROR;}
  const auto & p = info_.hardware_parameters;
  auto text = [&](const std::string & key, const std::string & fallback) {
      const auto it = p.find(key);
      return it == p.end() ? fallback : it->second;
    };
  auto num = [&](const std::string & key, double fallback) {
      const auto it = p.find(key);
      return it == p.end() || it->second.empty() ? fallback : std::stod(it->second);
    };
  try {
    arm_ = text("arm", "right");
    sdk_ = text("sdk", "fake");
    ip_ = text("ip", "");
    end_effector_ = text("end_effector", "wxai_v0_follower");
    flight_path_ = text("flight_path", "");
    rt_priority_ = static_cast<int>(num("rt_priority", 80));
    for (double c : numbers(text("rt_cpus", ""))) {rt_cpus_.push_back(static_cast<int>(c));}
    estop_settings_.source = estop::source_from_name(text("estop_source", "none"));
    if (estop_settings_.source < 0) {
      RCLCPP_FATAL(get_logger(), "estop_source must be none, serial or udp");
      return CallbackReturn::ERROR;
    }
    estop_settings_.device = text("estop_device", "/dev/tatbot-estop");
    estop_settings_.timeout_s = num("estop_timeout_s",
        estop_settings_.source == names::kEstopUdp ? 0.15 : 0.10);
    estop_settings_.udp_port = static_cast<int>(num("estop_udp_port", 7640));
    estop_settings_.relay_addr = text("estop_relay_addr", "");
    estop_settings_.debounce_frames = static_cast<int>(num("estop_debounce_frames", 3));
    probe_enabled_ = text("probe_enabled", "false") == "true" || text("probe_enabled", "") == "1";
    probe_settings_.udp_port = static_cast<int>(num("probe_udp_port", 7641));
    probe_settings_.relay_addr = text("probe_relay_addr", "");
    probe_settings_.timeout_s = num("probe_timeout_s", 0.2);
    Config & c = cfg_;
    c.carriage_qualified = text("carriage_qualified", "true") != "false";
    c.step_limit_rad = num("step_limit_rad", c.step_limit_rad);
    c.step_limit_m = num("step_limit_m", c.step_limit_m);
    c.feedback_warmup_s = num("feedback_warmup_s", c.feedback_warmup_s);
    c.stall_error_rad = num("stall_error_rad", c.stall_error_rad);
    c.stall_time_s = num("stall_time_s", c.stall_time_s);
    c.stall_progress_rad = num("stall_progress_rad", c.stall_progress_rad);
    c.over_velocity_rad_s = num("over_velocity_rad_s", c.over_velocity_rad_s);
    c.over_velocity_m_s = num("over_velocity_m_s", c.over_velocity_m_s);
    c.carriage_baseline_samples = static_cast<int>(num("carriage_baseline_samples", c.carriage_baseline_samples));
    c.carriage_trip_ticks = static_cast<int>(num("carriage_trip_ticks", c.carriage_trip_ticks));
    c.carriage_settle_ticks = static_cast<int>(num("carriage_settle_ticks", c.carriage_settle_ticks));
    c.carriage_rebaseline_samples =
      static_cast<int>(num("carriage_rebaseline_samples", c.carriage_rebaseline_samples));
    c.carriage_judge_below_rad_s = num("carriage_judge_below_rad_s", c.carriage_judge_below_rad_s);
    c.carriage_retract_s = num("carriage_retract_s", c.carriage_retract_s);
    c.carriage_contact_cap_n = num("carriage_contact_cap_n", c.carriage_contact_cap_n);
    c.carriage_contact_deflect_m = num("carriage_contact_deflect_m", c.carriage_contact_deflect_m);
    c.carriage_retract_m = num("carriage_retract_m", c.carriage_retract_m);
    c.tip_lag_trip_m = num("tip_lag_trip_m", c.tip_lag_trip_m);
    c.tip_lag_hold_s = num("tip_lag_hold_s", c.tip_lag_hold_s);
    c.tip_lag_arm_after_s = num("tip_lag_arm_after_s", c.tip_lag_arm_after_s);
    c.contact_force_n = num("contact_force_n", c.contact_force_n);
    c.contact_hold_s = num("contact_hold_s", c.contact_hold_s);
    c.contact_baseline_s = num("contact_baseline_s", c.contact_baseline_s);
    c.landing_takeover_s = num("landing_takeover_s", c.landing_takeover_s);
    c.landing_staged_s = num("landing_staged_s", c.landing_staged_s);
    c.landing_sleep_s = num("landing_sleep_s", c.landing_sleep_s);
    c.landing_verify_rad = num("landing_verify_rad", c.landing_verify_rad);
    c.landing_verify_carriage_m = num("landing_verify_carriage_m", c.landing_verify_carriage_m);
    c.landing_budget_s = num("landing_budget_s", c.landing_budget_s);
    const auto staged = numbers(text("staged_positions", ""));
    if (staged.size() == kJoints) {std::copy(staged.begin(), staged.end(), c.staged_positions.begin());}
    fake_page_ = p.count("fake_page_z") && !p.at("fake_page_z").empty();
    fake_page_z_ = num("fake_page_z", 0.0);
    fake_page_stiffness_ = num("fake_page_stiffness_n_m", 0.0);
    carriage_min_m_ = num("carriage_min_m", carriage_min_m_);
    carriage_max_m_ = num("carriage_max_m", carriage_max_m_);
    controller_query_period_s_ = num("controller_query_period_s", controller_query_period_s_);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "bad hardware parameter: %s", e.what());
    return CallbackReturn::ERROR;
  }
  if (sdk_ != "fake" && sdk_ != "real") {
    RCLCPP_FATAL(get_logger(), "sdk must be fake or real, not '%s'", sdk_.c_str());
    return CallbackReturn::ERROR;
  }

  // Joints by name: <arm>/joint_0..joint_5, then <arm>/left_carriage_joint.
  std::vector<std::string> joint_names(kJoints);
  for (size_t k = 0; k < kCarriage; ++k) {joint_names[k] = arm_ + "/joint_" + std::to_string(k);}
  joint_names[kCarriage] = arm_ + "/left_carriage_joint";
  if (info_.joints.size() != kJoints) {
    RCLCPP_FATAL(get_logger(), "expected 7 joints, got %zu", info_.joints.size());
    return CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < kJoints; ++i) {
    const auto it = std::find(joint_names.begin(), joint_names.end(), info_.joints[i].name);
    if (it == joint_names.end()) {
      RCLCPP_FATAL(get_logger(), "unexpected joint %s", info_.joints[i].name.c_str());
      return CallbackReturn::ERROR;
    }
    index_[i] = static_cast<size_t>(it - joint_names.begin());
    const auto limit = info_.limits.find(info_.joints[i].name);
    if (limit != info_.limits.end() && limit->second.has_position_limits) {
      cfg_.lower[index_[i]] = limit->second.min_position;
      cfg_.upper[index_[i]] = limit->second.max_position;
    }
  }
  std::string why;
  fk_ = make_tip_fk(info_.original_xml, text("base_frame", arm_ + "/base_link"),
      text("tcp_frame", arm_ + "/tcp"), joint_names, why);
  if (!fk_) {RCLCPP_ERROR(get_logger(), "tip-lag guard has no FK: %s", why.c_str());}

  const auto n = info_.joints.size();
  position_.assign(n, 0.0);
  velocity_.assign(n, 0.0);
  effort_.assign(n, 0.0);
  position_command_.assign(n, std::numeric_limits<double>::quiet_NaN());
  velocity_command_.assign(n, 0.0);
  safety_state_.assign(names::kSafetyState.size(), 0.0);
  safety_command_.assign(names::kSafetyCommand.size(), 0.0);
  core_ = std::make_unique<ArmCore>(cfg_, fk_);
  publish_safety();
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> TatbotArm::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_[i]);
    out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_[i]);
    out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &effort_[i]);
  }
  for (const auto & gpio : info_.gpios) {
    for (size_t i = 0; i < names::kSafetyState.size(); ++i) {
      out.emplace_back(gpio.name, std::string(names::kSafetyState[i]), &safety_state_[i]);
    }
  }
  return out;
}

std::vector<hardware_interface::CommandInterface> TatbotArm::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_command_[i]);
    out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_command_[i]);
  }
  for (const auto & gpio : info_.gpios) {
    for (size_t i = 0; i < names::kSafetyCommand.size(); ++i) {
      out.emplace_back(gpio.name, std::string(names::kSafetyCommand[i]), &safety_command_[i]);
    }
  }
  return out;
}

TatbotArm::CallbackReturn TatbotArm::on_configure(const State &)
{
  try {
    estop_ = estop::shared(estop_settings_);
    if (estop_ && !(estop_->settings() == estop_settings_)) {
      RCLCPP_WARN(get_logger(), "the other arm opened the e-stop reader with different settings; sharing it");
    }
    if (estop_settings_.source == names::kEstopUdp && estop_settings_.relay_addr.empty()) {
      RCLCPP_ERROR(get_logger(), "estop udp without estop_relay_addr accepts no frame: the arm stays held");
    }
    if (probe_enabled_) {
      probe_ = estop::shared_probe(probe_settings_);
      if (probe_settings_.relay_addr.empty()) {
        RCLCPP_ERROR(get_logger(), "probe without probe_relay_addr accepts no frame: GUARD_PROBE holds");
      }
    }
    if (sdk_ == "real") {
      lock_ = runtime::DriverLock::acquire();
      backend_ = make_sdk_backend(ip_, end_effector_);
      if (!backend_) {
        RCLCPP_FATAL(get_logger(), "built without the Trossen SDK (TATBOT_ARM_SDK off or offline)");
        return CallbackReturn::ERROR;
      }
    } else {
      FakeOptions options;
      options.start = cfg_.staged_positions;
      if (fake_page_) {
        options.fk = fk_;
        options.page_z = fake_page_z_;
        options.page_stiffness_n_m = fake_page_stiffness_;
      }
      auto fake = std::make_unique<FakeBackend>(options);
      fake_ = fake.get();
      backend_ = std::move(fake);
    }
    // SCHED_FIFO and the CPU pin first, on a helper thread, so the SDK's daemon thread that
    // configure() starts inherits both (cpp/teleop realtime::apply).
    std::exception_ptr failure;
    runtime::RtSetup rt;
    std::thread helper([&]() {
        rt = runtime::apply_realtime(rt_priority_, rt_cpus_);
        std::lock_guard<std::mutex> lock(connect_mutex());
        try {backend_->connect();} catch (...) {failure = std::current_exception();}
      });
    helper.join();
    if (failure) {std::rethrow_exception(failure);}
    {
      // A power-cycled controller boots with a -4 mm carriage floor, while a loaded carriage rests
      // on its stop near -4.7 mm: holding that pose idles motor 6 and every later command fails
      // with "modes different than configured modes" (cpp/teleop/arm_recover.cpp). Apply the
      // follower limits of config/trossen/follower.yaml before the first command.
      std::lock_guard<std::mutex> lock(connect_mutex());
      const auto before = backend_->set_carriage_limits(carriage_min_m_, carriage_max_m_);
      if (sdk_ == "real") {
        RCLCPP_INFO(get_logger(), "%s carriage limits [%.4f, %.4f] m (controller had [%.4f, %.4f])",
          arm_.c_str(), carriage_min_m_, carriage_max_m_, before.first, before.second);
      }
    }
    if (rt.fifo && rt.affinity) {
      RCLCPP_INFO(get_logger(), "%s arm connected (%s): SCHED_FIFO %d on %zu cores",
        arm_.c_str(), sdk_.c_str(), rt_priority_, rt.cpus.size());
    } else {
      RCLCPP_WARN(get_logger(), "%s arm connected (%s) without full real time: %s",
        arm_.c_str(), sdk_.c_str(), rt.error.c_str());
    }
    // configure() returns before the daemon has robot output: wait for a non-zero measurement,
    // then for feedback_warmup_s of it, so the seeded first command is judged against a real pose.
    const double until = now_s() + kFeedbackWaitS;
    double nonzero_since = -1;
    while (true) {
      fb_ = backend_->read();
      const double t = now_s();
      core_->observe_feedback(t, fb_);
      if (nonzero_since < 0 && std::any_of(fb_.q.begin(), fb_.q.end(), [](double x) {return x != 0.0;})) {
        nonzero_since = t;
      }
      if (nonzero_since >= 0 && t - nonzero_since >= cfg_.feedback_warmup_s) {break;}
      if (nonzero_since < 0 && t >= until) {
        throw std::runtime_error("no non-zero joint measurement within 1 s of connecting");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (size_t i = 0; i < info_.joints.size(); ++i) {position_[i] = fb_.q[index_[i]];}
    if (!flight_path_.empty()) {
      std::filesystem::create_directories(flight_path_);
      recorder_ = std::make_unique<runtime::FlightRecorder>(
        (std::filesystem::path(flight_path_) / (arm_ + "-flight.bin")).string());
      const std::string header = "tatbot-flight 1 " + arm_ + " " + std::to_string(sizeof(FlightRecord)) + "\n";
      recorder_->append(header.data(), header.size());
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "%s arm configure failed: %s", arm_.c_str(), e.what());
    release();
    return CallbackReturn::ERROR;
  }
  // TCP getters and the post-landing idle, off the control thread. Every SDK call, these included,
  // holds the driver's one data mutex for its whole round trip (trossen_arm.hpp "Mutex ownership"),
  // so each query stalls the 400 Hz loop by 1.5-2.5 ms: at 10 Hz that was ~120 late ticks a minute
  // on the pink arm (bench 2026-09-26). A controller error also surfaces as an SDK
  // exception on the next read or command (sticky_error_), so the query runs every
  // controller_query_period_s; an idle request is served within 100 ms.
  aux_ = std::make_shared<Aux>();
  aux_thread_ = std::thread([aux = aux_, backend = backend_.get(), logger = get_logger(), arm = arm_,
      period = controller_query_period_s_]() {
      sched_param other{};
      sched_setscheduler(0, SCHED_OTHER, &other);
      bool reported = false;
      auto next_query = std::chrono::steady_clock::now();
      while (!aux->stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const bool idle = aux->idle_request.load();
        const auto now = std::chrono::steady_clock::now();
        if (!idle && now < next_query) {continue;}
        next_query = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(period));
        std::string error;
        {
          std::lock_guard<std::timed_mutex> lock(aux->tcp);
          try {
            error = backend->error();
            if (aux->idle_request.exchange(false)) {backend->set_idle();}
          } catch (const std::exception & e) {
            error = std::string("controller unreachable: ") + e.what();
          }
        }
        aux->error.store(!error.empty());
        if (!error.empty() && !reported) {
          RCLCPP_ERROR(logger, "%s controller error: %s", arm.c_str(), error.c_str());
        }
        reported = !error.empty();
      }
      aux->done.store(true);
    });
  return CallbackReturn::SUCCESS;
}

TatbotArm::CallbackReturn TatbotArm::on_activate(const State &)
{
  if (!backend_) {return CallbackReturn::ERROR;}
  try {
    fb_ = backend_->read();
    core_->activate(now_s(), fb_);
    if (core_->phase() != Phase::kLanded) {
      std::unique_lock<std::timed_mutex> lock(aux_->tcp, std::chrono::milliseconds(500));
      if (!lock) {throw std::runtime_error("the controller is not answering (a TCP call is stuck)");}
      backend_->set_position_mode();
      backend_->command(core_->hold_pose(), Vec7{});
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "%s arm activate failed: %s", arm_.c_str(), e.what());
    return CallbackReturn::ERROR;
  }
  // The JTC starts from the command interfaces' values: the measured pose.
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    position_command_[i] = fb_.q[index_[i]];
    velocity_command_[i] = 0.0;
  }
  publish_safety();
  RCLCPP_INFO(get_logger(), "%s arm active, holding the measured pose", arm_.c_str());
  return CallbackReturn::SUCCESS;
}

void TatbotArm::hold(int reason)
{
  if (!core_ || !backend_) {return;}
  send(core_->hold_now(now_s(), fb_, reason));
  publish_safety();
  const std::string event = core_->take_event();
  if (!event.empty()) {RCLCPP_WARN(get_logger(), "%s: %s", arm_.c_str(), event.c_str());}
}

TatbotArm::CallbackReturn TatbotArm::on_deactivate(const State &)
{
  hold(names::kLatchDeactivated);   // hold, never idle
  return CallbackReturn::SUCCESS;
}

TatbotArm::CallbackReturn TatbotArm::on_error(const State &)
{
  hold(names::kLatchControllerError);
  return CallbackReturn::SUCCESS;
}

TatbotArm::CallbackReturn TatbotArm::on_cleanup(const State &)
{
  hold(names::kLatchDeactivated);
  release();
  return CallbackReturn::SUCCESS;
}

TatbotArm::CallbackReturn TatbotArm::on_shutdown(const State &)
{
  hold(names::kLatchDeactivated);
  release();
  return CallbackReturn::SUCCESS;
}

void TatbotArm::release()
{
  // Wait briefly for the aux thread: a TCP getter stuck on a vanished controller would otherwise
  // hold the stop until systemd's SIGKILL. A stuck thread is detached and keeps the backend.
  bool aux_stuck = false;
  if (aux_thread_.joinable()) {
    aux_->stop = true;
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (!aux_->done && std::chrono::steady_clock::now() < end) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (aux_->done) {
      aux_thread_.join();
    } else {
      aux_stuck = true;
      aux_thread_.detach();
      RCLCPP_ERROR(get_logger(), "%s controller not answering; leaving its session open", arm_.c_str());
    }
  }
  if (recorder_) {recorder_->finish();}
  recorder_.reset();
  const Phase phase = core_ ? core_->phase() : Phase::kInactive;
  if (backend_ && (aux_stuck || (phase != Phase::kLanded && phase != Phase::kInactive))) {
    // Commanded and not landed: closing the SDK session (cleanup(), also run by its destructor)
    // idles the controller and the arm falls. Leave the session open, holding the last sent pose,
    // with the driver lock, until the process ends (wxai_teleop's holding handover). A configure
    // that failed before any command (kInactive) closes normally.
    struct Held
    {
      std::unique_ptr<Backend> backend;
      std::shared_ptr<runtime::DriverLock> lock;
    };
    (void)new Held{std::move(backend_), lock_};   // never freed on purpose
  }
  if (backend_) {fake_ = nullptr;}   // a held fake stays readable by the tests
  backend_.reset();   // landed and idle, or never commanded: close the SDK session
  estop_.reset();
  probe_.reset();
  lock_.reset();
}

hardware_interface::return_type TatbotArm::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!backend_) {return return_type::OK;}
  const double t = now_s();
  // Longest read-to-read period over the last second: this window and the one before.
  if (last_read_ >= 0) {window_max_ = std::max(window_max_, t - last_read_);}
  last_read_ = t;
  if (t - window_start_ >= 1.0) {
    previous_max_ = window_max_;
    window_max_ = 0;
    window_start_ = t;
  }
  period_max_ms_ = std::max(previous_max_, window_max_) * 1e3;
  try {
    fb_ = backend_->read();
  } catch (const std::exception &) {
    sticky_error_ = true;   // the SDK expects the process to end after an exception
  }
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    position_[i] = fb_.q[index_[i]];
    velocity_[i] = fb_.qd[index_[i]];
    effort_[i] = fb_.effort[index_[i]];
  }
  return return_type::OK;
}

void TatbotArm::send(const Output & out)
{
  last_out_ = out;
  if (out.idle_request) {aux_->idle_request = true;}
  if (!out.send || !backend_) {return;}
  try {
    backend_->command(out.q, out.qd);
  } catch (const std::exception &) {
    sticky_error_ = true;
  }
}

hardware_interface::return_type TatbotArm::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!backend_) {return return_type::OK;}
  Commands cmd;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    cmd.q[index_[i]] = position_command_[i];
    cmd.qd[index_[i]] = velocity_command_[i];
  }
  cmd.guard_mode = safety_command_[0];
  cmd.unlatch = safety_command_[1];
  cmd.land = safety_command_[2];
  EstopStatus estop;
  const int64_t steady = estop::now_ns();
  if (estop_) {estop = estop_->status(steady);}
  ProbeStatus probe;
  if (probe_) {probe = probe_->status(steady);}
  const double t = now_s();
  const Output out = core_->tick(t, fb_, cmd, estop, aux_->error.load() || sticky_error_.load(), probe);
  if (out.reset_commands) {
    for (size_t i = 0; i < info_.joints.size(); ++i) {
      position_command_[i] = fb_.q[index_[i]];
      velocity_command_[i] = 0.0;
    }
  }
  send(out);
  publish_safety();
  if (recorder_) {
    FlightRecord r{};
    r.t = t;
    for (size_t k = 0; k < kJoints; ++k) {
      r.q[k] = static_cast<float>(fb_.q[k]);
      r.qd[k] = static_cast<float>(fb_.qd[k]);
      r.effort[k] = static_cast<float>(fb_.effort[k]);
      r.cmd_q[k] = static_cast<float>(cmd.q[k]);
      r.sent_q[k] = out.send ? static_cast<float>(out.q[k]) : NAN;
      r.sent_qd[k] = out.send ? static_cast<float>(out.qd[k]) : NAN;
    }
    r.estop_age_s = static_cast<float>(estop.age_s);
    r.period_ms = static_cast<float>(period_max_ms_);
    r.estop_ok = estop.ok;
    r.latched = static_cast<uint8_t>(safety_state_[4]);
    r.reason = static_cast<uint8_t>(safety_state_[5]);
    r.phase = static_cast<uint8_t>(core_->phase());
    recorder_->append(&r, sizeof(r));
  }
  const std::string event = core_->take_event();
  if (!event.empty()) {RCLCPP_WARN(get_logger(), "%s: %s", arm_.c_str(), event.c_str());}
  return return_type::OK;
}

void TatbotArm::publish_safety()
{
  const auto & s = core_->safety();
  std::copy(s.begin(), s.end(), safety_state_.begin());
  safety_state_.back() = period_max_ms_;
}

}  // namespace tatbot_hardware

PLUGINLIB_EXPORT_CLASS(tatbot_hardware::TatbotArm, hardware_interface::SystemInterface)
