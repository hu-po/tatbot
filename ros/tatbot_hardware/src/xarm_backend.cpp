// sdk=real on a UFACTORY xArm (internal plans, arm models section 3.2). Commands go through UFACTORY's C++ SDK
// (xArm-CPLUS-SDK, BSD-3-Clause, THIRD_PARTY.md) in servo mode: the 400 Hz control thread only writes the latest
// pose into a slot, and a sender thread one priority below it sends that slot every 4 ms with set_servo_angle_j, a
// TCP round trip, holding by resending the same pose. State comes from the controller's report streams through our
// own parser (xarm_report.hpp): the real-time stream for the joints, the normal stream for errors and the stored
// settings.
//
// Nothing is cleared or powered behind the operator's back. connect() (the stack's start, or a wake after a landing)
// opens the session and reads, never writes. set_position_mode() (every activation) is the explicit wake: it clears a
// standing error or warning, enables the motors holding where they are, sets the controller's TCP offset (zero: the
// kinematics are the URDF's), load and collision sensitivity where they differ, reads each back, and enters servo
// mode. The controller takes a setting only from a ready arm and each one leaves it reconfigured (state 5), so the
// settings follow the enable, one at a time, each followed by set_state(0) (2026-11-08: a load sent with the motors
// off, then one sent after the offset, came back STATE_NOT_READY). disconnect() leaves the motors on, holding the last
// pose: an unpowered xArm sags about 3 mm, so motors go off only on an explicit rig sleep.
#include "tatbot_hardware/backend.hpp"
#include "tatbot_hardware/xarm_report.hpp"

#ifdef TATBOT_HAVE_XARM_SDK
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <xarm/wrapper/xarm_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace tatbot_hardware
{
namespace
{
using Clock = std::chrono::steady_clock;

constexpr double kFirstFrameS = 3.0;      // both report streams answer within this of connecting
constexpr double kReportSilentS = 0.2;    // 50 real-time frames missing: the stream has stopped
constexpr double kModeChangeS = 2.0;      // servo mode and a ready state within this of the wake
constexpr int kRefusedSends = 25;         // 0.1 s of refused servo commands with no controller error to explain them
constexpr double kOffsetToleranceMm = 0.01, kLoadToleranceKg = 0.005, kComToleranceMm = 0.5;

double seconds_since(Clock::time_point t) {return std::chrono::duration<double>(Clock::now() - t).count();}

// What a controller error code means here; the rest are the controller's own (xarm_api_code.md).
std::string error_text(int code, int sensitivity)
{
  std::ostringstream out;
  out << "controller error C" << code;
  if (code == 1) {out << ": the control box's emergency stop";}
  if (code == 2) {out << ": the emergency stop input";}
  if (code == 31) {out << ": a collision at sensitivity " << sensitivity;}
  return out.str();
}

int open_report(const std::string & ip, int port, double timeout_s)
{
  const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (fd < 0) {throw std::runtime_error("report socket: " + std::string(std::strerror(errno)));}
  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &to.sin_addr) != 1) {
    ::close(fd);
    throw std::runtime_error("not an IPv4 address: " + ip);
  }
  int error = 0;
  if (::connect(fd, reinterpret_cast<sockaddr *>(&to), sizeof(to)) != 0) {
    error = errno;
    if (error == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      socklen_t len = sizeof(error);
      error = ::poll(&p, 1, static_cast<int>(timeout_s * 1000)) == 1 &&
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) == 0 ? error : ETIMEDOUT;
    }
  }
  if (error != 0) {
    ::close(fd);
    throw std::runtime_error("no report stream at " + ip + ":" + std::to_string(port) + ": " + std::strerror(error));
  }
  return fd;
}

class XArmBackend : public Backend
{
public:
  explicit XArmBackend(XArmOptions options) : o_(std::move(options)) {}
  ~XArmBackend() override {disconnect();}

  void connect() override
  {
    disconnect();
    // Radians; the controller checks its own joint and TCP limits; its own report for the SDK's ready checks.
    api_ = std::make_unique<XArmAPI>(o_.ip, true, false, false, true, true, false, true, true, -1, 512, 7, false,
        "normal");
    if (!api_->is_connected()) {
      api_.reset();
      throw std::runtime_error("no xArm controller answering at " + o_.ip);
    }
    for (auto [port, fd] : {std::pair{xarm::kRealtimePort, &realtime_fd_}, std::pair{xarm::kNormalPort, &normal_fd_}}) {
      *fd = open_report(o_.ip, port, kFirstFrameS);
    }
    connected_at_ = Clock::now();
    stop_.store(false);
    stream_lost_.store(false);
    realtime_frames_.store(0);
    normal_frames_.store(0);
    reports_ = std::thread([this]() {report_loop();});
    wait_for([this]() {return realtime_frames_ > 0 && normal_frames_ > 0;}, kFirstFrameS,
      "no report from the controller within 3 s of connecting");
    const xarm::Normal stored = normal();
    std::ostringstream note;
    note << "xArm at " << o_.ip << ", firmware " << reinterpret_cast<const char *>(api_->version) << ", state "
         << stored.state << ", error C" << stored.error_code << ", stored load " << std::fixed << std::setprecision(3)
         << stored.tcp_load[0] << " kg, TCP offset z " << std::setprecision(1) << stored.tcp_offset[2]
         << " mm, collision sensitivity " << stored.collision_sensitivity << " (the wake sets the arm's)";
    description_ = note.str();
  }

  Feedback read() override
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    Feedback fb;
    fb.q = realtime_.q;
    fb.qd = realtime_.qd;
    fb.effort = realtime_.torque;
    fb.controller_id = realtime_frames_;
    fb.controller_us = realtime_.timestamp_us;
    return fb;
  }

  void command(const Vec7 & q, const Vec7 &) override
  {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    slot_ = q;
    slot_set_ = true;
  }

  void set_position_mode() override
  {
    const xarm::Normal before = normal();
    std::lock_guard<std::mutex> lock(sdk_mutex_);
    woke_from_.clear();   // describe() names this wake's cleared error, not an earlier one's
    if (before.error_code != 0 || before.warn_code != 0) {
      woke_from_ = error_text(before.error_code, before.collision_sensitivity) + ", warning " +
        std::to_string(before.warn_code);
      check(api_->clean_error(), "clean_error");
      check(api_->clean_warn(), "clean_warn");
    }
    enter_mode(0, "position mode");   // motors on, holding where they stand
    configure();
    enter_mode(xarm::kModeServo, "servo mode");
    enabled_.store(true);
    if (!sender_.joinable()) {
      stop_send_.store(false);
      sender_ = std::thread([this]() {send_loop();});
    }
  }

  void disconnect() override
  {
    stop_send_.store(true);
    if (sender_.joinable()) {sender_.join();}
    stop_.store(true);
    if (reports_.joinable()) {reports_.join();}
    for (int * fd : {&realtime_fd_, &normal_fd_}) {
      if (*fd >= 0) {::close(*fd);}
      *fd = -1;
    }
    std::lock_guard<std::mutex> lock(sdk_mutex_);
    api_.reset();   // closes the SDK's sockets; the motors stay on, holding
    enabled_.store(false);
    slot_set_ = false;
  }

  std::string error() override
  {
    if (!api_) {return "";}
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (stream_lost_) {return "the controller's report stream closed or fell out of step";}
    const double silent = realtime_frames_ ? seconds_since(realtime_at_) : seconds_since(connected_at_);
    if (silent > kReportSilentS) {return "no real-time report for " + std::to_string(silent) + " s";}
    if (normal_.error_code != 0) {return error_text(normal_.error_code, normal_.collision_sensitivity);}
    if (enabled_.load() && (realtime_.state == xarm::kStateStopped || realtime_.state == xarm::kStateReconfigured)) {
      return "the controller left servo mode (state " + std::to_string(realtime_.state) + ") without an error code";
    }
    if (refused_sends_.load() >= kRefusedSends) {
      return "the controller refused " + std::to_string(refused_sends_.load()) + " servo commands (code " +
             std::to_string(last_refusal_.load()) + ")";
    }
    return "";
  }

  std::string describe() const override
  {
    return description_ + (woke_from_.empty() ? "" : "; woke from " + woke_from_);
  }

private:
  static void check(int code, const char * call)
  {
    if (code != 0) {throw std::runtime_error(std::string(call) + " returned " + std::to_string(code));}
  }

  xarm::Realtime realtime() const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return realtime_;
  }

  xarm::Normal normal() const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return normal_;
  }

  template<class Ready>
  void wait_for(Ready ready, double timeout_s, const char * what)
  {
    const auto until = Clock::now() + std::chrono::duration<double>(timeout_s);
    while (!ready()) {
      if (Clock::now() > until) {
        throw std::runtime_error(std::string(what) + (stream_lost_ ? ": the report stream closed" : ""));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  // Motors on in `mode`, state ready.
  void enter_mode(int mode, const char * what)
  {
    check(api_->motion_enable(true), "motion_enable");
    check(api_->set_mode(mode), "set_mode");
    ready(mode, what);
  }

  // set_state(0), then a real-time report sent after it shows `mode` and neither stopped nor reconfigured.
  void ready(int mode, const char * what)
  {
    check(api_->set_state(0), "set_state");
    const int seen = realtime_frames_.load();
    wait_for([this, mode, seen]() {
        const auto r = realtime();
        return realtime_frames_.load() > seen + 2 && r.mode == mode && r.state != xarm::kStateStopped &&
               r.state != xarm::kStateReconfigured;
      }, kModeChangeS, (std::string("the controller did not ready itself in ") + what).c_str());
  }

  // The controller's TCP offset to zero, its load and collision sensitivity to the arm's (a NaN mass or a negative
  // sensitivity keeps the controller's): each that differs written alone, without the SDK's unbounded wait, the arm
  // readied after it, and all of them read back from the normal report within kModeChangeS.
  void configure()
  {
    const xarm::Normal now = normal();
    const auto off = [](double v) {return std::abs(v) > kOffsetToleranceMm;};
    if (std::any_of(now.tcp_offset.begin(), now.tcp_offset.end(), off)) {
      fp32 zero[6] = {0, 0, 0, 0, 0, 0};   // a TCP pose; arm-model: not joints
      check(api_->set_tcp_offset(zero, false), "set_tcp_offset");
      ready(0, "position mode after the TCP offset");
    }
    if (!std::isnan(o_.payload_mass_kg) && unsettled(now).find("load") != std::string::npos) {
      fp32 com[3];
      for (size_t i = 0; i < 3; ++i) {com[i] = static_cast<fp32>(o_.payload_com_m[i] * 1000.0);}
      check(api_->set_tcp_load(static_cast<fp32>(o_.payload_mass_kg), com, false), "set_tcp_load");
      ready(0, "position mode after the load");
    }
    if (o_.collision_sensitivity >= 0 && now.collision_sensitivity != o_.collision_sensitivity) {
      check(api_->set_collision_sensitivity(o_.collision_sensitivity, false), "set_collision_sensitivity");
      ready(0, "position mode after the collision sensitivity");
    }
    const auto until = Clock::now() + std::chrono::duration<double>(kModeChangeS);
    for (std::string left = unsettled(normal()); !left.empty(); left = unsettled(normal())) {
      if (Clock::now() > until) {throw std::runtime_error("the controller did not take its settings:" + left);}
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  // What of the arm's settings a normal report does not show yet; "" when all of them.
  std::string unsettled(const xarm::Normal & now) const
  {
    std::string out;
    for (double v : now.tcp_offset) {
      if (std::abs(v) > kOffsetToleranceMm) {out = " TCP offset;";}
    }
    if (!std::isnan(o_.payload_mass_kg)) {
      if (std::abs(now.tcp_load[0] - o_.payload_mass_kg) > kLoadToleranceKg) {out += " load mass;";}
      for (size_t i = 0; i < 3; ++i) {
        if (std::abs(now.tcp_load[1 + i] - o_.payload_com_m[i] * 1000.0) > kComToleranceMm) {out += " load centre;";}
      }
    }
    if (o_.collision_sensitivity >= 0 && now.collision_sensitivity != o_.collision_sensitivity) {
      out += " collision sensitivity;";
    }
    return out;
  }

  void report_loop()
  {
    xarm::FrameReader realtime_reader, normal_reader;
    pollfd fds[2] = {{realtime_fd_, POLLIN, 0}, {normal_fd_, POLLIN, 0}};
    xarm::FrameReader * readers[2] = {&realtime_reader, &normal_reader};
    uint8_t buffer[16384];
    while (!stop_.load()) {
      if (::poll(fds, 2, 20) <= 0) {continue;}
      for (size_t s = 0; s < 2; ++s) {
        if (!(fds[s].revents & (POLLIN | POLLHUP | POLLERR))) {continue;}
        const ssize_t got = ::recv(fds[s].fd, buffer, sizeof(buffer), 0);
        if (got <= 0 && !(got < 0 && (errno == EAGAIN || errno == EINTR))) {
          stream_lost_.store(true);
          return;
        }
        if (got > 0) {readers[s]->feed(buffer, static_cast<size_t>(got));}
        while (auto frame = readers[s]->next()) {take(s == 0, *frame);}
        if (readers[s]->corrupt()) {
          stream_lost_.store(true);
          return;
        }
      }
    }
  }

  void take(bool realtime_stream, const std::vector<uint8_t> & frame)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (realtime_stream) {
      if (const auto r = xarm::decode_realtime(frame)) {
        realtime_ = *r;
        realtime_at_ = Clock::now();
        ++realtime_frames_;
      }
    } else if (const auto n = xarm::decode_normal(frame)) {
      normal_ = *n;
      ++normal_frames_;
    }
  }

  // One priority below the thread that connected (the control loop's, through the adapter's real-time helper).
  void send_loop()
  {
    int policy;
    sched_param param{};
    if (::pthread_getschedparam(::pthread_self(), &policy, &param) == 0 && policy == SCHED_FIFO &&
      param.sched_priority > 1)
    {
      --param.sched_priority;
      ::pthread_setschedparam(::pthread_self(), SCHED_FIFO, &param);
    }
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / o_.send_rate_hz));
    auto next = Clock::now();
    while (!stop_send_.load()) {
      next += period;
      std::this_thread::sleep_until(next);
      Vec7 q;
      {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        if (!slot_set_) {continue;}
        q = slot_;
      }
      fp32 angles[7];
      for (size_t j = 0; j < kJoints; ++j) {angles[j] = static_cast<fp32>(q[j]);}
      int code;
      {
        std::lock_guard<std::mutex> lock(sdk_mutex_);
        code = api_ ? api_->set_servo_angle_j(angles) : -1;
      }
      if (code == 0) {
        refused_sends_.store(0);
      } else {
        refused_sends_.fetch_add(1);
        last_refusal_.store(code);
      }
    }
  }

  XArmOptions o_;
  std::unique_ptr<XArmAPI> api_;
  std::mutex sdk_mutex_;   // one SDK call at a time: the sender's and the lifecycle's
  int realtime_fd_ = -1, normal_fd_ = -1;
  std::thread reports_, sender_;
  std::atomic<bool> stop_{false}, stop_send_{false}, enabled_{false};
  mutable std::mutex state_mutex_;
  xarm::Realtime realtime_;
  xarm::Normal normal_;
  std::atomic<int> realtime_frames_{0}, normal_frames_{0}, refused_sends_{0}, last_refusal_{0};
  Clock::time_point realtime_at_{}, connected_at_{};
  std::atomic<bool> stream_lost_{false};
  std::mutex slot_mutex_;
  Vec7 slot_{};
  bool slot_set_ = false;
  std::string description_, woke_from_;
};

}  // namespace

std::unique_ptr<Backend> make_xarm_backend(const XArmOptions & options)
{
  return std::make_unique<XArmBackend>(options);
}

}  // namespace tatbot_hardware

#else

namespace tatbot_hardware
{
std::unique_ptr<Backend> make_xarm_backend(const XArmOptions &) {return nullptr;}
}  // namespace tatbot_hardware

#endif
