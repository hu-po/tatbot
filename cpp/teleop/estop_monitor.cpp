#include "estop_monitor.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

namespace tatbot::estop
{

namespace
{
constexpr int DEBOUNCE_FRAMES = 3;
constexpr int HEARTBEAT_TIMEOUT_MS = 100;
constexpr int REOPEN_PERIOD_MS = 500;
constexpr int STATUS_PUBLISH_PERIOD_MS = 250;

std::filesystem::path status_path()
{
  if (const char * override_path = std::getenv("TATBOT_ESTOP_STATUS")) {
    return override_path;
  }
  if (const char * runtime = std::getenv("XDG_RUNTIME_DIR")) {
    return std::filesystem::path(runtime) / "tatbot" / "estop-status.json";
  }
  return std::filesystem::path("/tmp") / ("tatbot-" + std::to_string(getuid())) /
         "tatbot" / "estop-status.json";
}

std::string json_string(const std::string & value)
{
  std::string result = "\"";
  for (const char c : value) {
    if (static_cast<unsigned char>(c) < 0x20) {
      const char * hex = "0123456789abcdef";
      result += "\\u00";
      result += hex[(static_cast<unsigned char>(c) >> 4) & 15];
      result += hex[static_cast<unsigned char>(c) & 15];
      continue;
    }
    if (c == '\\' || c == '"') {result += '\\';}
    result += c;
  }
  return result + "\"";
}
}  // namespace

Monitor::Monitor(
  const std::string & device, bool required, std::atomic<int> & state)
: device_(device), state_(state)
{
  fd_ = open_device();
  if (fd_ < 0) {
    if (required) {
      throw std::runtime_error("cannot open e-stop device: " + device_);
    }
    state_.store(disabled);
    std::cout << "\nWARNING: e-stop device " << device_ << " not found — "
              << "running WITHOUT hardware e-stop.\n"
              << "         (plug it in and restart, or pass --estop PATH "
              << "to make it mandatory)\n" << std::endl;
    return;
  }
  state_.store(fault);
  thread_ = std::thread([this]() {run();});
  try {
    status_thread_ = std::thread([this]() {status_loop();});
  } catch (const std::system_error &) {
    // Telemetry is optional; a writer failure must not stop the serial reader.
  }
}

Monitor::~Monitor()
{
  stop_.store(true);
  if (thread_.joinable()) {thread_.join();}
  if (fd_ >= 0) {close(fd_);}
  if (status_thread_.joinable()) {status_thread_.join();}
  remove_status();
}

void Monitor::capture_status(long sequence, double heartbeat_age_ms)
{
  const double now = std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  snapshot_ = {state_.load(), sequence, heartbeat_age_ms, now};
}

void Monitor::status_loop()
{
  using clock = std::chrono::steady_clock;
  auto last_publish = clock::now();
  int last_state = disabled;
  while (!stop_.load()) {
    Snapshot snapshot;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      snapshot = snapshot_;
    }
    const auto now = clock::now();
    if (snapshot.updated_unix > 0 && (snapshot.state != last_state ||
      now - last_publish >= std::chrono::milliseconds(STATUS_PUBLISH_PERIOD_MS)))
    {
      publish_status(snapshot);  // No filesystem operation holds the snapshot lock.
      last_publish = now;
      last_state = snapshot.state;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void Monitor::publish_status(const Snapshot & snapshot)
{
  const auto path = status_path();
  std::string tmp;
  try {
    // Never chmod an existing directory supplied by the caller (e.g. /tmp).
    if (std::filesystem::create_directories(path.parent_path())) {
      chmod(path.parent_path().c_str(), 0700);
    }
    const char * name = snapshot.state == ok ? "ok" : snapshot.state == pressed ? "pressed" : "fault";
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"device\":" << json_string(device_) << ",\"engaged\":"
           << (snapshot.state == ok ? "false" : "true") << ",\"heartbeat_age_ms\":";
    if (snapshot.heartbeat_age_ms < 0) {output << "null";} else {output << snapshot.heartbeat_age_ms;}
    output << ",\"last_sequence\":";
    if (snapshot.sequence < 0) {output << "null";} else {output << snapshot.sequence;}
    output << ",\"pid\":" << getpid() << ",\"schema\":\"tatbot.estop-status/1\""
           << ",\"state\":\"" << name << "\",\"updated_unix\":" << snapshot.updated_unix << "}\n";
    // Exclusive creation is 0600 from the outset and cannot follow a stale symlink.
    tmp = (path.parent_path() / ("." + path.filename().string() + ".XXXXXX")).string();
    const int fd = mkstemp(tmp.data());
    if (fd < 0) {return;}
    const auto text = output.str();
    size_t sent = 0;
    while (sent < text.size()) {
      const auto n = write(fd, text.data() + sent, text.size() - sent);
      if (n < 0 && errno == EINTR) {continue;}
      if (n <= 0) {break;}
      sent += static_cast<size_t>(n);
    }
    const bool closed = close(fd) == 0;
    if (sent == text.size() && closed && !stop_.load()) {
      std::filesystem::rename(tmp, path);
    } else {
      std::filesystem::remove(tmp);
    }
  } catch (const std::filesystem::filesystem_error &) {
    std::error_code ignored;
    if (!tmp.empty()) {std::filesystem::remove(tmp, ignored);}
  }
}

void Monitor::remove_status()
{
  std::ifstream input(status_path());
  const std::string text((std::istreambuf_iterator<char>(input)), {});
  if (text.find("\"pid\":" + std::to_string(getpid()) + ",") != std::string::npos &&
    text.find("\"device\":" + json_string(device_)) != std::string::npos)
  {
    std::error_code ignored;
    std::filesystem::remove(status_path(), ignored);
  }
}

int Monitor::open_device()
{
  const int fd = open(device_.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
  if (fd >= 0 && isatty(fd)) {
    termios tio{};
    if (tcgetattr(fd, &tio) == 0) {
      cfmakeraw(&tio);
      tcsetattr(fd, TCSANOW, &tio);
    }
  }
  return fd;
}

void Monitor::run()
{
  using clock = std::chrono::steady_clock;
  auto last_frame = clock::now();
  auto last_reopen = clock::now();
  std::string buffer;
  long last_seq = -1;
  int raw_state = -1;
  int stable_state = -1;
  int stable_count = 0;

  while (!stop_.load()) {
    if (fd_ < 0) {
      state_.store(fault);
      capture_status(-1, -1);
      if (clock::now() - last_reopen > std::chrono::milliseconds(REOPEN_PERIOD_MS)) {
        last_reopen = clock::now();
        fd_ = open_device();
        if (fd_ >= 0) {
          buffer.clear();
          last_seq = -1;
          raw_state = stable_state = -1;
          stable_count = 0;
          last_frame = clock::now();
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    pollfd pfd = {fd_, POLLIN, 0};
    const int ready = poll(&pfd, 1, 20);
    if (ready > 0) {
      char chunk[256];
      const ssize_t n = read(fd_, chunk, sizeof(chunk));
      if (n > 0) {
        buffer.append(chunk, static_cast<size_t>(n));
      } else if (n == 0 || (errno != EAGAIN && errno != EINTR)) {
        close(fd_);
        fd_ = -1;
        continue;
      }
      size_t nl;
      while ((nl = buffer.find('\n')) != std::string::npos) {
        const std::string line = buffer.substr(0, nl);
        buffer.erase(0, nl + 1);
        std::istringstream input(line);
        std::string magic;
        std::string extra;
        long seq = -1;
        int button = -1;
        if ((input >> magic >> seq >> button) && !(input >> extra) &&
          magic == "EST1" && seq >= 0 && (button == 0 || button == 1))
        {
          if (last_seq >= 0 && seq <= last_seq) {
            raw_state = stable_state = -1;
            stable_count = 0;
          }
          last_seq = seq;
          last_frame = clock::now();
          if (button == raw_state) {
            if (stable_count < DEBOUNCE_FRAMES) {++stable_count;}
          } else {
            raw_state = button;
            stable_count = 1;
          }
          if (stable_count >= DEBOUNCE_FRAMES) {stable_state = button;}
        }
      }
      if (buffer.size() > 1024) {buffer.clear();}
    }

    if (clock::now() - last_frame > std::chrono::milliseconds(HEARTBEAT_TIMEOUT_MS)) {
      state_.store(fault);
    } else if (stable_state == 0) {
      state_.store(pressed);
    } else if (stable_state == 1) {
      state_.store(ok);
    }
    const double age_ms = last_seq < 0 ? -1.0 :
      std::chrono::duration<double, std::milli>(clock::now() - last_frame).count();
    capture_status(last_seq, age_ms);
  }
}

WaitResult wait_for_clear(
  const std::atomic<int> & state,
  const std::atomic<int> & stop_signals,
  int signals_at_hold)
{
  while (state.load() > ok) {
    if (stop_signals.load() > signals_at_hold) {return WaitResult::emergency;}
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return WaitResult::resume;
}

}  // namespace tatbot::estop
