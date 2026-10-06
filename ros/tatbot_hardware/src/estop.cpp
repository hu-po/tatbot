// EST1 reader, ported from cpp/teleop/estop_monitor.cpp (parser, 3-frame debounce, reopen every
// 0.5 s) with a UDP source added for the palette Pi's relay.
#include "tatbot_hardware/estop.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace tatbot_hardware::estop
{
namespace
{
constexpr int kReopenMs = 500;
constexpr size_t kMaxFrame = 128;   // the relay drops longer lines; so does the driver
}  // namespace

int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

int source_from_name(const std::string & name)
{
  if (name == "none" || name.empty()) {return names::kEstopNone;}
  if (name == "serial") {return names::kEstopSerial;}
  if (name == "udp") {return names::kEstopUdp;}
  return -1;
}

bool parse_frame(std::string_view line, long & seq, int & state)
{
  if (!line.empty() && line.back() == '\n') {line.remove_suffix(1);}
  if (!line.empty() && line.back() == '\r') {line.remove_suffix(1);}
  if (line.size() < 8 || line.substr(0, 5) != "EST1 ") {return false;}
  line.remove_prefix(5);
  const size_t space = line.find(' ');
  if (space == std::string_view::npos || space == 0 || space > 18) {return false;}
  long value = 0;
  for (size_t i = 0; i < space; ++i) {
    if (line[i] < '0' || line[i] > '9') {return false;}
    value = value * 10 + (line[i] - '0');
  }
  const std::string_view rest = line.substr(space + 1);
  if (rest != "0" && rest != "1") {return false;}
  seq = value;
  state = rest[0] - '0';
  return true;
}

Reader::Reader(const Settings & settings)
: settings_(settings),
  timeout_ns_(static_cast<int64_t>(settings.timeout_s * 1e9))
{
  if (settings_.source == names::kEstopUdp) {
    in_addr addr{};
    if (inet_pton(AF_INET, settings_.relay_addr.c_str(), &addr) == 1) {
      relay_ = addr.s_addr;
    } else if (!settings_.relay_addr.empty()) {
      throw std::runtime_error(
              "e-stop udp: relay address '" + settings_.relay_addr + "' is not an IPv4 address");
    }
    // No SO_REUSEADDR: a second reader on this port fails here instead of silently sharing it.
    fd_ = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port = htons(static_cast<uint16_t>(settings_.udp_port));
    if (fd_ < 0 || bind(fd_, reinterpret_cast<sockaddr *>(&bind_addr), sizeof(bind_addr)) != 0) {
      const std::string error = std::strerror(errno);
      if (fd_ >= 0) {close(fd_);}
      throw std::runtime_error(
              "e-stop udp: cannot bind port " + std::to_string(settings_.udp_port) + ": " + error);
    }
    socklen_t len = sizeof(bind_addr);
    getsockname(fd_, reinterpret_cast<sockaddr *>(&bind_addr), &len);
    bound_port_ = ntohs(bind_addr.sin_port);
    thread_ = std::thread([this]() {run_udp();});
  } else if (settings_.source == names::kEstopSerial) {
    thread_ = std::thread([this]() {run_serial();});
  }
}

Reader::~Reader()
{
  stop_.store(true);
  if (thread_.joinable()) {thread_.join();}
  if (fd_ >= 0) {close(fd_);}
}

void Reader::accept(long seq, int state, int64_t t)
{
  const int64_t last = last_ns_.load();
  const bool stale = last < 0 || t - last > timeout_ns_;
  if (!stale && seq <= last_seq_) {return;}   // reordered or replayed
  if (stale) {                                 // after silence the sequence re-seeds
    raw_ = -1;
    count_ = 0;
    stable_.store(-1);
  }
  last_seq_ = seq;
  if (state == raw_) {
    if (count_ < settings_.debounce_frames) {++count_;}
  } else {
    raw_ = state;
    count_ = 1;
  }
  if (count_ >= settings_.debounce_frames) {stable_.store(state);}
  last_ns_.store(t);
}

EstopStatus Reader::status(int64_t now) const
{
  EstopStatus s;
  s.source = settings_.source;
  const int64_t last = last_ns_.load();
  if (last < 0) {
    s.ok = false;
    return s;
  }
  const int64_t age = now - last;
  const int stable = stable_.load();
  s.age_s = static_cast<double>(age) * 1e-9;
  s.ok = age <= timeout_ns_ && stable == 1;
  s.pressed = age <= timeout_ns_ && stable == 0;
  return s;
}

void Reader::run_udp()
{
  char buffer[512];
  while (!stop_.load()) {
    pollfd pfd{fd_, POLLIN, 0};
    if (poll(&pfd, 1, 20) <= 0) {continue;}
    sockaddr_in from{};
    socklen_t len = sizeof(from);
    const ssize_t n = recvfrom(fd_, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr *>(&from), &len);
    const int64_t t = now_ns();
    long seq = 0;
    int state = 0;
    if (n <= 0 || static_cast<size_t>(n) > kMaxFrame || relay_ == 0 ||
      from.sin_addr.s_addr != relay_ ||
      !parse_frame(std::string_view(buffer, static_cast<size_t>(n)), seq, state))
    {
      continue;
    }
    accept(seq, state, t);
  }
}

void Reader::run_serial()
{
  std::string buffer;
  auto last_open = std::chrono::steady_clock::now() - std::chrono::milliseconds(kReopenMs);
  int fd = -1;
  while (!stop_.load()) {
    if (fd < 0) {
      if (std::chrono::steady_clock::now() - last_open >= std::chrono::milliseconds(kReopenMs)) {
        last_open = std::chrono::steady_clock::now();
        fd = open(settings_.device.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (fd >= 0 && isatty(fd)) {
          termios tio{};
          if (tcgetattr(fd, &tio) == 0) {
            cfmakeraw(&tio);
            tcsetattr(fd, TCSANOW, &tio);
          }
        }
        buffer.clear();
      }
      if (fd < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
    }
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 20);
    if (ready <= 0) {continue;}
    char chunk[256];
    const ssize_t n = read(fd, chunk, sizeof(chunk));
    if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR) || (pfd.revents & (POLLHUP | POLLERR))) {
      close(fd);
      fd = -1;
      continue;
    }
    if (n < 0) {continue;}
    const int64_t t = now_ns();
    buffer.append(chunk, static_cast<size_t>(n));
    size_t nl;
    while ((nl = buffer.find('\n')) != std::string::npos) {
      long seq = 0;
      int state = 0;
      if (nl <= kMaxFrame && parse_frame(std::string_view(buffer.data(), nl), seq, state)) {
        accept(seq, state, t);
      }
      buffer.erase(0, nl + 1);
    }
    if (buffer.size() > 1024) {buffer.clear();}
  }
  if (fd >= 0) {close(fd);}
}

std::shared_ptr<Reader> shared(const Settings & settings)
{
  static std::mutex mutex;
  static std::weak_ptr<Reader> instance;
  if (settings.source == names::kEstopNone) {return nullptr;}
  std::lock_guard<std::mutex> lock(mutex);
  auto reader = instance.lock();
  if (!reader) {
    reader = std::make_shared<Reader>(settings);
    instance = reader;
  }
  return reader;
}

// --- the station probe ----------------------------------------------------------------------------
namespace
{
// A decimal field of at most 19 digits ending at `end` (a space or the line's end); npos on failure.
size_t parse_field(std::string_view line, size_t at, int64_t & value)
{
  size_t i = at;
  value = 0;
  while (i < line.size() && line[i] != ' ') {
    if (line[i] < '0' || line[i] > '9' || i - at >= 19) {return std::string_view::npos;}
    value = value * 10 + (line[i] - '0');
    ++i;
  }
  return i == at ? std::string_view::npos : i;
}
}  // namespace

bool parse_probe_frame(std::string_view line, long & seq, int & state, int64_t & edge_ns, int64_t & sent_ns)
{
  if (!line.empty() && line.back() == '\n') {line.remove_suffix(1);}
  if (!line.empty() && line.back() == '\r') {line.remove_suffix(1);}
  if (line.substr(0, 5) != "PRB1 ") {return false;}
  int64_t fields[4];
  size_t at = 5;
  for (int k = 0; k < 4; ++k) {
    at = parse_field(line, at, fields[k]);
    if (at == std::string_view::npos || (k < 3 ? at >= line.size() : at != line.size())) {return false;}
    ++at;   // past the space
  }
  if (fields[1] != 0 && fields[1] != 1) {return false;}
  seq = static_cast<long>(fields[0]);
  state = static_cast<int>(fields[1]);
  edge_ns = fields[2];
  sent_ns = fields[3];
  return true;
}

ProbeReader::ProbeReader(const ProbeSettings & settings)
: settings_(settings),
  timeout_ns_(static_cast<int64_t>(settings.timeout_s * 1e9))
{
  in_addr addr{};
  if (inet_pton(AF_INET, settings_.relay_addr.c_str(), &addr) == 1) {
    relay_ = addr.s_addr;
  } else if (!settings_.relay_addr.empty()) {
    throw std::runtime_error("probe: relay address '" + settings_.relay_addr + "' is not an IPv4 address");
  }
  fd_ = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  sockaddr_in bind_addr{};
  bind_addr.sin_family = AF_INET;
  bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  bind_addr.sin_port = htons(static_cast<uint16_t>(settings_.udp_port));
  if (fd_ < 0 || bind(fd_, reinterpret_cast<sockaddr *>(&bind_addr), sizeof(bind_addr)) != 0) {
    const std::string error = std::strerror(errno);
    if (fd_ >= 0) {close(fd_);}
    throw std::runtime_error("probe: cannot bind port " + std::to_string(settings_.udp_port) + ": " + error);
  }
  socklen_t len = sizeof(bind_addr);
  getsockname(fd_, reinterpret_cast<sockaddr *>(&bind_addr), &len);
  bound_port_ = ntohs(bind_addr.sin_port);
  thread_ = std::thread([this]() {run();});
}

ProbeReader::~ProbeReader()
{
  stop_.store(true);
  if (thread_.joinable()) {thread_.join();}
  if (fd_ >= 0) {close(fd_);}
}

void ProbeReader::accept(long seq, int state, int64_t edge_ns, int64_t sent_ns, int64_t t)
{
  const int64_t last = last_ns_.load();
  const bool stale = last < 0 || t - last > timeout_ns_;
  if (!stale && seq <= last_seq_) {return;}   // reordered or replayed
  if (stale) {                                 // a restarted relay: a new clock and sequence
    offsets_n_ = 0;
    last_edge_ = 0;
    rise_relay_ns_.store(-1);
  }
  last_seq_ = seq;
  offsets_[offsets_head_] = t - sent_ns;
  offsets_head_ = (offsets_head_ + 1) % kClockWindow;
  offsets_n_ = std::min(offsets_n_ + 1, kClockWindow);
  offset_ns_.store(*std::min_element(offsets_.begin(), offsets_.begin() + static_cast<long>(offsets_n_)));
  if (edge_ns != last_edge_ && edge_ns > 0) {   // a new edge: a rising one when the line reads touched
    last_edge_ = edge_ns;
    if (state == 1) {rise_relay_ns_.store(edge_ns);}
  }
  state_.store(state);
  last_ns_.store(t);
}

ProbeStatus ProbeReader::status(int64_t now) const
{
  ProbeStatus s;
  s.enabled = true;
  const int64_t last = last_ns_.load();
  s.fresh = last >= 0 && now - last <= timeout_ns_;
  s.triggered = s.fresh ? state_.load() == 1 : false;
  const int64_t rise = rise_relay_ns_.load();
  if (rise > 0) {s.rise_age_s = static_cast<double>(now - (rise + offset_ns_.load())) * 1e-9;}
  return s;
}

void ProbeReader::run()
{
  char buffer[512];
  while (!stop_.load()) {
    pollfd pfd{fd_, POLLIN, 0};
    if (poll(&pfd, 1, 20) <= 0) {continue;}
    sockaddr_in from{};
    socklen_t len = sizeof(from);
    const ssize_t n = recvfrom(fd_, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr *>(&from), &len);
    const int64_t t = now_ns();
    long seq = 0;
    int state = 0;
    int64_t edge_ns = 0, sent_ns = 0;
    if (n <= 0 || static_cast<size_t>(n) > kMaxFrame || relay_ == 0 || from.sin_addr.s_addr != relay_ ||
      !parse_probe_frame(std::string_view(buffer, static_cast<size_t>(n)), seq, state, edge_ns, sent_ns))
    {
      continue;
    }
    accept(seq, state, edge_ns, sent_ns, t);
  }
}

std::shared_ptr<ProbeReader> shared_probe(const ProbeSettings & settings)
{
  static std::mutex mutex;
  static std::weak_ptr<ProbeReader> instance;
  std::lock_guard<std::mutex> lock(mutex);
  auto reader = instance.lock();
  if (!reader) {
    reader = std::make_shared<ProbeReader>(settings);
    instance = reader;
  }
  return reader;
}

}  // namespace tatbot_hardware::estop
