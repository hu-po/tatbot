#include "telemetry_udp.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tatbot::telemetry
{
namespace
{

class FixedJsonWriter
{
public:
  FixedJsonWriter(char * data, size_t capacity)
  : cursor_(data), end_(data + capacity) {}

  bool append(const char * value)
  {
    const size_t bytes = std::strlen(value);
    if (static_cast<size_t>(end_ - cursor_) < bytes) {return false;}
    std::memcpy(cursor_, value, bytes);
    cursor_ += bytes;
    return true;
  }

  template<typename Number>
  bool number(Number value)
  {
    const auto result = std::to_chars(cursor_, end_, value);
    if (result.ec != std::errc{}) {return false;}
    cursor_ = result.ptr;
    return true;
  }

  bool number(double value)
  {
    const auto result = std::to_chars(
      cursor_, end_, value, std::chars_format::general, 10);
    if (result.ec != std::errc{}) {return false;}
    cursor_ = result.ptr;
    return true;
  }

  bool array(const std::vector<double> & values)
  {
    if (!append("[")) {return false;}
    for (size_t index = 0; index < values.size(); ++index) {
      if ((index != 0 && !append(",")) || !number(values[index])) {return false;}
    }
    return append("]");
  }

  size_t size(const char * begin) const {return static_cast<size_t>(cursor_ - begin);}

private:
  char * cursor_;
  char * end_;
};

std::pair<std::string, std::string> split_endpoint(const std::string & endpoint)
{
  const size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
    throw std::runtime_error("telemetry endpoint must be HOST:PORT, got " + endpoint);
  }
  return {endpoint.substr(0, colon), endpoint.substr(colon + 1)};
}

}  // namespace

UdpPublisher::UdpPublisher(const std::string & endpoint, double max_fps)
: endpoint_(endpoint)
{
  if (!std::isfinite(max_fps) || max_fps <= 0.0 || max_fps > 120.0) {
    throw std::runtime_error("telemetry FPS must be in (0, 120]");
  }
  const auto [host, port] = split_endpoint(endpoint);
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo * addresses = nullptr;
  const int status = getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses);
  if (status != 0 || addresses == nullptr) {
    throw std::runtime_error(
      "cannot resolve telemetry endpoint " + endpoint + ": " + gai_strerror(status));
  }
  std::memcpy(&destination_, addresses->ai_addr, sizeof(destination_));
  freeaddrinfo(addresses);
  socket_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
  if (socket_ < 0) {
    throw std::runtime_error("cannot create telemetry UDP socket: " + std::string(std::strerror(errno)));
  }
  minimum_interval_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(1.0 / max_fps));
}

UdpPublisher::~UdpPublisher()
{
  if (socket_ >= 0) {close(socket_);}
}

void UdpPublisher::publish(
  int64_t timestamp_ns,
  uint64_t sequence,
  const std::vector<double> & leader_pos,
  const std::vector<double> & follower_pos,
  const std::vector<double> & target,
  const std::vector<double> & follower_eff) noexcept
{
  const size_t joints = leader_pos.size();
  const auto valid = [joints](const std::vector<double> & values) {
      return values.size() == joints &&
             std::all_of(values.begin(), values.end(), [](double value) {
               return std::isfinite(value);
             });
    };
  if (joints == 0 || !valid(leader_pos) || !valid(follower_pos) || !valid(target) ||
    !valid(follower_eff))
  {
    ++stats_.send_errors;
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (last_sent_ != std::chrono::steady_clock::time_point{} &&
    now - last_sent_ < minimum_interval_)
  {
    ++stats_.rate_limited;
    return;
  }
  FixedJsonWriter message(payload_.data(), payload_.size());
  const bool encoded =
    message.append("{\"magic\":\"tatbot-teleop-joints\",\"version\":1,\"timestamp_ns\":") &&
    message.number(timestamp_ns) && message.append(",\"sequence\":") &&
    message.number(sequence) && message.append(",\"leader_pos\":") &&
    message.array(leader_pos) && message.append(",\"follower_pos\":") &&
    message.array(follower_pos) && message.append(",\"target\":") &&
    message.array(target) && message.append(",\"follower_eff\":") &&
    message.array(follower_eff) && message.append("}");
  if (!encoded) {
    ++stats_.send_errors;
    return;
  }
  const size_t payload_size = message.size(payload_.data());
  const ssize_t written = sendto(
    socket_, payload_.data(), payload_size, MSG_DONTWAIT | MSG_NOSIGNAL,
    reinterpret_cast<const sockaddr *>(&destination_), sizeof(destination_));
  if (written == static_cast<ssize_t>(payload_size)) {
    ++stats_.sent;
    last_sent_ = now;
  } else {
    ++stats_.send_errors;
  }
}

}  // namespace tatbot::telemetry
