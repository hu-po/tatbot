#include "telemetry_udp.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tatbot::telemetry
{
namespace
{

void append_array(std::ostringstream & output, const std::vector<double> & values)
{
  output << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {output << ',';}
    output << values[index];
  }
  output << ']';
}

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
  try {
    std::ostringstream message;
    message << std::setprecision(10)
            << "{\"magic\":\"tatbot-teleop-joints\",\"version\":1"
            << ",\"timestamp_ns\":" << timestamp_ns
            << ",\"sequence\":" << sequence << ",\"leader_pos\":";
    append_array(message, leader_pos);
    message << ",\"follower_pos\":";
    append_array(message, follower_pos);
    message << ",\"target\":";
    append_array(message, target);
    message << ",\"follower_eff\":";
    append_array(message, follower_eff);
    message << '}';
    const std::string payload = message.str();
    const ssize_t written = sendto(
      socket_, payload.data(), payload.size(), MSG_DONTWAIT | MSG_NOSIGNAL,
      reinterpret_cast<const sockaddr *>(&destination_), sizeof(destination_));
    if (written == static_cast<ssize_t>(payload.size())) {
      ++stats_.sent;
      last_sent_ = now;
    } else {
      ++stats_.send_errors;
    }
  } catch (...) {
    ++stats_.send_errors;
  }
}

}  // namespace tatbot::telemetry
