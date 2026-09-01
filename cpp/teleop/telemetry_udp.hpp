#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <netinet/in.h>

namespace tatbot::telemetry
{

struct Stats
{
  uint64_t sent = 0;
  uint64_t rate_limited = 0;
  uint64_t send_errors = 0;
};

/// Best-effort, nonblocking joint-state telemetry. This class never throws
/// from publish(): visualization failure cannot perturb the control loop.
class UdpPublisher
{
public:
  UdpPublisher(const std::string & endpoint, double max_fps);
  ~UdpPublisher();

  UdpPublisher(const UdpPublisher &) = delete;
  UdpPublisher & operator=(const UdpPublisher &) = delete;

  void publish(
    int64_t timestamp_ns,
    uint64_t sequence,
    const std::vector<double> & leader_pos,
    const std::vector<double> & follower_pos,
    const std::vector<double> & target,
    const std::vector<double> & follower_eff) noexcept;

  const Stats & stats() const {return stats_;}
  const std::string & endpoint() const {return endpoint_;}

private:
  int socket_ = -1;
  sockaddr_in destination_{};
  std::string endpoint_;
  std::chrono::steady_clock::duration minimum_interval_{};
  std::chrono::steady_clock::time_point last_sent_{};
  Stats stats_;
};

}  // namespace tatbot::telemetry
