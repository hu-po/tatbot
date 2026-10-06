#pragma once
// Process plumbing for the driver, ported from cpp/teleop: real-time setup (realtime.cpp), the
// arm-driver lock (driver_lease.hpp, exclusive mode only) and the flight recorder
// (flight_recorder.cpp).
#include <pthread.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tatbot_hardware::runtime
{

struct RtSetup
{
  bool fifo = false, affinity = false;
  std::vector<int> cpus;
  std::string error;   // empty when both applied
};

// Cores within 10% of the highest cpuinfo_max_freq (all cores on a homogeneous CPU).
std::vector<int> fastest_cpus(const std::map<int, long> & max_freq_khz);
std::map<int, long> read_max_frequencies(const std::string & sysfs_root = "/sys/devices/system/cpu");
// Pin the CALLING thread (cpus, or the fastest class when empty) and request SCHED_FIFO at
// priority. Threads it creates afterwards (the SDK's daemon) inherit both. Never throws.
RtSetup apply_realtime(int priority, const std::vector<int> & cpus);

// /tmp/tatbot-arm-driver.lock, the same flock that wxai_teleop, arm_recover and LeRobot
// take: exclusive, never unlinked. One per process, shared by both arms. Throws "driver busy".
class DriverLock
{
public:
  static std::shared_ptr<DriverLock> acquire(const std::string & path = "/tmp/tatbot-arm-driver.lock");
  explicit DriverLock(const std::string & path);
  ~DriverLock();
  DriverLock(const DriverLock &) = delete;
  DriverLock & operator=(const DriverLock &) = delete;

private:
  int fd_ = -1;
};

// Bounded, nonblocking flight log for the control thread: each record is one atomic O_NONBLOCK
// pipe write; a SCHED_OTHER worker drains the pipe to disk. A slow disk drops whole records and
// never stalls the 400 Hz loop.
class FlightRecorder
{
public:
  explicit FlightRecorder(const std::string & path);
  ~FlightRecorder();
  FlightRecorder(const FlightRecorder &) = delete;
  FlightRecorder & operator=(const FlightRecorder &) = delete;
  bool append(const void * data, size_t bytes) noexcept;
  uint64_t dropped() const noexcept {return dropped_.load(std::memory_order_relaxed);}
  void finish() noexcept;

private:
  static void * entry(void * self) noexcept;
  void drain() noexcept;
  int file_fd_ = -1, read_fd_ = -1, write_fd_ = -1;
  size_t atomic_limit_ = 512;
  pthread_t worker_{};
  bool started_ = false;
  std::atomic<uint64_t> dropped_{0};
};

}  // namespace tatbot_hardware::runtime
