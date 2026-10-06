#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <thread>

namespace tatbot::estop
{

enum State : int {
  disabled = -1,
  ok = 0,
  pressed = 1,
  fault = 2,
};

class Monitor
{
public:
  Monitor(const std::string & device, bool required, std::atomic<int> & state);
  ~Monitor();

  Monitor(const Monitor &) = delete;
  Monitor & operator=(const Monitor &) = delete;

private:
  int open_device();
  void run();
  struct Snapshot {
    int state{fault};
    long sequence{-1};
    double heartbeat_age_ms{-1};
    double updated_unix{0};
  };
  void capture_status(long sequence, double heartbeat_age_ms);
  void status_loop();
  void publish_status(const Snapshot & snapshot);
  void remove_status();

  std::string device_;
  std::atomic<int> & state_;
  int fd_{-1};
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::thread status_thread_;
  std::mutex snapshot_mutex_;
  Snapshot snapshot_;
};

enum class WaitResult { resume, emergency };

WaitResult wait_for_clear(
  const std::atomic<int> & state,
  const std::atomic<int> & stop_signals,
  int signals_at_hold);

}  // namespace tatbot::estop
