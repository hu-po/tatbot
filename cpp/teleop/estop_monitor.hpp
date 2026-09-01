#pragma once

#include <atomic>
#include <string>
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

  std::string device_;
  std::atomic<int> & state_;
  int fd_{-1};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

enum class WaitResult { resume, emergency };

WaitResult wait_for_clear(
  const std::atomic<int> & state,
  const std::atomic<int> & stop_signals,
  int signals_at_hold);

}  // namespace tatbot::estop
