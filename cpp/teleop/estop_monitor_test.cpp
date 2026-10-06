#include "estop_monitor.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <cmath>
#include <unistd.h>

namespace
{
using namespace std::chrono_literals;

void require(bool condition, const std::string & message)
{
  if (!condition) {throw std::runtime_error(message);}
}

bool wait_state(const std::atomic<int> & state, int expected, std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (state.load() == expected) {return true;}
    std::this_thread::sleep_for(5ms);
  }
  return state.load() == expected;
}

void frames(int fd, int & seq, int state, int count)
{
  for (int i = 0; i < count; ++i) {
    const std::string frame =
      "EST1 " + std::to_string(seq++) + " " + std::to_string(state) + "\n";
    require(write(fd, frame.data(), frame.size()) == static_cast<ssize_t>(frame.size()),
      "failed to write PTY heartbeat frame");
    std::this_thread::sleep_for(10ms);
  }
}
}  // namespace

int main()
{
  try {
    const std::string status_path = "/tmp/tatbot-estop-monitor-test-" + std::to_string(getpid()) + ".json";
    setenv("TATBOT_ESTOP_STATUS", status_path.c_str(), 1);
    const int master = posix_openpt(O_RDWR | O_NOCTTY);
    require(master >= 0, "posix_openpt failed");
    require(grantpt(master) == 0 && unlockpt(master) == 0, "PTY setup failed");
    const char * slave = ptsname(master);
    require(slave != nullptr, "ptsname failed");

    {
      std::atomic<int> state{tatbot::estop::disabled};
      tatbot::estop::Monitor monitor(slave, true, state);
      require(state.load() == tatbot::estop::fault, "monitor must start fail-safe");

      int seq = 0;
      frames(master, seq, 1, 3);
      require(wait_state(state, tatbot::estop::ok, 200ms), "three OK frames not accepted");
      frames(master, seq, 1, 30);
      std::ifstream snapshot(status_path);
      const std::string snapshot_text((std::istreambuf_iterator<char>(snapshot)), {});
      require(snapshot_text.find("\"state\":\"ok\"") != std::string::npos,
        "monitor did not publish healthy passive status");

      const auto timestamp_at = snapshot_text.find("\"updated_unix\":");
      require(timestamp_at != std::string::npos, "snapshot has no timestamp");
      const double updated = std::stod(snapshot_text.substr(timestamp_at + 15));
      const double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
      require(std::abs(now - updated) < 1.0, "snapshot timestamp lost subsecond precision");
      struct stat permissions{};
      require(stat(status_path.c_str(), &permissions) == 0 &&
        (permissions.st_mode & 0777) == 0600, "snapshot must be private from creation");

      const std::string garbage = "EST1 99 1 trailing\nnot-a-frame\n";
      require(
        write(master, garbage.data(), garbage.size()) ==
        static_cast<ssize_t>(garbage.size()),
        "failed to write malformed test frame");
      std::this_thread::sleep_for(30ms);
      require(state.load() == tatbot::estop::ok, "malformed frames changed state");

      frames(master, seq, 0, 3);
      require(wait_state(state, tatbot::estop::pressed, 200ms), "press was not debounced");

      frames(master, seq, 1, 3);
      require(wait_state(state, tatbot::estop::ok, 200ms), "release was not debounced");
      require(wait_state(state, tatbot::estop::fault, 250ms), "heartbeat silence did not fault");
    }
    require(access(status_path.c_str(), F_OK) != 0, "monitor left a stale status snapshot");
    close(master);

    std::atomic<int> flow_state{tatbot::estop::pressed};
    std::atomic<int> signals{0};
    std::thread release([&flow_state]() {
      std::this_thread::sleep_for(30ms);
      flow_state.store(tatbot::estop::ok);
    });
    require(
      tatbot::estop::wait_for_clear(flow_state, signals, 0) ==
      tatbot::estop::WaitResult::resume,
      "clear did not return automatic resume");
    release.join();

    flow_state.store(tatbot::estop::pressed);
    std::thread interrupt([&signals]() {
      std::this_thread::sleep_for(30ms);
      signals.store(1);
    });
    require(
      tatbot::estop::wait_for_clear(flow_state, signals, 0) ==
      tatbot::estop::WaitResult::emergency,
      "stop signal did not interrupt wait");
    interrupt.join();
    std::cout << "estop_monitor_test PASS" << std::endl;
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "estop_monitor_test FAIL: " << error.what() << std::endl;
    return 1;
  }
}
