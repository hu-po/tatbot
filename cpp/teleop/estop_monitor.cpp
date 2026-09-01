#include "estop_monitor.hpp"

#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>

namespace tatbot::estop
{

namespace
{
constexpr int DEBOUNCE_FRAMES = 3;
constexpr int HEARTBEAT_TIMEOUT_MS = 100;
constexpr int REOPEN_PERIOD_MS = 500;
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
}

Monitor::~Monitor()
{
  stop_.store(true);
  if (thread_.joinable()) {thread_.join();}
  if (fd_ >= 0) {close(fd_);}
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
