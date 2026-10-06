// End to end: an emulated Pico on a pty -> the real relay (tatbot_estop_relay.py, a child
// process) -> UDP -> the driver's e-stop reader. Loss, delay, reorder, wrong source, relay death
// and a rebooted Pico.
#include <arpa/inet.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <pty.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "tatbot_hardware/estop.hpp"

using namespace tatbot_hardware;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace
{
EstopStatus status(const estop::Reader & r) {return r.status(estop::now_ns());}

template<class F>
double wait_ms(const estop::Reader & r, F f, std::chrono::milliseconds budget)
{
  const auto t0 = Clock::now();
  while (Clock::now() - t0 < budget) {
    if (f(status(r))) {return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();}
    std::this_thread::sleep_for(1ms);
  }
  return -1;
}

struct Pico
{
  int master = -1, slave = -1;
  std::string path;
  std::atomic<bool> run{true}, paused{false};
  std::atomic<int> state{1}, gap_ms{10};
  std::atomic<long> seq{0};
  std::atomic<int64_t> last_write_ns{0};
  std::thread thread;
  Pico()
  {
    char name[128];
    EXPECT_EQ(openpty(&master, &slave, name, nullptr, nullptr), 0);
    termios tio{};
    tcgetattr(slave, &tio);
    cfmakeraw(&tio);
    tcsetattr(slave, TCSANOW, &tio);
    path = name;
    fcntl(master, F_SETFL, fcntl(master, F_GETFL) | O_NONBLOCK);   // never block on a dead relay
    thread = std::thread([this]() {
        while (run) {
          if (!paused) {
            const std::string f = "EST1 " + std::to_string(seq++) + " " + std::to_string(state.load()) + "\n";
            if (write(master, f.data(), f.size()) > 0) {last_write_ns = estop::now_ns();}
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms.load()));
        }
      });
  }
  ~Pico()
  {
    run = false;
    thread.join();
    close(master);
    close(slave);
  }
};

pid_t start_relay(const std::string & device, int port)
{
  const pid_t pid = fork();
  if (pid == 0) {
    const std::string dest = "127.0.0.1:" + std::to_string(port);
    execlp("python3", "python3", RELAY_SCRIPT, "--device", device.c_str(), "--dest", dest.c_str(),
      static_cast<char *>(nullptr));
    _exit(127);
  }
  return pid;
}

void stop_relay(pid_t pid)
{
  kill(pid, SIGTERM);
  int status = 0;
  waitpid(pid, &status, 0);
}

void inject(const char * from, int port, const std::string & frame, int times)
{
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  inet_pton(AF_INET, from, &a.sin_addr);
  bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a));
  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
  for (int i = 0; i < times; ++i) {
    sendto(fd, frame.data(), frame.size(), 0, reinterpret_cast<sockaddr *>(&to), sizeof(to));
  }
  close(fd);
}
}  // namespace

TEST(HwRelay, EmulatedPicoThroughTheRelayIntoTheDriverReader)
{
  estop::Settings s;
  s.source = names::kEstopUdp;
  s.udp_port = 0;
  s.timeout_s = 0.15;
  s.relay_addr = "127.0.0.1";
  estop::Reader reader(s);
  Pico pico;
  pid_t relay = start_relay(pico.path, reader.bound_port());
  auto ok = [](const EstopStatus & e) {return e.ok;};
  auto stale = [](const EstopStatus & e) {return !e.ok && !e.pressed;};
  auto pressed = [](const EstopStatus & e) {return e.pressed;};

  ASSERT_GE(wait_ms(reader, ok, 3000ms), 0) << "no heartbeat through the relay";
  EXPECT_LT(status(reader).age_s, 0.05);

  // Press and release through the whole path, within a few frames.
  pico.state = 0;
  const double press_ms = wait_ms(reader, pressed, 500ms);
  EXPECT_GE(press_ms, 0);
  EXPECT_LT(press_ms, 100);
  pico.state = 1;
  EXPECT_GE(wait_ms(reader, ok, 500ms), 0);

  // Loss: the Pico goes quiet; stale once the last frame is 150 ms old.
  pico.paused = true;
  ASSERT_GE(wait_ms(reader, stale, 1000ms), 0);
  const double silent_ms = (estop::now_ns() - pico.last_write_ns.load()) * 1e-6;
  EXPECT_GE(silent_ms, 150);
  EXPECT_LT(silent_ms, 200);
  pico.paused = false;   // the sequence advanced: accepted again after 3 frames
  EXPECT_GE(wait_ms(reader, ok, 500ms), 0);

  // Delay: frames 120 ms apart keep it released.
  pico.gap_ms = 120;
  std::this_thread::sleep_for(150ms);
  bool always_ok = true;
  for (int i = 0; i < 100; ++i) {
    always_ok = always_ok && status(reader).ok;
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(always_ok);
  pico.gap_ms = 10;
  std::this_thread::sleep_for(50ms);

  // Reorder / replay from the relay's address: dropped by sequence.
  inject("127.0.0.1", reader.bound_port(), "EST1 3 0\n", 10);
  std::this_thread::sleep_for(30ms);
  EXPECT_TRUE(status(reader).ok);
  // Wrong source: ignored whatever it says.
  inject("127.0.0.2", reader.bound_port(), "EST1 99999999 0\n", 10);
  std::this_thread::sleep_for(30ms);
  EXPECT_TRUE(status(reader).ok);
  EXPECT_FALSE(status(reader).pressed);

  // Relay death is silence: stale.
  stop_relay(relay);
  EXPECT_GE(wait_ms(reader, stale, 1000ms), 0);

  // A rebooted Pico (sequence from 0) behind a restarted relay recovers after the silence.
  pico.paused = true;
  std::this_thread::sleep_for(50ms);
  tcflush(pico.slave, TCIOFLUSH);   // drop what queued while no relay read
  pico.seq = 0;
  pico.paused = false;
  relay = start_relay(pico.path, reader.bound_port());
  EXPECT_GE(wait_ms(reader, ok, 3000ms), 0);
  stop_relay(relay);
}
