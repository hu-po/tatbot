// The e-stop reader: the EST1 parser, the serial source on a pseudo-terminal, and the UDP source
// (loss, delay, reorder, wrong source, re-seed after silence).
#include <arpa/inet.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <pty.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

#include "tatbot_hardware/estop.hpp"

using namespace tatbot_hardware;
using namespace std::chrono_literals;

namespace
{
EstopStatus now_status(const estop::Reader & r) {return r.status(estop::now_ns());}

// Wait up to `budget` for the predicate on the status.
template<class F>
bool eventually(const estop::Reader & r, F f, std::chrono::milliseconds budget = 1000ms)
{
  const auto end = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < end) {
    if (f(now_status(r))) {return true;}
    std::this_thread::sleep_for(2ms);
  }
  return f(now_status(r));
}

struct Pty
{
  int master = -1, slave = -1;
  std::string name;
  Pty()
  {
    char path[128];
    EXPECT_EQ(openpty(&master, &slave, path, nullptr, nullptr), 0);
    termios tio{};
    tcgetattr(slave, &tio);
    cfmakeraw(&tio);
    tcsetattr(slave, TCSANOW, &tio);
    name = path;
  }
  ~Pty() {close(master); close(slave);}
  void put(const std::string & s) {ASSERT_EQ(write(master, s.data(), s.size()), static_cast<ssize_t>(s.size()));}
};

struct Sender
{
  int fd;
  explicit Sender(const char * from)
  {
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    inet_pton(AF_INET, from, &a.sin_addr);
    EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)), 0);
  }
  ~Sender() {close(fd);}
  void send(int port, const std::string & s)
  {
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
    sendto(fd, s.data(), s.size(), 0, reinterpret_cast<sockaddr *>(&to), sizeof(to));
  }
  void frames(int port, long first, int count, int state, std::chrono::milliseconds gap = 10ms)
  {
    for (int i = 0; i < count; ++i) {
      send(port, "EST1 " + std::to_string(first + i) + " " + std::to_string(state) + "\n");
      std::this_thread::sleep_for(gap);
    }
  }
};

estop::Settings udp_settings()
{
  estop::Settings s;
  s.source = names::kEstopUdp;
  s.udp_port = 0;
  s.timeout_s = 0.15;
  s.relay_addr = "127.0.0.1";
  return s;
}
}  // namespace

TEST(HwEstop, ParsesOnlyWellFormedFrames)
{
  long seq = -1;
  int state = -1;
  EXPECT_TRUE(estop::parse_frame("EST1 4711 1\n", seq, state));
  EXPECT_EQ(seq, 4711);
  EXPECT_EQ(state, 1);
  EXPECT_TRUE(estop::parse_frame("EST1 0 0", seq, state));
  EXPECT_EQ(state, 0);
  EXPECT_TRUE(estop::parse_frame("EST1 12 1\r\n", seq, state));
  for (const char * bad : {"EST1 12EST1 13 0", "EST1 -1 1", "EST1 5 2", "EST2 5 1", "EST1 5 1 x",
      "EST1  5 1", "EST1 5", "", "EST1 99999999999999999999 1", "est1 5 1"})
  {
    EXPECT_FALSE(estop::parse_frame(bad, seq, state)) << bad;
  }
}

TEST(HwEstop, SerialFramesSilenceAndGarbage)
{
  Pty pty;
  estop::Settings s;
  s.source = names::kEstopSerial;
  s.device = pty.name;
  s.timeout_s = 0.10;
  estop::Reader reader(s);
  EXPECT_FALSE(now_status(reader).ok);   // nothing yet: stopped
  EXPECT_EQ(now_status(reader).age_s, -1);
  std::atomic<bool> run{true};
  std::atomic<int> state{1};
  std::atomic<bool> garbage{false};
  std::thread pico([&]() {
      long seq = 0;
      while (run) {
        if (garbage) {pty.put("EST1 xx 1\nnoise\x01\x02\nEST1 12EST1 13 1\n");} else {
          pty.put("EST1 " + std::to_string(seq++) + " " + std::to_string(state.load()) + "\n");
        }
        std::this_thread::sleep_for(10ms);
      }
    });
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}));
  EXPECT_EQ(now_status(reader).source, names::kEstopSerial);
  state = 0;
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.pressed && !e.ok;}, 200ms));
  state = 1;
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}, 200ms));
  // Garbage is not a heartbeat: stale after 100 ms.
  garbage = true;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return !e.ok && !e.pressed;}, 500ms));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_GE(ms, 60.0);    // the last frame may predate the switch by one period
  EXPECT_LE(ms, 160.0);
  garbage = false;
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}, 500ms));
  // Silence: stale past 100 ms, and age reported.
  run = false;
  pico.join();
  std::this_thread::sleep_for(60ms);
  EXPECT_TRUE(now_status(reader).ok);
  std::this_thread::sleep_for(80ms);
  EXPECT_FALSE(now_status(reader).ok);
  EXPECT_GT(now_status(reader).age_s, 0.10);
}

TEST(HwEstop, SerialDebouncesThreeFrames)
{
  Pty pty;
  estop::Settings s;
  s.source = names::kEstopSerial;
  s.device = pty.name;
  estop::Reader reader(s);
  std::this_thread::sleep_for(50ms);
  pty.put("EST1 1 1\nEST1 2 1\n");
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(now_status(reader).ok);    // two frames are not enough
  pty.put("EST1 3 1\n");
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}, 100ms));
  pty.put("EST1 4 0\nEST1 5 0\n");
  std::this_thread::sleep_for(20ms);
  EXPECT_TRUE(now_status(reader).ok);     // still the stable released state
  pty.put("EST1 6 0\n");
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.pressed;}, 100ms));
}

TEST(HwEstop, UdpLossDelayReorderWrongSourceAndReseed)
{
  estop::Reader reader(udp_settings());
  const int port = reader.bound_port();
  ASSERT_GT(port, 0);
  Sender relay("127.0.0.1"), stranger("127.0.0.2");
  relay.frames(port, 100, 3, 1);
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}, 100ms));
  // Delay: frames 120 ms apart stay under the 150 ms limit.
  for (int i = 0; i < 6; ++i) {
    relay.frames(port, 103 + i, 1, 1, 0ms);
    for (int k = 0; k < 12; ++k) {
      EXPECT_TRUE(now_status(reader).ok);
      std::this_thread::sleep_for(10ms);
    }
  }
  // Reorder / replay: older sequence numbers are dropped, even when they say pressed.
  relay.frames(port, 109, 1, 1, 1ms);
  relay.frames(port, 50, 5, 0, 1ms);
  EXPECT_TRUE(now_status(reader).ok);
  EXPECT_FALSE(now_status(reader).pressed);
  // Wrong source: ignored, however new its sequence.
  stranger.frames(port, 100000, 5, 0, 1ms);
  EXPECT_TRUE(now_status(reader).ok);
  EXPECT_FALSE(now_status(reader).pressed);
  // Oversized datagram: ignored.
  relay.send(port, "EST1 200000 0\n" + std::string(200, ' '));
  // Loss: the stranger's frames never refreshed the age; stale past 150 ms.
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return !e.ok && !e.pressed;}, 300ms));
  EXPECT_GT(now_status(reader).age_s, 0.15);
  // A rebooted Pico starts again at seq 0: accepted after the silence, ok after 3 frames.
  relay.frames(port, 0, 2, 1);
  EXPECT_FALSE(now_status(reader).ok);
  relay.frames(port, 2, 1, 1);
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.ok;}, 100ms));
  // Pressed through the relay.
  relay.frames(port, 3, 3, 0);
  EXPECT_TRUE(eventually(reader, [](const EstopStatus & e) {return e.pressed;}, 100ms));
}

TEST(HwEstop, UdpWithoutARelayAddressAcceptsNothing)
{
  auto s = udp_settings();
  s.relay_addr = "";
  estop::Reader reader(s);
  Sender relay("127.0.0.1");
  relay.frames(reader.bound_port(), 0, 5, 1);
  EXPECT_FALSE(now_status(reader).ok);
  EXPECT_EQ(now_status(reader).age_s, -1);
}

TEST(HwEstop, UdpRefusesAHostnameAndASharedPort)
{
  auto s = udp_settings();
  s.relay_addr = "estop-relay";   // a name, not an address
  EXPECT_THROW(estop::Reader{s}, std::runtime_error);
  estop::Reader first(udp_settings());
  auto again = udp_settings();
  again.udp_port = first.bound_port();
  EXPECT_THROW(estop::Reader{again}, std::runtime_error);   // no SO_REUSEADDR sharing
}

TEST(HwEstop, SharedReaderIsOnePerProcess)
{
  auto s = udp_settings();
  auto a = estop::shared(s);
  auto b = estop::shared(s);
  EXPECT_EQ(a.get(), b.get());
  estop::Settings none;
  EXPECT_EQ(estop::shared(none), nullptr);
}

// --- the station probe's reader --------------------------------------------------------------------
TEST(HwProbe, ParsesOnlyWellFormedFrames)
{
  long seq = 0;
  int state = 0;
  int64_t edge = 0, sent = 0;
  ASSERT_TRUE(estop::parse_probe_frame("PRB1 7 1 123456789 987654321\n", seq, state, edge, sent));
  EXPECT_EQ(seq, 7);
  EXPECT_EQ(state, 1);
  EXPECT_EQ(edge, 123456789);
  EXPECT_EQ(sent, 987654321);
  EXPECT_TRUE(estop::parse_probe_frame("PRB1 0 0 0 5", seq, state, edge, sent));
  for (const char * bad : {"PRB1 7 2 1 2", "PRB1 7 1 1", "PRB1 7 1 1 2 3", "EST1 7 1", "PRB1 -7 1 1 2",
      "PRB1 7 1 1 x", "PRB1 7  1 1 2", "PRB1 7 1 1 2 ", "PRB1 12345678901234567890 1 1 2"})
  {
    EXPECT_FALSE(estop::parse_probe_frame(bad, seq, state, edge, sent)) << bad;
  }
}

TEST(HwProbe, MapsTheRelayClockByTheLeastDelayedFrameAndKeepsTheRisingEdge)
{
  estop::ProbeSettings s;
  s.udp_port = 0;
  s.relay_addr = "127.0.0.1";
  estop::ProbeReader reader(s);
  // The relay's clock runs 5 s behind this one; frames take 1-3 ms, one of them only 0.2 ms.
  const int64_t offset = 5'000'000'000, ms = 1'000'000;
  int64_t relay = 1'000'000'000;
  long seq = 0;
  for (int i = 0; i < 50; ++i, relay += 20 * ms) {
    const int64_t delay = i == 17 ? ms / 5 : ms + (i % 3) * ms;
    reader.accept(seq++, 0, 0, relay, relay + offset + delay);
  }
  EXPECT_FALSE(reader.status(relay + offset).triggered);
  EXPECT_TRUE(std::isnan(reader.status(relay + offset).rise_age_s));
  // Touched at relay time `edge`, reported 2 ms later with 1 ms of network delay.
  const int64_t edge = relay + 5 * ms;
  reader.accept(seq++, 1, edge, edge + 2 * ms, edge + 2 * ms + offset + ms);
  const int64_t now = edge + offset + 10 * ms;
  const ProbeStatus p = reader.status(now);
  EXPECT_TRUE(p.enabled && p.fresh && p.triggered);
  // The map carries only the least delay (0.2 ms), so the edge reads 9.8 ms old, not 10.
  EXPECT_NEAR(p.rise_age_s, 0.0098, 1e-9);
  // Released: the rising edge is kept; heartbeats with the same stamp change nothing.
  reader.accept(seq++, 0, edge + 30 * ms, edge + 31 * ms, edge + 31 * ms + offset + ms);
  reader.accept(seq++, 0, edge + 30 * ms, edge + 51 * ms, edge + 51 * ms + offset + ms);
  EXPECT_FALSE(reader.status(edge + 52 * ms + offset).triggered);
  EXPECT_NEAR(reader.status(now).rise_age_s, 0.0098, 1e-9);
  // Silence: stale; a restarted relay (new clock, sequence from 0) re-seeds the map and the edge.
  EXPECT_FALSE(reader.status(edge + offset + 400 * ms).fresh);
  reader.accept(0, 0, 0, 7 * ms, edge + offset + 500 * ms);
  EXPECT_TRUE(reader.status(edge + offset + 501 * ms).fresh);
  EXPECT_TRUE(std::isnan(reader.status(edge + offset + 501 * ms).rise_age_s));
}

TEST(HwProbe, UdpAcceptsTheRelayOnlyAndGoesStaleInSilence)
{
  estop::ProbeSettings s;
  s.udp_port = 0;
  s.relay_addr = "127.0.0.1";
  s.timeout_s = 0.1;
  estop::ProbeReader reader(s);
  Sender relay("127.0.0.1"), stranger("127.0.0.2");
  auto frame = [](long seq, int state) {
      return "PRB1 " + std::to_string(seq) + " " + std::to_string(state) + " 0 " +
             std::to_string(estop::now_ns()) + "\n";
    };
  for (long i = 0; i < 5; ++i) {relay.send(reader.bound_port(), frame(i, 0)); std::this_thread::sleep_for(5ms);}
  std::this_thread::sleep_for(20ms);
  EXPECT_TRUE(reader.status(estop::now_ns()).fresh);
  EXPECT_FALSE(reader.status(estop::now_ns()).triggered);
  stranger.send(reader.bound_port(), frame(1000, 1));   // ignored, however new
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(reader.status(estop::now_ns()).triggered);
  std::this_thread::sleep_for(150ms);
  EXPECT_FALSE(reader.status(estop::now_ns()).fresh);
  estop::ProbeSettings bad = s;
  bad.relay_addr = "palette-pi";
  EXPECT_THROW(estop::ProbeReader{bad}, std::runtime_error);
}
