#pragma once
// The e-stop reader: the Pico's EST1 heartbeat from a local serial port or from the palette Pi's
// UDP relay (ros/tatbot_estop_relay). Ported from cpp/teleop/estop_monitor.cpp. One reader per
// process, shared by both arms. The status is judged by the age of the last valid frame at every
// call, never by a latched flag: silence is a stop.
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include "tatbot_hardware/core.hpp"

namespace tatbot_hardware::estop
{

// "EST1 <seq> <state>" with an optional trailing "\n" (or "\r\n"); state 1 released, 0 pressed.
bool parse_frame(std::string_view line, long & seq, int & state);

struct Settings
{
  int source = names::kEstopNone;     // none | serial | udp
  std::string device = "/dev/tatbot-estop";
  double timeout_s = 0.10;            // serial 0.10, udp 0.15
  int udp_port = 7640;                // 0 binds an ephemeral port (tests)
  std::string relay_addr;             // udp: the only source address accepted
  int debounce_frames = 3;
  bool operator==(const Settings & o) const
  {
    return source == o.source && device == o.device && timeout_s == o.timeout_s &&
           udp_port == o.udp_port && relay_addr == o.relay_addr &&
           debounce_frames == o.debounce_frames;
  }
};

int source_from_name(const std::string & name);   // -1 when unknown
int64_t now_ns();                                 // steady clock

class Reader
{
public:
  explicit Reader(const Settings & settings);
  ~Reader();
  Reader(const Reader &) = delete;
  Reader & operator=(const Reader &) = delete;

  EstopStatus status(int64_t now) const;
  const Settings & settings() const {return settings_;}
  int bound_port() const {return bound_port_;}
  // Accept one parsed frame received at t (reader thread; public for tests).
  void accept(long seq, int state, int64_t t);

private:
  void run_serial();
  void run_udp();

  Settings settings_;
  int64_t timeout_ns_;
  uint32_t relay_ = 0;       // network order; 0 accepts nothing
  int fd_ = -1;
  int bound_port_ = 0;
  long last_seq_ = -1;
  int raw_ = -1, count_ = 0;
  std::atomic<int64_t> last_ns_{-1};
  std::atomic<int> stable_{-1};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// The process-wide reader (nullptr for source none). A second caller gets the first reader.
std::shared_ptr<Reader> shared(const Settings & settings);

// --- the station probe ----------------------------------------------------------------------------
// "PRB1 <seq> <state> <edge_ns> <now_ns>" from the probe's own relay instance on the palette Pi
// (ros/tatbot_estop_relay --probe): state 1 touched (or a broken wire), 0 at rest; edge_ns the relay
// kernel's CLOCK_MONOTONIC stamp of the latest edge (0 before the first); now_ns the relay's clock
// when it sent.
bool parse_probe_frame(std::string_view line, long & seq, int & state, int64_t & edge_ns, int64_t & sent_ns);

struct ProbeSettings
{
  int udp_port = 7641;          // 0 binds an ephemeral port (tests)
  std::string relay_addr;       // the only source address accepted
  double timeout_s = 0.2;       // frames come at 50 Hz
  bool operator==(const ProbeSettings & o) const
  {
    return udp_port == o.udp_port && relay_addr == o.relay_addr && timeout_s == o.timeout_s;
  }
};

// Relay time maps onto this host's steady clock by the smallest (receive - send) over the last
// kClockWindow frames: the least-delayed frame bounds the offset, and the window follows drift. The
// latest rising edge is kept in relay time and mapped at every status(), so a later, better offset
// improves it.
class ProbeReader
{
public:
  static constexpr size_t kClockWindow = 100;   // 2 s of frames

  explicit ProbeReader(const ProbeSettings & settings);
  ~ProbeReader();
  ProbeReader(const ProbeReader &) = delete;
  ProbeReader & operator=(const ProbeReader &) = delete;

  ProbeStatus status(int64_t now) const;
  const ProbeSettings & settings() const {return settings_;}
  int bound_port() const {return bound_port_;}
  // Accept one parsed frame received at t on this clock (reader thread; public for tests).
  void accept(long seq, int state, int64_t edge_ns, int64_t sent_ns, int64_t t);

private:
  void run();

  ProbeSettings settings_;
  int64_t timeout_ns_;
  uint32_t relay_ = 0;       // network order; 0 accepts nothing
  int fd_ = -1;
  int bound_port_ = 0;
  long last_seq_ = -1;
  std::array<int64_t, kClockWindow> offsets_{};
  size_t offsets_n_ = 0, offsets_head_ = 0;
  int64_t last_edge_ = 0;
  std::atomic<int64_t> last_ns_{-1}, offset_ns_{0}, rise_relay_ns_{-1};
  std::atomic<int> state_{-1};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// The process-wide probe reader, shared by both arms (one touches at a time).
std::shared_ptr<ProbeReader> shared_probe(const ProbeSettings & settings);

}  // namespace tatbot_hardware::estop
