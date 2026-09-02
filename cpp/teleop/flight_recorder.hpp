#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include <pthread.h>

namespace tatbot::flight
{

struct Stats
{
  uint64_t records_enqueued = 0;
  uint64_t records_dropped = 0;
  uint64_t write_errors = 0;
};

/// Bounded, nonblocking flight-log handoff for the real-time control thread.
///
/// Each append is one atomic O_NONBLOCK pipe write. A normal-priority worker
/// drains the bounded kernel pipe to local disk, so a slow filesystem can drop
/// complete records but can never stall or corrupt the 400 Hz producer.
class Recorder
{
public:
  explicit Recorder(const std::string & path);
  ~Recorder();

  Recorder(const Recorder &) = delete;
  Recorder & operator=(const Recorder &) = delete;

  bool write_header(const void * data, size_t bytes) noexcept;
  bool append_record(const void * data, size_t bytes) noexcept;

  /// Drain queued bytes and close the file. Safe to call more than once.
  Stats finish() noexcept;
  Stats stats() const noexcept;

private:
  bool enqueue(const void * data, size_t bytes, bool record) noexcept;
  static void * thread_entry(void * self) noexcept;
  void drain() noexcept;

  int file_fd_ = -1;
  int read_fd_ = -1;
  int write_fd_ = -1;
  size_t atomic_write_limit_ = 0;
  pthread_t worker_{};
  bool worker_started_ = false;
  bool finished_ = false;
  std::atomic<uint64_t> records_enqueued_{0};
  std::atomic<uint64_t> records_dropped_{0};
  std::atomic<uint64_t> write_errors_{0};
};

}  // namespace tatbot::flight
