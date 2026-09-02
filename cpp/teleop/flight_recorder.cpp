#include "flight_recorder.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace tatbot::flight
{

Recorder::Recorder(const std::string & path)
{
  file_fd_ = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0666);
  if (file_fd_ < 0) {
    throw std::runtime_error("cannot open log file " + path + ": " + std::strerror(errno));
  }

  int pipe_fds[2] = {-1, -1};
  if (pipe2(pipe_fds, O_CLOEXEC | O_NONBLOCK) != 0) {
    const int failure = errno;
    close(file_fd_);
    file_fd_ = -1;
    throw std::runtime_error("cannot create flight-recorder pipe: " + std::string(std::strerror(failure)));
  }
  read_fd_ = pipe_fds[0];
  write_fd_ = pipe_fds[1];
  const long pipe_limit = fpathconf(write_fd_, _PC_PIPE_BUF);
  atomic_write_limit_ = pipe_limit > 0 ? static_cast<size_t>(pipe_limit) : 512U;

  // The caller may already be SCHED_FIFO. The disk worker must explicitly be
  // SCHED_OTHER so slow I/O cannot inherit real-time priority and compete with
  // the arm driver/control threads.
  pthread_attr_t attributes;
  int status = pthread_attr_init(&attributes);
  const bool attributes_initialized = status == 0;
  if (status == 0) {status = pthread_attr_setinheritsched(&attributes, PTHREAD_EXPLICIT_SCHED);}
  if (status == 0) {status = pthread_attr_setschedpolicy(&attributes, SCHED_OTHER);}
  sched_param scheduling{};
  if (status == 0) {status = pthread_attr_setschedparam(&attributes, &scheduling);}
  if (status == 0) {status = pthread_create(&worker_, &attributes, &Recorder::thread_entry, this);}
  if (attributes_initialized) {pthread_attr_destroy(&attributes);}
  if (status != 0) {
    close(write_fd_);
    close(read_fd_);
    close(file_fd_);
    write_fd_ = read_fd_ = file_fd_ = -1;
    throw std::runtime_error("cannot start flight-recorder worker: " + std::string(std::strerror(status)));
  }
  worker_started_ = true;
}

Recorder::~Recorder()
{
  finish();
}

bool Recorder::write_header(const void * data, size_t bytes) noexcept
{
  return enqueue(data, bytes, false);
}

bool Recorder::append_record(const void * data, size_t bytes) noexcept
{
  return enqueue(data, bytes, true);
}

bool Recorder::enqueue(const void * data, size_t bytes, bool record) noexcept
{
  if (write_fd_ < 0 || data == nullptr || bytes == 0 || bytes > atomic_write_limit_) {
    if (record) {records_dropped_.fetch_add(1, std::memory_order_relaxed);}
    write_errors_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  ssize_t written;
  do {
    written = write(write_fd_, data, bytes);
  } while (written < 0 && errno == EINTR);
  if (written == static_cast<ssize_t>(bytes)) {
    if (record) {records_enqueued_.fetch_add(1, std::memory_order_relaxed);}
    return true;
  }
  if (record) {records_dropped_.fetch_add(1, std::memory_order_relaxed);}
  if (written >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
    write_errors_.fetch_add(1, std::memory_order_relaxed);
  }
  return false;
}

Stats Recorder::stats() const noexcept
{
  return Stats{
    records_enqueued_.load(std::memory_order_relaxed),
    records_dropped_.load(std::memory_order_relaxed),
    write_errors_.load(std::memory_order_relaxed)};
}

Stats Recorder::finish() noexcept
{
  if (finished_) {return stats();}
  finished_ = true;
  if (write_fd_ >= 0) {
    close(write_fd_);
    write_fd_ = -1;
  }
  if (worker_started_) {
    const int status = pthread_join(worker_, nullptr);
    if (status != 0) {write_errors_.fetch_add(1, std::memory_order_relaxed);}
    worker_started_ = false;
  }
  if (read_fd_ >= 0) {
    close(read_fd_);
    read_fd_ = -1;
  }
  if (file_fd_ >= 0) {
    if (close(file_fd_) != 0) {write_errors_.fetch_add(1, std::memory_order_relaxed);}
    file_fd_ = -1;
  }
  return stats();
}

void * Recorder::thread_entry(void * self) noexcept
{
  static_cast<Recorder *>(self)->drain();
  return nullptr;
}

void Recorder::drain() noexcept
{
  std::array<char, 64 * 1024> buffer{};
  bool file_healthy = true;
  while (true) {
    pollfd ready{read_fd_, POLLIN | POLLHUP, 0};
    int poll_status;
    do {
      poll_status = poll(&ready, 1, -1);
    } while (poll_status < 0 && errno == EINTR);
    if (poll_status < 0) {
      write_errors_.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    while (true) {
      const ssize_t count = read(read_fd_, buffer.data(), buffer.size());
      if (count == 0) {return;}
      if (count < 0) {
        if (errno == EINTR) {continue;}
        if (errno == EAGAIN || errno == EWOULDBLOCK) {break;}
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (!file_healthy) {continue;}
      size_t offset = 0;
      while (offset < static_cast<size_t>(count)) {
        const ssize_t written = write(
          file_fd_, buffer.data() + offset, static_cast<size_t>(count) - offset);
        if (written > 0) {
          offset += static_cast<size_t>(written);
        } else if (written < 0 && errno == EINTR) {
          continue;
        } else {
          write_errors_.fetch_add(1, std::memory_order_relaxed);
          file_healthy = false;
          break;
        }
      }
    }
  }
}

}  // namespace tatbot::flight
