#include "tatbot_hardware/runtime.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace tatbot_hardware::runtime
{

std::vector<int> fastest_cpus(const std::map<int, long> & max_freq_khz)
{
  long best = 0;
  for (const auto & [cpu, khz] : max_freq_khz) {best = std::max(best, khz);}
  std::vector<int> cpus;
  for (const auto & [cpu, khz] : max_freq_khz) {
    if (best > 0 && static_cast<double>(khz) >= 0.90 * static_cast<double>(best)) {cpus.push_back(cpu);}
  }
  return cpus;
}

std::map<int, long> read_max_frequencies(const std::string & sysfs_root)
{
  std::map<int, long> result;
  std::error_code error;
  for (const auto & entry : std::filesystem::directory_iterator(sysfs_root, error)) {
    const std::string name = entry.path().filename().string();
    if (name.size() <= 3 || name.rfind("cpu", 0) != 0 ||
      !std::all_of(name.begin() + 3, name.end(), [](char c) {return c >= '0' && c <= '9';}))
    {
      continue;
    }
    std::ifstream file(entry.path() / "cpufreq" / "cpuinfo_max_freq");
    long khz = 0;
    if (file >> khz && khz > 0) {result[std::stoi(name.substr(3))] = khz;}
  }
  return result;
}

RtSetup apply_realtime(int priority, const std::vector<int> & cpus)
{
  RtSetup setup;
  setup.cpus = cpus.empty() ? fastest_cpus(read_max_frequencies()) : cpus;
  if (!setup.cpus.empty()) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (const int cpu : setup.cpus) {CPU_SET(cpu, &mask);}
    setup.affinity = sched_setaffinity(0, sizeof(mask), &mask) == 0;
    if (!setup.affinity) {setup.error = std::string("affinity: ") + std::strerror(errno) + "; ";}
  }
  sched_param param{};
  param.sched_priority = priority;
  setup.fifo = sched_setscheduler(0, SCHED_FIFO, &param) == 0;
  if (!setup.fifo) {
    rlimit limit{};
    getrlimit(RLIMIT_RTPRIO, &limit);
    setup.error += std::string("SCHED_FIFO ") + std::to_string(priority) + ": " +
      std::strerror(errno) + " (RLIMIT_RTPRIO " + std::to_string(limit.rlim_cur) + ")";
  }
  return setup;
}

DriverLock::DriverLock(const std::string & path)
{
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600);
  if (fd_ < 0) {throw std::runtime_error("driver lock " + path + ": " + std::strerror(errno));}
  struct stat info {};
  if (::fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != ::getuid()) {
    ::close(fd_);
    throw std::runtime_error("driver lock " + path + " must be a regular file owned by this user");
  }
  if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    const int error = errno;
    ::close(fd_);
    throw std::runtime_error(
            error == EWOULDBLOCK ? "driver busy: another process holds " + path :
            "driver lock " + path + ": " + std::strerror(error));
  }
}

DriverLock::~DriverLock()
{
  if (fd_ >= 0) {::close(fd_);}   // never unlink the locked inode
}

std::shared_ptr<DriverLock> DriverLock::acquire(const std::string & path)
{
  static std::mutex mutex;
  static std::weak_ptr<DriverLock> instance;
  std::lock_guard<std::mutex> lock(mutex);
  auto held = instance.lock();
  if (!held) {
    held = std::make_shared<DriverLock>(path);
    instance = held;
  }
  return held;
}

FlightRecorder::FlightRecorder(const std::string & path)
{
  file_fd_ = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
  if (file_fd_ < 0) {throw std::runtime_error("flight log " + path + ": " + std::strerror(errno));}
  int fds[2] = {-1, -1};
  if (pipe2(fds, O_CLOEXEC | O_NONBLOCK) != 0) {
    ::close(file_fd_);
    throw std::runtime_error(std::string("flight pipe: ") + std::strerror(errno));
  }
  read_fd_ = fds[0];
  write_fd_ = fds[1];
  const long limit = fpathconf(write_fd_, _PC_PIPE_BUF);
  atomic_limit_ = limit > 0 ? static_cast<size_t>(limit) : 512U;
  // The caller may be SCHED_FIFO: the disk worker is explicitly SCHED_OTHER.
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
  sched_param none{};
  pthread_attr_setschedparam(&attr, &none);
  started_ = pthread_create(&worker_, &attr, &FlightRecorder::entry, this) == 0;
  pthread_attr_destroy(&attr);
  if (!started_) {
    ::close(write_fd_);
    ::close(read_fd_);
    ::close(file_fd_);
    throw std::runtime_error("flight recorder: cannot start its worker");
  }
}

FlightRecorder::~FlightRecorder() {finish();}

bool FlightRecorder::append(const void * data, size_t bytes) noexcept
{
  if (write_fd_ < 0 || bytes == 0 || bytes > atomic_limit_) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  ssize_t n;
  do {n = ::write(write_fd_, data, bytes);} while (n < 0 && errno == EINTR);
  if (n == static_cast<ssize_t>(bytes)) {return true;}
  dropped_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

void FlightRecorder::finish() noexcept
{
  if (write_fd_ >= 0) {::close(write_fd_); write_fd_ = -1;}
  if (started_) {pthread_join(worker_, nullptr); started_ = false;}
  if (read_fd_ >= 0) {::close(read_fd_); read_fd_ = -1;}
  if (file_fd_ >= 0) {::close(file_fd_); file_fd_ = -1;}
}

void * FlightRecorder::entry(void * self) noexcept
{
  static_cast<FlightRecorder *>(self)->drain();
  return nullptr;
}

void FlightRecorder::drain() noexcept
{
  std::array<char, 64 * 1024> buffer{};
  bool healthy = true;
  while (true) {
    pollfd ready{read_fd_, POLLIN | POLLHUP, 0};
    if (::poll(&ready, 1, -1) < 0 && errno != EINTR) {return;}
    while (true) {
      const ssize_t count = ::read(read_fd_, buffer.data(), buffer.size());
      if (count == 0) {return;}   // writer closed and drained
      if (count < 0) {
        if (errno == EINTR) {continue;}
        if (errno == EAGAIN || errno == EWOULDBLOCK) {break;}
        return;
      }
      size_t offset = 0;
      while (healthy && offset < static_cast<size_t>(count)) {
        const ssize_t w = ::write(file_fd_, buffer.data() + offset, static_cast<size_t>(count) - offset);
        if (w > 0) {offset += static_cast<size_t>(w);} else if (!(w < 0 && errno == EINTR)) {healthy = false;}
      }
    }
  }
}

}  // namespace tatbot_hardware::runtime
