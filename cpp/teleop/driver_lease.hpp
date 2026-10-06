#pragma once
// Shared with tatbot-arm::lease and lerobot_robot_tatbot.driver_lease.
// Never unlink the locked inode. A lease is ownership, not motion authority.
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "driver_takeover.hpp"

namespace tatbot {
class DriverLease {
 public:
  enum class Mode { exclusive, recover };
  explicit DriverLease(const char * path = "/tmp/tatbot-arm-driver.lock",
                       Mode mode = Mode::exclusive) {
    fd_ = ::open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600);
    if (fd_ < 0) throw std::runtime_error(std::string("driver lease: ") + std::strerror(errno));
    struct stat info {};
    if (::fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != ::getuid()) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("driver lease must be a regular file owned by this user");
    }
    try {
      if (mode == Mode::recover) {
        driver_takeover::acquire(fd_);
      } else if (!driver_takeover::try_lock(fd_)) {
        throw std::runtime_error("driver busy: Resource temporarily unavailable");
      }
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }
  ~DriverLease() { if (fd_ >= 0) ::close(fd_); }
  DriverLease(const DriverLease &) = delete;
  DriverLease & operator=(const DriverLease &) = delete;
 private:
  int fd_ = -1;
};
}  // namespace tatbot
