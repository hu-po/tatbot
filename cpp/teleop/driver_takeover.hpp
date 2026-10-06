#pragma once
// Recovery-only takeover of the existing driver lease. Keep the inode and
// acquire its flock before constructing a driver; never infer ownership from
// a process name, a stale PID file, or merely having the lock file open.
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace tatbot::driver_takeover {

inline bool try_lock(int fd) {
  if (::flock(fd, LOCK_EX | LOCK_NB) == 0) return true;
  if (errno != EWOULDBLOCK && errno != EAGAIN) {
    throw std::runtime_error(std::string("driver lease: ") + std::strerror(errno));
  }
  return false;
}

// True when a descriptor's fdinfo carries an flock on inode `ino`. Reading
// fdinfo never reaches the file's filesystem, unlike stat() through fd/N, which
// blocks uninterruptibly on a descriptor into a hung hard-mounted NFS share.
inline bool fdinfo_flocks(const std::string &path, ino_t ino) {
  std::ifstream info(path);
  std::string line;
  while (std::getline(info, line)) {
    // lock:  1: FLOCK  ADVISORY  WRITE <pid> <maj>:<min>:<ino> <start> <end>
    std::istringstream fields(line);
    std::string label, id, type, advisory, access, pid, file;
    if (!(fields >> label >> id >> type >> advisory >> access >> pid >> file) ||
        label != "lock:" || type != "FLOCK" || advisory != "ADVISORY" ||
        (access != "WRITE" && access != "READ")) continue;
    const auto colon = file.rfind(':');
    if (colon != std::string::npos && file.substr(colon + 1) == std::to_string(ino)) return true;
  }
  return false;
}

inline bool holds_lock(pid_t pid, const struct stat &lease) {
  const std::string root = "/proc/" + std::to_string(pid);
  struct stat process {};
  if (::stat(root.c_str(), &process) != 0 || process.st_uid != ::getuid()) return false;
  std::error_code error;
  std::filesystem::directory_iterator entry(root + "/fdinfo", error), end;
  for (; !error && entry != end; entry.increment(error)) {
    // fdinfo reports flock ownership for *each* inherited descriptor, even
    // after the original locking process exits (/proc/locks can omit it).
    // Only a descriptor already flocking the lease's inode number is statted
    // to confirm device and inode.
    if (!fdinfo_flocks(entry->path().string(), lease.st_ino)) continue;
    const std::string fd = root + "/fd/" + entry->path().filename().string();
    struct stat file {};
    if (::stat(fd.c_str(), &file) == 0 &&
        file.st_dev == lease.st_dev && file.st_ino == lease.st_ino) return true;
  }
  return false;
}

inline void signal_holders(int fd, int signal) {
  struct stat lease {};
  if (::fstat(fd, &lease) != 0) throw std::runtime_error("cannot stat driver lease");
  for (const auto &entry : std::filesystem::directory_iterator("/proc")) {
    const std::string name = entry.path().filename().string();
    if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) continue;
    const pid_t pid = static_cast<pid_t>(std::stol(name));
    if (!holds_lock(pid, lease)) continue;
    if (pid == ::getpid()) throw std::runtime_error("recovery already holds another driver lease");
    // Pin identity before rechecking ownership: a PID recycled between /proc
    // inspection and signaling must never redirect a signal to another task.
    const int process_fd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
    if (process_fd < 0) {
      if (errno == ESRCH) continue;
      throw std::runtime_error(std::string("cannot pin driver owner: ") + std::strerror(errno));
    }
    pollfd exited{process_fd, POLLIN, 0};
    const bool owner = holds_lock(pid, lease) && ::poll(&exited, 1, 0) == 0;
    int result = 0;
    int error = 0;
    if (owner) {
      std::cerr << "arm_recover: sending " << (signal == SIGTERM ? "SIGTERM" : "SIGKILL")
                << " to driver lease holder pid=" << pid << std::endl;
      result = static_cast<int>(::syscall(SYS_pidfd_send_signal, process_fd, signal, nullptr, 0));
      error = errno;
    }
    ::close(process_fd);
    if (result != 0 && error != ESRCH) {
      throw std::runtime_error(std::string("cannot stop driver owner: ") + std::strerror(error));
    }
  }
}

inline bool wait_for_lock(int fd, std::chrono::milliseconds budget) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  do {
    if (try_lock(fd)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (std::chrono::steady_clock::now() < deadline);
  return try_lock(fd);
}

inline void acquire(int fd, std::chrono::milliseconds grace = std::chrono::seconds(10),
                    std::chrono::milliseconds kill_wait = std::chrono::seconds(2)) {
  if (try_lock(fd)) return;
  signal_holders(fd, SIGTERM);
  if (wait_for_lock(fd, grace)) return;
  // Reinspect instead of trusting the earlier owner list: a stuck recovery
  // can leave a forked SDK child holding the same open-file description.
  signal_holders(fd, SIGKILL);
  if (wait_for_lock(fd, kill_wait)) return;
  throw std::runtime_error("driver busy: ownership was not released after termination; recovery refused");
}

}  // namespace tatbot::driver_takeover
