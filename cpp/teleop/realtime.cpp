#include "realtime.hpp"

#include <sched.h>
#include <sys/resource.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace tatbot::realtime
{
namespace
{

// Cores whose advertised max frequency is within this fraction of the fastest
// core count as the same (fast) tier. P-core boost clocks differ by ~6% across
// a die; the gap to the next tier down is >20%, so 10% separates them cleanly.
constexpr double same_tier_ratio = 0.90;

}  // namespace

std::vector<int> fastest_cpus(const std::map<int, long> & max_freq_khz)
{
  std::vector<int> cpus;
  if (max_freq_khz.empty()) {
    return cpus;
  }
  long best = 0;
  for (const auto & [cpu, khz] : max_freq_khz) {
    (void)cpu;
    best = std::max(best, khz);
  }
  if (best <= 0) {
    return cpus;
  }
  const double threshold = static_cast<double>(best) * same_tier_ratio;
  for (const auto & [cpu, khz] : max_freq_khz) {
    if (static_cast<double>(khz) >= threshold) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

std::map<int, long> read_max_frequencies(const std::string & sysfs_root)
{
  std::map<int, long> result;
  std::error_code error;
  if (!std::filesystem::is_directory(sysfs_root, error)) {
    return result;
  }
  for (const auto & entry : std::filesystem::directory_iterator(sysfs_root, error)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("cpu", 0) != 0 || name.size() <= 3) {
      continue;
    }
    if (!std::all_of(name.begin() + 3, name.end(), [](char c) {return c >= '0' && c <= '9';})) {
      continue;  // skip cpufreq/, cpuidle/, ...
    }
    std::ifstream file(entry.path() / "cpufreq" / "cpuinfo_max_freq");
    long khz = 0;
    if (file >> khz && khz > 0) {
      result[std::stoi(name.substr(3))] = khz;
    }
  }
  return result;
}

std::string format_cpus(const std::vector<int> & cpus)
{
  if (cpus.empty()) {
    return "none";
  }
  std::vector<int> sorted = cpus;
  std::sort(sorted.begin(), sorted.end());
  std::ostringstream out;
  size_t i = 0;
  while (i < sorted.size()) {
    size_t j = i;
    while (j + 1 < sorted.size() && sorted[j + 1] == sorted[j] + 1) {
      ++j;
    }
    if (i != 0) {
      out << ',';
    }
    out << sorted[i];
    if (j > i) {
      out << '-' << sorted[j];
    }
    i = j + 1;
  }
  return out.str();
}

Setup apply(int priority)
{
  Setup setup;

  setup.cpus = fastest_cpus(read_max_frequencies());
  if (setup.cpus.empty()) {
    setup.affinity_error = "no cpufreq information; leaving affinity to the scheduler";
  } else {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (const int cpu : setup.cpus) {
      CPU_SET(cpu, &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) == 0) {
      setup.affinity_applied = true;
    } else {
      setup.affinity_error = std::strerror(errno);
      setup.cpus.clear();
    }
  }

  sched_param param{};
  param.sched_priority = priority;
  if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
    setup.fifo_applied = true;
  } else {
    const int failure = errno;
    rlimit limit{};
    const bool capped = getrlimit(RLIMIT_RTPRIO, &limit) == 0 &&
      limit.rlim_cur < static_cast<rlim_t>(priority);
    setup.fifo_error = std::strerror(failure);
    if (failure == EPERM && capped) {
      setup.fifo_error += " (RLIMIT_RTPRIO is " + std::to_string(limit.rlim_cur) + ")";
    }
  }
  return setup;
}

uint64_t advance_deadline(
  std::chrono::steady_clock::time_point & next,
  std::chrono::steady_clock::duration period,
  std::chrono::steady_clock::time_point now) noexcept
{
  if (period <= std::chrono::steady_clock::duration::zero()) {
    next = now;
    return 0;
  }
  next += period;
  if (next > now) {return 0;}
  const auto missed = static_cast<uint64_t>((now - next) / period) + 1;
  next = now + period;
  return missed;
}

}  // namespace tatbot::realtime
