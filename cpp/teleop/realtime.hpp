#pragma once

#include <map>
#include <string>
#include <vector>

namespace tatbot::realtime
{

/// What apply() managed to do, so the caller can tell the operator plainly.
struct Setup
{
  bool affinity_applied = false;
  bool fifo_applied = false;
  std::vector<int> cpus;        // CPUs the loop was pinned to (empty if not pinned)
  std::string affinity_error;   // empty when affinity_applied
  std::string fifo_error;       // empty when fifo_applied
};

/// Pick the fastest class of cores from a cpu -> cpuinfo_max_freq map.
///
/// On Intel hybrid parts (e.g. a Core Ultra 9 185H: P-cores at 4.8-5.1 GHz,
/// E-cores at 3.8 GHz, LP E-cores at 2.5 GHz) the kernel will happily migrate a
/// mostly-sleeping 400 Hz loop onto an E-core, where the same work takes ~3x
/// longer. Cores within 10% of the highest max frequency count as the fast
/// class, which keeps hyperthread siblings of slightly-different-boost P-cores
/// together while excluding a whole slower tier.
///
/// A homogeneous CPU yields every core, which is the correct no-op.
std::vector<int> fastest_cpus(const std::map<int, long> & max_freq_khz);

/// Read cpu -> cpuinfo_max_freq (kHz) for every online CPU. Empty if the
/// kernel does not expose cpufreq (VMs, some ARM boards).
std::map<int, long> read_max_frequencies(const std::string & sysfs_root = "/sys/devices/system/cpu");

/// Pin this thread to the fastest cores and request SCHED_FIFO at `priority`.
///
/// Must be called BEFORE the arm drivers are constructed: pthreads inherit both
/// the affinity mask and (with the default PTHREAD_INHERIT_SCHED) the policy,
/// so the SDK's UDP daemon threads get the same treatment as the control loop.
///
/// Never throws and never aborts: each half is best-effort and reported in the
/// returned Setup, because an unprivileged bench run must still work.
Setup apply(int priority);

/// "0-11" style rendering of a CPU list, for logs.
std::string format_cpus(const std::vector<int> & cpus);

}  // namespace tatbot::realtime
