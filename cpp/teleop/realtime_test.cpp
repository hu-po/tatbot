#include "realtime.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

namespace {

void require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << "realtime_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main()
{
  using tatbot::realtime::fastest_cpus;
  using tatbot::realtime::format_cpus;

  // A Core Ultra 9 185H: P-cores 0-11 (4.8-5.1 GHz), E-cores 12-19
  // (3.8 GHz), LP E-cores 20-21 (2.5 GHz). Only the P-cores may be chosen —
  // an E-core makes the same 400 Hz tick take ~3x longer.
  std::map<int, long> hybrid;
  const long p_core[] = {4800000, 5100000, 5100000, 5100000, 5100000, 4800000,
    4800000, 4800000, 4800000, 4800000, 4800000, 4800000};
  for (int cpu = 0; cpu < 12; ++cpu) {hybrid[cpu] = p_core[cpu];}
  for (int cpu = 12; cpu < 20; ++cpu) {hybrid[cpu] = 3800000;}
  for (int cpu = 20; cpu < 22; ++cpu) {hybrid[cpu] = 2500000;}
  const std::vector<int> fast = fastest_cpus(hybrid);
  require(fast.size() == 12, "hybrid CPU: expected the 12 P-core threads");
  require(fast.front() == 0 && fast.back() == 11, "hybrid CPU: expected CPUs 0-11");
  require(format_cpus(fast) == "0-11", "hybrid CPU: expected the range to render as 0-11");

  // Hyperthread siblings that boost a little differently stay in the same tier.
  require(fastest_cpus({{0, 5100000}, {1, 4800000}}).size() == 2,
    "a 6% boost gap must not split the P-core tier");

  // A homogeneous CPU must yield every core: pinning is then a no-op, not a
  // way to strand the loop on core 0.
  std::map<int, long> uniform;
  for (int cpu = 0; cpu < 8; ++cpu) {uniform[cpu] = 3000000;}
  require(fastest_cpus(uniform).size() == 8, "homogeneous CPU: expected every core");

  // No cpufreq at all (VMs, containers): report nothing rather than guessing.
  require(fastest_cpus({}).empty(), "empty input must yield no CPUs");
  require(fastest_cpus({{0, 0}, {1, 0}}).empty(), "zero frequencies must yield no CPUs");
  require(format_cpus({}) == "none", "an empty CPU list must render as none");

  // Discontiguous sets render as ranges plus singletons.
  require(format_cpus({0, 1, 2, 5, 7, 8}) == "0-2,5,7-8", "unexpected CPU list rendering");

  // Reading the real sysfs must either work or degrade quietly; on a missing
  // path it must not throw.
  require(tatbot::realtime::read_max_frequencies("/nonexistent/cpu/path").empty(),
    "a missing sysfs root must yield no frequencies");

  using clock = std::chrono::steady_clock;
  using namespace std::chrono_literals;
  auto deadline = clock::time_point{};
  require(tatbot::realtime::advance_deadline(deadline, 2500us, clock::time_point{}) == 0,
    "on-time tick unexpectedly skipped a deadline");
  require(deadline == clock::time_point{} + 2500us, "on-time deadline did not advance");
  require(tatbot::realtime::advance_deadline(deadline, 2500us, clock::time_point{} + 10ms) == 3,
    "large stall reported the wrong number of skipped deadlines");
  require(deadline == clock::time_point{} + 12500us,
    "late deadline was not resynchronized one period after now");
  require(tatbot::realtime::advance_deadline(deadline, 0us, clock::time_point{} + 20ms) == 0,
    "invalid period should not divide by zero");
  require(deadline == clock::time_point{} + 20ms,
    "invalid period should reset the deadline to now");

  std::cout << "realtime_test: ok" << std::endl;
  return 0;
}
