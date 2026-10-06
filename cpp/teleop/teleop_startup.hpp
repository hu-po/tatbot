#pragma once

#include <atomic>
#include <poll.h>
#include <unistd.h>

namespace tatbot::teleop
{
enum class StopChoice { release, emergency, resume, estop };

// Losing a console cannot authorize a landing or a release. Stay holding;
// explicit signals and the hardware E-stop remain observable after EOF.
inline StopChoice wait_for_hold_choice(
  int fd, const std::atomic<int> & stops, int accepted_stops,
  const std::atomic<int> & estop, int healthy_estop)
{
  bool closed = false;
  bool resume = false;
  while (true) {
    if (stops.load() > accepted_stops) {return StopChoice::emergency;}
    if (estop.load() > healthy_estop) {return StopChoice::estop;}
    pollfd input{closed ? -1 : fd, POLLIN, 0};
    const int ready = poll(&input, 1, 20);
    if (ready > 0 && (input.revents & (POLLERR | POLLNVAL))) {closed = true;}
    if (ready > 0 && (input.revents & (POLLIN | POLLHUP))) {
      char key = '\0';
      if (read(fd, &key, 1) <= 0) {closed = true; continue;}
      if (stops.load() > accepted_stops) {return StopChoice::emergency;}
      if (estop.load() > healthy_estop) {return StopChoice::estop;}
      if (key == '\n') {return resume ? StopChoice::resume : StopChoice::release;}
      if (key == 'r' || key == 'R') {resume = true;}
    }
  }
}

// A missing console is cancellation, never consent to an alignment move.
// The caller passes the stop count already acknowledged by an explicit resume;
// startup always passes zero, so an earlier interrupt stays pending.
inline bool wait_for_alignment_confirmation(
  int fd, const std::atomic<int> & stops, int accepted_stops,
  const std::atomic<int> & estop, int healthy_estop)
{
  while (stops.load() == accepted_stops && estop.load() <= healthy_estop) {
    pollfd input{fd, POLLIN, 0};
    const int ready = poll(&input, 1, 20);
    if (ready < 0) {continue;}
    if (input.revents & (POLLERR | POLLNVAL)) {return false;}
    if (ready > 0 && (input.revents & (POLLIN | POLLHUP))) {
      char key = '\0';
      if (read(fd, &key, 1) <= 0) {return false;}
      if (key == '\n') {
        return stops.load() == accepted_stops && estop.load() <= healthy_estop;
      }
    }
  }
  return false;
}
}  // namespace tatbot::teleop
