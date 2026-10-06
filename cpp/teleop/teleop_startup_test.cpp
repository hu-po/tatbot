#include "teleop_startup.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
void check(bool value, const char * message)
{
  if (!value) {std::cerr << message << '\n'; std::exit(1);}
}
}

int main()
{
  using tatbot::teleop::wait_for_alignment_confirmation;
  for (int scenario = 0; scenario < 6; ++scenario) {
    int pipefd[2];
    check(pipe(pipefd) == 0, "pipe failed");
    std::atomic<int> stops{scenario == 0 ? 1 : 0};
    std::atomic<int> estop{0};
    const auto start = std::chrono::steady_clock::now();
    std::thread input([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      if (scenario == 1) {stops.fetch_add(1);}
      if (scenario == 2) {estop.store(1);}
      if (scenario == 4 || scenario == 5) {
        if (scenario == 5) {stops.store(2);}
        const char key = '\n';
        check(write(pipefd[1], &key, 1) == 1, "write failed");
      }
      if (scenario == 3) {close(pipefd[1]);}
    });
    const bool allowed = wait_for_alignment_confirmation(pipefd[0], stops, 0, estop, 0);
    input.join();
    check(allowed == (scenario == 4), "stop, E-stop or EOF approved motion");
    check(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(250),
      "startup cancellation was delayed");
    close(pipefd[0]);
    if (scenario != 3) {close(pipefd[1]);}
  }
  // Resume can acknowledge an old stop; a new one must still cancel.
  int pipefd[2];
  check(pipe(pipefd) == 0, "resume pipe failed");
  std::atomic<int> stops{1}, estop{0};
  const char key = '\n';
  check(write(pipefd[1], &key, 1) == 1, "resume write failed");
  check(wait_for_alignment_confirmation(pipefd[0], stops, 1, estop, 0),
    "an acknowledged stop prevented explicit resume");
  stops.store(2);
  check(!wait_for_alignment_confirmation(pipefd[0], stops, 1, estop, 0),
    "resume swallowed a new stop");
  close(pipefd[0]); close(pipefd[1]);
  using tatbot::teleop::StopChoice;
  using tatbot::teleop::wait_for_hold_choice;
  // A second interrupt already pending when the hold prompt opens is still
  // an emergency; EOF must not silently turn that prompt into a landing.
  check(pipe(pipefd) == 0, "hold pipe failed");
  check(wait_for_hold_choice(pipefd[0], stops, 1, estop, 0) == StopChoice::emergency,
    "hold swallowed an already-pending second interrupt");
  stops.store(1);
  close(pipefd[1]);
  std::thread release_hold([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stops.store(2);
  });
  check(wait_for_hold_choice(pipefd[0], stops, 1, estop, 0) == StopChoice::emergency,
    "closed console authorized release or landing");
  release_hold.join();
  close(pipefd[0]);
  for (bool resume_hold : {false, true}) {
    check(pipe(pipefd) == 0, "hold confirmation pipe failed");
    const char * command = resume_hold ? "r\n" : "\n";
    const size_t length = resume_hold ? 2 : 1;
    check(write(pipefd[1], command, length) == static_cast<ssize_t>(length),
      "hold confirmation write failed");
    check(wait_for_hold_choice(pipefd[0], stops, 2, estop, 0) ==
      (resume_hold ? StopChoice::resume : StopChoice::release), "explicit hold choice failed");
    estop.store(1);
    check(wait_for_hold_choice(pipefd[0], stops, 2, estop, 0) == StopChoice::estop,
      "hold ignored the E-stop");
    estop.store(0);
    close(pipefd[0]); close(pipefd[1]);
  }
  std::cout << "startup: pending stop, active stop, E-stop, EOF, Enter and resume PASS\n";
}
