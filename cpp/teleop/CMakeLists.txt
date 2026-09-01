cmake_minimum_required(VERSION 3.15)
project(wxai_teleop CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Official vendor SDK, pinned to the same version as the repo's python dep
# (pyproject.toml: trossen-arm==1.8.5). Public (BSD-3-Clause), ships a
# prebuilt static library, fetched from Trossen's GitHub. The SDK stays in
# the default build (decision 2026-08-31) — but when the fetch is impossible
# (no network, cold cache), the hardware-independent libraries and their
# tests still configure, build, and pass (plan Phase 2 exit gate). The
# SDK-linked targets are simply skipped, loudly.
include(FetchContent)
FetchContent_Declare(
  trossen_arm_sdk
  GIT_REPOSITORY https://github.com/TrossenRobotics/trossen_arm.git
  GIT_TAG v1.8.5
  # The SDK's CMakeLists assumes it is the top-level project (its CPack config
  # breaks under add_subdirectory). We only need its headers and prebuilt lib,
  # so point SOURCE_SUBDIR at a dir without a CMakeLists to skip it.
  SOURCE_SUBDIR include
)
set(TROSSEN_SDK_AVAILABLE ON)
set(_trossen_cached "${FETCHCONTENT_BASE_DIR}")
if(NOT _trossen_cached)
  set(_trossen_cached "${CMAKE_BINARY_DIR}/_deps")
endif()
if(NOT EXISTS "${_trossen_cached}/trossen_arm_sdk-src/include")
  # Cold cache: probe the remote before FetchContent, whose failure is fatal.
  execute_process(
    COMMAND git ls-remote -q https://github.com/TrossenRobotics/trossen_arm.git HEAD
    RESULT_VARIABLE _trossen_probe OUTPUT_QUIET ERROR_QUIET TIMEOUT 20)
  if(NOT _trossen_probe EQUAL 0)
    set(TROSSEN_SDK_AVAILABLE OFF)
  endif()
endif()
if(TROSSEN_SDK_AVAILABLE)
  FetchContent_MakeAvailable(trossen_arm_sdk)
endif()
if(NOT TROSSEN_SDK_AVAILABLE)
  message(WARNING
    "trossen_arm SDK unavailable (no network and no cached _deps): building "
    "only the hardware-independent targets (estop_monitor, telemetry_udp, "
    "realtime and their tests). wxai_teleop, friction_tune and arm_set_ip "
    "are skipped.")
else()
  # The SDK's own imported target is directory-scoped, so define our own
  # against its prebuilt lib for this platform.
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    set(TROSSEN_ARCH "x86_64")
  else()
    set(TROSSEN_ARCH "aarch64")
  endif()
  add_library(trossen::arm STATIC IMPORTED)
  set_target_properties(trossen::arm PROPERTIES
    IMPORTED_LOCATION "${trossen_arm_sdk_SOURCE_DIR}/lib/linux/${TROSSEN_ARCH}/libtrossen_arm.a"
    INTERFACE_INCLUDE_DIRECTORIES "${trossen_arm_sdk_SOURCE_DIR}/include"
  )
endif()

add_compile_options(-Wall -Wextra)

add_library(estop_monitor STATIC estop_monitor.cpp)
target_link_libraries(estop_monitor PUBLIC pthread)
target_include_directories(estop_monitor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(telemetry_udp STATIC telemetry_udp.cpp)
target_include_directories(telemetry_udp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(realtime STATIC realtime.cpp)
target_include_directories(realtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(TROSSEN_SDK_AVAILABLE)
  add_executable(wxai_teleop wxai_teleop.cpp)
  target_link_libraries(wxai_teleop PRIVATE trossen::arm estop_monitor telemetry_udp realtime pthread)

  add_executable(friction_tune friction_tune.cpp)
  target_link_libraries(friction_tune PRIVATE trossen::arm estop_monitor pthread)

  add_executable(arm_set_ip arm_set_ip.cpp)
  target_link_libraries(arm_set_ip PRIVATE trossen::arm pthread)
endif()

enable_testing()
add_executable(estop_monitor_test estop_monitor_test.cpp)
target_link_libraries(estop_monitor_test PRIVATE estop_monitor pthread)
add_test(NAME estop_monitor_test COMMAND estop_monitor_test)
add_executable(telemetry_udp_test telemetry_udp_test.cpp)
target_link_libraries(telemetry_udp_test PRIVATE telemetry_udp)
add_test(NAME telemetry_udp_test COMMAND telemetry_udp_test)
add_executable(realtime_test realtime_test.cpp)
target_link_libraries(realtime_test PRIVATE realtime)
add_test(NAME realtime_test COMMAND realtime_test)
