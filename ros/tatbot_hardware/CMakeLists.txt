cmake_minimum_required(VERSION 3.16)
project(tatbot_hardware CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

# The Trossen SDK (headers + prebuilt static lib) by FetchContent, as cpp/teleop does, cached under
# FETCHCONTENT_BASE_DIR (~/tatbot-ros/deps on the ros node). Its major.minor must match the
# controllers' firmware (v1.11.1 since 2026-10-07). A cache checked out at the tag is used as is, so
# incremental and offline rebuilds never touch the network; any other checkout is re-fetched. With no cache and no
# network, or with TATBOT_ARM_SDK OFF, only sdk=fake is built.
set(TATBOT_ARM_SDK_TAG v1.11.0)
option(TATBOT_ARM_SDK "Build the real Trossen SDK backend (fetches trossen_arm ${TATBOT_ARM_SDK_TAG})" ON)
set(TROSSEN_SDK_AVAILABLE OFF)
if(TATBOT_ARM_SDK)
  include(FetchContent)
  set(_deps "${FETCHCONTENT_BASE_DIR}")
  if(NOT _deps)
    set(_deps "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(_cached "${_deps}/trossen_arm_sdk-src")
  set(_cached_tag "")
  if(EXISTS "${_cached}/include/libtrossen_arm/trossen_arm.hpp")
    execute_process(COMMAND git -C "${_cached}" describe --tags --exact-match
      OUTPUT_VARIABLE _cached_tag OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT _cached_tag STREQUAL TATBOT_ARM_SDK_TAG)
      message(WARNING "cached trossen_arm SDK at ${_cached} is '${_cached_tag}', not ${TATBOT_ARM_SDK_TAG}: re-fetching")
    endif()
  endif()
  if(_cached_tag STREQUAL TATBOT_ARM_SDK_TAG)
    set(FETCHCONTENT_SOURCE_DIR_TROSSEN_ARM_SDK "${_cached}")
    set(TROSSEN_SDK_AVAILABLE ON)
  else()
    execute_process(
      COMMAND git ls-remote -q https://github.com/TrossenRobotics/trossen_arm.git refs/tags/${TATBOT_ARM_SDK_TAG}
      RESULT_VARIABLE _probe OUTPUT_QUIET ERROR_QUIET TIMEOUT 20)
    if(_probe EQUAL 0)
      set(TROSSEN_SDK_AVAILABLE ON)
    endif()
  endif()
  if(TROSSEN_SDK_AVAILABLE)
    FetchContent_Declare(trossen_arm_sdk
      GIT_REPOSITORY https://github.com/TrossenRobotics/trossen_arm.git
      GIT_TAG ${TATBOT_ARM_SDK_TAG}
      SOURCE_SUBDIR include)   # its own CMakeLists assumes it is top level; headers + lib only
    FetchContent_MakeAvailable(trossen_arm_sdk)
  else()
    message(WARNING "trossen_arm SDK unavailable (no cache, no network): sdk=real is not built")
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/core.cpp src/estop.cpp src/runtime.cpp src/fake_backend.cpp src/sdk_backend.cpp
  src/tatbot_arm.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(${PROJECT_NAME} PUBLIC
  hardware_interface::hardware_interface kdl_parser::kdl_parser orocos-kdl pluginlib::pluginlib
  rclcpp::rclcpp rclcpp_lifecycle::rclcpp_lifecycle pthread)
if(TROSSEN_SDK_AVAILABLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(_arch x86_64)
  else()
    set(_arch aarch64)
  endif()
  target_include_directories(${PROJECT_NAME} PRIVATE "${trossen_arm_sdk_SOURCE_DIR}/include")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    "${trossen_arm_sdk_SOURCE_DIR}/lib/linux/${_arch}/libtrossen_arm.a")
  target_compile_definitions(${PROJECT_NAME} PRIVATE TATBOT_HAVE_SDK)
  message(STATUS "tatbot_hardware: sdk=real built against trossen_arm ${TATBOT_ARM_SDK_TAG} (${_arch})")
endif()
# sdk=sim, the simulator's lockstep backend, is private: a tree without src/sim_backend.cpp builds the stub in
# tatbot_arm.cpp and refuses sdk=sim at configure.
set(TATBOT_SIM_BACKEND OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_backend.cpp")
  find_package(rosgraph_msgs REQUIRED)
  target_sources(${PROJECT_NAME} PRIVATE src/sim_backend.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${rosgraph_msgs_TARGETS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE TATBOT_HAVE_SIM)
  set(TATBOT_SIM_BACKEND ON)
endif()
pluginlib_export_plugin_description_file(hardware_interface tatbot_hardware.xml)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_names test/test_names.cpp)
  target_include_directories(test_names PRIVATE include)
  foreach(t test_hw_core test_hw_estop test_hw_plugin)
    ament_add_gtest(${t} test/${t}.cpp TIMEOUT 120)
    target_link_libraries(${t} ${PROJECT_NAME} util)
  endforeach()
  if(TROSSEN_SDK_AVAILABLE)   # the session's end effector, gains and carriage offset, against the real SDK
    ament_add_gtest(test_sdk_options test/test_sdk_options.cpp TIMEOUT 60)
    target_link_libraries(test_sdk_options ${PROJECT_NAME})
    target_include_directories(test_sdk_options PRIVATE "${trossen_arm_sdk_SOURCE_DIR}/include")
    target_compile_definitions(test_sdk_options PRIVATE TATBOT_HAVE_SDK)
  endif()
  if(TATBOT_SIM_BACKEND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_hw_sim.cpp")
    ament_add_gtest(test_hw_sim test/test_hw_sim.cpp TIMEOUT 120 SKIP_LINKING_MAIN_LIBRARIES)   # main: its router
    target_link_libraries(test_hw_sim ${PROJECT_NAME} ${rosgraph_msgs_TARGETS} gtest)
    target_compile_definitions(test_hw_sim PRIVATE
      SIM_WIRE_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/test/sim_wire_golden.bin")
  endif()
  ament_add_gtest(test_hw_relay test/test_hw_relay.cpp TIMEOUT 120)
  target_link_libraries(test_hw_relay ${PROJECT_NAME} util)
  target_compile_definitions(test_hw_relay PRIVATE
    RELAY_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/../tatbot_estop_relay/tatbot_estop_relay.py")
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(hardware_interface kdl_parser orocos_kdl_vendor pluginlib rclcpp rclcpp_lifecycle)
ament_package()
