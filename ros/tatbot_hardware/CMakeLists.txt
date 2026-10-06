cmake_minimum_required(VERSION 3.16)
project(tatbot_hardware CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

# The Trossen SDK v1.8.5 (headers + prebuilt static lib) by FetchContent, as cpp/teleop does,
# cached under FETCHCONTENT_BASE_DIR (~/tatbot-ros/deps on the ros node). A cache checked out at
# v1.8.5 is used as is, so incremental and offline rebuilds never touch the network; any other
# checkout is re-fetched. With no cache and no
# network, or with TATBOT_ARM_SDK OFF, only sdk=fake is built.
option(TATBOT_ARM_SDK "Build the real Trossen SDK backend (fetches trossen_arm v1.8.5)" ON)
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
    if(NOT _cached_tag STREQUAL "v1.8.5")
      message(WARNING "cached trossen_arm SDK at ${_cached} is '${_cached_tag}', not v1.8.5: re-fetching")
    endif()
  endif()
  if(_cached_tag STREQUAL "v1.8.5")
    set(FETCHCONTENT_SOURCE_DIR_TROSSEN_ARM_SDK "${_cached}")
    set(TROSSEN_SDK_AVAILABLE ON)
  else()
    execute_process(
      COMMAND git ls-remote -q https://github.com/TrossenRobotics/trossen_arm.git refs/tags/v1.8.5
      RESULT_VARIABLE _probe OUTPUT_QUIET ERROR_QUIET TIMEOUT 20)
    if(_probe EQUAL 0)
      set(TROSSEN_SDK_AVAILABLE ON)
    endif()
  endif()
  if(TROSSEN_SDK_AVAILABLE)
    FetchContent_Declare(trossen_arm_sdk
      GIT_REPOSITORY https://github.com/TrossenRobotics/trossen_arm.git
      GIT_TAG v1.8.5
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
  message(STATUS "tatbot_hardware: sdk=real built against trossen_arm v1.8.5 (${_arch})")
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
  ament_add_gtest(test_hw_relay test/test_hw_relay.cpp TIMEOUT 120)
  target_link_libraries(test_hw_relay ${PROJECT_NAME} util)
  target_compile_definitions(test_hw_relay PRIVATE
    RELAY_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/../tatbot_estop_relay/tatbot_estop_relay.py")
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(hardware_interface kdl_parser orocos_kdl_vendor pluginlib rclcpp rclcpp_lifecycle)
ament_package()
