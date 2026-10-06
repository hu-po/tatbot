cmake_minimum_required(VERSION 3.16)
project(tatbot_bringup NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config systemd scripts DESTINATION share/${PROJECT_NAME} USE_SOURCE_PERMISSIONS)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(_env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tatbot_description:$ENV{PYTHONPATH}
    TATBOT_REPO=${CMAKE_CURRENT_SOURCE_DIR}/../..)
  ament_add_pytest_test(test_config test/test_config.py ENV ${_env})
  ament_add_pytest_test(test_stack_launch test/test_stack_launch.py ENV ${_env})
  ament_add_pytest_test(test_flight_summary test/test_flight_summary.py ENV ${_env})
  # The whole launch on mock hardware, in its own zenoh router on a free port (scripts/mock_check.py).
  ament_add_pytest_test(test_mock_stack test/test_mock_stack.py ENV ${_env} TATBOT_CHECK_HARDWARE=mock TIMEOUT 240)
  # The same checks on TatbotArm with its fake SDK: the driver's interlocks wired through the launch.
  ament_add_pytest_test(test_fake_stack test/test_mock_stack.py ENV ${_env} TATBOT_CHECK_HARDWARE=fake TIMEOUT 240)
endif()

ament_package()
