cmake_minimum_required(VERSION 3.16)
project(tatbot_description NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_description test/test_description.py
    ENV TATBOT_REPO=${CMAKE_CURRENT_SOURCE_DIR}/../..)
  ament_add_pytest_test(test_names_parity test/test_names_parity.py)
endif()

ament_package()
