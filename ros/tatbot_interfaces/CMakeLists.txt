cmake_minimum_required(VERSION 3.16)
project(tatbot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)

# The whole custom surface of the stack: at most 8 types (ros/README.md section 1).
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/Draw.action"
  "action/Touch.action"
  "action/Land.action"
  "srv/Decide.srv"
  "msg/Event.msg"
  "msg/SafetyState.msg"
  "msg/Page.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
