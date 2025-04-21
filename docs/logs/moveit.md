# MoveIt

following the MoveIt setup assistant

https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html#getting-started

presentation on using docker for moveit

https://picknik.ai/ros/robotics/docker/2021/07/20/Vatan-Aksoy-Tezer-Docker.html

creating a dockerfile which will be used to run the setup assistant in oop

https://moveit.picknik.ai/main/doc/how_to_guides/how_to_setup_docker_containers_in_ubuntu.html

moveit containers for ros2 humble can be found here:

https://hub.docker.com/r/moveit/moveit2/tags

```bash
cd ~/dev/tatbot-dev
./scripts/oop/launch.oop-moveit-setup.sh
```

now we can use moveit on the rpi, though maybe not since the rpi is arm based
it would probably be easier to do it on the meerkat given it is x86 based
that would also allow us to plug in the realsense camera and use it within ROS
so basically ojo is the model stuff, meerkat is the ROS master (moveit and cameras), and rpis are for DNS or something else
hard to think about what the rpis will actually be usefull for, the combination of arm and tiny is not very useful
maybe rpis have ROS nodes that run the things like MCP server, SQLite db, AprilTag tracking

try to run the moveit release container on meerkat

```bash
docker run -it --rm moveit/moveit2:humble-release bash
```

interesting that they also have a script entrypoint in the root of the docker image

```bash
root@57ec33cc0dcf:/# ls
bin  boot  dev  etc  home  lib  lib32  lib64  libx32  log  media  mnt  opt  proc  root  ros_entrypoint.sh  run  sbin  srv  sys  tmp  usr  var
root@57ec33cc0dcf:/# cat ros_entrypoint.sh 
#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
exec "$@"
root@57ec33cc0dcf:/# source "/opt/ros/$ROS_DISTRO/setup.bash"
```

https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/motion_planning_python_api/scripts/motion_planning_python_api_tutorial.py