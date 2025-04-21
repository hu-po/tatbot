# SoloAI

Setting up the Trossen Robotics SoloAI arm

https://docs.trossenrobotics.com/trossen_arm/main/getting_started/hardware_setup.html

arm driver takes in end effector properties

https://docs.trossenrobotics.com/trossen_arm/main/api/classtrossen__arm_1_1TrossenArmDriver.html

```cpp
void configure(Model model, EndEffectorProperties end_effector, const std::string serv_ip, bool clear_error)
model – Model of the robot
end_effector – End effector properties
serv_ip – IP address of the robot
clear_error – Whether to clear the error state of the robot
```

```cpp
void set_end_effector(const EndEffectorProperties &end_effector)
```

```cpp
void set_arm_modes(Mode mode = Mode::idle)
void set_gripper_mode(Mode mode = Mode::idle)

enum class trossen_arm::Mode : uint8_t
Operation modes of a joint.

Values:

enumerator idle
Arm joints are braked, the gripper joint closing with a safe force.

enumerator position
Control the joint to a desired position.

enumerator velocity
Control the joint to a desired velocity.

enumerator external_effort
Control the joint to a desired external effort.
```

# tatbot URDF creation

http://urdf.robotsfan.com/

https://marketplace.visualstudio.com/items?itemName=morningfrog.urdf-visualizer

the official urdf for the trossen arm is here:

https://github.com/TrossenRobotics/trossen_arm_description/tree/main

the urdf is just for the individual arm, and then there is an xacro file that has the conditional logic for the leader and follower arms, as well as loading it into rviz

## Initial Testing

following https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/teleoperation.py

```bash
ping 192.168.1.2
ping 192.168.1.3
cd /home/trossen-ai/libtrossen_arm/demos/python
conda env list
conda activate trossen_ai_data_collection_ui_env
python3 configure_cleanup.py
python3 teleoperation.py
```

arm controller boxes can be either `dhcp` mode or `manual` mode, manual mode is default

https://docs.trossenrobotics.com/trossen_arm/v1.7/getting_started/configuration.html#ip-method

## Arm Networking

with the trossen pc and both arm controllers on an isolated network switch, I can connect to the arms, which have a static IP address.

```bash
sudo ip addr add 192.168.1.100/24 dev enp86s0
sudo ip link set enp86s0 up
sudo ip route add 192.168.1.0/24 dev enp86s0
ping 192.168.1.3
ping 192.168.1.2
```

get the mac address of the two arm control boxes

```bash
ip neigh show dev enp86s0
> 192.168.1.2 04:e9:e5:18:f8:5d
> 192.168.1.3 04:e9:e5:18:f8:5c
```

Go to the home network page (router page on browser 192.168.1.1)

Advanced settings -> Setup -> LAN Setup -> Address Reservation

added resevations for `trossen-arm-follower` and `trossen-arm-leader`

testing arm

```bash
cd /home/trossen-ai/libtrossen_arm/demos/python
conda activate trossen_ai_data_collection_ui_env
```

leader arm works

```bash
python simple_move.py
```

now follower arm

```bash
vim simple_move.py
```

change ip to 192.168.1.2 and ee config to follower

```python
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_follower,
    "192.168.1.2",
    False
)
```

follower works

test teleop

```bash
vim teleoperation.py
```

run into bug

```bash
(trossen_ai_data_collection_ui_env) trossen-ai@trossen-ai-pc:~/libtrossen_arm/demos/python$ python teleoperation.py 
Initializing the drivers...
Configuring the drivers...
[INFO] [trossen_arm] Driver version: 'v1.7.4'
[INFO] [trossen_arm] Controller firmware version: 'v1.7.2'
[INFO] [trossen_arm] Driver version: 'v1.7.4'
[INFO] [trossen_arm] Controller firmware version: 'v1.7.2'
Moving to home positions...
setting continuity factor
continuity [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
Starting to teleoperate the robots...
Traceback (most recent call last):
  File "/home/trossen-ai/libtrossen_arm/demos/python/teleoperation.py", line 114, in <module>
    driver_follower.set_all_positions(
RuntimeError: [ERROR] [trossen_arm] [Driver] Error occurred: Discontinuous robot input received
Latest log on the arm controller: [ERROR] [Motor Interface] Joint 6 position discontinuity: 0.000000 -> 0.003285
```

## Data Collection UI

https://docs.trossenrobotics.com/trossen_arm/v1.3/tutorials/trossen_data_collection_ui.html

## Calibration of arms

putting arm in gravity compensation mode and tuning joints one by one

https://docs.trossenrobotics.com/trossen_arm/v1.3/getting_started/configuration.html#friction-transition-velocities-friction-constant-terms-friction-coulomb-coefs-and-friction-viscous-coefs

```bash
python ~/libtrossen_arm/demos/python/gravity_compensation.py
```

had to install pyyaml

```bash
pip install PyYAML
```

need to use the `set_gripper_force_limit_scaling_factor` function to set the gripper force limit scaling factor
https://docs.trossenrobotics.com/trossen_arm/v1.3/api/classtrossen__arm_1_1TrossenArmDriver.html#_CPPv4N11trossen_arm16TrossenArmDriver38set_gripper_force_limit_scaling_factorEf

it is a runtime config setting, so it will not persist, which means we need to set it every time we start the arm driver

lerobot/common/robot_devices/motors/trossen_arm_driver.py