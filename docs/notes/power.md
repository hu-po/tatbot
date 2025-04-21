# Theoretical Power

| Device        | Quantity | Minimum Power (W) | Maximum Power (W) |
|---------------|----------|-------------------|-------------------|
| ojo           | 1        | 15                | 60                |
| rpi           | 2        | 3                 | 10                |
| switch-poe    | 1        | 5                 | 10                |
| switch-main   | 1        | 5                 | 10                |
| cameras       | 8        | 5                 | 10                |
| realsense     | 2        | 0.035             | 1.55              |
| WidowXAI V0   | 2        | 20                | 360               |

```python
total_min = 15 + 2*3 + 5 + 5 + 8*5 + 2*0.035 + 2*20
total_max = 60 + 2*10 + 10 + 10 + 8*10 + 2*1.55 + 2*360
print(f"min: {total_min}W max: {total_max}W")
```

the total power consumption when idle (min) and running (max):

> min: 112W max: 905W

everything is plugged into a power strip (1875W max) in the back of the robot
[powerstrip](notes/powerstrip.md)

everything can be put behind battery (1800W max) to prevent power outages
[battery](notes/battery.md)

# Power Logs

idle:
- 24W: switch-poe 2xrpi, 4x cameras
- 24W: both switches, 2xrpi
- 39W: both switches, ojo, 2xrpi
- 84W: both switches, ojo, 2xrpi, 2xarms, 1xtrossen-ai, screen

active:
- 116W: recording teleop dataset, screen off, ojo and rpis idle