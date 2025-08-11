import evdev
from lerobot.teleoperators.gamepad import AtariTeleoperatorConfig

print("Default expected name:", repr(AtariTeleoperatorConfig().device_name))
for path in evdev.list_devices():
    d = evdev.InputDevice(path)
    print(f"{path} -> {repr(d.name)} (strip={repr(d.name.strip())})")
print("Exact match exists?",
      any(evdev.InputDevice(p).name.strip()==AtariTeleoperatorConfig().device_name
          for p in evdev.list_devices()))