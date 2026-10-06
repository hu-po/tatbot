# One arm's driver safety state, republished by tatbot_session on /tatbot/safety (one message per arm,
# 20 Hz and on change) from the <arm>_safety GPIO state interfaces (ros/README.md "Names").

uint8 ESTOP_SOURCE_NONE=0
uint8 ESTOP_SOURCE_SERIAL=1
uint8 ESTOP_SOURCE_UDP=2

uint8 LATCH_NONE=0
uint8 LATCH_ESTOP=1             # a valid frame reports the button pressed
uint8 LATCH_ESTOP_STALE=2       # no valid frame within the source's limit (serial 0.10 s, udp 0.15 s)
uint8 LATCH_CARRIAGE_CONTACT=3  # carriage contact over its cap; retracted, then held
uint8 LATCH_STALL=4             # tracking stall
uint8 LATCH_OVER_VELOCITY=5     # measured over-velocity
uint8 LATCH_GUARD_TIP_LAG=6     # tip-lag guard trip (a stop, not a fault)
uint8 LATCH_GUARD_PROBE=7       # probe guard trip (M2)
uint8 LATCH_CONTROLLER_ERROR=8  # the arm controller reported an error
uint8 LATCH_DEACTIVATED=9       # hardware component deactivated: hold, never idle
uint8 LATCH_STEP_REFUSED=10     # a command more than the step limit from the held pose

uint8 GUARD_NONE=0
uint8 GUARD_TIP_LAG=1
uint8 GUARD_PROBE=2

builtin_interfaces/Time stamp
string arm
uint8 estop_source          # ESTOP_SOURCE_*
bool estop_ok               # released and fresh (always true for ESTOP_SOURCE_NONE)
float64 estop_age_s         # age of the last valid frame, s; -1 for none or before the first frame
bool probe_triggered        # M2; false until then
bool latched                # the driver holds the latched pose and ignores commands
uint8 latch_reason          # LATCH_* (the first reason since the last unlatch)
bool guard_tripped
uint8 guard_mode            # GUARD_* currently armed
float64[] trip_joints       # measured joints at the guard trip: 6 revolute (rad) then carriage (m); empty if none
bool landing                # the landing sequence is running
bool landed                 # verified landed and idle
bool controller_error
bool machine_off            # passive tool, or fresh relay reply confirming commanded power off
float64 rt_period_max_ms    # longest read-to-read period over the last second
