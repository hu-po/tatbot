# Operator-facing events, published by tatbot_session on /tatbot/events.
string KIND_RUN_START="run_start"
string KIND_RUN_END="run_end"
string KIND_OP_SENT="op_sent"
string KIND_OP_DONE="op_done"
string KIND_OP_ABORTED="op_aborted"
string KIND_REPLAN="replan"
string KIND_PAUSE="pause"              # a pause op: text is its reason; waiting for Decide CONTINUE
string KIND_LATCHED="latched"          # text names the latch reason
string KIND_RELEASED="released"        # e-stop released while latched; waiting for Decide CONTINUE or LAND
string KIND_UNCERTAIN="uncertain"      # an op left `sent`; waiting for Decide REDRAW or SKIP
string KIND_DECISION="decision"
string KIND_TOUCH="touch"
string KIND_PAGE="page"
string KIND_LANDED="landed"
string KIND_MACHINE="machine"            # the tattoo machine switched on or off (ros/README.md 4.4)
string KIND_PEN_TRIM="pen_trim"          # the operator's pen trim for the cartridge on the arm (ros/README.md 4.4)
string KIND_ERROR="error"

builtin_interfaces/Time stamp
string arm        # "" when not arm-specific
string kind       # KIND_*
string op_id      # "" when not op-specific
string run_id
string text
