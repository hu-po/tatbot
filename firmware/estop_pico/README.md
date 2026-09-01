# E-stop heartbeat firmware

This directory contains the small firmware endpoint used by motion consumers.
It emits a versioned heartbeat and treats an open or unreadable input as a
stop. The protocol and timeout behavior are tested by the host-side consumers.

## Protocol

```text
EST1 <sequence> <state>\n
```

`state=1` means released/healthy; `state=0` means stopped. Consumers must
debounce frames, time out a silent stream, and fail closed on malformed data.
Keep the protocol constants synchronized with the C++ and Python readers.

## Bench workflow

Flash the board using the official CircuitPython instructions, then run the
host parser tests with no arm attached. Wiring, device paths, and powered
acceptance evidence are deployment-specific and are not documented here.

The e-stop is a motion stop, not a promise that motor power is removed. See
the public [safety contract](../../docs/estop.md).
