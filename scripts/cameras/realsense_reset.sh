#!/bin/bash

# === Configuration ===
TARGET_VID="8086"  # Vendor ID for Intel RealSense
TARGET_PID="0b5b"  # Product ID for your RealSense D405 cameras
RESET_DELAY=2      # Seconds to wait between unbind and bind
# === End Configuration ===

# Check for root privileges
if [[ $EUID -ne 0 ]]; then
   echo "ERROR: This script must be run as root (use sudo)."
   exit 1
fi

echo "Starting USB reset script for VID=${TARGET_VID}, PID=${TARGET_PID}..."
echo "Looking for devices in /sys/bus/usb/devices/..."
echo "DEBUG: Target VID='${TARGET_VID}', Target PID='${TARGET_PID}'" # Debug: Show target values

# Check if core sysfs paths exist and are writable by root
usb_drivers_dir="/sys/bus/usb/drivers/usb"
if [[ ! -d "$usb_drivers_dir" ]]; then
    echo "ERROR: USB drivers directory not found at $usb_drivers_dir." >&2
    echo "Ensure USB kernel modules are loaded." >&2
    exit 1
fi
unbind_file="$usb_drivers_dir/unbind"
bind_file="$usb_drivers_dir/bind"
if [[ ! -w "$unbind_file" ]] || [[ ! -w "$bind_file" ]]; then
    echo "ERROR: Cannot write to $unbind_file or $bind_file." >&2
    echo "This script requires root privileges (which seem active, but check file permissions)." >&2
    exit 1
fi

found_devices=0
overall_success=true

# Iterate through all directories in the USB devices path
echo "DEBUG: Iterating through /sys/bus/usb/devices/* ..." # Debug: Mark loop start
for device_path in /sys/bus/usb/devices/*; do
    # Check if it's a directory and contains the necessary ID files
    if [[ -d "$device_path" ]] && [[ -f "$device_path/idVendor" ]] && [[ -f "$device_path/idProduct" ]]; then

        # ===>>> CHANGE: Use 'cat' command explicitly <<<===
        vid_raw=$(cat "$device_path/idVendor" 2>/dev/null)
        pid_raw=$(cat "$device_path/idProduct" 2>/dev/null)

        # DEBUG: Print the raw values read from the files
        echo "DEBUG: Read from '$device_path': VID_raw='${vid_raw}', PID_raw='${pid_raw}'"

        # Perform comparison (using lowercase conversion for robustness)
        # Check if variables are not empty before proceeding
        if [[ -n "$vid_raw" ]] && [[ -n "$pid_raw" ]]; then
            vid_lc="${vid_raw,,}" # Requires Bash 4.0+
            pid_lc="${pid_raw,,}" # Requires Bash 4.0+
            target_vid_lc="${TARGET_VID,,}"
            target_pid_lc="${TARGET_PID,,}"

            if [[ "$vid_lc" == "$target_vid_lc" ]] && [[ "$pid_lc" == "$target_pid_lc" ]]; then
                device_name=$(basename "$device_path") # Get the device identifier (e.g., "3-2", "4-1.1")
                echo -e "\nFound matching device: $device_name ($device_path)"
                found_devices=$((found_devices + 1))
                device_success=true

                # --- Unbind Device ---
                echo "  Attempting to unbind '$device_name'..."
                if echo "$device_name" > "$unbind_file"; then
                    echo "  Successfully unbound '$device_name'."
                    echo "  Waiting ${RESET_DELAY}s..."
                    sleep "$RESET_DELAY"

                    # --- Bind Device ---
                    echo "  Attempting to bind '$device_name'..."
                    if echo "$device_name" > "$bind_file"; then
                        echo "  Successfully bound '$device_name'."
                        sleep 1 # Short delay after bind
                    else
                        echo "  ERROR: Failed to bind '$device_name'. Check 'dmesg' for kernel messages." >&2
                        device_success=false; overall_success=false
                    fi
                else
                    echo "  WARN: Failed to unbind '$device_name' (maybe already unbound?). Trying bind anyway..." >&2
                    echo "  Attempting to bind '$device_name'..."
                    if echo "$device_name" > "$bind_file"; then
                        echo "  Successfully bound '$device_name'."
                        sleep 1
                    else
                        echo "  ERROR: Failed to bind '$device_name' after unbind failed/warned. Check 'dmesg'." >&2
                        device_success=false; overall_success=false
                    fi
                fi

                if [[ "$device_success" == "true" ]]; then echo "  Reset cycle for '$device_name' completed."; else echo "  Reset cycle for '$device_name' failed." >&2; fi
            fi # end VID/PID match
        fi # end check for non-empty reads
    fi # end check for valid device dir
done # end loop through devices
echo "DEBUG: Finished iteration." # Debug: Mark loop end

echo -e "\nScript finished."
if [[ "$found_devices" -eq 0 ]]; then
    echo "No active devices found matching VID=${TARGET_VID}, PID=${TARGET_PID}."
else
    echo "Processed $found_devices matching device(s)."
    if [[ "$overall_success" == "true" ]]; then
        echo "All operations appeared successful (check 'dmesg' for related kernel messages)."
    else
        echo "One or more reset operations failed. Please review output and 'dmesg'." >&2
    fi
fi

exit 0