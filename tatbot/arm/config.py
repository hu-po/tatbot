import argparse
import os
import sys

import yaml

import trossen_arm

# --- Constants ---
SERVER_IP_LEADER = '192.168.1.3'
SERVER_IP_FOLLOWER = '192.168.1.2'
_CONFIG_FILE_LEADER_RAW = '~/dev/tatbot-dev/config/trossen/leader.yaml'
_CONFIG_FILE_FOLLOWER_RAW = '~/dev/tatbot-dev/config/trossen/follower.yaml'
CONFIG_FILE_LEADER = os.path.expanduser(_CONFIG_FILE_LEADER_RAW)
CONFIG_FILE_FOLLOWER = os.path.expanduser(_CONFIG_FILE_FOLLOWER_RAW)

# Custom key for the gripper force setting in the YAML file.
# NOTE: This setting is NON-PERSISTENT on the robot and resets on power cycle.
# It is applied at runtime when using the --push command.
GRIPPER_FORCE_KEY = 'gripper_force_limit_scaling_factor'

# --- Functions ---
def print_configurations(driver: trossen_arm.TrossenArmDriver, arm_name: str):
    """Prints the configuration values for a given arm driver."""
    print(f"\n--- {arm_name} Configurations ---")
    configs = {}
    try:
        # Persistent EEPROM settings
        configs['EEPROM factory reset flag'] = driver.get_factory_reset_flag()
        configs['EEPROM IP method'] = driver.get_ip_method()
        configs['EEPROM manual IP'] = driver.get_manual_ip()
        configs['EEPROM DNS'] = driver.get_dns()
        configs['EEPROM gateway'] = driver.get_gateway()
        configs['EEPROM subnet'] = driver.get_subnet()
        configs['EEPROM effort corrections'] = driver.get_effort_corrections()
        configs['EEPROM friction transition velocities'] = driver.get_friction_transition_velocities()
        configs['EEPROM friction constant terms'] = driver.get_friction_constant_terms()
        configs['EEPROM friction coulomb coefs'] = driver.get_friction_coulomb_coefs()
        configs['EEPROM friction viscous coefs'] = driver.get_friction_viscous_coefs()
        configs['EEPROM continuity factors'] = driver.get_continuity_factors()

        # Non-Persistent runtime setting
        try:
             # Display the CURRENT runtime value
             configs[GRIPPER_FORCE_KEY] = f"{driver.get_gripper_force_limit_scaling_factor():.2f} (Runtime)"
        except AttributeError:
             configs[GRIPPER_FORCE_KEY] = "Not supported"
        except Exception as e_grip:
             configs[GRIPPER_FORCE_KEY] = f"Error retrieving ({e_grip})"

        for key, value in configs.items():
             print(f"  {key}: {value}")

    except Exception as e:
        print(f"  Major error retrieving configurations for {arm_name}: {e}")
        print("  Ensure the arm is connected and powered on.")
        if configs:
             print("  Partial configs retrieved:")
             for key, value in configs.items():
                 print(f"    {key}: {value}")
    print("-" * (len(arm_name) + 24))


def read_gripper_force_from_yaml(filepath: str) -> float | None:
    """Reads the custom gripper force key from the YAML file."""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        if data and GRIPPER_FORCE_KEY in data:
            factor = float(data[GRIPPER_FORCE_KEY])
            if 0.0 <= factor <= 1.0:
                return factor
            else:
                print(f"  Warning: {GRIPPER_FORCE_KEY} ({factor}) in {filepath} out of range [0.0, 1.0].", file=sys.stderr)
                return None
    except FileNotFoundError:
        # This is okay if pushing for the first time, file might only have EEPROM
        pass # print(f"  Info: Config file {filepath} not found for reading gripper force.", file=sys.stderr)
    except (yaml.YAMLError, ValueError, TypeError) as e:
        print(f"  Error reading or parsing {GRIPPER_FORCE_KEY} from {filepath}: {e}", file=sys.stderr)
    return None


def add_gripper_force_to_yaml(filepath: str, factor: float):
    """Adds/Updates the custom gripper force key in the YAML file after trossen lib saved it."""
    try:
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        if config_data is not None:
            config_data[GRIPPER_FORCE_KEY] = factor
            with open(filepath, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=None, sort_keys=False)
            # Modify print slightly to reflect it stores the *current* runtime value
            print(f"  + Current runtime {GRIPPER_FORCE_KEY} ({factor:.2f}) saved to {filepath}.")
        else:
            print(f"  Warning: Could not reload {filepath} after saving to add {GRIPPER_FORCE_KEY}.", file=sys.stderr)

    except (yaml.YAMLError, IOError, TypeError) as e:
        print(f"  Error processing {filepath} to add {GRIPPER_FORCE_KEY}: {e}", file=sys.stderr)


# --- Main Execution ---
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Manage Trossen Arm configurations. Default: Save EEPROM config + current runtime gripper force to YAML. --push: Load EEPROM config from YAML and apply runtime gripper force from YAML.'
    )
    parser.add_argument(
        '--push',
        action='store_true',
        help=f'Push EEPROM config from YAML files AND apply {GRIPPER_FORCE_KEY} from YAML as runtime setting.'
    )
    args = parser.parse_args()

    print("Initializing drivers...")
    driver_leader = trossen_arm.TrossenArmDriver()
    driver_follower = trossen_arm.TrossenArmDriver()
    leader_configured = False
    follower_configured = False

    print("Configuring drivers (connecting to arms)...")
    # (Configuration logic remains the same)
    try:
        driver_leader.configure(
            trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_leader,
            SERVER_IP_LEADER, False
        )
        print(f"Leader arm driver configured for IP: {SERVER_IP_LEADER}")
        leader_configured = True
    except Exception as e:
        print(f"Error configuring leader arm driver ({SERVER_IP_LEADER}): {e}. Check connection and IP.", file=sys.stderr)

    try:
        driver_follower.configure(
            trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_follower,
            SERVER_IP_FOLLOWER, False
        )
        print(f"Follower arm driver configured for IP: {SERVER_IP_FOLLOWER}")
        follower_configured = True
    except Exception as e:
        print(f"Error configuring follower arm driver ({SERVER_IP_FOLLOWER}): {e}. Check connection and IP.", file=sys.stderr)

    if not leader_configured and not follower_configured:
        print("\nError: Failed to configure any arm drivers. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Mode Logic ---
    if args.push:
        # --- PUSH MODE ---
        print("\n*** PUSH MODE ACTIVATED ***")
        print(f"Applying EEPROM config and runtime {GRIPPER_FORCE_KEY} from YAML files.")
        config_pushed = False # Tracks if EEPROM config was pushed

        # Leader Arm Push
        if leader_configured:
            if os.path.exists(CONFIG_FILE_LEADER):
                try:
                    # 1. Load EEPROM settings (library ignores GRIPPER_FORCE_KEY)
                    print(f"\nLoading main config for Leader arm from {CONFIG_FILE_LEADER}...")
                    driver_leader.load_configs_from_file(CONFIG_FILE_LEADER)
                    print("Leader arm main configuration loaded.")
                    config_pushed = True

                    # 2. Read GRIPPER_FORCE_KEY from YAML and apply as runtime setting
                    gripper_force_to_set = read_gripper_force_from_yaml(CONFIG_FILE_LEADER)
                    if gripper_force_to_set is not None:
                        try:
                             print(f"  Applying runtime {GRIPPER_FORCE_KEY} ({gripper_force_to_set:.2f}) for Leader arm...")
                             driver_leader.set_gripper_force_limit_scaling_factor(gripper_force_to_set)
                             print(f"  Leader arm runtime {GRIPPER_FORCE_KEY} applied.")
                        except Exception as e_set_grip:
                             print(f"  Error applying runtime {GRIPPER_FORCE_KEY} for Leader arm: {e_set_grip}", file=sys.stderr)

                    # 3. Print final state (shows EEPROM values + current runtime gripper force)
                    print_configurations(driver_leader, "Leader Arm (After Push)")

                except Exception as e:
                    print(f"Error during push for Leader arm: {e}", file=sys.stderr)
            else:
                print(f"\nWarning: Config file {CONFIG_FILE_LEADER} not found. Skipping push for Leader arm.", file=sys.stderr)
        else:
             print("\nSkipping push for Leader arm (driver not configured).")

        # Follower Arm Push (Repeat logic)
        if follower_configured:
            if os.path.exists(CONFIG_FILE_FOLLOWER):
                try:
                    print(f"\nLoading main config for Follower arm from {CONFIG_FILE_FOLLOWER}...")
                    driver_follower.load_configs_from_file(CONFIG_FILE_FOLLOWER)
                    print("Follower arm main configuration loaded.")
                    config_pushed = True

                    gripper_force_to_set = read_gripper_force_from_yaml(CONFIG_FILE_FOLLOWER)
                    if gripper_force_to_set is not None:
                        try:
                            print(f"  Applying runtime {GRIPPER_FORCE_KEY} ({gripper_force_to_set:.2f}) for Follower arm...")
                            driver_follower.set_gripper_force_limit_scaling_factor(gripper_force_to_set)
                            print(f"  Follower arm runtime {GRIPPER_FORCE_KEY} applied.")
                        except Exception as e_set_grip:
                            print(f"  Error applying runtime {GRIPPER_FORCE_KEY} for Follower arm: {e_set_grip}", file=sys.stderr)

                    print_configurations(driver_follower, "Follower Arm (After Push)")

                except Exception as e:
                     print(f"Error during push for Follower arm: {e}", file=sys.stderr)
            else:
                print(f"\nWarning: Config file {CONFIG_FILE_FOLLOWER} not found. Skipping push for Follower arm.", file=sys.stderr)
        else:
            print("\nSkipping push for Follower arm (driver not configured).")

        # Power cycle reminder is still relevant for EEPROM IP changes etc.
        if config_pushed:
             print("\nIMPORTANT: Power cycle the robot(s) if persistent EEPROM settings like IP method were changed.")

    else:
        # --- SAVE MODE (Default) ---
        print("\n*** SAVE CURRENT MODE (Default) ***")
        print(f"Saving EEPROM config and current runtime {GRIPPER_FORCE_KEY} to YAML files...")

        # Leader Arm Save
        if leader_configured:
            leader_dir = os.path.dirname(CONFIG_FILE_LEADER)
            os.makedirs(leader_dir, exist_ok=True)
            current_gripper_force = None
            try:
                # 1. Get current RUNTIME gripper force
                try:
                    current_gripper_force = driver_leader.get_gripper_force_limit_scaling_factor()
                except Exception as e_get_grip:
                     print(f"  Warning: Could not get current runtime {GRIPPER_FORCE_KEY} for Leader: {e_get_grip}", file=sys.stderr)

                # 2. Save EEPROM config (overwrites file)
                print(f"\nSaving Leader arm EEPROM configuration to {CONFIG_FILE_LEADER}...")
                driver_leader.save_configs_to_file(CONFIG_FILE_LEADER)
                print(f"Leader arm EEPROM configuration saved.")

                # 3. Add current RUNTIME gripper force back to the file
                if current_gripper_force is not None:
                    add_gripper_force_to_yaml(CONFIG_FILE_LEADER, current_gripper_force)

                # 4. Print the overall state (EEPROM values + current runtime gripper force)
                print_configurations(driver_leader, "Leader Arm (Saved State)")

            except Exception as e:
                print(f"Error during save for Leader arm: {e}", file=sys.stderr)
        else:
            print("\nSkipping save for Leader arm (driver not configured).")

        # Follower Arm Save (Repeat logic)
        if follower_configured:
            follower_dir = os.path.dirname(CONFIG_FILE_FOLLOWER)
            os.makedirs(follower_dir, exist_ok=True)
            current_gripper_force = None
            try:
                try:
                    current_gripper_force = driver_follower.get_gripper_force_limit_scaling_factor()
                except Exception as e_get_grip:
                     print(f"  Warning: Could not get current runtime {GRIPPER_FORCE_KEY} for Follower: {e_get_grip}", file=sys.stderr)

                print(f"\nSaving Follower arm EEPROM configuration to {CONFIG_FILE_FOLLOWER}...")
                driver_follower.save_configs_to_file(CONFIG_FILE_FOLLOWER)
                print(f"Follower arm EEPROM configuration saved.")

                if current_gripper_force is not None:
                    add_gripper_force_to_yaml(CONFIG_FILE_FOLLOWER, current_gripper_force)

                print_configurations(driver_follower, "Follower Arm (Saved State)")

            except Exception as e:
                print(f"Error during save for Follower arm: {e}", file=sys.stderr)
        else:
            print("\nSkipping save for Follower arm (driver not configured).")

    print("\nScript finished.")