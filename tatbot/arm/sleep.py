import numpy as np
import trossen_arm

if __name__=='__main__':
    server_ip_leader = '192.168.1.3'
    server_ip_follower = '192.168.1.2'

    print("Initializing the drivers...")
    driver_leader = trossen_arm.TrossenArmDriver()
    driver_follower = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver_leader.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        server_ip_leader,
        False
    )
    driver_follower.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_follower,
        server_ip_follower,
        False
    )

    print("Moving to sleep positions...")
    driver_leader.set_all_modes(trossen_arm.Mode.position)
    driver_leader.set_all_positions(
        np.zeros(driver_leader.get_num_joints()),
        2.0,
        True
    )
    driver_follower.set_all_modes(trossen_arm.Mode.position)
    driver_follower.set_all_positions(
        np.zeros(driver_follower.get_num_joints()),
        2.0,
        True
    )

    print("Set arm to idle")
    driver_leader.set_all_modes(trossen_arm.Mode.idle)
    driver_follower.set_all_modes(trossen_arm.Mode.idle)