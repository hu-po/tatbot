"""A hardware-free stand-in for the Trossen arm driver.

The mock backend of plan Phase 2: same getter/setter surface as
``trossen_arm.TrossenArmDriver`` for everything the tuning cockpit, golden
loader, and recovery paths touch, holding state in plain Python. It exists so
teleop/recording/tuning/tool code paths can run and be tested from a fresh
clone with no vendor hardware — it is NOT a simulator and never satisfies the
hardware-profile gate (use the 'example' profile with it).

Values are neutral vendor-shaped placeholders, not Tatbot measurements.
"""

from __future__ import annotations

import trossen_arm

N_JOINTS = 7


class _PID:
    def __init__(self, kp: float):
        self.kp, self.ki, self.kd, self.imax = float(kp), 0.0, 0.0, 0.0


class _Motor:
    def __init__(self, pos_kp: float, vel_kp: float):
        self.position = _PID(pos_kp)
        self.velocity = _PID(vel_kp)


class _Limit:
    def __init__(self):
        self.position_min, self.position_max = -3.14, 3.14
        self.position_tolerance = 0.2
        self.velocity_max, self.velocity_tolerance = 6.28, 0.0
        self.effort_max, self.effort_tolerance = 27.0, 5.4


class _Characteristic:
    def __init__(self):
        self.effort_correction = 1.0
        self.friction_transition_velocity = 0.02
        self.friction_constant_term = 0.0
        self.friction_coulomb_coef = 0.0
        self.friction_viscous_coef = 0.0
        self.position_offset = 0.0


class _Algo:
    def __init__(self):
        self.singularity_threshold = 0.0


class MockDriver:
    """Stores vectors like the real driver; same getter/setter names.

    The interface parity test pins this against the real TrossenArmDriver.
    """

    def __init__(self):
        self.friction_constant_terms = [0.0] * N_JOINTS
        self.friction_coulomb_coefs = [0.0] * N_JOINTS
        self.friction_viscous_coefs = [0.0] * N_JOINTS
        self.friction_transition_velocities = [0.02] * N_JOINTS
        self.effort_corrections = [1.0] * N_JOINTS
        self.motor_parameters = [
            {trossen_arm.Mode.position: _Motor(100.0, 4.0)} for _ in range(N_JOINTS)
        ]
        self.joint_limits = [_Limit() for _ in range(N_JOINTS)]
        self.characteristics = [_Characteristic() for _ in range(N_JOINTS)]
        self.algorithm = _Algo()
        # kinematic state, for code that reads the arm back
        self.positions = [0.0] * N_JOINTS
        self.velocities = [0.0] * N_JOINTS
        self.external_efforts = [0.0] * N_JOINTS

    @staticmethod
    def _vec(name: str):
        def get(self):
            return list(getattr(self, name))

        def set_(self, vals):
            setattr(self, name, [float(v) for v in vals])

        return get, set_

    get_friction_constant_terms, set_friction_constant_terms = _vec("friction_constant_terms")
    get_friction_coulomb_coefs, set_friction_coulomb_coefs = _vec("friction_coulomb_coefs")
    get_friction_viscous_coefs, set_friction_viscous_coefs = _vec("friction_viscous_coefs")
    get_friction_transition_velocities, set_friction_transition_velocities = _vec(
        "friction_transition_velocities")
    get_effort_corrections, set_effort_corrections = _vec("effort_corrections")
    get_positions, set_positions = _vec("positions")
    get_velocities, set_velocities = _vec("velocities")
    get_external_efforts, set_external_efforts = _vec("external_efforts")

    def get_motor_parameters(self):
        return self.motor_parameters

    def set_motor_parameters(self, mp):
        self.motor_parameters = mp

    def get_joint_limits(self):
        return self.joint_limits

    def set_joint_limits(self, jl):
        self.joint_limits = list(jl)

    def get_joint_characteristics(self):
        return self.characteristics

    def set_joint_characteristics(self, jc):
        self.characteristics = list(jc)

    def get_algorithm_parameter(self):
        return self.algorithm

    def set_algorithm_parameter(self, ap):
        self.algorithm = ap

    def get_error_information(self):
        return ""

    # --- the surface recovery.land_arm touches -------------------------------

    def configure(self, *args, **kwargs):
        return None

    def cleanup(self, *args, **kwargs):
        return None

    def get_all_positions(self):
        return list(self.positions)

    def get_all_velocities(self):
        return list(self.velocities)

    def set_all_modes(self, *args, **kwargs):
        return None

    def set_all_positions(self, *args, **kwargs):
        return None

    def set_arm_positions(self, *args, **kwargs):
        return None

    def set_arm_modes(self, *args, **kwargs):
        return None

    def set_joint_position(self, *args, **kwargs):
        return None

    def set_end_effector(self, *args, **kwargs):
        return None

    def get_all_external_efforts(self):
        return list(self.external_efforts)

    def set_all_external_efforts(self, *args, **kwargs):
        return None

    def set_arm_external_efforts(self, *args, **kwargs):
        return None

    def get_robot_output(self):
        return None
