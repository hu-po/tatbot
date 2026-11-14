"""ros_budget's joint rule: ros/ takes which joint is which from the arm model, never from a joint's number."""
import ros_budget


def test_a_joint_by_number_is_flagged_and_a_line_marked_as_no_joint_is_not():
    text = "\n".join([
        "speed = abs(qd[:, :6]).max()",                                          # the WXAI's chain by number
        "carriage = q[6]",                                                       # its carriage by number
        "names = [f'{arm}/joint_{i}' for i in range(6)]",                        # its joint names by number
        "for _ in range(6):",                                                    # six tries: no joint
        "pose = camera.moved(x[:6])   # arm-model: not joints, the camera's pose",
        "chain = q[model.arm_index]",
        "if (i == kCarriage) {",
    ])
    assert ros_budget.joint_indices(text) == [1, 2, 3, 7]
