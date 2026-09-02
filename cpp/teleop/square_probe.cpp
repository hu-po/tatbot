#include "square_probe.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace tatbot::square
{

std::array<Pose, 4> targets(const Pose & start, double side_m)
{
  if (!std::isfinite(side_m) || side_m <= 0.0) {
    throw std::invalid_argument("square side must be finite and positive");
  }
  const double x_step = std::signbit(start[0]) ? side_m : -side_m;
  const double y_step = std::signbit(start[1]) ? side_m : -side_m;
  std::array<Pose, 4> result{start, start, start, start};
  result[0][0] += x_step;
  result[1][0] += x_step;
  result[1][1] += y_step;
  result[2][1] += y_step;
  return result;
}

double translation_error_mm(const Pose & measured, const Pose & target)
{
  double sum = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    const double delta = measured[i] - target[i];
    sum += delta * delta;
  }
  return std::sqrt(sum) * 1000.0;
}

SegmentSample quintic_segment(
  const Pose & start, const Pose & target, double elapsed_s, double duration_s)
{
  if (!std::isfinite(elapsed_s) || !std::isfinite(duration_s) || duration_s <= 0.0) {
    throw std::invalid_argument("segment time must be finite and duration positive");
  }
  const double u = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double blend_rate = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / duration_s;

  SegmentSample sample{start, Pose{}};
  for (size_t axis = 0; axis < 3; ++axis) {
    const double delta = target[axis] - start[axis];
    sample.position[axis] += blend * delta;
    sample.feedforward_velocity[axis] = blend_rate * delta;
  }
  return sample;
}

namespace
{

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;
using Mat6 = std::array<std::array<double, 6>, 6>;
using Mat67 = std::array<std::array<double, 7>, 6>;

// Canonical arm geometry: urdf/tatbot.urdf right/joint_0..5 and
// right/ee_gripper. The live FK agreement gate in wxai_teleop makes any drift
// from the vendor controller a refusal before motion.
constexpr std::array<Vec3, 6> JOINT_ORIGINS{{
  {0.0, 0.0, 0.05725},
  {0.02, 0.0, 0.04625},
  {-0.264, 0.0, 0.0},
  {0.245, 0.0, 0.06},
  {0.06775, 0.0, 0.0455},
  {0.02895, 0.0, -0.0455}}};
constexpr std::array<Vec3, 6> JOINT_AXES{{
  {0.0, 0.0, 1.0},
  {0.0, 1.0, 0.0},
  {0.0, -1.0, 0.0},
  {0.0, -1.0, 0.0},
  {0.0, 0.0, -1.0},
  {1.0, 0.0, 0.0}}};
constexpr Vec3 TCP_IN_LINK6{0.156062, 0.0, 0.0};
// config/workspace.yaml lutin-ballpoint-dot measured tip transformed through
// urdf/tatbot.urdf right/left_carriage_joint and right/tool_mount_joint into
// link_6 at carriage=0. The carriage adds link_6-local +Y translation.
constexpr Vec3 BALLPOINT_TIP_IN_LINK6{0.20550927, 0.01083364, -0.00149001};
constexpr Vec3 CARRIAGE_AXIS_IN_LINK6{0.0, 1.0, 0.0};
constexpr JointPose JOINT_LOWER{
  -3.0543261909900767, 0.0, 0.0, -1.5707963267948966,
  -1.5707963267948966, -3.141592653589793};
constexpr JointPose JOINT_UPPER{
  3.0543261909900767, 3.141592653589793, 2.356194490192345,
  1.5707963267948966, 1.5707963267948966, 3.141592653589793};
constexpr double JOINT_LIMIT_MARGIN_RAD = 0.05;
constexpr double PLAN_MAX_JOINT_VELOCITY_RAD_S = 0.25;
constexpr double PLAN_MAX_MODEL_ERROR_M = 0.0001;
// Pen up (orbit, approach, lift): the damped solve trails a 10 mm/s reference
// by a few tenths of a millimetre, which is nothing at standoff; the drawing
// cap above still applies to every pen-down sample.
constexpr double PLAN_MAX_MODEL_ERROR_PEN_UP_M = 0.001;
// Pen-down samples of a draw path (plan_path_samples). The 0.1 mm cap above stays
// on the executor's own spiral; a surface path follows the local normal, and
// with the wrist near its singularity (joint 4 through zero on the first bottle
// draw) the damped solve trails a 3.5 mm/s reference by 0.18 mm -- nothing
// against a pen line. Mirrors draw_kinematics.PLAN_MAX_MODEL_ERROR_DRAW_M; the
// preflight reports the actual value.
constexpr double PLAN_MAX_MODEL_ERROR_DRAW_M = 0.00025;
constexpr double PLAN_MAX_ORIENTATION_ERROR_RAD = 0.001;
constexpr double DLS_DAMPING = 0.02;
constexpr double POSITION_ERROR_GAIN_S = 4.0;
constexpr double ORIENTATION_ERROR_GAIN_S = 2.0;
constexpr double CARRIAGE_DLS_WEIGHT = 2.0;
constexpr double CARRIAGE_CENTER_GAIN_S = 2.0;
constexpr double PLAN_MAX_CARRIAGE_VELOCITY_M_S = 0.001;
constexpr double PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 = 0.02;

Mat3 identity3()
{
  return Mat3{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

Vec3 add(const Vec3 & a, const Vec3 & b)
{
  return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract(const Vec3 & a, const Vec3 & b)
{
  return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return Vec3{
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]};
}

Vec3 multiply(const Mat3 & matrix, const Vec3 & vector)
{
  Vec3 result{};
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {result[row] += matrix[row][col] * vector[col];}
  }
  return result;
}

Mat3 multiply(const Mat3 & left, const Mat3 & right)
{
  Mat3 result{};
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      for (size_t inner = 0; inner < 3; ++inner) {
        result[row][col] += left[row][inner] * right[inner][col];
      }
    }
  }
  return result;
}

Mat3 axis_rotation(const Vec3 & axis, double angle)
{
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double one_c = 1.0 - c;
  const double x = axis[0];
  const double y = axis[1];
  const double z = axis[2];
  return Mat3{{
    {x * x * one_c + c, x * y * one_c - z * s, x * z * one_c + y * s},
    {y * x * one_c + z * s, y * y * one_c + c, y * z * one_c - x * s},
    {z * x * one_c - y * s, z * y * one_c + x * s, z * z * one_c + c}}};
}

struct KinematicState
{
  Vec3 position{};
  Mat3 rotation{};
  Mat6 jacobian{};
};

KinematicState evaluate_at(const JointPose & joints, const Vec3 & tcp_in_link6)
{
  Mat3 rotation = identity3();
  Vec3 position{};
  std::array<Vec3, 6> joint_positions{};
  std::array<Vec3, 6> joint_axes{};
  for (size_t joint = 0; joint < joints.size(); ++joint) {
    position = add(position, multiply(rotation, JOINT_ORIGINS[joint]));
    joint_positions[joint] = position;
    joint_axes[joint] = multiply(rotation, JOINT_AXES[joint]);
    rotation = multiply(rotation, axis_rotation(JOINT_AXES[joint], joints[joint]));
  }
  position = add(position, multiply(rotation, tcp_in_link6));

  Mat6 jacobian{};
  for (size_t joint = 0; joint < joints.size(); ++joint) {
    const Vec3 linear = cross(joint_axes[joint], subtract(position, joint_positions[joint]));
    for (size_t axis = 0; axis < 3; ++axis) {
      jacobian[axis][joint] = linear[axis];
      jacobian[axis + 3][joint] = joint_axes[joint][axis];
    }
  }
  return KinematicState{position, rotation, jacobian};
}

KinematicState evaluate(const JointPose & joints)
{
  return evaluate_at(joints, TCP_IN_LINK6);
}

struct CarriageKinematicState
{
  Vec3 position{};
  Mat3 rotation{};
  Mat67 jacobian{};
};

CarriageKinematicState evaluate_ballpoint(const JointPose & joints, double carriage_m)
{
  Vec3 tip = BALLPOINT_TIP_IN_LINK6;
  for (size_t axis = 0; axis < 3; ++axis) {
    tip[axis] += carriage_m * CARRIAGE_AXIS_IN_LINK6[axis];
  }
  const KinematicState arm = evaluate_at(joints, tip);
  Mat67 jacobian{};
  for (size_t axis = 0; axis < 6; ++axis) {
    for (size_t joint = 0; joint < 6; ++joint) {
      jacobian[axis][joint] = arm.jacobian[axis][joint];
    }
  }
  const Vec3 carriage_axis = multiply(arm.rotation, CARRIAGE_AXIS_IN_LINK6);
  for (size_t axis = 0; axis < 3; ++axis) {jacobian[axis][6] = carriage_axis[axis];}
  return CarriageKinematicState{arm.position, arm.rotation, jacobian};
}

Vec3 orientation_error(const Mat3 & current, const Mat3 & target)
{
  Vec3 error{};
  for (size_t col = 0; col < 3; ++col) {
    const Vec3 current_axis{current[0][col], current[1][col], current[2][col]};
    const Vec3 target_axis{target[0][col], target[1][col], target[2][col]};
    const Vec3 term = cross(current_axis, target_axis);
    for (size_t axis = 0; axis < 3; ++axis) {error[axis] += 0.5 * term[axis];}
  }
  return error;
}

std::array<double, 6> solve(Mat6 matrix, std::array<double, 6> rhs)
{
  for (size_t col = 0; col < 6; ++col) {
    size_t pivot = col;
    for (size_t row = col + 1; row < 6; ++row) {
      if (std::fabs(matrix[row][col]) > std::fabs(matrix[pivot][col])) {pivot = row;}
    }
    if (std::fabs(matrix[pivot][col]) < 1e-12) {
      throw std::runtime_error("square DLS solve is singular");
    }
    if (pivot != col) {
      std::swap(matrix[pivot], matrix[col]);
      std::swap(rhs[pivot], rhs[col]);
    }
    for (size_t row = col + 1; row < 6; ++row) {
      const double factor = matrix[row][col] / matrix[col][col];
      for (size_t inner = col; inner < 6; ++inner) {
        matrix[row][inner] -= factor * matrix[col][inner];
      }
      rhs[row] -= factor * rhs[col];
    }
  }
  std::array<double, 6> solution{};
  for (int row = 5; row >= 0; --row) {
    double value = rhs[static_cast<size_t>(row)];
    for (size_t col = static_cast<size_t>(row) + 1; col < 6; ++col) {
      value -= matrix[static_cast<size_t>(row)][col] * solution[col];
    }
    solution[static_cast<size_t>(row)] =
      value / matrix[static_cast<size_t>(row)][static_cast<size_t>(row)];
  }
  return solution;
}

JointPose damped_least_squares(const Mat6 & jacobian, const std::array<double, 6> & twist)
{
  Mat6 normal{};
  for (size_t row = 0; row < 6; ++row) {
    for (size_t col = 0; col < 6; ++col) {
      for (size_t joint = 0; joint < 6; ++joint) {
        normal[row][col] += jacobian[row][joint] * jacobian[col][joint];
      }
      if (row == col) {normal[row][col] += DLS_DAMPING * DLS_DAMPING;}
    }
  }
  const auto intermediate = solve(normal, twist);
  JointPose joint_velocity{};
  for (size_t joint = 0; joint < 6; ++joint) {
    for (size_t axis = 0; axis < 6; ++axis) {
      joint_velocity[joint] += jacobian[axis][joint] * intermediate[axis];
    }
  }
  return joint_velocity;
}

FullJointPose weighted_carriage_dls(
  const Mat67 & jacobian, const std::array<double, 6> & twist,
  double carriage_centering_velocity_m_s)
{
  const std::array<double, 7> inverse_weights{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0 / (CARRIAGE_DLS_WEIGHT * CARRIAGE_DLS_WEIGHT)};
  auto apply_pseudoinverse = [&](const std::array<double, 6> & task) {
      Mat6 normal{};
      for (size_t row = 0; row < 6; ++row) {
        for (size_t col = 0; col < 6; ++col) {
          for (size_t joint = 0; joint < 7; ++joint) {
            normal[row][col] += jacobian[row][joint] * inverse_weights[joint] *
              jacobian[col][joint];
          }
          if (row == col) {normal[row][col] += DLS_DAMPING * DLS_DAMPING;}
        }
      }
      const auto intermediate = solve(normal, task);
      FullJointPose result{};
      for (size_t joint = 0; joint < 7; ++joint) {
        for (size_t axis = 0; axis < 6; ++axis) {
          result[joint] += inverse_weights[joint] * jacobian[axis][joint] *
            intermediate[axis];
        }
      }
      return result;
    };

  FullJointPose velocity = apply_pseudoinverse(twist);
  FullJointPose centering{};
  centering[6] = carriage_centering_velocity_m_s;
  std::array<double, 6> centering_task{};
  for (size_t axis = 0; axis < 6; ++axis) {
    centering_task[axis] = jacobian[axis][6] * centering[6];
  }
  const FullJointPose projected = apply_pseudoinverse(centering_task);
  for (size_t joint = 0; joint < 7; ++joint) {
    velocity[joint] += (centering[joint] - projected[joint]);
  }
  return velocity;
}

double norm(const Vec3 & vector)
{
  return std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}

void check_joint_limits(const JointPose & joints)
{
  for (size_t joint = 0; joint < joints.size(); ++joint) {
    if (!std::isfinite(joints[joint]) ||
      joints[joint] < JOINT_LOWER[joint] + JOINT_LIMIT_MARGIN_RAD ||
      joints[joint] > JOINT_UPPER[joint] - JOINT_LIMIT_MARGIN_RAD)
    {
      throw std::runtime_error(
              "square joint plan reaches the guarded limit on joint " + std::to_string(joint));
    }
  }
}

}  // namespace

std::array<double, 3> wxai_tcp_translation(const JointPose & joints)
{
  return evaluate(joints).position;
}

std::array<double, 3> wxai_ballpoint_tip_translation(
  const JointPose & joints, double carriage_m)
{
  return evaluate_ballpoint(joints, carriage_m).position;
}

JointPlan plan_joint_square(
  const JointPose & start_joints,
  const std::array<Pose, 4> & cartesian_targets,
  double edge_s,
  double period_s)
{
  if (!std::isfinite(edge_s) || edge_s <= 0.0 ||
    !std::isfinite(period_s) || period_s <= 0.0)
  {
    throw std::invalid_argument("square plan times must be finite and positive");
  }
  check_joint_limits(start_joints);
  const size_t ticks_per_edge = static_cast<size_t>(std::ceil(edge_s / period_s));
  if (ticks_per_edge == 0 || ticks_per_edge > 100000) {
    throw std::invalid_argument("square plan sample count is outside the guarded range");
  }

  JointPlan plan;
  plan.positions.reserve(ticks_per_edge * cartesian_targets.size());
  plan.velocities.reserve(ticks_per_edge * cartesian_targets.size());
  plan.cartesian_references.reserve(ticks_per_edge * cartesian_targets.size());
  JointPose joints = start_joints;
  const KinematicState initial = evaluate(joints);
  Pose edge_start{};
  for (size_t axis = 0; axis < 3; ++axis) {edge_start[axis] = initial.position[axis];}

  for (size_t edge = 0; edge < cartesian_targets.size(); ++edge) {
    for (size_t tick = 1; tick <= ticks_per_edge; ++tick) {
      const double elapsed_s = std::min(edge_s, static_cast<double>(tick) * period_s);
      const SegmentSample reference = quintic_segment(
        edge_start, cartesian_targets[edge], elapsed_s, edge_s);
      const Vec3 feedforward{
        reference.feedforward_velocity[0], reference.feedforward_velocity[1],
        reference.feedforward_velocity[2]};
      plan.max_cartesian_velocity_m_s = std::max(
        plan.max_cartesian_velocity_m_s, norm(feedforward));
      const KinematicState state = evaluate(joints);
      const Vec3 rotation_error = orientation_error(state.rotation, initial.rotation);
      std::array<double, 6> twist{};
      for (size_t axis = 0; axis < 3; ++axis) {
        twist[axis] = reference.feedforward_velocity[axis] +
          POSITION_ERROR_GAIN_S * (reference.position[axis] - state.position[axis]);
        twist[axis + 3] = ORIENTATION_ERROR_GAIN_S * rotation_error[axis];
      }
      const JointPose joint_velocity = damped_least_squares(state.jacobian, twist);
      for (size_t joint = 0; joint < joints.size(); ++joint) {
        plan.max_joint_velocity_rad_s = std::max(
          plan.max_joint_velocity_rad_s, std::fabs(joint_velocity[joint]));
        if (std::fabs(joint_velocity[joint]) > PLAN_MAX_JOINT_VELOCITY_RAD_S) {
          throw std::runtime_error(
                  "square joint plan exceeds its velocity cap on joint " +
                  std::to_string(joint));
        }
        joints[joint] += period_s * joint_velocity[joint];
      }
      check_joint_limits(joints);
      const KinematicState integrated = evaluate(joints);
      const Vec3 reference_position{
        reference.position[0], reference.position[1], reference.position[2]};
      const double model_error_m = norm(subtract(reference_position, integrated.position));
      plan.max_model_error_mm = std::max(plan.max_model_error_mm, model_error_m * 1000.0);
      if (model_error_m > PLAN_MAX_MODEL_ERROR_M) {
        throw std::runtime_error("square joint plan exceeds its Cartesian model-error cap");
      }
      const double orientation_error_rad = norm(orientation_error(
        integrated.rotation, initial.rotation));
      plan.max_orientation_error_rad = std::max(
        plan.max_orientation_error_rad, orientation_error_rad);
      if (orientation_error_rad > PLAN_MAX_ORIENTATION_ERROR_RAD) {
        throw std::runtime_error("square joint plan exceeds its orientation-error cap");
      }
      plan.positions.push_back(joints);
      plan.velocities.push_back(joint_velocity);
      plan.cartesian_references.push_back({
        reference.position[0], reference.position[1], reference.position[2]});
    }
    const KinematicState endpoint = evaluate(joints);
    const Vec3 target{
      cartesian_targets[edge][0], cartesian_targets[edge][1], cartesian_targets[edge][2]};
    if (norm(subtract(target, endpoint.position)) > PLAN_MAX_MODEL_ERROR_M) {
      throw std::runtime_error("square joint plan endpoint does not converge");
    }
    plan.edge_end_ticks[edge] = plan.positions.size();
    edge_start = cartesian_targets[edge];
  }
  return plan;
}

JointPlan plan_joint_spiral(
  const JointPose & start_joints,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s)
{
  if (!std::isfinite(radius_m) || radius_m <= 0.0 ||
    !std::isfinite(turns) || turns <= 0.0 ||
    !std::isfinite(duration_s) || duration_s <= 0.0 ||
    !std::isfinite(ease_s) || ease_s <= 0.0 || ease_s * 2.0 >= duration_s ||
    !std::isfinite(period_s) || period_s <= 0.0)
  {
    throw std::invalid_argument("spiral geometry and times must be finite and positive");
  }
  check_joint_limits(start_joints);
  const size_t ticks = static_cast<size_t>(std::ceil(duration_s / period_s));
  if (ticks == 0 || ticks > 250000) {
    throw std::invalid_argument("spiral plan sample count is outside the guarded range");
  }

  JointPlan plan;
  plan.positions.reserve(ticks);
  plan.velocities.reserve(ticks);
  plan.cartesian_references.reserve(ticks);
  JointPose joints = start_joints;
  const KinematicState initial = evaluate(joints);
  const Vec3 center = initial.position;
  constexpr double pi = 3.14159265358979323846;
  const double total_angle = 2.0 * pi * turns;
  const double spiral_scale = radius_m / total_angle;
  const double path_length = 0.5 * spiral_scale * (
    total_angle * std::sqrt(1.0 + total_angle * total_angle) +
    std::asinh(total_angle));
  const double cruise_speed = path_length / (duration_s - ease_s);
  plan.path_length_m = path_length;

  for (size_t tick = 1; tick <= ticks; ++tick) {
    const double elapsed_s = std::min(duration_s, static_cast<double>(tick) * period_s);
    double distance = 0.0;
    double path_speed = 0.0;
    if (elapsed_s < ease_s) {
      const double u = elapsed_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    } else if (elapsed_s <= duration_s - ease_s) {
      distance = 0.5 * cruise_speed * ease_s +
        cruise_speed * (elapsed_s - ease_s);
      path_speed = cruise_speed;
    } else {
      const double remaining_s = duration_s - elapsed_s;
      const double u = remaining_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = path_length - cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    }
    distance = std::clamp(distance, 0.0, path_length);
    double angle = total_angle * distance / path_length;
    for (size_t iteration = 0; iteration < 6; ++iteration) {
      const double root = std::sqrt(1.0 + angle * angle);
      const double integrated_length = 0.5 * spiral_scale * (
        angle * root + std::asinh(angle));
      angle -= (integrated_length - distance) / (spiral_scale * root);
      angle = std::clamp(angle, 0.0, total_angle);
    }
    const double radius = spiral_scale * angle;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const Vec3 reference{
      center[0] + radius * cosine,
      center[1] + radius * sine,
      center[2]};
    const Vec3 feedforward{
      path_speed * (cosine - angle * sine) / std::sqrt(1.0 + angle * angle),
      path_speed * (sine + angle * cosine) / std::sqrt(1.0 + angle * angle),
      0.0};
    plan.max_cartesian_velocity_m_s = std::max(
      plan.max_cartesian_velocity_m_s, norm(feedforward));

    const KinematicState state = evaluate(joints);
    const Vec3 rotation_error = orientation_error(state.rotation, initial.rotation);
    std::array<double, 6> twist{};
    for (size_t axis = 0; axis < 3; ++axis) {
      twist[axis] = feedforward[axis] +
        POSITION_ERROR_GAIN_S * (reference[axis] - state.position[axis]);
      twist[axis + 3] = ORIENTATION_ERROR_GAIN_S * rotation_error[axis];
    }
    const JointPose joint_velocity = damped_least_squares(state.jacobian, twist);
    for (size_t joint = 0; joint < joints.size(); ++joint) {
      plan.max_joint_velocity_rad_s = std::max(
        plan.max_joint_velocity_rad_s, std::fabs(joint_velocity[joint]));
      if (std::fabs(joint_velocity[joint]) > PLAN_MAX_JOINT_VELOCITY_RAD_S) {
        throw std::runtime_error(
                "spiral joint plan exceeds its velocity cap on joint " +
                std::to_string(joint));
      }
      joints[joint] += period_s * joint_velocity[joint];
    }
    check_joint_limits(joints);
    const KinematicState integrated = evaluate(joints);
    const double model_error_m = norm(subtract(reference, integrated.position));
    plan.max_model_error_mm = std::max(plan.max_model_error_mm, model_error_m * 1000.0);
    if (model_error_m > PLAN_MAX_MODEL_ERROR_M) {
      throw std::runtime_error("spiral joint plan exceeds its Cartesian model-error cap");
    }
    const double orientation_error_rad = norm(orientation_error(
      integrated.rotation, initial.rotation));
    plan.max_orientation_error_rad = std::max(
      plan.max_orientation_error_rad, orientation_error_rad);
    if (orientation_error_rad > PLAN_MAX_ORIENTATION_ERROR_RAD) {
      throw std::runtime_error("spiral joint plan exceeds its orientation-error cap");
    }
    plan.positions.push_back(joints);
    plan.velocities.push_back(joint_velocity);
    plan.cartesian_references.push_back(reference);
  }

  const KinematicState endpoint = evaluate(joints);
  const Vec3 target{center[0] + radius_m, center[1], center[2]};
  if (norm(subtract(target, endpoint.position)) > PLAN_MAX_MODEL_ERROR_M) {
    throw std::runtime_error("spiral joint plan endpoint does not converge");
  }
  plan.edge_end_ticks[0] = plan.positions.size();
  return plan;
}

Rotation wxai_link6_rotation(const JointPose & joints)
{
  return evaluate(joints).rotation;
}

std::vector<PathSample> spiral_path_samples(
  const std::array<double, 3> & center,
  const Rotation & rotation,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s)
{
  if (!std::isfinite(radius_m) || radius_m <= 0.0 ||
    !std::isfinite(turns) || turns <= 0.0 ||
    !std::isfinite(duration_s) || duration_s <= 0.0 ||
    !std::isfinite(ease_s) || ease_s <= 0.0 || ease_s * 2.0 >= duration_s ||
    !std::isfinite(period_s) || period_s <= 0.0)
  {
    throw std::invalid_argument("carriage-IK spiral geometry and times must be finite and positive");
  }
  const size_t ticks = static_cast<size_t>(std::ceil(duration_s / period_s));
  if (ticks == 0 || ticks > 250000) {
    throw std::invalid_argument("carriage-IK spiral sample count is outside the guarded range");
  }
  constexpr double pi = 3.14159265358979323846;
  const double total_angle = 2.0 * pi * turns;
  const double spiral_scale = radius_m / total_angle;
  const double path_length = 0.5 * spiral_scale * (
    total_angle * std::sqrt(1.0 + total_angle * total_angle) +
    std::asinh(total_angle));
  const double cruise_speed = path_length / (duration_s - ease_s);

  std::vector<PathSample> samples;
  samples.reserve(ticks);
  for (size_t tick = 1; tick <= ticks; ++tick) {
    const double elapsed_s = std::min(duration_s, static_cast<double>(tick) * period_s);
    double distance = 0.0;
    double path_speed = 0.0;
    if (elapsed_s < ease_s) {
      const double u = elapsed_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    } else if (elapsed_s <= duration_s - ease_s) {
      distance = 0.5 * cruise_speed * ease_s +
        cruise_speed * (elapsed_s - ease_s);
      path_speed = cruise_speed;
    } else {
      const double remaining_s = duration_s - elapsed_s;
      const double u = remaining_s / ease_s;
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double u4 = u3 * u;
      const double u5 = u4 * u;
      const double u6 = u5 * u;
      const double speed_blend = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
      const double distance_blend = 2.5 * u4 - 3.0 * u5 + u6;
      distance = path_length - cruise_speed * ease_s * distance_blend;
      path_speed = cruise_speed * speed_blend;
    }
    distance = std::clamp(distance, 0.0, path_length);
    double angle = total_angle * distance / path_length;
    for (size_t iteration = 0; iteration < 6; ++iteration) {
      const double root = std::sqrt(1.0 + angle * angle);
      const double integrated_length = 0.5 * spiral_scale * (
        angle * root + std::asinh(angle));
      angle -= (integrated_length - distance) / (spiral_scale * root);
      angle = std::clamp(angle, 0.0, total_angle);
    }
    const double radius = spiral_scale * angle;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    PathSample sample;
    sample.t_s = elapsed_s;
    sample.position = {
      center[0] + radius * cosine,
      center[1] + radius * sine,
      center[2]};
    sample.velocity = {
      path_speed * (cosine - angle * sine) / std::sqrt(1.0 + angle * angle),
      path_speed * (sine + angle * cosine) / std::sqrt(1.0 + angle * angle),
      0.0};
    sample.rotation = rotation;
    sample.pen = true;
    samples.push_back(sample);
  }
  return samples;
}

namespace
{

double spiral_path_length(double radius_m, double turns)
{
  constexpr double pi = 3.14159265358979323846;
  const double total_angle = 2.0 * pi * turns;
  const double spiral_scale = radius_m / total_angle;
  return 0.5 * spiral_scale * (
    total_angle * std::sqrt(1.0 + total_angle * total_angle) + std::asinh(total_angle));
}

bool orthonormal(const Rotation & r)
{
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      double dot = 0.0;
      for (size_t k = 0; k < 3; ++k) {dot += r[k][i] * r[k][j];}
      const double expected = (i == j) ? 1.0 : 0.0;
      if (!std::isfinite(dot) || std::fabs(dot - expected) > 1e-6) {return false;}
    }
  }
  const Vec3 c0{r[0][0], r[1][0], r[2][0]};
  const Vec3 c1{r[0][1], r[1][1], r[2][1]};
  const Vec3 c2{r[0][2], r[1][2], r[2][2]};
  const Vec3 c0xc1 = cross(c0, c1);
  double det = 0.0;
  for (size_t k = 0; k < 3; ++k) {det += c0xc1[k] * c2[k];}
  return std::fabs(det - 1.0) < 1e-6;
}

std::vector<std::string> split_csv(const std::string & line)
{
  std::vector<std::string> fields;
  std::string field;
  std::stringstream stream(line);
  while (std::getline(stream, field, ',')) {fields.push_back(field);}
  if (!line.empty() && line.back() == ',') {fields.emplace_back();}
  return fields;
}

double parse_double(const std::string & text, const std::string & what)
{
  try {
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
      throw std::runtime_error("bad");
    }
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error("samples file: " + what + " is not a finite number: '" + text + "'");
  }
}

}  // namespace

PathFile load_path_file(const std::string & path, double period_s)
{
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("samples file cannot be opened: " + path);
  }
  static const std::vector<std::string> expected_columns{
    "t_s", "px", "py", "pz", "vx", "vy", "vz",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "pen", "capture"};
  PathFile out;
  bool schema_ok = false;
  bool frame_ok = false;
  bool period_seen = false;
  bool tip_seen[3] = {false, false, false};
  size_t declared_samples = 0;
  bool in_rows = false;
  std::string line;
  size_t line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {line.pop_back();}
    if (line.empty()) {continue;}
    const auto fields = split_csv(line);
    if (!in_rows) {
      if (fields.size() < 2) {
        throw std::runtime_error("samples file: malformed header line " + std::to_string(line_number));
      }
      const std::string & key = fields[0];
      const std::string & value = fields[1];
      if (key == "schema") {
        if (value != "tatbot.draw-samples/1") {
          throw std::runtime_error("samples file: unsupported schema '" + value + "'");
        }
        schema_ok = true;
      } else if (key == "kind") {
        out.kind = value;
      } else if (key == "frame") {
        if (value != "right/base_link") {
          throw std::runtime_error("samples file: frame must be right/base_link, got '" + value + "'");
        }
        frame_ok = true;
      } else if (key == "period_s") {
        const double declared = parse_double(value, "period_s");
        if (std::fabs(declared - period_s) > 1e-9) {
          throw std::runtime_error(
                  "samples file: period " + value + " s does not match the control period " +
                  std::to_string(period_s) + " s");
        }
        out.period_s = declared;
        period_seen = true;
      } else if (key == "tip_x_m" || key == "tip_y_m" || key == "tip_z_m") {
        const size_t axis = static_cast<size_t>(key[4] - 'x');
        out.tip_in_link6[axis] = parse_double(value, key);
        tip_seen[axis] = true;
      } else if (key == "sample_count") {
        declared_samples = static_cast<size_t>(parse_double(value, key));
      } else if (key == "capture_count") {
        out.capture_count = static_cast<size_t>(parse_double(value, key));
      } else if (key == "start_tolerance_m") {
        out.start_tolerance_m = parse_double(value, key);
      } else if (key == "carriage_ik") {
        const double flag = parse_double(value, key);
        if (flag != 0.0 && flag != 1.0) {
          throw std::runtime_error("samples file: carriage_ik must be 0 or 1");
        }
        out.carriage_ik = flag == 1.0;
      } else if (key == "columns") {
        if (std::vector<std::string>(fields.begin() + 1, fields.end()) != expected_columns) {
          throw std::runtime_error("samples file: unexpected column layout: " + line);
        }
        in_rows = true;
      } else {
        out.report.emplace_back(key, value);
      }
      continue;
    }
    if (fields.size() != expected_columns.size()) {
      throw std::runtime_error(
              "samples file: row " + std::to_string(line_number) + " has " +
              std::to_string(fields.size()) + " fields, expected " +
              std::to_string(expected_columns.size()));
    }
    PathSample sample;
    sample.t_s = parse_double(fields[0], "t_s");
    for (size_t axis = 0; axis < 3; ++axis) {
      sample.position[axis] = parse_double(fields[1 + axis], "position");
      sample.velocity[axis] = parse_double(fields[4 + axis], "velocity");
    }
    for (size_t row = 0; row < 3; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        sample.rotation[row][col] = parse_double(fields[7 + row * 3 + col], "rotation");
      }
    }
    if (!orthonormal(sample.rotation)) {
      throw std::runtime_error(
              "samples file: row " + std::to_string(line_number) + " rotation is not orthonormal");
    }
    const double pen = parse_double(fields[16], "pen");
    const double capture = parse_double(fields[17], "capture");
    if ((pen != 0.0 && pen != 1.0) || capture < 0.0 || capture != std::floor(capture)) {
      throw std::runtime_error("samples file: row " + std::to_string(line_number) + " has bad pen/capture flags");
    }
    sample.pen = pen == 1.0;
    sample.capture = static_cast<size_t>(capture);
    out.samples.push_back(sample);
  }
  if (!schema_ok || !frame_ok || !period_seen || !tip_seen[0] || !tip_seen[1] || !tip_seen[2]) {
    throw std::runtime_error("samples file: missing schema, frame, period_s or tip_*_m header");
  }
  if (!in_rows || out.samples.empty()) {
    throw std::runtime_error("samples file: no sample rows");
  }
  if (declared_samples != 0 && declared_samples != out.samples.size()) {
    throw std::runtime_error(
            "samples file: sample_count " + std::to_string(declared_samples) + " but " +
            std::to_string(out.samples.size()) + " rows");
  }
  if (out.samples.size() > 250000) {
    throw std::runtime_error("samples file: more than 250000 rows");
  }
  const double tip_error = norm(subtract(out.tip_in_link6, BALLPOINT_TIP_IN_LINK6));
  if (tip_error > 1e-4) {
    throw std::runtime_error(
            "samples file: tip model differs from the executor's ballpoint constant by " +
            std::to_string(tip_error * 1e3) + " mm; re-derive one of them");
  }
  size_t expected_capture = 1;
  for (const auto & sample : out.samples) {
    if (sample.capture == 0) {continue;}
    if (sample.capture != expected_capture) {
      throw std::runtime_error("samples file: capture indices must run 1..K in order");
    }
    ++expected_capture;
  }
  if (expected_capture - 1 != out.capture_count) {
    throw std::runtime_error(
            "samples file: capture_count " + std::to_string(out.capture_count) + " but " +
            std::to_string(expected_capture - 1) + " capture rows");
  }
  if (!std::isfinite(out.start_tolerance_m) || out.start_tolerance_m <= 0.0 ||
    out.start_tolerance_m > 0.005)
  {
    throw std::runtime_error("samples file: start_tolerance_m must be in (0, 5] mm");
  }
  return out;
}

CarriageJointPlan plan_joint_path(
  const JointPose & start_joints,
  double start_carriage_m,
  const std::vector<PathSample> & samples,
  double period_s,
  double start_tolerance_m,
  bool carriage_ik)
{
  if (!std::isfinite(start_carriage_m) ||
    start_carriage_m < CARRIAGE_IK_MIN_M || start_carriage_m > CARRIAGE_IK_MAX_M)
  {
    throw std::invalid_argument("carriage-IK start is outside its guarded drawing envelope");
  }
  if (!std::isfinite(period_s) || period_s <= 0.0) {
    throw std::invalid_argument("path plan period must be finite and positive");
  }
  if (samples.empty() || samples.size() > 250000) {
    throw std::invalid_argument("path plan sample count is outside the guarded range");
  }
  check_joint_limits(start_joints);

  CarriageJointPlan plan;
  plan.positions.reserve(samples.size());
  plan.velocities.reserve(samples.size());
  plan.cartesian_references.reserve(samples.size());
  plan.min_carriage_m = start_carriage_m;
  plan.max_carriage_m = start_carriage_m;
  JointPose joints = start_joints;
  double carriage_m = start_carriage_m;
  double previous_carriage_velocity = 0.0;
  const CarriageKinematicState initial = evaluate_ballpoint(joints, carriage_m);
  const double start_error_m = norm(subtract(samples.front().position, initial.position));
  if (start_error_m > start_tolerance_m) {
    throw std::runtime_error(
            "path plan starts " + std::to_string(start_error_m * 1e3) +
            " mm from the current tip (tolerance " + std::to_string(start_tolerance_m * 1e3) +
            " mm)");
  }
  const double start_rotation_error = norm(orientation_error(
    initial.rotation, samples.front().rotation));
  if (start_rotation_error > 0.02) {
    throw std::runtime_error(
            "path plan starts " + std::to_string(start_rotation_error) +
            " rad from the current rotation (tolerance 0.02 rad)");
  }
  double previous_t = -period_s;

  for (size_t tick = 0; tick < samples.size(); ++tick) {
    const PathSample & sample = samples[tick];
    if (!std::isfinite(sample.t_s) || sample.t_s < previous_t) {
      throw std::runtime_error("path plan sample times must be finite and non-decreasing");
    }
    previous_t = sample.t_s;
    const Vec3 & reference = sample.position;
    const Vec3 & feedforward = sample.velocity;
    if (tick > 0) {
      plan.path_length_m += norm(subtract(reference, samples[tick - 1].position));
    }
    plan.max_cartesian_velocity_m_s = std::max(
      plan.max_cartesian_velocity_m_s, norm(feedforward));

    const CarriageKinematicState state = evaluate_ballpoint(joints, carriage_m);
    const Vec3 rotation_error = orientation_error(state.rotation, sample.rotation);
    // Angular feedforward from the next sample's rotation (small-angle rotation
    // vector over one tick). Without it a moving rotation target lags the
    // proportional loop by omega / K and trips the 1 mrad cap on the orbit's
    // tilts. Exactly zero for a constant rotation, so the spiral is untouched.
    Vec3 omega_ff{};
    if (tick + 1 < samples.size()) {
      const Vec3 step = orientation_error(sample.rotation, samples[tick + 1].rotation);
      for (size_t axis = 0; axis < 3; ++axis) {omega_ff[axis] = step[axis] / period_s;}
    }
    std::array<double, 6> twist{};
    for (size_t axis = 0; axis < 3; ++axis) {
      twist[axis] = feedforward[axis] +
        POSITION_ERROR_GAIN_S * (reference[axis] - state.position[axis]);
      twist[axis + 3] = omega_ff[axis] + ORIENTATION_ERROR_GAIN_S * rotation_error[axis];
    }
    FullJointPose velocity{};
    if (sample.pen && carriage_ik) {
      velocity = weighted_carriage_dls(
        state.jacobian, twist,
        CARRIAGE_CENTER_GAIN_S * (CARRIAGE_IK_BIAS_M - carriage_m));
      // Slew-limit the carriage: at the pen-up -> pen-down handover the weighted
      // solve would hand the carriage its whole share of the accumulated position
      // correction in one tick (first live bottle path: -26 mm/s^2 against the
      // 20 mm/s^2 cap, 0.9 s into the spiral). Hold it to 90 % of the cap and
      // let the arm take the remainder; the caps below still apply unchanged,
      // and the flat spiral never engages this (peak 0.43 mm/s^2).
      // The same clamp bounds its speed: drawing on a 40 mm cylinder turns the
      // wrist at ~0.7 deg/s, and cancelling that rotation's 200 mm lever arm at
      // the tip would otherwise recruit the carriage past its 1 mm/s cap while
      // the tip itself moves at 0.5 mm/s. The arm cancels it instead.
      const double max_step = 0.9 * PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 * period_s;
      const double max_speed = 0.9 * PLAN_MAX_CARRIAGE_VELOCITY_M_S;
      const double slewed = std::clamp(
        std::clamp(velocity[6], -max_speed, max_speed),
        previous_carriage_velocity - max_step, previous_carriage_velocity + max_step);
      if (slewed != velocity[6]) {
        Mat6 arm_jacobian{};
        std::array<double, 6> arm_twist{};
        for (size_t axis = 0; axis < 6; ++axis) {
          for (size_t joint = 0; joint < 6; ++joint) {
            arm_jacobian[axis][joint] = state.jacobian[axis][joint];
          }
          arm_twist[axis] = twist[axis] - state.jacobian[axis][6] * slewed;
        }
        const JointPose arm_velocity = damped_least_squares(arm_jacobian, arm_twist);
        for (size_t joint = 0; joint < 6; ++joint) {velocity[joint] = arm_velocity[joint];}
        velocity[6] = slewed;
      }
    } else {
      // Pen up, or a path that keeps the carriage out of the drawing solve
      // (carriage_ik 0 -- every draw session: on a curved surface the wrist
      // turns to follow the normal and the weighted solve walked the carriage
      // out of its envelope within 25 s on the first live bottle path; the
      // paper-validated seven-joint spiral keeps carriage_ik 1):
      // the carriage is not a drawing DOF. The weighted solve hands it
      // most of any motion along the tool axis, and a 100 mm lift would walk
      // it into its 3.5 mm stop, so bring it to rest at half its acceleration
      // cap and hold it; the arm takes the whole task minus what the carriage
      // still contributes. Mirrors draw_kinematics.plan_joints.
      const double step = 0.5 * PLAN_MAX_CARRIAGE_ACCELERATION_M_S2 * period_s;
      const double held = previous_carriage_velocity -
        std::clamp(previous_carriage_velocity, -step, step);
      Mat6 arm_jacobian{};
      std::array<double, 6> arm_twist{};
      for (size_t axis = 0; axis < 6; ++axis) {
        for (size_t joint = 0; joint < 6; ++joint) {
          arm_jacobian[axis][joint] = state.jacobian[axis][joint];
        }
        arm_twist[axis] = twist[axis] - state.jacobian[axis][6] * held;
      }
      const JointPose arm_velocity = damped_least_squares(arm_jacobian, arm_twist);
      for (size_t joint = 0; joint < 6; ++joint) {velocity[joint] = arm_velocity[joint];}
      velocity[6] = held;
    }
    for (size_t joint = 0; joint < joints.size(); ++joint) {
      plan.max_joint_velocity_rad_s = std::max(
        plan.max_joint_velocity_rad_s, std::fabs(velocity[joint]));
      if (!std::isfinite(velocity[joint]) ||
        std::fabs(velocity[joint]) > PLAN_MAX_JOINT_VELOCITY_RAD_S)
      {
        throw std::runtime_error(
                "path plan exceeds its arm velocity cap on joint " +
                std::to_string(joint) + " at sample " + std::to_string(tick));
      }
      joints[joint] += period_s * velocity[joint];
    }
    const double carriage_velocity = velocity[6];
    plan.max_carriage_velocity_m_s = std::max(
      plan.max_carriage_velocity_m_s, std::fabs(carriage_velocity));
    if (!std::isfinite(carriage_velocity) ||
      std::fabs(carriage_velocity) > PLAN_MAX_CARRIAGE_VELOCITY_M_S)
    {
      throw std::runtime_error(
              "path plan exceeds its carriage velocity cap at sample " + std::to_string(tick));
    }
    const double carriage_acceleration =
      (carriage_velocity - previous_carriage_velocity) / period_s;
    plan.max_carriage_acceleration_m_s2 = std::max(
      plan.max_carriage_acceleration_m_s2, std::fabs(carriage_acceleration));
    if (!std::isfinite(carriage_acceleration) ||
      std::fabs(carriage_acceleration) > PLAN_MAX_CARRIAGE_ACCELERATION_M_S2)
    {
      throw std::runtime_error(
              "path plan exceeds its carriage acceleration cap at sample " + std::to_string(tick));
    }
    carriage_m += period_s * carriage_velocity;
    previous_carriage_velocity = carriage_velocity;
    if (!std::isfinite(carriage_m) ||
      carriage_m < CARRIAGE_IK_MIN_M || carriage_m > CARRIAGE_IK_MAX_M)
    {
      throw std::runtime_error(
              "path plan leaves its guarded carriage envelope at sample " + std::to_string(tick));
    }
    plan.min_carriage_m = std::min(plan.min_carriage_m, carriage_m);
    plan.max_carriage_m = std::max(plan.max_carriage_m, carriage_m);
    check_joint_limits(joints);

    const CarriageKinematicState integrated = evaluate_ballpoint(joints, carriage_m);
    const double model_error_m = norm(subtract(reference, integrated.position));
    plan.max_model_error_mm = std::max(plan.max_model_error_mm, model_error_m * 1000.0);
    if (model_error_m > (sample.pen ? PLAN_MAX_MODEL_ERROR_DRAW_M : PLAN_MAX_MODEL_ERROR_PEN_UP_M)) {
      throw std::runtime_error(
              "path plan exceeds its Cartesian model-error cap at sample " + std::to_string(tick));
    }
    const double orientation_error_rad = norm(orientation_error(
      integrated.rotation, sample.rotation));
    plan.max_orientation_error_rad = std::max(
      plan.max_orientation_error_rad, orientation_error_rad);
    if (orientation_error_rad > PLAN_MAX_ORIENTATION_ERROR_RAD) {
      throw std::runtime_error(
              "path plan exceeds its orientation-error cap at sample " + std::to_string(tick));
    }
    FullJointPose positions{};
    for (size_t joint = 0; joint < joints.size(); ++joint) {
      positions[joint] = joints[joint];
    }
    positions[6] = carriage_m;
    plan.positions.push_back(positions);
    plan.velocities.push_back(velocity);
    plan.cartesian_references.push_back(reference);
    if (sample.capture > 0) {
      plan.capture_ticks.emplace_back(tick, sample.capture);
    }
  }

  const CarriageKinematicState endpoint = evaluate_ballpoint(joints, carriage_m);
  if (norm(subtract(samples.back().position, endpoint.position)) > PLAN_MAX_MODEL_ERROR_M) {
    throw std::runtime_error("path plan endpoint does not converge");
  }
  plan.endpoint_tick = plan.positions.size();
  return plan;
}

CarriageJointPlan plan_joint_spiral_with_carriage(
  const JointPose & start_joints,
  double start_carriage_m,
  double radius_m,
  double turns,
  double duration_s,
  double ease_s,
  double period_s)
{
  if (!std::isfinite(start_carriage_m) ||
    start_carriage_m < CARRIAGE_IK_MIN_M || start_carriage_m > CARRIAGE_IK_MAX_M)
  {
    throw std::invalid_argument("carriage-IK start is outside its guarded drawing envelope");
  }
  check_joint_limits(start_joints);
  const CarriageKinematicState initial = evaluate_ballpoint(start_joints, start_carriage_m);
  const auto samples = spiral_path_samples(
    initial.position, initial.rotation, radius_m, turns, duration_s, ease_s, period_s);
  CarriageJointPlan plan = plan_joint_path(start_joints, start_carriage_m, samples, period_s);
  // The spiral's path length is the closed-form arc length, as the A/B reported it.
  plan.path_length_m = spiral_path_length(radius_m, turns);
  const Vec3 target{
    initial.position[0] + radius_m, initial.position[1], initial.position[2]};
  JointPose end_joints{};
  std::copy_n(plan.positions.back().begin(), 6, end_joints.begin());
  const auto end_tip = evaluate_ballpoint(end_joints, plan.positions.back()[6]).position;
  if (norm(subtract(target, end_tip)) > PLAN_MAX_MODEL_ERROR_M) {
    throw std::runtime_error("carriage-IK spiral endpoint does not converge");
  }
  return plan;
}

MotionGuard::MotionGuard(
  double velocity_limit,
  double overforce_limit,
  double overforce_window_s,
  double overforce_fraction,
  size_t overforce_min_samples)
: velocity_limit_(velocity_limit),
  overforce_limit_(overforce_limit),
  overforce_window_s_(overforce_window_s),
  overforce_fraction_(overforce_fraction),
  overforce_min_samples_(overforce_min_samples)
{
}

void MotionGuard::reset()
{
  overforce_.clear();
}

std::optional<GuardTrip> MotionGuard::observe(
  double now_s,
  const std::vector<double> & arm_velocities,
  const std::vector<double> & arm_efforts)
{
  if (arm_velocities.empty() || arm_velocities.size() != arm_efforts.size()) {
    return GuardTrip{"telemetry_width", 0, static_cast<double>(arm_velocities.size()),
      static_cast<double>(arm_efforts.size())};
  }
  for (size_t i = 0; i < arm_velocities.size(); ++i) {
    if (!std::isfinite(arm_velocities[i]) || !std::isfinite(arm_efforts[i])) {
      return GuardTrip{"non_finite_telemetry", i, arm_velocities[i], 0.0};
    }
  }

  const auto fastest = std::max_element(
    arm_velocities.begin(), arm_velocities.end(),
    [](double a, double b) {return std::fabs(a) < std::fabs(b);});
  if (velocity_limit_ > 0.0 && std::fabs(*fastest) > velocity_limit_) {
    return GuardTrip{"measured_velocity",
      static_cast<size_t>(std::distance(arm_velocities.begin(), fastest)),
      *fastest, velocity_limit_};
  }

  const auto loaded = std::max_element(
    arm_efforts.begin(), arm_efforts.end(),
    [](double a, double b) {return std::fabs(a) < std::fabs(b);});
  overforce_.emplace_back(now_s, std::fabs(*loaded) > overforce_limit_);
  while (!overforce_.empty() && overforce_.front().first < now_s - overforce_window_s_) {
    overforce_.pop_front();
  }
  const bool ready = overforce_.size() >= overforce_min_samples_ &&
    now_s - overforce_.front().first >= overforce_window_s_ * 0.8;
  if (overforce_limit_ > 0.0 && ready) {
    const size_t over = static_cast<size_t>(std::count_if(
      overforce_.begin(), overforce_.end(), [](const auto & sample) {return sample.second;}));
    const double fraction = static_cast<double>(over) / static_cast<double>(overforce_.size());
    if (fraction >= overforce_fraction_) {
      return GuardTrip{"rolling_overforce",
        static_cast<size_t>(std::distance(arm_efforts.begin(), loaded)),
        *loaded, overforce_limit_};
    }
  }
  return std::nullopt;
}

}  // namespace tatbot::square
