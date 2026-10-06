// The ros2_control adapter with sdk=fake: parameters, interface export in URDF joint order,
// activation in position mode at the measured pose, velocity feed-forward, deactivate/error hold,
// the tip-lag guard with KDL FK from the robot description, the UDP e-stop, the flight log, and a
// landed arm idle until a restart.
#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>

#include "hardware_interface/component_parser.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tatbot_hardware/tatbot_arm.hpp"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

using namespace tatbot_hardware;
using namespace std::chrono_literals;

namespace
{
std::string urdf(const std::map<std::string, std::string> & params)
{
  std::string links, joints;
  const char * axes[] = {"0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"};
  std::string parent = "right/base_link";
  for (int i = 0; i < 6; ++i) {
    const std::string link = "right/link_" + std::to_string(i);
    links += "<link name=\"" + link + "\"/>";
    joints += "<joint name=\"right/joint_" + std::to_string(i) + "\" type=\"revolute\"><parent link=\"" +
      parent + "\"/><child link=\"" + link + "\"/><origin xyz=\"" + (i == 0 ? "0 0 0.1" : "0.06 0 0") +
      "\"/><axis xyz=\"" + axes[i] + "\"/><limit lower=\"-3\" upper=\"3\" effort=\"10\" velocity=\"3\"/></joint>";
    parent = link;
  }
  std::string hw;
  for (const auto & [k, v] : params) {hw += "<param name=\"" + k + "\">" + v + "</param>";}
  std::string gpio = "<gpio name=\"right_safety\">";
  for (auto n : names::kSafetyCommand) {gpio += "<command_interface name=\"" + std::string(n) + "\"/>";}
  for (auto n : names::kSafetyState) {gpio += "<state_interface name=\"" + std::string(n) + "\"/>";}
  gpio += "</gpio>";
  auto joint = [](const std::string & name) {
      return "<joint name=\"" + name + "\"><command_interface name=\"position\"/><command_interface name=\"velocity\"/>"
             "<state_interface name=\"position\"/><state_interface name=\"velocity\"/><state_interface name=\"effort\"/></joint>";
    };
  std::string control = joint("right/left_carriage_joint");   // not in controller order on purpose
  for (int i = 0; i < 6; ++i) {control += joint("right/joint_" + std::to_string(i));}
  return "<?xml version=\"1.0\"?><robot name=\"t\"><link name=\"world\"/><link name=\"right/base_link\"/>"
         "<joint name=\"right/world_joint\" type=\"fixed\"><parent link=\"world\"/><child link=\"right/base_link\"/></joint>" +
         links + joints +
         "<link name=\"right/tool_mount\"/><link name=\"right/tcp\"/>"
         "<joint name=\"right/left_carriage_joint\" type=\"prismatic\"><parent link=\"right/link_5\"/><child link=\"right/tool_mount\"/>"
         "<axis xyz=\"0 1 0\"/><limit lower=\"-0.006\" upper=\"0.040\" effort=\"100\" velocity=\"0.25\"/></joint>"
         "<joint name=\"right/tcp_joint\" type=\"fixed\"><parent link=\"right/tool_mount\"/><child link=\"right/tcp\"/>"
         "<origin xyz=\"0.1 0 0\" rpy=\"3.141592653589793 0 0\"/></joint>"   // tool points down
         "<ros2_control name=\"right_arm\" type=\"system\"><hardware><plugin>tatbot_hardware/TatbotArm</plugin>" + hw +
         "</hardware>" + control + gpio + "</ros2_control></robot>";
}

struct Plugin
{
  TatbotArm arm;
  std::map<std::string, hardware_interface::StateInterface *> state;
  std::map<std::string, hardware_interface::CommandInterface *> command;
  std::vector<hardware_interface::StateInterface> states;
  std::vector<hardware_interface::CommandInterface> commands;
  rclcpp_lifecycle::State none{lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "unconfigured"};

  explicit Plugin(std::map<std::string, std::string> params)
  {
    params.emplace("arm", "right");
    params.emplace("sdk", "fake");
    hardware_interface::HardwareComponentParams p;
    p.hardware_info = hardware_interface::parse_control_resources_from_urdf(urdf(params)).at(0);
    p.clock = std::make_shared<rclcpp::Clock>();
    EXPECT_EQ(arm.init(p), hardware_interface::CallbackReturn::SUCCESS);
    states = arm.export_state_interfaces();
    commands = arm.export_command_interfaces();
    for (auto & s : states) {state[s.get_name()] = &s;}
    for (auto & c : commands) {command[c.get_name()] = &c;}
  }
  double get(const std::string & name) {return state.at(name)->get_optional().value();}
  void set(const std::string & name, double v) {(void)command.at(name)->set_value(v);}
  void cycle(int n = 1)
  {
    for (int i = 0; i < n; ++i) {
      arm.read(rclcpp::Time(), rclcpp::Duration(0, 2500000));
      arm.write(rclcpp::Time(), rclcpp::Duration(0, 2500000));
      std::this_thread::sleep_for(2500us);
    }
  }
  void start()
  {
    // Activate straight after configure, as ros2_control_node does.
    ASSERT_EQ(arm.on_configure(none), hardware_interface::CallbackReturn::SUCCESS);
    ASSERT_EQ(arm.on_activate(none), hardware_interface::CallbackReturn::SUCCESS);
  }
};
}  // namespace

TEST(HwPlugin, ActivatesHoldingInPositionModeAndPassesVelocityFeedForward)
{
  Plugin p({{"staged_positions", "[0, 0.2, 0, 0, 0, 1.5707963267948966, 0.001]"}});
  EXPECT_EQ(p.states.size(), 7u * 3 + names::kSafetyState.size());
  EXPECT_EQ(p.commands.size(), 7u * 2 + names::kSafetyCommand.size());
  EXPECT_EQ(p.state.count("right_safety/rt_period_max_ms"), 1u);
  EXPECT_TRUE(p.arm.fake() == nullptr);
  p.start();
  ASSERT_NE(p.arm.fake(), nullptr);
  EXPECT_FALSE(p.arm.fake()->idle());
  // Command interfaces start at the measured pose; the carriage is exported first.
  EXPECT_DOUBLE_EQ(p.command.at("right/left_carriage_joint/position")->get_optional().value(), 0.001);
  EXPECT_DOUBLE_EQ(p.command.at("right/joint_1/position")->get_optional().value(), 0.2);
  p.cycle(10);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  EXPECT_EQ(p.get("right_safety/estop_ok"), 1);
  EXPECT_EQ(p.get("right_safety/estop_age_s"), -1);
  double q = 0.2;
  for (int i = 0; i < 100; ++i) {
    q += 0.1 * 0.0025;
    p.set("right/joint_1/position", q);
    p.set("right/joint_1/velocity", 0.1);
    p.set("right/left_carriage_joint/velocity", 0.0);
    p.cycle();
  }
  EXPECT_DOUBLE_EQ(p.arm.fake()->last_qd()[1], 0.1);
  EXPECT_NEAR(p.get("right/joint_1/position"), q, 2e-3);
  EXPECT_GT(p.get("right_safety/rt_period_max_ms"), 0.0);
  // Deactivate holds the measured pose, never idles.
  const double held = p.get("right/joint_1/position");
  p.arm.on_deactivate(p.none);
  EXPECT_EQ(p.get("right_safety/latched"), 1);
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchDeactivated);
  EXPECT_NEAR(p.arm.fake()->last_q()[1], held, 1e-9);
  EXPECT_DOUBLE_EQ(p.arm.fake()->last_qd()[1], 0.0);
  EXPECT_FALSE(p.arm.fake()->idle());
  p.arm.on_error(p.none);
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchDeactivated);   // first reason kept
  EXPECT_FALSE(p.arm.fake()->idle());
}

TEST(HwPlugin, ErrorHoldsAndTheAuxThreadCachesControllerErrors)
{
  Plugin p({});
  p.start();
  p.cycle(10);
  p.arm.fake()->inject_error("motor 2 overtemperature");
  std::this_thread::sleep_for(250ms);   // the 10 Hz aux thread
  p.cycle(2);
  EXPECT_EQ(p.get("right_safety/controller_error"), 1);
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchControllerError);
  EXPECT_FALSE(p.arm.fake()->idle());
}

TEST(HwPlugin, TipLagGuardUsesFkFromTheRobotDescription)
{
  // At the start pose the tip is ~0.1 m above base z with the tool (tcp +z) pointing straight
  // down; joint_1 (+y) swings it down, along the tool axis, onto the page.
  Plugin p({{"staged_positions", "0,0,0,0,0,0.001,0"}, {"fake_page_z", "0.098"}});
  p.start();
  p.cycle(4);
  p.set("right_safety/guard_mode", names::kGuardTipLag);
  double q = 0;
  for (int i = 0; i < 1200 && p.get("right_safety/latched") == 0; ++i) {
    q += 0.02 * 0.0025;   // ~5 mm/s at the tip
    p.set("right/joint_1/position", q);
    p.set("right/joint_1/velocity", 0.02);
    p.cycle();
  }
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchGuardTipLag);
  EXPECT_EQ(p.get("right_safety/guard_tripped"), 1);
  EXPECT_NEAR(p.get("right_safety/trip_q1"), p.get("right/joint_1/position"), 1e-3);
  EXPECT_GT(p.get("right_safety/trip_q1"), 0.005);
}

TEST(HwPlugin, UdpEstopHoldsUntilReleasedAndUnlatched)
{
  Plugin p({{"estop_source", "udp"}, {"estop_udp_port", "0"}, {"estop_relay_addr", "127.0.0.1"},
    {"flight_path", ::testing::TempDir() + "tatbot-flight"}});
  p.start();
  p.cycle(4);
  EXPECT_EQ(p.get("right_safety/estop_source"), names::kEstopUdp);
  EXPECT_EQ(p.get("right_safety/latched"), 1);
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchEstopStale);
  estop::Settings s;
  s.source = names::kEstopUdp;
  s.udp_port = 0;
  s.timeout_s = 0.15;
  s.relay_addr = "127.0.0.1";
  const int port = estop::shared(s)->bound_port();
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
  for (int seq = 0; seq < 5; ++seq) {
    const std::string f = "EST1 " + std::to_string(seq) + " 1\n";
    sendto(fd, f.data(), f.size(), 0, reinterpret_cast<sockaddr *>(&to), sizeof(to));
    p.cycle(4);
  }
  EXPECT_EQ(p.get("right_safety/estop_ok"), 1);
  EXPECT_EQ(p.get("right_safety/latched"), 1);   // release alone never unlatches
  p.set("right_safety/unlatch", 1);
  p.cycle();
  EXPECT_EQ(p.get("right_safety/unlatch_ack"), 1);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  close(fd);
  p.arm.on_cleanup(p.none);
  const auto path = std::filesystem::path(::testing::TempDir()) / "tatbot-flight" / "right-flight.bin";
  std::ifstream log(path, std::ios::binary);
  std::string header;
  std::getline(log, header);
  EXPECT_EQ(header.rfind("tatbot-flight 1 right ", 0), 0u);
  EXPECT_GT(std::filesystem::file_size(path), header.size() + 20 * 100);
}

TEST(HwPlugin, StartsUnlatchedWithEstopNoneWhenActivatedRightAfterConfigure)
{
  Plugin p({{"estop_source", "none"}});
  p.start();
  p.cycle(10);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  EXPECT_EQ(p.get("right_safety/latch_reason"), names::kLatchNone);
  // The seeded command is tracked: a small move follows without a Decide.
  const double q = p.get("right/joint_2/position") + 0.01;
  p.set("right/joint_2/position", q);
  p.cycle(100);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  EXPECT_NEAR(p.get("right/joint_2/position"), q, 1e-3);
}

TEST(HwPlugin, ShutdownBeforeLandingLeavesTheSessionHoldingNeverIdle)
{
  Plugin p({});
  p.start();
  p.cycle(10);
  const double q = p.get("right/joint_1/position") + 0.02;
  p.set("right/joint_1/position", q);
  p.cycle(100);
  const double held = p.get("right/joint_1/position");
  FakeBackend * fake = p.arm.fake();
  const int sent = fake->commands();
  p.arm.on_deactivate(p.none);
  p.arm.on_shutdown(p.none);   // SIGINT, `tatbot ros down`, a unit restart
  ASSERT_EQ(p.arm.fake(), fake);   // the session is left open, not closed
  EXPECT_FALSE(fake->idle());
  EXPECT_GT(fake->commands(), sent);
  EXPECT_NEAR(fake->last_q()[1], held, 1e-9);
  EXPECT_DOUBLE_EQ(fake->last_qd()[1], 0.0);
}

TEST(HwPlugin, ShutdownAfterLandingClosesTheSession)
{
  Plugin p({{"landing_takeover_s", "0.02"}, {"landing_staged_s", "0.1"}, {"landing_sleep_s", "0.1"}});
  p.start();
  p.cycle(10);
  p.set("right_safety/land", 1);
  for (int i = 0; i < 400 && p.get("right_safety/landed") == 0; ++i) {p.cycle();}
  ASSERT_EQ(p.get("right_safety/landed"), 1);
  std::this_thread::sleep_for(250ms);   // the aux thread idles the motors
  EXPECT_TRUE(p.arm.fake()->idle());
  p.arm.on_shutdown(p.none);
  EXPECT_EQ(p.arm.fake(), nullptr);   // landed and idle: the session closes
}

TEST(HwPlugin, ALandedArmIgnoresCommandsUntilItsHardwareAloneIsCycled)
{
  // What tatbot_session refuses a landed arm's goals on: the driver keeps it landed and idle whatever it is
  // commanded. A deactivate and activate of this arm's hardware alone (`client wake`, 2026-09-30: the other arm
  // keeps running) takes it back where it rests, and a first command from there is tracked.
  Plugin p({{"landing_takeover_s", "0.02"}, {"landing_staged_s", "0.1"}, {"landing_sleep_s", "0.1"}});
  p.start();
  p.cycle(10);
  p.set("right_safety/land", 1);
  for (int i = 0; i < 400 && p.get("right_safety/landed") == 0; ++i) {p.cycle();}
  ASSERT_EQ(p.get("right_safety/landed"), 1);
  std::this_thread::sleep_for(250ms);   // the aux thread idles the motors
  FakeBackend * fake = p.arm.fake();
  const int sent = fake->commands();
  const double rest = p.get("right/joint_1/position");
  auto stays_put = [&]() {
      p.set("right/joint_1/position", rest + 0.02);   // a trajectory controller's goal
      p.cycle(100);
      EXPECT_EQ(fake->commands(), sent);
      EXPECT_TRUE(fake->idle());
      EXPECT_DOUBLE_EQ(p.get("right/joint_1/position"), rest);
      EXPECT_EQ(p.get("right_safety/landed"), 1);
      EXPECT_EQ(p.get("right_safety/latched"), 0);
    };
  stays_put();
  p.arm.on_deactivate(p.none);
  ASSERT_EQ(p.arm.on_activate(p.none), hardware_interface::CallbackReturn::SUCCESS);
  p.cycle();
  EXPECT_EQ(p.get("right_safety/landed"), 0);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  EXPECT_FALSE(fake->idle());
  EXPECT_NEAR(p.get("right/joint_1/position"), rest, 1e-6);
  p.set("right/joint_1/position", rest + 0.002);   // the trajectory controller's first command, from the rest
  p.cycle(200);
  EXPECT_EQ(p.get("right_safety/latched"), 0);
  EXPECT_GT(fake->commands(), sent);
  EXPECT_NEAR(p.get("right/joint_1/position"), rest + 0.002, 5e-4);
}
