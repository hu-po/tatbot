#pragma once
// tatbot_hardware/TatbotArm: the thin ros2_control adapter around ArmCore for one Trossen WXAI arm.
// It owns the backend (SDK or fake), the shared e-stop reader and driver lock, the 10 Hz aux
// thread for TCP getters, the flight recorder and the <arm>_safety GPIO. ros/README.md section 13.
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_component_interface_params.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "tatbot_hardware/backend.hpp"
#include "tatbot_hardware/core.hpp"
#include "tatbot_hardware/estop.hpp"
#include "tatbot_hardware/runtime.hpp"

namespace tatbot_hardware
{

// base_frame -> tcp_frame FK over the URDF (KDL), mapping chain joints onto the 7 arm joints
// named in joint_names; empty (with why set) when the chain is missing.
TipFk make_tip_fk(
  const std::string & urdf, const std::string & base_frame, const std::string & tcp_frame,
  const std::vector<std::string> & joint_names, std::string & why);

class TatbotArm : public hardware_interface::SystemInterface
{
public:
  using CallbackReturn = hardware_interface::CallbackReturn;
  using State = rclcpp_lifecycle::State;
  ~TatbotArm() override;

  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  CallbackReturn on_configure(const State & previous) override;
  CallbackReturn on_activate(const State & previous) override;
  CallbackReturn on_deactivate(const State & previous) override;
  CallbackReturn on_cleanup(const State & previous) override;
  CallbackReturn on_shutdown(const State & previous) override;
  CallbackReturn on_error(const State & previous) override;
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  FakeBackend * fake() const {return fake_;}   // sdk=fake only (tests)
  const ArmCore * core() const {return core_.get();}

private:
  void send(const Output & out);
  void hold(int reason);
  void release();
  void publish_safety();

  std::string arm_, sdk_, ip_, end_effector_, flight_path_;
  int rt_priority_ = 80;
  std::vector<int> rt_cpus_;
  estop::Settings estop_settings_;
  estop::ProbeSettings probe_settings_;
  bool probe_enabled_ = false;
  Config cfg_;
  TipFk fk_;
  double fake_page_z_ = 0, fake_page_stiffness_ = 0;
  double carriage_min_m_ = -0.006, carriage_max_m_ = 0.040;   // config/trossen/follower.yaml
  double controller_query_period_s_ = 1.0;
  bool fake_page_ = false;
  std::array<size_t, kJoints> index_{};   // info_.joints[i] is core joint index_[i]

  std::unique_ptr<ArmCore> core_;
  std::unique_ptr<Backend> backend_;
  FakeBackend * fake_ = nullptr;
  std::shared_ptr<estop::Reader> estop_;
  std::shared_ptr<estop::ProbeReader> probe_;
  std::shared_ptr<runtime::DriverLock> lock_;
  std::unique_ptr<runtime::FlightRecorder> recorder_;

  // The 10 Hz TCP getter thread's state, shared with it: a thread stuck in a blocking SDK call
  // (the controller gone) is detached by release() and keeps its state and the backend alive.
  struct Aux
  {
    std::timed_mutex tcp;
    std::atomic<bool> stop{false}, done{false}, idle_request{false}, error{false};
  };
  std::shared_ptr<Aux> aux_ = std::make_shared<Aux>();
  std::thread aux_thread_;
  std::atomic<bool> sticky_error_{false};

  Feedback fb_;
  Output last_out_;
  std::vector<double> position_, velocity_, effort_, position_command_, velocity_command_;
  std::vector<double> safety_state_, safety_command_;
  double last_read_ = -1, window_start_ = 0, window_max_ = 0, previous_max_ = 0, period_max_ms_ = 0;
};

}  // namespace tatbot_hardware
