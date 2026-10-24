#pragma once
// Each physical arm's controller model for the C++ tools, by the rule the ros stack
// (tatbot_description.hardware_params) applies:
//
// - the dated Trossen preset the arms have always run with (driver 1.11 moved the undated follower
//   name to its 2026-06-26 model), per PHYSICAL arm: left the leader's, right the follower's, whatever
//   teleop role either plays;
// - its palm replaced by config/workspace.yaml `<arm>.payload` when that load was identified with the
//   tool the calibration names (`<arm>.tool_id`): the stock palm left 3-4 Nm unmodelled at both
//   elbows (2026-10-07);
// - Trossen's 2025-05-09 position-mode gains, which firmware 1.10+ no longer boots with (the goldens
//   hold the same numbers; a tool that loads no golden sets them by name).
#include <libtrossen_arm/trossen_arm.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace arm_model
{

inline const trossen_arm::EndEffector & preset(const std::string & physical_arm)
{
  return physical_arm == "left" ? trossen_arm::StandardEndEffector::wxai_v0_leader_20250509 :
         trossen_arm::StandardEndEffector::wxai_v0_follower_20250509;
}

inline const std::string preset_name(const std::string & physical_arm)
{
  return physical_arm == "left" ? "wxai_v0_leader_20250509" : "wxai_v0_follower_20250509";
}

struct Payload
{
  std::string tool_id, identified_with, preset;
  double mass_kg = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 3> com_m{};
};

inline std::string scalar(std::string text)
{
  const auto hash = text.find('#');
  if (hash != std::string::npos) {text = text.substr(0, hash);}
  const auto first = text.find_first_not_of(" \t\r\n\"'");
  if (first == std::string::npos) {return {};}
  return text.substr(first, text.find_last_not_of(" \t\r\n\"'") - first + 1);
}

// The `<arm>:` block's tool_id and payload, scraped line by line like the C++ tools' other reads of
// workspace.yaml (two-space keys under the arm, four-space keys under `payload:`).
inline Payload scrape_payload(const std::filesystem::path & workspace, const std::string & physical_arm)
{
  Payload out;
  std::ifstream file(workspace);
  std::string line;
  bool in_arm = false, in_payload = false;
  while (std::getline(file, line)) {
    if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {continue;}
    const auto indent = line.find_first_not_of(' ');
    if (indent == 0) {
      in_arm = line.rfind(physical_arm + ":", 0) == 0;
      in_payload = false;
      continue;
    }
    if (!in_arm || line[indent] == '#') {continue;}
    const auto body = line.substr(indent);
    const auto colon = body.find(':');
    if (colon == std::string::npos) {continue;}
    const auto key = body.substr(0, colon), value = scalar(body.substr(colon + 1));
    if (indent == 2) {
      in_payload = key == "payload";
      if (key == "tool_id") {out.tool_id = value;}
    } else if (in_payload && indent == 4) {
      if (key == "identified_with") {out.identified_with = value;}
      if (key == "preset") {out.preset = value;}
      if (key == "palm_mass_kg") {out.mass_kg = std::stod(value);}
      if (key == "palm_com_m") {
        std::string list = value;
        for (char & c : list) {
          if (c == '[' || c == ']' || c == ',') {c = ' ';}
        }
        std::istringstream in(list);
        for (double & v : out.com_m) {in >> v;}
      }
    }
  }
  return out;
}

// The arm's end effector; `note` (if given) says which palm it carries.
inline trossen_arm::EndEffector end_effector(
  const std::filesystem::path & repo, const std::string & physical_arm, std::string * note = nullptr)
{
  trossen_arm::EndEffector ee = preset(physical_arm);
  std::ostringstream said;
  said << preset_name(physical_arm);
  const auto payload = repo.empty() ? Payload{} : scrape_payload(repo / "config" / "workspace.yaml", physical_arm);
  if (std::isfinite(payload.mass_kg) && !payload.tool_id.empty() && payload.identified_with == payload.tool_id &&
    payload.preset == preset_name(physical_arm))
  {
    ee.palm.mass = payload.mass_kg;
    ee.palm.origin_xyz = payload.com_m;
    said << " with the palm identified with " << payload.tool_id << " (" << payload.mass_kg << " kg)";
  } else if (std::isfinite(payload.mass_kg)) {
    said << " (its payload was identified with " << payload.identified_with << ", not " << payload.tool_id
         << ": the preset's palm)";
  }
  if (note) {*note = said.str();}
  return ee;
}

inline const std::vector<std::map<trossen_arm::Mode, trossen_arm::MotorParameter>> & gains()
{
  return trossen_arm::StandardMotorParameters::wxai_v0_20250509;
}

}  // namespace arm_model
