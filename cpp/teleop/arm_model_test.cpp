// arm_model.hpp against the real SDK with synthetic workspace inputs. Also
// check the deployment's workspace when it exists, without requiring it in an export.
#include "arm_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
int failures = 0;
void check(bool ok, const char * what)
{
  if (!ok) {
    std::cerr << "FAIL: " << what << std::endl;
    ++failures;
  }
}
}  // namespace

void check_palms(const std::filesystem::path & repo)
{
  for (const char * arm : {"left", "right"}) {
    const auto payload = arm_model::scrape_payload(repo / "config" / "workspace.yaml", arm);
    std::string note;
    const auto ee = arm_model::end_effector(repo, arm, &note);
    std::cout << arm << ": " << note << std::endl;
    check(payload.identified_with == payload.tool_id, "the payload belongs to its stated tool");
    check(ee.palm.mass == payload.mass_kg && ee.palm.origin_xyz == payload.com_m, "the palm is the payload's");
    check(ee.palm.inertia == arm_model::preset(arm).palm.inertia, "the rest of the palm is the preset's");
    check(ee.t_flange_tool == arm_model::preset(arm).t_flange_tool, "the tool frame is the preset's");
  }
}

int main()
{
  auto pattern = (std::filesystem::temp_directory_path() / "arm_model_test_XXXXXX").string();
  const auto created = mkdtemp(pattern.data());
  if (!created) {std::perror("mkdtemp"); return 1;}
  const std::filesystem::path dir = created;
  std::filesystem::create_directories(dir / "config");
  std::ofstream(dir / "config" / "workspace.yaml") <<
    "left:\n  tool_id: test-left\n  payload:\n    identified_with: test-left\n"
    "    preset: wxai_v0_leader_20250509\n    palm_mass_kg: 1.234\n    palm_com_m: [0.01, -0.02, 0.03]\n"
    "right:\n  tool_id: test-right\n  payload:\n    identified_with: test-right\n"
    "    preset: wxai_v0_follower_20250509\n    palm_mass_kg: 1.567\n    palm_com_m: [-0.02, 0.03, 0.01]\n";
  check_palms(dir);
  const std::filesystem::path repo = TATBOT_REPO_ROOT;
  if (std::filesystem::is_regular_file(repo / "config" / "workspace.yaml")) {
    check_palms(repo);
  }
  check(arm_model::preset("left").palm.mass == trossen_arm::StandardEndEffector::wxai_v0_leader_20250509.palm.mass,
    "the left arm's preset is the 2025-05-09 leader");

  // A payload identified with another tool leaves the preset's palm.
  std::ofstream(dir / "config" / "workspace.yaml") <<
    "right:\n  tool_id: pen-b\n  payload:   # note\n    identified_with: pen-a\n"
    "    preset: wxai_v0_follower_20250509\n    palm_mass_kg: 1.5\n    palm_com_m: [0.01, 0.02, 0.03]\n";
  std::string note;
  const auto stock = arm_model::end_effector(dir, "right", &note);
  check(stock.palm.mass == arm_model::preset("right").palm.mass, "another tool's payload is not applied");
  check(note.find("pen-a") != std::string::npos, "and the note says why");
  const auto scraped = arm_model::scrape_payload(dir / "config" / "workspace.yaml", "right");
  check(scraped.com_m[2] == 0.03 && scraped.mass_kg == 1.5, "the payload block scrapes");
  std::filesystem::remove_all(dir);

  check(arm_model::gains()[1].at(trossen_arm::Mode::position).position.kp == 120.0, "the 2025-05-09 gains");
  if (failures) {return 1;}
  std::cout << "arm_model_test: ok" << std::endl;
  return 0;
}
