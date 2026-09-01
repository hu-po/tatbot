// Emergency EEPROM network fix: connect to an arm at its current IP and
// write a new manual IP (+ sane LAN gateway/dns/subnet), then reboot the
// controller so it takes effect.
//   arm_set_ip <current_ip> <new_ip>
#include <cstdlib>
#include <iostream>
#include "libtrossen_arm/trossen_arm.hpp"
int main(int argc, char ** argv)
{
  if (argc != 3) { std::cerr << "usage: arm_set_ip <current_ip> <new_ip>\n"; return 1; }
  try {
    trossen_arm::TrossenArmDriver driver;
    driver.configure(
      trossen_arm::Model::wxai_v0,
      trossen_arm::StandardEndEffector::wxai_v0_base,
      argv[1], true, 10.0);
    driver.set_ip_method(trossen_arm::IPMethod::manual);
    driver.set_manual_ip(argv[2]);
    // Gateway/DNS for the arm's LAN: from env, defaulting to nothing baked
    // in beyond the RFC 1918 gateway convention the deployment states.
    const char * gw = std::getenv("TATBOT_ARM_GATEWAY");
    const char * dns = std::getenv("TATBOT_ARM_DNS");
    if (!gw || !*gw) {
      std::cerr << "arm_set_ip: set TATBOT_ARM_GATEWAY (and optionally "
                   "TATBOT_ARM_DNS) for your LAN before rewriting EEPROM"
                << std::endl;
      return 2;
    }
    driver.set_gateway(gw);
    driver.set_dns((dns && *dns) ? dns : "8.8.8.8");
    driver.set_subnet("255.255.255.0");
    std::cout << "EEPROM now: ip=" << driver.get_manual_ip()
              << " gw=" << driver.get_gateway()
              << " subnet=" << driver.get_subnet() << std::endl;
    driver.reboot_controller();  // cleanup(true): apply on next boot
    std::cout << "controller rebooting with new address " << argv[2] << std::endl;
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
