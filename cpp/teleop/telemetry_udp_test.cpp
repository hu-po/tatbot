#include "telemetry_udp.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << "telemetry_udp_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main()
{
  const int receiver = socket(AF_INET, SOCK_DGRAM, 0);
  require(receiver >= 0, "socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  require(bind(receiver, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
          "bind failed");
  socklen_t address_size = sizeof(address);
  require(getsockname(receiver, reinterpret_cast<sockaddr *>(&address), &address_size) == 0,
          "getsockname failed");

  tatbot::telemetry::UdpPublisher publisher(
    "127.0.0.1:" + std::to_string(ntohs(address.sin_port)), 120.0);
  publisher.publish(1, 0, {1.0}, {2.0, 3.0}, {4.0}, {5.0});
  require(publisher.stats().send_errors == 1, "invalid vectors were not rejected");
  publisher.publish(123456789, 7, {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0});

  pollfd ready{receiver, POLLIN, 0};
  require(poll(&ready, 1, 500) == 1, "datagram was not received");
  char buffer[2048]{};
  const ssize_t size = recv(receiver, buffer, sizeof(buffer), 0);
  require(size > 0, "empty datagram");
  const std::string message(buffer, static_cast<size_t>(size));
  require(message.find("\"magic\":\"tatbot-teleop-joints\"") != std::string::npos,
          "magic missing");
  require(message.find("\"timestamp_ns\":123456789") != std::string::npos,
          "timestamp missing");
  require(message.find("\"leader_pos\":[1,2]") != std::string::npos,
          "leader positions missing");
  require(message.find("\"follower_eff\":[7,8]") != std::string::npos,
          "follower effort missing");
  require(publisher.stats().sent == 1, "publisher did not count the datagram");
  close(receiver);
}
