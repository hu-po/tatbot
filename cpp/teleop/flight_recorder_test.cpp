#include "flight_recorder.hpp"

#include <unistd.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "flight_recorder_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main()
{
  const auto path = std::filesystem::temp_directory_path() /
    ("tatbot-flight-recorder-test-" + std::to_string(getpid()) + ".bin");
  std::filesystem::remove(path);
  const std::array<unsigned char, 4> header{1, 2, 3, 4};
  const std::array<unsigned char, 3> record{5, 6, 7};
  tatbot::flight::Stats stats;
  {
    tatbot::flight::Recorder recorder(path.string());
    require(recorder.write_header(header.data(), header.size()), "header was not queued");
    for (int i = 0; i < 100; ++i) {
      require(recorder.append_record(record.data(), record.size()), "record was not queued");
    }
    stats = recorder.finish();
    require(recorder.finish().records_enqueued == 100, "finish was not idempotent");
  }
  require(stats.records_enqueued == 100, "wrong enqueue count");
  require(stats.records_dropped == 0, "records unexpectedly dropped");
  require(stats.write_errors == 0, "writer reported an error");

  std::ifstream input(path, std::ios::binary);
  const std::vector<unsigned char> bytes(
    (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  require(bytes.size() == header.size() + 100 * record.size(), "wrong file length");
  require(std::equal(header.begin(), header.end(), bytes.begin()), "header changed");
  for (size_t offset = header.size(); offset < bytes.size(); offset += record.size()) {
    require(std::equal(record.begin(), record.end(), bytes.begin() + offset), "record changed");
  }
  std::filesystem::remove(path);
  std::cout << "flight_recorder_test: ok" << std::endl;
}
