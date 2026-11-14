// The xArm report parser against frames recorded from the arm (firmware 2.7.0, 2026-11-08: motors off at its
// rest, state 4): xarm_report_30000.bin holds the real-time stream's first 12 frames, xarm_report_30001.bin the
// normal stream's first 3, as the sockets delivered them.
#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "tatbot_hardware/xarm_report.hpp"

using tatbot_hardware::xarm::decode_normal;
using tatbot_hardware::xarm::decode_realtime;
using tatbot_hardware::xarm::FrameReader;

namespace
{

std::vector<uint8_t> fixture(const std::string & name)
{
  std::ifstream in(std::string(XARM_FIXTURES) + "/" + name, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Every frame of `bytes`, fed `chunk` bytes at a time as a socket may split them.
std::vector<std::vector<uint8_t>> frames(const std::vector<uint8_t> & bytes, size_t chunk)
{
  FrameReader reader;
  std::vector<std::vector<uint8_t>> out;
  for (size_t at = 0; at < bytes.size(); at += chunk) {
    reader.feed(bytes.data() + at, std::min(chunk, bytes.size() - at));
    while (auto frame = reader.next()) {out.push_back(*frame);}
  }
  EXPECT_FALSE(reader.corrupt());
  return out;
}

}  // namespace

TEST(XArmReport, TheRealtimeStreamSplitsIntoItsFramesHoweverTheSocketCutsIt)
{
  const auto bytes = fixture("xarm_report_30000.bin");
  for (size_t chunk : {1, 7, 784, 1000, 9408}) {
    const auto split = frames(bytes, chunk);
    ASSERT_EQ(split.size(), 12u) << chunk;
    for (const auto & frame : split) {EXPECT_EQ(frame.size(), 784u);}
  }
}

TEST(XArmReport, ARealtimeFrameCarriesTheArmAtRestWithItsMotorsOff)
{
  const auto split = frames(fixture("xarm_report_30000.bin"), 784);
  const auto first = decode_realtime(split.front());
  const auto last = decode_realtime(split.back());
  ASSERT_TRUE(first && last);
  EXPECT_EQ(first->state, tatbot_hardware::xarm::kStateStopped);
  EXPECT_EQ(first->mode, 0);
  EXPECT_EQ(first->timestamp_us, 180866826u);
  EXPECT_EQ(last->timestamp_us - first->timestamp_us, 44154u);   // 11 periods of ~4 ms: about 250 frames a second
  const double q[7] = {0.0010200971737504005, -0.24074870347976685, -0.00481286458671093, 0.9012117981910706,
    -5.7524280236975756e-06, 1.1563951969146729, 1.1504856047395151e-05};
  for (size_t j = 0; j < 7; ++j) {
    EXPECT_FLOAT_EQ(first->q[j], q[j]) << j;
    EXPECT_FLOAT_EQ(first->target_q[j], q[j]) << j;   // stopped: the target is where it stands
    EXPECT_EQ(first->qd[j], 0.0) << j;
  }
  EXPECT_FLOAT_EQ(first->torque[5], -1.7057288885116577);
}

TEST(XArmReport, TheNormalFrameCarriesTheControllersErrorsMotorsAndStoredToolSettings)
{
  const auto split = frames(fixture("xarm_report_30001.bin"), 50);
  ASSERT_EQ(split.size(), 3u);
  const auto normal = decode_normal(split.front());
  ASSERT_TRUE(normal);
  EXPECT_EQ(normal->state, tatbot_hardware::xarm::kStateStopped);
  EXPECT_EQ(normal->error_code, 0);
  EXPECT_EQ(normal->warn_code, 0);
  EXPECT_EQ(normal->enables, 0);   // motors off
  EXPECT_EQ(normal->collision_sensitivity, 3);
  // What the arm had stored: a 172 mm TCP and a 0.95 kg load 60 mm out, which connect() replaces.
  EXPECT_FLOAT_EQ(normal->tcp_offset[2], 172.0);
  EXPECT_FLOAT_EQ(normal->tcp_load[0], 0.95);
  EXPECT_FLOAT_EQ(normal->tcp_load[3], 60.0);
  EXPECT_FLOAT_EQ(normal->q[3], 0.9012117981910706);
}

TEST(XArmReport, ALengthOutOfRangeMarksTheStreamOutOfStep)
{
  FrameReader reader;
  const uint8_t junk[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
  reader.feed(junk, sizeof(junk));
  EXPECT_FALSE(reader.next());
  EXPECT_TRUE(reader.corrupt());
  const auto bytes = fixture("xarm_report_30000.bin");
  reader.feed(bytes.data(), bytes.size());
  EXPECT_FALSE(reader.next());   // nothing more until the caller reconnects
}

TEST(XArmReport, AShortFrameDecodesToNothing)
{
  const std::vector<uint8_t> short_frame(100, 0);
  EXPECT_FALSE(decode_realtime(short_frame));
  EXPECT_FALSE(decode_normal(short_frame));
}
