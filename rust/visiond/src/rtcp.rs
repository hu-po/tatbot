//! Minimal RTCP/RTP timestamp handling used to preserve camera time through
//! GStreamer depayloading.
//!
//! GStreamer can receive and use RTCP sender reports internally while still
//! dropping the resulting `GstReferenceTimestampMeta` at a depayloader
//! boundary. This module only parses the standards-defined fields needed to
//! reconstruct the sender-clock timestamp; media parsing remains GStreamer's
//! responsibility.

use std::collections::VecDeque;

const NTP_UNIX_EPOCH_SECONDS: u64 = 2_208_988_800;
const RTP_CLOCK_RATE: i128 = 90_000;
const MAX_SENDER_REPORT_DISCONTINUITY_NS: i128 = 500_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtcpSenderReport {
    pub ssrc: u32,
    pub ntp_unix_ns: i128,
    pub rtp_timestamp: u32,
}

/// Parse all RTCP Sender Reports in a compound packet.
pub fn parse_sender_reports(packet: &[u8]) -> Vec<RtcpSenderReport> {
    let mut reports = Vec::new();
    let mut offset = 0;
    while offset + 4 <= packet.len() {
        let first = packet[offset];
        if first >> 6 != 2 {
            break;
        }
        let packet_type = packet[offset + 1];
        let packet_length_words =
            u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]) as usize;
        let packet_length = (packet_length_words + 1).saturating_mul(4);
        if packet_length < 4 || offset + packet_length > packet.len() {
            break;
        }
        if packet_type == 200 && packet_length >= 28 {
            let base = offset;
            let ssrc = u32::from_be_bytes([
                packet[base + 4],
                packet[base + 5],
                packet[base + 6],
                packet[base + 7],
            ]);
            let ntp_seconds = u32::from_be_bytes([
                packet[base + 8],
                packet[base + 9],
                packet[base + 10],
                packet[base + 11],
            ]) as u64;
            let ntp_fraction = u32::from_be_bytes([
                packet[base + 12],
                packet[base + 13],
                packet[base + 14],
                packet[base + 15],
            ]) as u128;
            let rtp_timestamp = u32::from_be_bytes([
                packet[base + 16],
                packet[base + 17],
                packet[base + 18],
                packet[base + 19],
            ]);
            if let Some(ntp_unix_ns) = ntp_to_unix_ns(ntp_seconds, ntp_fraction) {
                reports.push(RtcpSenderReport {
                    ssrc,
                    ntp_unix_ns,
                    rtp_timestamp,
                });
            }
        }
        offset += packet_length;
    }
    reports
}

fn ntp_to_unix_ns(seconds: u64, fraction: u128) -> Option<i128> {
    if seconds < NTP_UNIX_EPOCH_SECONDS {
        return None;
    }
    let whole_seconds = (seconds - NTP_UNIX_EPOCH_SECONDS) as i128;
    let fractional_ns = ((fraction * 1_000_000_000) >> 32) as i128;
    Some(whole_seconds.saturating_mul(1_000_000_000) + fractional_ns)
}

/// Parse the fixed RTP header fields needed for timestamp association.
pub fn parse_rtp_header(packet: &[u8]) -> Option<(u32, u32)> {
    if packet.len() < 12 || packet[0] >> 6 != 2 {
        return None;
    }
    let csrc_count = (packet[0] & 0x0f) as usize;
    let has_extension = packet[0] & 0x10 != 0;
    let mut header_length = 12 + csrc_count.saturating_mul(4);
    if header_length > packet.len() {
        return None;
    }
    if has_extension {
        if header_length + 4 > packet.len() {
            return None;
        }
        let extension_words =
            u16::from_be_bytes([packet[header_length + 2], packet[header_length + 3]]) as usize;
        header_length = header_length.saturating_add(4 + extension_words.saturating_mul(4));
        if header_length > packet.len() {
            return None;
        }
    }
    let _ = header_length;
    let timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);
    let ssrc = u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]]);
    Some((ssrc, timestamp))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtpTimestampEstimate {
    pub ssrc: u32,
    pub rtp_timestamp: u32,
    pub ntp_unix_ns: i128,
}

/// Short-history RTP clock model. RTCP sender reports are sparse, so each
/// report anchors the RTP clock and frames are projected using the negotiated
/// 90 kHz H.264 clock rate.
#[derive(Debug, Clone, Default)]
pub struct RtpClockMapper {
    reports: VecDeque<RtcpSenderReport>,
    capacity: usize,
}

impl RtpClockMapper {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("RTP clock report capacity must be positive".into());
        }
        Ok(Self {
            reports: VecDeque::with_capacity(capacity),
            capacity,
        })
    }

    /// Add a sender report if it is continuous with the current RTP/NTP
    /// mapping. A camera that abruptly changes its wall clock must not move
    /// every subsequent frame by seconds; the prior anchor remains usable and
    /// the caller can surface the rejected report in health telemetry.
    pub fn observe(&mut self, report: RtcpSenderReport) -> bool {
        if let Some(previous) = self.reports.back() {
            let rtp_delta =
                report.rtp_timestamp.wrapping_sub(previous.rtp_timestamp) as i32 as i128;
            let expected_ntp =
                previous.ntp_unix_ns + rtp_delta.saturating_mul(1_000_000_000) / RTP_CLOCK_RATE;
            if (report.ntp_unix_ns - expected_ntp).unsigned_abs()
                > MAX_SENDER_REPORT_DISCONTINUITY_NS as u128
            {
                return false;
            }
        }
        if self.reports.len() == self.capacity {
            self.reports.pop_front();
        }
        self.reports.push_back(report);
        true
    }

    pub fn latest_report(&self, ssrc: u32) -> Option<RtcpSenderReport> {
        self.reports
            .iter()
            .rev()
            .find(|report| report.ssrc == ssrc)
            .copied()
            // A few RTSP cameras have been observed to put zero or a
            // rewritten SSRC in their RTCP sender report while retaining the
            // correct RTP clock mapping. Each PoeRtspCapture owns one video
            // stream, so the latest report is a safe single-stream fallback.
            .or_else(|| self.reports.back().copied())
    }

    pub fn estimate(&self, ssrc: u32, rtp_timestamp: u32) -> Option<RtpTimestampEstimate> {
        let report = self.latest_report(ssrc)?;
        let signed_delta = rtp_timestamp.wrapping_sub(report.rtp_timestamp) as i32 as i128;
        Some(RtpTimestampEstimate {
            ssrc,
            rtp_timestamp,
            ntp_unix_ns: report.ntp_unix_ns
                + signed_delta.saturating_mul(1_000_000_000) / RTP_CLOCK_RATE,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sender_report(ssrc: u32, ntp_seconds: u32, rtp_timestamp: u32) -> Vec<u8> {
        let mut packet = vec![0_u8; 28];
        packet[0] = 0x80;
        packet[1] = 200;
        packet[2..4].copy_from_slice(&6_u16.to_be_bytes());
        packet[4..8].copy_from_slice(&ssrc.to_be_bytes());
        packet[8..12].copy_from_slice(&ntp_seconds.to_be_bytes());
        packet[12..16].copy_from_slice(&0x8000_0000_u32.to_be_bytes());
        packet[16..20].copy_from_slice(&rtp_timestamp.to_be_bytes());
        packet
    }

    #[test]
    fn parses_sender_report_and_fractional_ntp_time() {
        let reports = parse_sender_reports(&sender_report(7, 2_208_988_801, 90_000));
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].ssrc, 7);
        assert_eq!(reports[0].rtp_timestamp, 90_000);
        assert_eq!(reports[0].ntp_unix_ns, 1_500_000_000);
    }

    #[test]
    fn projects_rtp_timestamp_from_sender_report() {
        let mut mapper = RtpClockMapper::new(4).unwrap();
        mapper.observe(RtcpSenderReport {
            ssrc: 7,
            ntp_unix_ns: 10_000_000_000,
            rtp_timestamp: 90_000,
        });
        let estimate = mapper.estimate(7, 180_000).unwrap();
        assert_eq!(estimate.ntp_unix_ns, 11_000_000_000);
    }

    #[test]
    fn rejects_sender_report_clock_jump() {
        let mut mapper = RtpClockMapper::new(4).unwrap();
        assert!(mapper.observe(RtcpSenderReport {
            ssrc: 7,
            ntp_unix_ns: 10_000_000_000,
            rtp_timestamp: 90_000,
        }));
        assert!(!mapper.observe(RtcpSenderReport {
            ssrc: 7,
            ntp_unix_ns: 15_000_000_000,
            rtp_timestamp: 180_000,
        }));
        assert_eq!(
            mapper.estimate(7, 180_000).unwrap().ntp_unix_ns,
            11_000_000_000
        );
    }

    #[test]
    fn parses_rtp_ssrc_and_timestamp_with_extension() {
        let mut packet = vec![0_u8; 20];
        packet[0] = 0x90;
        packet[4..8].copy_from_slice(&123_u32.to_be_bytes());
        packet[8..12].copy_from_slice(&456_u32.to_be_bytes());
        packet[14..16].copy_from_slice(&0_u16.to_be_bytes());
        assert_eq!(parse_rtp_header(&packet), Some((456, 123)));
    }
}
