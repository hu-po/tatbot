//! Fleet trackers on the camera node. The wrist trackers (`trackd`) and the
//! stencil observer (`stencild`) share the frame-socket merge, the bounded
//! evidence file and the supervised Python estimator. No camera backend or
//! arm driver lives here; every binary is a subscriber to an owner's socket.
pub mod evidence;
pub mod worker;

use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use tatbot_visiond::ReceivedFrameSet;

/// Merge the fresh auxiliary sets that share the cadence set's timestamp
/// basis and fall within `tolerance_ns` of it. Returns the merged set, how
/// many sources it holds, and each source's sequence number.
pub fn merge_auxiliary(
    mut primary: ReceivedFrameSet,
    pending: &mut BTreeMap<usize, ReceivedFrameSet>,
    tolerance_ns: u128,
) -> Result<(ReceivedFrameSet, usize, BTreeMap<usize, u64>)> {
    let mut sequences = BTreeMap::from([(0, primary.sequence)]);
    let mut used = 1;
    for source in pending.keys().copied().collect::<Vec<_>>() {
        let Some(candidate) = pending.get(&source) else {
            continue;
        };
        if candidate.timestamp_basis != primary.timestamp_basis {
            pending.remove(&source);
            continue;
        }
        let delta = candidate.timestamp_ns.abs_diff(primary.timestamp_ns);
        if delta > tolerance_ns {
            // An older auxiliary can never match a later cadence frame. A
            // newer one may match the next primary, so retain only that case.
            if candidate.timestamp_ns < primary.timestamp_ns {
                pending.remove(&source);
            }
            continue;
        }
        let candidate = pending.remove(&source).expect("candidate exists");
        for frame in &candidate.frames {
            ensure!(
                primary
                    .frames
                    .iter()
                    .all(|existing| existing.metadata.sensor_name != frame.metadata.sensor_name),
                "camera owners published duplicate sensor names"
            );
        }
        primary.maximum_skew_ns = primary
            .maximum_skew_ns
            .max(candidate.maximum_skew_ns)
            .max(delta);
        sequences.insert(source, candidate.sequence);
        primary.frames.extend(candidate.frames);
        used += 1;
    }
    Ok((primary, used, sequences))
}

#[cfg(test)]
mod auxiliary_tests {
    use super::*;

    fn set(sequence: u64, timestamp_ns: i128, basis: &str) -> ReceivedFrameSet {
        ReceivedFrameSet {
            envelope: None,
            sequence,
            timestamp_basis: basis.into(),
            timestamp_ns,
            maximum_skew_ns: 2,
            frames: Vec::new(),
        }
    }

    #[test]
    fn merges_only_timestamp_compatible_auxiliary_sets() {
        let mut pending = BTreeMap::from([(1, set(7, 1_040, "normalized_source"))]);
        let (merged, used, sequences) =
            merge_auxiliary(set(3, 1_000, "normalized_source"), &mut pending, 50).unwrap();
        assert_eq!(used, 2);
        assert_eq!(merged.maximum_skew_ns, 40);
        assert_eq!(sequences, BTreeMap::from([(0, 3), (1, 7)]));
        assert!(pending.is_empty());
    }

    #[test]
    fn discards_old_auxiliary_but_retains_future_candidate() {
        let mut old = BTreeMap::from([(1, set(1, 900, "normalized_source"))]);
        let (_, used, _) =
            merge_auxiliary(set(2, 1_000, "normalized_source"), &mut old, 50).unwrap();
        assert_eq!(used, 1);
        assert!(old.is_empty());

        let mut future = BTreeMap::from([(1, set(4, 1_100, "normalized_source"))]);
        let (_, used, _) =
            merge_auxiliary(set(3, 1_000, "normalized_source"), &mut future, 50).unwrap();
        assert_eq!(used, 1);
        assert_eq!(future.get(&1).unwrap().sequence, 4);
    }
}
