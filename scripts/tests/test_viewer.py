"""The fleet viewer's recording names, fixed blueprint and data-only RRD import."""
import sys
from pathlib import Path

import pytest

# One vector for both twins: rerun_viewer::recording_name holds the same cases.
NAME_VECTOR = [
    ('live-cameras', 'Rig preview'),
    ('cockpit-20260917T201500Z', 'Cockpit 20:15'),
    ('sweep-20260917_150212', 'Sweep 15:02'),
    ('record-20260917-150212', 'Record 15:02'),
    ('draw-20260917T174304Z-arm-7e3c', 'Draw 17:43'),
    ('replay-flight_12-20260917T174304Z', 'Replay flight_12 17:43'),
    ('draw-shadow-squiggle_v3', 'Draw shadow squiggle_v3'),
    ('20260917T174304Z-arm-7e3c', None),
    ('part-1234567890123456', 'Part 1234567890123456'),
    ('x20260917T174304Z', 'X20260917T174304Z'),
    ('', None),
    (None, None),
]


@pytest.mark.parametrize(('recording_id', 'expected'), NAME_VECTOR)
def test_sources_are_named_from_their_id_and_run_ids_are_left_to_the_session(recording_id, expected):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
    import tatbot_rerun as tr
    assert tr.recording_name(recording_id) == expected
    assert tr.STANDING_RECORDING == 'live-cameras'
    minted = tr.recording_id('cockpit')
    assert tr.recording_name(minted).startswith('Cockpit ') and minted.endswith('Z')


def test_every_viewer_server_generation_installs_one_fixed_blueprint():
    repo = Path(__file__).resolve().parents[2]
    server = (repo / 'config/systemd/tatbot-viewer@.service').read_text()
    launcher = (repo / 'scripts/viewer.sh').read_text()
    assert 'ExecStartPost=' in server and 'scripts/viewer.sh blueprint' in server
    assert 'send-blueprint --connect "$proxy" --recording-id live-cameras' in launcher
    assert 'Rerun did not release ports $RERUN_PORT and ${RERUN_WEB_PORT:-9090}' in launcher


def test_rrd_import_is_data_only_before_it_reaches_the_fleet():
    repo = Path(__file__).resolve().parents[2]
    launcher = (repo / 'scripts/viewer.sh').read_text()
    assert 'sanitize-rrd --input "$target"' in launcher
    assert 'rrd verify "$import_tmp/data-only.rrd"' in launcher
    assert 'log_file_from_path(path)' in launcher
    assert 'rr.send_recording_name(recording_name(rid))' in launcher
