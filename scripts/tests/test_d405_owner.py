"""Camera handoff never starts a second owner or changes arm signal handling."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_shell(body):
    script = f'source "{ROOT}/scripts/lib/d405_owner.sh"\n' + body
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True)


def test_active_owner_stops_before_borrow_and_restores_once():
    result = run_shell('''
active=1
systemctl() {
  if [[ "$1" == show ]]; then echo loaded; return 0; fi
  [[ "$active" == 1 ]]
}
sudo() { echo "$*"; if [[ "$3" == stop ]]; then active=0; else active=1; fi; }
pgrep() { return 1; }
d405_owner::borrow || exit $?
[[ "$active" == 0 ]] || exit 10
d405_owner::restore || exit $?
d405_owner::restore
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['-n systemctl stop tatbot-visiond-d405.service',
                                         '-n systemctl start tatbot-visiond-d405.service']


def test_missing_service_never_uses_privilege_or_starts_camera():
    result = run_shell('''
systemctl() { echo not-found; }
sudo() { echo unexpected; return 99; }
d405_owner::borrow && d405_owner::restore
''')
    assert result.returncode == 0
    assert not result.stdout


def test_failed_stop_refuses_handoff_and_does_not_restore():
    result = run_shell('''
systemctl() { [[ "$1" == show ]] && echo loaded; return 0; }
sudo() { echo "$*"; return 1; }
d405_owner::borrow
rc=$?
d405_owner::restore
exit "$rc"
''')
    assert result.returncode == 5
    assert 'start' not in result.stdout


def test_live_lerobot_child_blocks_service_restoration():
    result = run_shell('''
D405_OWNER_RESTORE=1
pgrep() { return 0; }
sudo() { echo unexpected; return 99; }
d405_owner::restore
''')
    assert result.returncode == 6
    assert not result.stdout
    assert 'remains stopped' in result.stderr
