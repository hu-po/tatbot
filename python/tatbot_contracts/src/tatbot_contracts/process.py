"""Linux process identity shared by runtime evidence and driver admission."""
from pathlib import Path

PROC = Path('/proc')


def process_stamp(pid, *, proc=PROC):
    """Boot and start ticks disambiguate PID reuse; a zombie is not a live runtime."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid process ID')
    fields = (proc/str(pid)/'stat').read_text().rpartition(')')[2].split()
    if fields[0] in ('Z', 'X', 'x'):
        raise ValueError('process has exited')
    return {'pid': pid, 'process_start': fields[19],
            'boot_id': (proc/'sys/kernel/random/boot_id').read_text().strip()}


def alive(stamp, *, proc=PROC):
    try:
        return process_stamp(stamp['pid'], proc=proc) == {k: stamp[k] for k in ('pid', 'process_start', 'boot_id')}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def process_record(pid):
    try:
        return process_stamp(pid)
    except (OSError, ValueError, IndexError) as error:
        return {'pid': pid, 'process_start': None, 'boot_id': None, 'error': str(error)}
