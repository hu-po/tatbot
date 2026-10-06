"""Bounded passive controller packet capture; never construct an arm driver."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import pwd
import shutil
import subprocess
import time
from pathlib import Path

import tatbot_profile

REPO = Path(__file__).resolve().parents[2]


def controller_filter(driver):
    addresses = [str(ipaddress.IPv4Address(driver[role + '_ip'])) for role in ('leader', 'follower')]
    if len(set(addresses)) != 2:
        raise ValueError('controller addresses must be distinct')
    hosts = '(' + ' or '.join('host ' + ip for ip in addresses) + ')'
    return f'{hosts} and (arp or udp port 50000 or tcp port 50001)'


def capture_command(driver, output, seconds, tcpdump, timeout, sudo, username):
    if not 15 <= seconds <= 120:
        raise ValueError('network trace duration must be 15..120 seconds')
    # Ring is bounded to four ~4 MB files; snapshot covers transport headers.
    # sudo never prompts or runs a shell. Drop capture privileges to the caller.
    return [sudo, '-n', timeout, '--signal=INT', '--kill-after=3', str(seconds),
            tcpdump, '-i', 'any', '-p', '-nn', '-s', '128', '-B', '4096', '-U',
            '-C', '4', '-W', '4', '-Z', username, '-w', str(output / 'controllers.pcap'),
            controller_filter(driver)]


def network_snapshot(output, name):
    snapshot = {'observed_unix': time.time()}
    for label, argv in (('routes', ['ip', '-j', 'route']), ('links', ['ip', '-j', '-s', 'link'])):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
        snapshot[label] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    snapshot['snmp'] = Path('/proc/net/snmp').read_text()
    (output / name).write_text(json.dumps(snapshot, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args(argv)
    try:
        profile = tatbot_profile.load(REPO)
        driver = profile['driver']
        output = Path(os.environ['TATBOT_RUN_DIR']).resolve()
        if output.is_relative_to(REPO) or not output.is_dir():
            raise ValueError('network trace needs an existing run directory outside the checkout')
        binaries = {name: shutil.which(name) for name in ('tcpdump', 'timeout', 'sudo')}
        if not all(binaries.values()):
            raise ValueError('network trace needs tcpdump, timeout and sudo installed on the arm owner')
        command = capture_command(driver, output, args.seconds, username=pwd.getpwuid(os.getuid()).pw_name,
                                  **binaries)
        os.umask(0o077)
        network_snapshot(output, 'network-before.json')
        (output / 'trace.json').write_text(json.dumps({
            'schema': 'tatbot.teleop-network/1', 'filter': command[-1], 'seconds': args.seconds,
            'profile_sha256': profile['_sha256'], 'motion_commands': False}, indent=2) + '\n')
        print(f'Passive controller trace for {args.seconds} seconds: {output}', flush=True)
        print('Wait for tcpdump to print "listening on any", then use a separate terminal for manual teleop.\n'
              'This command does not start, stop or release the arms.', flush=True)
        result = subprocess.run(command, timeout=args.seconds + 10, check=False)
        network_snapshot(output, 'network-after.json')
        files = sorted(output.glob('controllers.pcap*'))
        captured = any(path.stat().st_size > 24 for path in files)
        success = result.returncode in (0, 124) and captured
        (output / 'result.json').write_text(json.dumps({
            'exit_code': result.returncode, 'packets_retained': captured,
            'files': [{'name': p.name, 'bytes': p.stat().st_size} for p in files],
            'communication_qualified': False}, indent=2) + '\n')
        print('Trace retained: ' + str(output), flush=True)
        if not success:
            print('No usable trace. Check the capture error above; no arm controls were changed.', flush=True)
        return 0 if success else 3
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, tatbot_profile.ProfileError) as error:
        print('Network trace refused: ' + str(error), flush=True)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
