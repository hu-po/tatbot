"""deploy — immutable source builds and manifested fleet services."""
from fleet_deploy import arguments

from tatbot_cli.registry import REMOTE, verb
from tatbot_cli.verbs._common import sh


def _deploy_effects(effects, ns, rest):
    if ns.build_only:
        return effects - {"write_config", "stop_process", "sensor_read"}
    return effects


@verb(refine_effects=_deploy_effects,
      effects=('read_files', 'write_files', 'network', 'remote_exec', 'remote_write',
               'environment_setup', 'start_process', 'write_config', 'stop_process', 'sensor_read'), noun='deploy', verb='', tier=REMOTE,
      role='operator', args=arguments, example=('all', '--build-only'),
      summary='build pushed origin/main and deploy manifested camera, tracker and bus services',
      wraps=('scripts/fleet_deploy.sh', 'scripts/lib/fleet_deploy.py', 'scripts/lib/fleet_install.py', 'scripts/lib/fleet_release.py', 'scripts/lib/tatbot_cli_install.py', 'scripts/fleet_service.sh', 'scripts/lib/fleet_source.py', 'scripts/lib/fleet_toolchain.sh'),
      invariants=('Only a pushed origin/main source archive is deployed.',
                  'Build every selected node successfully before any install; preserve tracked edits and untracked files.',
                  'Every installed node exposes the deployed launcher as /usr/local/bin/tatbot.',
                  'Never start, stop or replace an arm process or a Rerun viewer.'), doc='docs/fleet.md')
def deploy(ctx, ns, rest):
    if ns.service and ns.node == 'all':
        from tatbot_cli.cli import UsageError
        raise UsageError('--service requires a specific node')
    service_args = [arg for unit in ns.service for arg in ('--service', unit)]
    return sh(ctx, 'scripts/fleet_deploy.sh', ns.node,
              *(['--build-only'] if ns.build_only else []), *service_args)
