"""Runtime evidence for paired research: startup sources and loaded controller code.

This is provenance, not a motion interlock. Missing or changing evidence makes
a research comparison inadmissible; ordinary drawing keeps its existing rules.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

from tatbot_contracts.process import PROC, alive, process_record, process_stamp  # noqa: F401 -- runtime API
from tatbot_contracts.ros_runtime import (  # noqa: F401 -- existing session/launch API
    configuration_digest,
    controller_identity,
    publish,
    record_controller,
    software_digest,
    source_digest,
    source_status,
    workspace_description,
    workspace_record,
)


def session_identity(repo, controller_reference=None, configuration=None, workspace=None):
    """Snapshot module and owner-library source trees; check current bytes at admission/readback."""
    import numpy
    import tatbot_bridge
    import tatbot_contracts
    import tatbot_description
    import tatbot_motion

    import tatbot_session
    from tatbot_session import config

    roots = {module.__name__: str(Path(module.__file__).resolve().parent)
             for module in (tatbot_session, tatbot_motion, tatbot_description, tatbot_bridge, tatbot_contracts)}
    roots.update({f'scripts/{folder}': str(Path(repo).resolve()/'scripts'/folder) for folder in ('lib', 'vision')})
    return {'schema': 'tatbot.ros-runtime/2', 'revision_at_start': config.revision(repo),
            'source_roots': roots, 'sources_sha256': source_digest(roots),
            'python': platform.python_version(), 'numpy': numpy.__version__,
            **process_record(os.getpid()), 'controller_reference': controller_reference,
            'configuration_sha256': configuration, 'workspace_sha256': workspace}


def current(identity):
    if identity is None:
        return None
    return {**identity, 'controller': controller_identity(identity.get('controller_reference')), 'source_status': source_status(identity)}


def validate_research(program, observed, recorded):
    if not program.get('research'):
        return
    if observed.get('schema') != 'tatbot.ros-runtime/2' or not observed.get('controller', {}).get('complete'):
        raise ValueError('research requires identified loaded controller code')
    if not observed.get('source_status', {}).get('complete'):
        raise ValueError('research source differs from its startup snapshot')
    if observed != recorded:
        raise ValueError('research runtime changed since the run started; comparison is inconclusive')
