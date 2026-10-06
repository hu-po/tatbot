from __future__ import annotations

import json
import subprocess
from copy import deepcopy

import pytest
from tatbot_sim.human_rep.contracts import canonical_digest
from tatbot_sim.inkmap.project import validate_project
from tatbot_sim.repo import repo_root


def test_browser_project_roundtrips_in_python_and_rejects_rehashed_bad_history():
    script = """
      import {readFileSync} from 'node:fs';
      import {makeProject} from './web/inkmap/src/core/project.ts';
      const file=JSON.parse(readFileSync('config/inkmap/examples/forearm-placement-v6.json','utf8'));
      const project=await makeProject({name:'Cross-language fixture',placement_file:file,
        editor:{pose_id:'supine',skin_tone:'#804030',show_atlas:false,camera:null},
        history:{past:[[]],future:[]},edit_before:null,selected_id:null});
      process.stdout.write(JSON.stringify(project));
    """
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
        cwd=repo_root(), capture_output=True, text=True, timeout=30, check=True,
    )
    project = json.loads(result.stdout)
    assert validate_project(project) == project
    broken = deepcopy(project)
    broken["history"]["future"] = [[{**project["placement_file"]["placements"][0], "design_id": "missing"}]]
    broken["content_sha256"] = canonical_digest(broken)
    with pytest.raises(ValueError, match="missing"):
        validate_project(broken)


@pytest.mark.parametrize("schema", ["tatbot.inkmap-project/1", "tatbot.inkmap-project/2"])
def test_old_project_versions_require_explicit_migration(schema):
    with pytest.raises(ValueError, match="project_invalid"):
        validate_project({"schema": schema})


def test_acquired_paths_survive_browser_python_project_and_bundle_handoff():
    from tatbot_sim.inkmap.bundle import validate_simulation_bundle
    from tatbot_sim.inkmap.contracts import validate_placement

    script = """
      import {readFileSync} from 'node:fs';
      import {embeddedFromArtwork} from './web/inkmap/src/core/body-design.ts';
      import {EMPTY_DRAFT,newChartItem,chartDocumentFromDraft} from './web/inkmap/src/core/chart-draft.ts';
      import {makeProject} from './web/inkmap/src/core/project.ts';
      import {makeSimulationBundle,simulationRequest} from './web/inkmap/src/core/sim-bundle.ts';
      const file=JSON.parse(readFileSync('config/inkmap/examples/forearm-placement-v6.json','utf8'));
      const art=JSON.parse(readFileSync('web/inkmap/public/designs/dbv3-orbit/artwork.json','utf8'));
      file.designs['line-v1']=embeddedFromArtwork(art);
      const chart=chartDocumentFromDraft({...EMPTY_DRAFT,items:[newChartItem('item','line-v1',art,[30,30])]});
      const project=await makeProject({name:'Frozen artwork',placement_file:file,
        editor:{pose_id:'supine',skin_tone:'#804030',show_atlas:false,camera:null},
        history:{past:[],future:[]},edit_before:null,selected_id:null,chart,chart_parked:{...chart,kind:'cylinder'}});
      const atlas=JSON.parse(readFileSync('web/inkmap/public/bodies/mhr-soma-v1.regions.json','utf8'));
      const bundle=await makeSimulationBundle(file,atlas,simulationRequest('supine','#804030',null,'lutin-3rl-bugpin',42));
      process.stdout.write(JSON.stringify({project,bundle,art}));
    """
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
        cwd=repo_root(), capture_output=True, text=True, timeout=30, check=True,
    )
    record = json.loads(result.stdout)
    project, bundle, art = record["project"], record["bundle"], record["art"]
    assert validate_project(project) == project
    assert validate_simulation_bundle(bundle)["artworks"]["line-v1"] == art
    broken = deepcopy(bundle)
    broken["placement_file"]["designs"]["line-v1"]["program"]["layers"][0]["elements"][0]["width_m"] = .0004
    broken["content_sha256"] = canonical_digest(broken)
    with pytest.raises(ValueError, match="wrong_hash"):
        validate_simulation_bundle(broken)
    broken = deepcopy(project["placement_file"])
    broken["designs"]["line-v1"]["conversion"]["strokes"] = "centerline"
    with pytest.raises(ValueError, match="fields"):
        validate_placement(broken)
    for location in ("body", "chart", "chart_parked"):
        broken = deepcopy(project)
        design = broken["placement_file"]["designs"]["line-v1"] if location == "body" else broken[location]["artwork"]["line-v1"]
        design["program"]["canvas_m"]["width"] = .04
        broken["content_sha256"] = canonical_digest(broken)
        with pytest.raises(ValueError, match="wrong_hash"):
            validate_project(broken)


@pytest.mark.parametrize("location", ["body", "chart", "chart_parked"])
def test_rehashed_legacy_saved_artwork_requires_dbv3_regeneration(location):
    from tatbot_sim.inkmap.cli import _require_acquired_input
    file = json.loads((repo_root()/'config/inkmap/examples/forearm-placement-v6.json').read_text())
    legacy = deepcopy(file['designs']['line-v1'])
    legacy['conversion'].update(adapter='tatbot-svg-paint/1', recipe_sha256=None)
    legacy['content_sha256'] = canonical_digest(legacy)
    with pytest.raises(ValueError, match='DrawingBot V3'):
        _require_acquired_input({'designs': {'legacy': legacy}})
    with pytest.raises(ValueError, match='DrawingBot V3'):
        _require_acquired_input({'schema': 'tatbot.inkmap-sim-bundle/1', 'artworks': {'legacy': legacy}})
    # The body and both chart registries enforce the same admission boundary.
    if location == 'body':
        from tatbot_contracts.artwork import require_acquired_artwork
        with pytest.raises(ValueError, match='DrawingBot V3'):
            require_acquired_artwork(legacy)
    else:
        from tatbot_sim.inkmap.project import _validate_chart
        with pytest.raises(ValueError, match='DrawingBot V3'):
            _validate_chart({'items': [], 'selected_id': None, 'artwork': {'legacy': legacy}})
