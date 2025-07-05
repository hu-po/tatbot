import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Any, Optional

import polyscope as ps
import polyscope.imgui as psim
import numpy as np

from tatbot.utils.log import get_logger

log = get_logger('map.base', "📐")

@dataclass
class ViewConfig:
    debug: bool = False
    config_dir: str = "~/tatbot/config/polyscope"
    ini_files: tuple = (".polyscope.ini", "imgui.ini")
    mesh_path: Optional[str] = "~/tatbot/nfs/3d/fakeskin-lowpoly/fakeskin-lowpoly.obj"
    pointcloud_path: Optional[str] = "~/tatbot/nfs/3d/skin.ply"

@dataclass
class AppState:
    geodesic_source: int = -1
    extension_sources: list = field(default_factory=list)
    transport_source: int = -1
    transport_vector: list = field(default_factory=lambda: [6., 6.])
    logmap_source: int = -1
    curves: list = field(default_factory=lambda: [[]])

def symlink_ini_files(config: ViewConfig):
    log.info("🔗 Symlink .ini files from config to current directory")
    for ini_file in config.ini_files:
        target_path = os.path.expanduser(os.path.join(config.config_dir, ini_file))
        link_path = os.path.join(os.getcwd(), ini_file)
        if os.path.lexists(link_path):
            if not os.path.islink(link_path) or os.readlink(link_path) != target_path:
                os.remove(link_path)
                try:
                    os.symlink(target_path, link_path)
                    log.info(f"Updated symlink for {ini_file}")
                except OSError as e:
                    log.error(f"Failed to create symlink for {ini_file}: {e}")
        else:
            try:
                os.symlink(target_path, link_path)
                log.info(f"Created symlink for {ini_file}")
            except OSError as e:
                log.error(f"Failed to create symlink for {ini_file}: {e}")

def cleanup_ini_files(config: ViewConfig):
    for ini_file in config.ini_files:
        link_path = os.path.join(os.getcwd(), ini_file)
        if os.path.exists(link_path):
            log.info(f"🗑️ Removing symlink {link_path}")
            os.remove(link_path)

def remove_polyscope_ini_files(config: ViewConfig):
    # Remove .polyscope.ini and imgui.ini if they exist as real files (not just symlinks)
    for ini_file in config.ini_files:
        link_path = os.path.join(os.getcwd(), ini_file)
        if os.path.exists(link_path) and not os.path.islink(link_path):
            try:
                os.remove(link_path)
                log.info(f"🗑️ Removed real file {link_path} after Polyscope exit")
            except Exception as e:
                log.error(f"Failed to remove {link_path}: {e}")

def polyscope_view(
    state: AppState,
    solver: Any,
    structure: Any,
    basisX: np.ndarray,
    basisY: np.ndarray,
    basisN: np.ndarray,
    get_name: Callable[[], str],
    add_scalar_quantity: Callable[[str, np.ndarray, bool], None],
    add_vector_quantity: Callable[[str, np.ndarray, bool], None],
    selection_label: str = "Point",
    log_prefix: str = "",
):
    ui_scale_set = False
    def callback():
        nonlocal ui_scale_set
        if not ui_scale_set:
            psim.GetIO().FontGlobalScale = 1.5
            ui_scale_set = True
        log.debug("--- GUI callback ---")
        # Geodesic Distance
        if psim.TreeNode("Geodesic Distance"):
            log.debug("Rendering 'Geodesic Distance' UI")
            if psim.Button(f"Set Source {selection_label}"):
                log.debug(f"Button 'Set Source {selection_label}' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == get_name():
                        state.geodesic_source = idx
                        log.info(f"Set geodesic source to {idx}")
                else:
                    log.warning(f"'Set Source {selection_label}' clicked but no selection was made.")
            psim.TextUnformatted(f"Source: {state.geodesic_source if state.geodesic_source != -1 else 'None'}")
            if psim.Button("Compute Geodesic Distance"):
                log.debug("Button 'Compute Geodesic Distance' clicked")
                if state.geodesic_source != -1:
                    dists = solver.compute_distance(state.geodesic_source)
                    add_scalar_quantity("Geodesic Distance", dists, enabled=True)
                    ps.screenshot()
                else:
                    log.warning("No source selected for geodesic distance.")
            psim.TreePop()
        # Scalar Extension
        if psim.TreeNode("Scalar Extension"):
            log.debug("Rendering 'Scalar Extension' UI")
            if psim.Button("Add Source"):
                log.debug("Button 'Add Source' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == get_name():
                        state.extension_sources.append(idx)
                        log.info(f"Added extension source {idx}")
                else:
                    log.warning("'Add Source' clicked but no selection was made.")
            if psim.Button("Clear Sources"):
                log.debug("Button 'Clear Sources' clicked")
                state.extension_sources.clear()
                log.info("Cleared extension sources")
            psim.TextUnformatted(f"Sources: {state.extension_sources}")
            if psim.Button("Compute Scalar Extension"):
                log.debug("Button 'Compute Scalar Extension' clicked")
                if len(state.extension_sources) >= 2:
                    points = state.extension_sources
                    values = np.linspace(0., 1., len(points)).tolist()
                    ext = solver.extend_scalar(points, values)
                    add_scalar_quantity("Scalar Extension", ext, enabled=True)
                else:
                    log.warning("Need at least 2 sources for scalar extension.")
            psim.TreePop()
        # Vector Transport
        if psim.TreeNode("Vector Transport"):
            log.debug("Rendering 'Vector Transport' UI")
            if psim.Button("Set Source"):
                log.debug("Button 'Set Source' for Vector Transport clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == get_name():
                        state.transport_source = idx
                        log.info(f"Set transport source to {idx}")
                else:
                    log.warning("'Set Source' clicked but no selection was made.")
            psim.TextUnformatted(f"Source: {state.transport_source if state.transport_source != -1 else 'None'}")
            _, state.transport_vector[0] = psim.InputFloat("vx", state.transport_vector[0])
            _, state.transport_vector[1] = psim.InputFloat("vy", state.transport_vector[1])
            if psim.Button("Compute Vector Transport"):
                log.debug("Button 'Compute Vector Transport' clicked")
                if state.transport_source != -1:
                    ext = solver.transport_tangent_vector(state.transport_source, state.transport_vector)
                    ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
                    add_vector_quantity("Transported Vector", ext3D, enabled=True)
                else:
                    log.warning("No source selected for vector transport.")
            psim.TreePop()
        # Log Map
        if psim.TreeNode("Log Map"):
            log.debug("Rendering 'Log Map' UI")
            if psim.Button("Set Source##LogMap"):
                log.debug("Button 'Set Source##LogMap' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == get_name():
                        state.logmap_source = idx
                        log.info(f"Set logmap source to {idx}")
                else:
                    log.warning("'Set Source##LogMap' clicked but no selection was made.")
            psim.TextUnformatted(f"Source: {state.logmap_source if state.logmap_source != -1 else 'None'}")
            if psim.Button("Compute Log Map"):
                log.debug("Button 'Compute Log Map' clicked")
                if state.logmap_source != -1:
                    logmap = solver.compute_log_map(state.logmap_source)
                    logmap3D = logmap[:,0,np.newaxis] * basisX +  logmap[:,1,np.newaxis] * basisY
                    add_vector_quantity("Log Map", logmap3D, enabled=True)
                else:
                    log.warning("No source selected for log map.")
            psim.TreePop()
        # Signed Distance
        if psim.TreeNode("Signed Distance"):
            log.debug("Rendering 'Signed Distance' UI")
            for i, curve in enumerate(state.curves):
                psim.TextUnformatted(f"Curve {i}: {curve}")
            if psim.Button(f"Add {selection_label} to Last Curve"):
                log.debug(f"Button 'Add {selection_label} to Last Curve' clicked")
                if ps.have_selection():
                    selection = ps.get_selection()
                    log.debug(f"Got selection: {selection}")
                    name = selection.structure_name
                    idx = selection.local_index
                    if name == get_name():
                        if not state.curves:
                            state.curves.append([])
                        state.curves[-1].append(idx)
                else:
                    log.warning(f"'Add {selection_label} to Last Curve' clicked but no selection was made.")
            if psim.Button("Add New Curve"):
                log.debug("Button 'Add New Curve' clicked")
                state.curves.append([])
            if psim.Button("Clear Curves"):
                log.debug("Button 'Clear Curves' clicked")
                state.curves = [[]]
            if psim.Button("Compute Signed Distance"):
                log.debug("Button 'Compute Signed Distance' clicked")
                valid_curves = [c for c in state.curves if c]
                if valid_curves:
                    signed_dist = solver.compute_signed_distance(valid_curves, basisN)
                    add_scalar_quantity("Signed Distance", signed_dist, enabled=True)
                else:
                    log.warning("No valid curves to compute signed distance.")
            psim.TreePop()
    ps.set_user_callback(callback)
    ps.show()
