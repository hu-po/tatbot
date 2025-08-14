
from typing import List, Optional

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from tatbot.config_schema import AppConfig
from tatbot.data.scene import Scene
from tatbot.utils.exceptions import ConfigurationError


class AppConstants:
    """Configuration constants for the main application."""
    CONFIG_PATH: str = "../conf"
    DEFAULT_CONFIG_NAME: str = "config"
    DEFAULT_SCENE_NAME: str = "default"


def load_scene_from_config(cfg: DictConfig) -> Scene:
    """Load and validate a Scene from Hydra configuration."""
    app_config = AppConfig(**OmegaConf.to_object(cfg))
    return app_config.scene




def compose_and_validate_scene(
    name: str = AppConstants.DEFAULT_SCENE_NAME,
    meta: Optional[str] = None,
) -> Scene:
    """Compose a scene configuration and validate it.

    Parameters:
    - name: scene config name under `conf/scenes`
    - meta: optional meta config name under `conf/meta` to merge
    """
    from hydra.core.global_hydra import GlobalHydra

    override_list: List[str] = [f"scenes={name}"]
    if meta:
        override_list.append(f"meta={meta}")

    if GlobalHydra().is_initialized():
        cfg = compose(config_name=AppConstants.DEFAULT_CONFIG_NAME, overrides=override_list)
        return load_scene_from_config(cfg)
    else:
        with initialize(
            config_path=AppConstants.CONFIG_PATH, 
            version_base=None
        ):
            cfg = compose(config_name=AppConstants.DEFAULT_CONFIG_NAME, overrides=override_list)
            return load_scene_from_config(cfg)


@hydra.main(
    version_base=None, 
    config_path=AppConstants.CONFIG_PATH, 
    config_name=AppConstants.DEFAULT_CONFIG_NAME
)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for the tatbot application."""
    print("🚀 Tatbot starting with Hydra configuration")
    print("Configuration keys:", list(cfg.keys()))
    
    try:
        scene = load_scene_from_config(cfg)
        print(f"✅ Scene loaded: {scene.name}")
        print(f"🦾 Arms: {scene.arms.ip_address_l} / {scene.arms.ip_address_r}")
        print(f"📷 Cameras: {len(scene.cams.realsenses)} RealSense, {len(scene.cams.ipcameras)} IP")
        print(f"🎨 Inks: {len(scene.inks.inkcaps)} inkcaps")
        print("✅ Configuration validated successfully!")
    except (ValueError, TypeError, KeyError) as validation_error:
        print(f"❌ Configuration validation failed: {validation_error}")
        raise ConfigurationError(f"Failed to load scene configuration: {validation_error}") from validation_error


if __name__ == "__main__":
    main()
