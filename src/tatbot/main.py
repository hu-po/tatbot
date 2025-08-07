import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from tatbot.config_schema import AppConfig
from tatbot.data.scene import Scene
from tatbot.utils.exceptions import ConfigurationError


def load_scene_from_config(cfg: DictConfig) -> Scene:
    """Load and validate a Scene from Hydra configuration."""
    app_config = AppConfig(**OmegaConf.to_object(cfg))
    return app_config.scene


def compose_and_validate_scene(name: str = "default") -> Scene:
    """Compose a scene configuration and validate it."""
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        cfg = compose(config_name="config", overrides=[f"scenes={name}"])
        return load_scene_from_config(cfg)
    else:
        with initialize(
            config_path="../conf", 
            version_base=None
        ):
            cfg = compose(config_name="config", overrides=[f"scenes={name}"])
            return load_scene_from_config(cfg)


@hydra.main(
    version_base=None, 
    config_path="../conf", 
    config_name="config"
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
    except (ValueError, TypeError, KeyError) as e:
        print(f"❌ Configuration validation failed: {e}")
        raise ConfigurationError(f"Failed to load scene configuration: {e}") from e


if __name__ == "__main__":
    main()
