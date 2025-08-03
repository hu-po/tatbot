"""Test Hydra configuration validation for all YAML files."""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from tatbot.config_schema import AppConfig


class TestHydraConfigs:
    """Test Hydra configuration loading and validation."""
    
    def test_all_config_yamls_parse(self):
        """Test that all YAML files in conf/ directory parse correctly."""
        conf_dir = Path(__file__).parent.parent / "conf"
        
        # Find all YAML files
        yaml_files = list(conf_dir.rglob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found in conf directory"
        
        errors = []
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                errors.append(f"{yaml_file.relative_to(conf_dir)}: {e}")
        
        assert not errors, "YAML parsing errors:\n" + "\n".join(errors)
    
    def test_all_scene_configs_load(self):
        """Test that all scene configurations can be loaded with Hydra."""
        scenes_dir = Path(__file__).parent.parent / "conf" / "scenes"
        scene_files = list(scenes_dir.glob("*.yaml"))
        
        errors = []
        for scene_file in scene_files:
            scene_name = scene_file.stem
            try:
                with initialize(
                    config_path=str(Path(__file__).parent.parent / "conf"),
                    version_base=None
                ):
                    cfg = compose(
                        config_name="config", 
                        overrides=[f"scenes={scene_name}"],
                        return_hydra_config=True
                    )
                    # Just test that it composes without errors
                    assert cfg is not None
            except Exception as e:
                errors.append(f"Scene '{scene_name}': {e}")
        
        assert not errors, "Scene configuration errors:\n" + "\n".join(errors)
    
    def test_all_scene_configs_validate_with_pydantic(self):
        """Test that all scene configurations validate with Pydantic models."""
        scenes_dir = Path(__file__).parent.parent / "conf" / "scenes"
        scene_files = list(scenes_dir.glob("*.yaml"))
        
        errors = []
        for scene_file in scene_files:
            scene_name = scene_file.stem
            try:
                with initialize(
                    config_path=str(Path(__file__).parent.parent / "conf"),
                    version_base=None
                ):
                    cfg = compose(
                        config_name="config", 
                        overrides=[f"scenes={scene_name}"]
                    )
                    
                    # Test Pydantic validation
                    app_config = AppConfig(**OmegaConf.to_object(cfg))
                    assert app_config.scene is not None
                    assert app_config.scene.name == scene_name
                    
            except Exception as e:
                errors.append(f"Scene '{scene_name}' validation: {e}")
        
        assert not errors, "Scene validation errors:\n" + "\n".join(errors)
    
    def test_group_directories_match_names(self):
        """Test that Hydra group names match directory names for autocompletion."""
        conf_dir = Path(__file__).parent.parent / "conf"
        config_file = conf_dir / "config.yaml"
        
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        defaults = config_data.get("defaults", [])
        group_overrides = [item for item in defaults if isinstance(item, dict)]
        
        for override in group_overrides:
            for group_name, config_name in override.items():
                group_dir = conf_dir / group_name
                assert group_dir.is_dir(), f"Group directory '{group_name}' does not exist"
                
                config_file_path = group_dir / f"{config_name}.yaml"
                assert config_file_path.exists(), f"Config file '{config_name}.yaml' does not exist in group '{group_name}'"
    
    def test_config_no_run_mode(self):
        """Test config loading in no-run mode to ensure all configs parse."""
        scenes_dir = Path(__file__).parent.parent / "conf" / "scenes"
        scene_files = list(scenes_dir.glob("*.yaml"))
        
        for scene_file in scene_files:
            scene_name = scene_file.stem
            # This simulates running with --config-path and --config-name but no execution
            with initialize(
                config_path=str(Path(__file__).parent.parent / "conf"),
                version_base=None
            ):
                try:
                    cfg = compose(
                        config_name="config",
                        overrides=[f"scenes={scene_name}", "hydra.mode=RUN"]
                    )
                    # Just ensure it can be converted to dict without errors
                    config_dict = OmegaConf.to_object(cfg)
                    assert isinstance(config_dict, dict)
                except Exception as e:
                    pytest.fail(f"Config parsing failed for scene '{scene_name}': {e}")