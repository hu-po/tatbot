from pydantic import BaseModel, model_validator
from typing import Optional, Dict, Any
from tatbot.data.arms import Arms
from tatbot.data.cams import Cams
from tatbot.data.inks import Inks
from tatbot.data.scene import Scene
from tatbot.data.skin import Skin
from tatbot.data.tags import Tags
from tatbot.data.urdf import URDF

class AppConfig(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    arms: Arms
    cams: Cams
    inks: Inks
    poses: Optional[Dict[str, Any]] = None  # Poses are loaded individually by Scene
    scenes: dict  # Raw scene config data
    skins: Skin
    tags: Tags
    urdf: URDF
    
    # Computed scene object
    scene: Scene = None
    
    @model_validator(mode='after')
    def create_scene(self) -> 'AppConfig':
        """Compose the full Scene object from all components."""
        scene_data = self.scenes.copy()
        
        # Inject the actual component objects into the scene data
        scene_data['arms'] = self.arms
        scene_data['cams'] = self.cams
        scene_data['inks'] = self.inks
        scene_data['skin'] = self.skins
        scene_data['tags'] = self.tags
        scene_data['urdf'] = self.urdf
        
        # Create the scene with all dependencies
        self.scene = Scene(**scene_data)
        return self
