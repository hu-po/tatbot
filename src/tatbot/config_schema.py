from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator

from tatbot.data.arms import Arms
from tatbot.data.cams import Cams
from tatbot.data.inks import Inks
from tatbot.data.scene import Scene
from tatbot.data.skin import Skin
from tatbot.data.tags import Tags
from tatbot.data.urdf import URDF


class AppConfig(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    
    arms: Dict[str, Any]
    cams: Dict[str, Any]
    inks: Dict[str, Any]
    poses: Optional[Dict[str, Any]] = None
    scenes: Dict[str, Any]
    skins: Dict[str, Any]
    tags: Dict[str, Any]
    urdf: Dict[str, Any]
    
    scene: Optional[Scene] = None
    
    @model_validator(mode='after')
    def create_scene(self) -> 'AppConfig':
        """Compose the full Scene object from all components.
        
        Instantiate objects from pure config data to maintain Hydra's
        guarantee that configs contain only pure data.
        """
        # Instantiate component objects from config data
        arms_obj = Arms(**self.arms)
        cams_obj = Cams(**self.cams)
        inks_obj = Inks(**self.inks)
        skin_obj = Skin(**self.skins)
        tags_obj = Tags(**self.tags)
        urdf_obj = URDF(**self.urdf)
        
        # Compose scene data with instantiated objects
        scene_data = self.scenes.copy()
        scene_data['arms'] = arms_obj
        scene_data['cams'] = cams_obj
        scene_data['inks'] = inks_obj
        scene_data['skin'] = skin_obj
        scene_data['tags'] = tags_obj
        scene_data['urdf'] = urdf_obj
        
        # Create the scene with all dependencies
        self.scene = Scene(**scene_data)
        return self
