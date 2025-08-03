from tatbot.data.scene import Scene
from tatbot.main import compose_and_validate_scene


def load_scene(name="default") -> Scene:
    """Legacy compatibility function for loading scenes.
    
    This function provides backward compatibility for existing code that uses
    Scene.from_name(). Use compose_and_validate_scene() for new code.
    """
    return compose_and_validate_scene(name)
