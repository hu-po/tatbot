# Tatbot Configuration Documentation

## Overview

Tatbot uses a hierarchical YAML-based configuration system that manages all aspects of the robotic tattooing system. This document explains the configuration structure, file organization, and how to customize the system for different setups.

## Configuration Directory Structure

```
~/tatbot/config/
├── arms/           # Robot arm configurations
├── cams/           # Camera configurations  
├── dbv3/           # DrawingBotV3 configurations
├── inks/           # Ink and inkcap definitions
├── nodes.yaml      # Network node definitions
├── polyscope/      # Polyscope visualization configs
├── poses/          # Robot pose definitions
├── scenes/         # Scene configurations (main)
├── skins/          # Skin surface definitions
├── tags/           # AprilTag configurations
└── trossen/        # Trossen robot specific configs
```

## Core Configuration System

### Yaml Base Class

The configuration system is built around the `Yaml` base class in `src/tatbot/data/__init__.py`:

```python
class Yaml:
    yaml_dir: str = "~/tatbot/config"
    default: str = os.path.join(yaml_dir, "default.yaml")
    
    @classmethod
    def from_name(cls: Type[T], name: str) -> T:
        """Load configuration from YAML file by name"""
        
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
```

### Key Features

- **Automatic Serialization**: Dataclasses are automatically converted to/from YAML
- **Type Safety**: NumPy arrays and JAX arrays are handled correctly
- **Validation**: Configuration loading includes validation and error checking
- **Inheritance**: All configuration classes inherit from `Yaml` base class

## Scene Configuration

The `Scene` class is the main configuration aggregator that brings together all system components:

### Scene Structure

```yaml
# scenes/default.yaml
name: "default"
arms_config_name: "default"
cams_config_name: "default" 
urdf_config_name: "default"
skin_config_name: "default"
inks_config_name: "default"
tags_config_name: "default"

sleep_pos_l_name: "left/sleep"
sleep_pos_r_name: "right/sleep"
ready_pos_l_name: "left/ready"
ready_pos_r_name: "right/ready"

pen_names_l: ["black", "red"]
pen_names_r: ["blue", "green"]
pens_config_path: "~/tatbot/config/dbv3/pens.json"

stroke_length: 100
design_dir_path: "~/tatbot/designs/default"
```

### Scene Loading Process

1. **Component Loading**: Each component (arms, cams, etc.) is loaded from its respective config
2. **Validation**: Cross-references between components are validated
3. **Pose Computation**: URDF-based pose calculations for inkcaps and widgets
4. **Design Loading**: Design images and G-code files are loaded
5. **Mesh Preparation**: Skin mesh directories are created and prepared

## Component Configurations

### Arms Configuration (`arms/`)

Defines robot arm network settings and parameters:

```yaml
# arms/default.yaml
ip_address_l: "192.168.1.3"
ip_address_r: "192.168.1.2"
arm_l_config_filepath: "~/tatbot/config/trossen/arm_l.yaml"
arm_r_config_filepath: "~/tatbot/config/trossen/arm_r.yaml"
goal_time_slow: 5.0
goal_time_fast: 2.0
connection_timeout: 10.0
```

### Cameras Configuration (`cams/`)

Defines camera setups and parameters:

```yaml
# cams/default.yaml
realsenses:
  - name: "rs_left"
    serial_number: "123456789"
    fps: 30
    width: 640
    height: 480
  - name: "rs_right" 
    serial_number: "987654321"
    fps: 30
    width: 640
    height: 480
```

### Skin Configuration (`skins/`)

Defines the 3D surface for tattooing:

```yaml
# skins/default.yaml
name: "default"
image_width_m: 0.2
image_height_m: 0.15
plymesh_dir: "~/tatbot/skin_meshes/default"
mesh_file: "skin.ply"
```

### Inks Configuration (`inks/`)

Defines ink types and inkcap locations:

```yaml
# inks/default.yaml
inkcaps:
  - name: "left_black"
    diameter_m: 0.02
    depth_m: 0.01
    ink:
      name: "black"
      color: [0, 0, 0]
  - name: "right_blue"
    diameter_m: 0.02
    depth_m: 0.01
    ink:
      name: "blue"
      color: [0, 0, 255]
```

### Poses Configuration (`poses/`)

Defines robot arm poses for different operations:

```yaml
# poses/left/sleep.yaml
name: "left/sleep"
joints: [0.0, -1.57, 0.0, -1.57, 0.0, 0.0, 0.0]

# poses/left/ready.yaml  
name: "left/ready"
joints: [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
```

## Network Configuration

### Nodes Configuration (`nodes.yaml`)

Defines all network nodes in the system:

```yaml
nodes:
  - name: "trossen-ai"
    hostname: "192.168.1.100"
    username: "tatbot"
    roles: ["robot", "compute"]
    
  - name: "ook"
    hostname: "192.168.1.101" 
    username: "tatbot"
    roles: ["compute", "storage"]
    
  - name: "rpi1"
    hostname: "192.168.1.102"
    username: "tatbot"
    roles: ["camera", "sensor"]
```

## Configuration Management

### Loading Configurations

```python
from tatbot.data import Scene, Arms, Cams

# Load complete scene
scene = Scene.from_name("default")

# Load individual components
arms = Arms.from_name("default")
cams = Cams.from_name("default")
```

### Saving Configurations

```python
# Save modified configuration
scene.to_yaml("~/tatbot/config/scenes/custom.yaml")
```

### Validation

The configuration system includes several validation checks:

1. **File Existence**: All referenced files must exist
2. **Cross-references**: Components must reference valid configurations
3. **Type Checking**: Arrays and numerical values are validated
4. **URDF Validation**: Robot poses are validated against URDF constraints

## Environment-Specific Configurations

### Development Environment

```yaml
# scenes/dev.yaml
name: "dev"
arms_config_name: "dev"
cams_config_name: "dev"
# ... minimal configuration for development
```

### Production Environment

```yaml
# scenes/prod.yaml  
name: "prod"
arms_config_name: "prod"
cams_config_name: "prod"
# ... full production configuration
```

### Testing Environment

```yaml
# scenes/test.yaml
name: "test"
arms_config_name: "sim"
cams_config_name: "mock"
# ... configuration for testing without hardware
```

## Configuration Best Practices

### 1. Naming Conventions

- Use descriptive names for configurations
- Include environment in configuration names
- Use consistent naming across related configs

### 2. File Organization

- Keep related configurations together
- Use subdirectories for complex components
- Maintain clear separation between environments

### 3. Validation

- Always validate configurations before deployment
- Test configurations in development environment
- Use type hints and validation in custom configs

### 4. Version Control

- Track configuration changes in version control
- Use environment-specific branches
- Document configuration changes

## Troubleshooting

### Common Issues

1. **Missing Files**: Ensure all referenced files exist
2. **Invalid References**: Check cross-references between components
3. **Type Errors**: Verify array types and numerical values
4. **Network Issues**: Validate network configurations

### Debugging

```python
from tatbot.utils.log import get_logger
log = get_logger("config", "⚙️")

# Enable debug logging
log.setLevel(logging.DEBUG)
```

### Validation Tools

```python
# Validate scene configuration
scene = Scene.from_name("default")
print(f"Scene loaded successfully: {scene}")

# Check component references
print(f"Arms config: {scene.arms}")
print(f"Cameras config: {scene.cams}")
```

## Advanced Configuration

### Custom Components

To add custom configuration components:

1. Create dataclass inheriting from `Yaml`
2. Define configuration structure
3. Add to scene configuration
4. Implement loading logic

### Dynamic Configuration

For runtime configuration changes:

```python
# Modify configuration at runtime
scene.arms.goal_time_slow = 3.0

# Save modified configuration
scene.to_yaml("~/tatbot/config/scenes/modified.yaml")
```

### Configuration Templates

Create configuration templates for common setups:

```yaml
# templates/minimal.yaml
name: "minimal"
arms_config_name: "basic"
cams_config_name: "single"
# ... minimal working configuration
```

This configuration system provides a flexible, type-safe way to manage all aspects of the tatbot system while maintaining clear separation between different environments and use cases. 