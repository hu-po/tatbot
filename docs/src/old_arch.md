# Tatbot Architecture Documentation

## Overview

Tatbot is a robotic tattooing system with a modular architecture organized into several key submodules. This document provides a comprehensive analysis of each module's functionality, important abstractions, potential bottlenecks, and optimization opportunities.

## Module Analysis

### 1. Data Module (`src/tatbot/data/`)

**Purpose**: Core data structures and YAML serialization framework for the entire system.

**Key Files**:
- `__init__.py` (144 lines) - Core YAML serialization framework with `Yaml` base class
- `scene.py` (167 lines) - Main scene configuration aggregator
- `stroke.py` (94 lines) - Stroke data structures
- `arms.py` (49 lines) - Robot arm configurations
- `cams.py` (85 lines) - Camera configurations
- `inks.py` (35 lines) - Ink and inkcap definitions
- `pose.py` (42 lines) - Pose and position representations
- `skin.py` (38 lines) - Skin surface definitions
- `tags.py` (19 lines) - AprilTag configurations
- `urdf.py` (26 lines) - URDF file configurations
- `node.py` (23 lines) - Network node definitions

**Important Abstractions**:
- `Yaml` base class with automatic serialization/deserialization
- `Scene` class that aggregates all configuration components
- `FLOAT_TYPE = np.float32` for consistent numerical precision
- Dataclass-based configuration system

**Key Functions**:
- `dataclass_to_dict()` - Recursive conversion with JAX/NumPy array handling
- `Yaml.from_name()` / `Yaml.to_yaml()` - File-based persistence
- `Scene.__post_init__()` - Complex initialization with validation

**Potential Bottlenecks**:
- **Memory**: Large YAML files with embedded arrays could consume significant memory
- **Compute**: Complex dataclass validation and conversion during loading
- **I/O**: File system access for configuration loading

**Optimization Opportunities**:
- Implement lazy loading for large configuration objects
- Add caching for frequently accessed configurations
- Consider binary serialization for large datasets
- Parallelize configuration validation

### 2. Generation Module (`src/tatbot/gen/`)

**Purpose**: Converts design data into robot-executable trajectories and G-code.

**Key Files**:
- `gcode.py` (375 lines) - G-code parsing and generation
- `map.py` (249 lines) - 3D surface mapping with geodesic paths
- `batch.py` (107 lines) - Stroke batching operations
- `align.py` (56 lines) - Alignment operations
- `inkdip.py` (54 lines) - Ink dipping trajectory generation
- `ik.py` (97 lines) - Inverse kinematics
- `strokes.py` (34 lines) - Stroke processing utilities

**Important Abstractions**:
- G-code parsing with coordinate transformations
- Geodesic path computation on 3D meshes
- Stroke resampling to uniform length
- Batch processing for multiple strokes

**Key Functions**:
- `parse_gcode_file()` - Parses G-code with coordinate transformations
- `map_strokes_to_mesh()` - Maps 2D strokes to 3D surface
- `resample_path()` - Even arc-length resampling
- `make_gcode_strokes()` - Generates robot trajectories

**Potential Bottlenecks**:
- **Memory**: Large mesh data and stroke arrays
- **Compute**: Geodesic path computation (O(n²) complexity)
- **I/O**: Large G-code file parsing
- **Algorithm**: Potpourri3D geodesic tracer initialization

**Optimization Opportunities**:
- Implement mesh caching and LOD (Level of Detail)
- Parallelize geodesic computations
- Use spatial indexing for mesh operations
- Optimize G-code parsing with streaming

### 3. Utils Module (`src/tatbot/utils/`)

**Purpose**: Shared utilities and helper functions across the system.

**Key Files**:
- `net.py` (303 lines) - Network management and SSH operations
- `plymesh.py` (346 lines) - PLY mesh file handling
- `mode_toggle.py` (193 lines) - Mode switching utilities
- `log.py` (51 lines) - Logging configuration
- `colors.py` (35 lines) - Color utilities
- `jnp_types.py` (17 lines) - JAX NumPy type utilities

**Important Abstractions**:
- `NetworkManager` - SSH-based distributed operations
- `PlyMesh` - 3D mesh file format handling
- Centralized logging system
- Color management system

**Key Functions**:
- `NetworkManager.setup_network()` - SSH key distribution
- `NetworkManager.test_all_nodes()` - Connectivity testing
- `save_ply()` / `load_ply()` - Mesh file I/O
- `get_logger()` - Centralized logging

**Potential Bottlenecks**:
- **Network**: SSH operations and file transfers
- **I/O**: Large PLY file operations
- **Memory**: Mesh data loading
- **Concurrency**: Network operations blocking

**Optimization Opportunities**:
- Implement connection pooling for SSH
- Add mesh compression and streaming
- Parallelize network operations
- Implement mesh caching

### 4. Bot Module (`src/tatbot/bot/`)

**Purpose**: Robot hardware abstraction and control.

**Key Files**:
- `trossen_config.py` (188 lines) - Trossen robot configuration
- `trossen_homing.py` (208 lines) - Homing procedures
- `urdf.py` (48 lines) - URDF loading and manipulation

**Important Abstractions**:
- `TrossenConfig` - Robot configuration management
- URDF-based robot representation
- Homing and calibration procedures

**Key Functions**:
- `driver_from_arms()` - Robot driver initialization
- `configure_arm()` - Arm configuration
- `get_link_poses()` - URDF pose computation
- Homing and calibration routines

**Potential Bottlenecks**:
- **Hardware**: Robot communication latency
- **Safety**: Homing and calibration procedures
- **Configuration**: Complex robot parameter management
- **Real-time**: Joint limit checking and safety

**Optimization Opportunities**:
- Implement robot state caching
- Optimize communication protocols
- Add predictive safety checks
- Parallelize configuration loading

### 5. Operations Module (`src/tatbot/ops/`)

**Purpose**: High-level robot operations and recording.

**Key Files**:
- `record_stroke.py` (223 lines) - Stroke recording operations
- `record.py` (203 lines) - General recording operations
- `record_align.py` (82 lines) - Alignment recording
- `record_sense.py` (121 lines) - Sensor data recording
- `reset.py` (48 lines) - System reset operations
- `base.py` (55 lines) - Base operation class

**Important Abstractions**:
- `RecordOp` base class for all operations
- Async operation framework with progress reporting
- Episode logging and data collection

**Key Functions**:
- `StrokeOp._run()` - Main stroke recording loop
- `RecordOp.run()` - Async operation execution
- Episode logging and data management
- Real-time data collection

**Potential Bottlenecks**:
- **Real-time**: Camera data collection and processing
- **Storage**: Large dataset generation
- **Memory**: Real-time data buffering
- **Network**: Distributed data collection

**Optimization Opportunities**:
- Implement data streaming and compression
- Add real-time data filtering
- Optimize camera frame processing
- Implement distributed data collection

### 6. Visualization Module (`src/tatbot/viz/`)

**Purpose**: 3D visualization and debugging interface.

**Key Files**:
- `base.py` (230 lines) - Base visualization framework
- `stroke.py` (251 lines) - Stroke visualization
- `map.py` (274 lines) - Mapping visualization
- `teleop.py` (123 lines) - Teleoperation interface

**Important Abstractions**:
- `BaseViz` - Core visualization class
- Viser-based 3D rendering
- Real-time visualization updates

**Key Functions**:
- `BaseViz.__init__()` - Visualization setup
- `BaseViz.step()` - Real-time updates
- Stroke and mapping visualization
- Teleoperation interface

**Potential Bottlenecks**:
- **Graphics**: 3D rendering performance
- **Memory**: Large mesh and point cloud data
- **Network**: Real-time data streaming
- **UI**: Complex GUI interactions

**Optimization Opportunities**:
- Implement LOD for large datasets
- Add GPU acceleration for rendering
- Optimize real-time data streaming
- Implement viewport culling

### 7. MCP (Model Context Protocol) Module (`src/tatbot/mcp/`)

**Purpose**: External API interface for remote control and automation.

**Key Files**:
- `base.py` (49 lines) - Base MCP server
- `ook.py` (67 lines) - OOK-specific MCP server
- `oop.py` (26 lines) - OOP-specific MCP server
- `trossen-ai.py` (25 lines) - Trossen AI MCP server
- `ojo.py` (39 lines) - OJO-specific MCP server
- `rpi1.py` (74 lines) - Raspberry Pi 1 MCP server
- `rpi2.py` (42 lines) - Raspberry Pi 2 MCP server

**Important Abstractions**:
- `MCPConfig` - MCP server configuration
- Node-specific MCP implementations
- Async operation execution

**Key Functions**:
- `_run_op()` - Async operation execution
- Node-specific MCP handlers
- Progress reporting and error handling

**Potential Bottlenecks**:
- **Network**: MCP protocol overhead
- **Latency**: Remote operation execution
- **Concurrency**: Multiple client handling
- **Error handling**: Complex error propagation

**Optimization Opportunities**:
- Implement connection pooling
- Add operation caching
- Optimize protocol serialization
- Implement operation queuing

### 8. Camera Module (`src/tatbot/cam/`)

**Purpose**: Camera hardware abstraction and depth processing.

**Key Files**:
- `depth.py` (82 lines) - Depth camera processing
- `intrinsics_rs.py` (120 lines) - RealSense intrinsics
- `extrinsics.py` (140 lines) - Camera extrinsic calibration
- `intrinsics_ip.py` (19 lines) - IP camera intrinsics
- `tracker.py` (94 lines) - Camera tracking

**Important Abstractions**:
- `DepthCamera` - RealSense depth processing
- Camera calibration and intrinsics
- Point cloud processing

**Key Functions**:
- `DepthCamera.get_pointcloud()` - Depth data processing
- Camera calibration routines
- Point cloud transformation and filtering

**Potential Bottlenecks**:
- **Hardware**: Camera frame rate limitations
- **Processing**: Real-time point cloud processing
- **Memory**: Large point cloud data
- **I/O**: PLY file saving

**Optimization Opportunities**:
- Implement frame rate optimization
- Add point cloud compression
- Optimize real-time processing
- Implement streaming point cloud saving

## System-Wide Analysis

### Memory Bottlenecks
1. **Large Configuration Files**: YAML files with embedded arrays
2. **Mesh Data**: 3D surface meshes for mapping
3. **Point Clouds**: Real-time depth camera data
4. **Stroke Arrays**: Large stroke datasets
5. **Real-time Buffers**: Camera and sensor data

### Compute Bottlenecks
1. **Geodesic Computation**: O(n²) complexity in surface mapping
2. **G-code Parsing**: Large file processing
3. **Real-time Processing**: Camera data and robot control
4. **Mesh Operations**: Spatial queries and transformations
5. **Network Operations**: SSH and distributed computing

### I/O Bottlenecks
1. **File System**: Large YAML and PLY files
2. **Network**: SSH operations and data transfer
3. **Hardware**: Camera frame capture
4. **Storage**: Dataset recording and playback

## Optimization Recommendations

### Immediate Improvements
1. **Implement Caching**: Add LRU caches for frequently accessed data
2. **Parallel Processing**: Use multiprocessing for independent operations
3. **Memory Management**: Implement object pooling for large data structures
4. **Streaming**: Replace batch processing with streaming where possible

### Medium-term Optimizations
1. **GPU Acceleration**: Use GPU for mesh operations and rendering
2. **Database Integration**: Replace file-based storage with databases
3. **Compression**: Implement data compression for large datasets
4. **Distributed Computing**: Better utilize network resources

### Long-term Architectural Changes
1. **Microservices**: Split into smaller, focused services
2. **Event-driven Architecture**: Implement pub/sub for real-time data
3. **Containerization**: Docker-based deployment for better resource management
4. **API-first Design**: RESTful APIs for all external interactions

## Development Team Discussion Points

### Code Quality
- **Type Safety**: Consider adding more type hints and validation
- **Error Handling**: Implement comprehensive error handling and recovery
- **Testing**: Add unit tests for critical components
- **Documentation**: Improve inline documentation and examples

### Performance Monitoring
- **Profiling**: Add performance monitoring and metrics
- **Logging**: Implement structured logging for better debugging
- **Metrics**: Add system health monitoring
- **Alerting**: Implement performance alerts

### Scalability
- **Modularity**: Improve module independence
- **Configuration**: Centralize configuration management
- **Deployment**: Implement CI/CD pipelines
- **Monitoring**: Add comprehensive system monitoring

This documentation provides a foundation for discussing system architecture, identifying bottlenecks, and planning optimizations with the development team. 