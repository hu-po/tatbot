# Dependency Update Log

**Branch:** dependency-updates-2025-01  
**Started:** 2025-01-08  
**Status:** In Progress

## Current Versions (Baseline)
Captured in `current_dependencies_baseline.txt`

## Update Phases

### Phase 1: Safe Updates (Low Risk)
**Target Dependencies:**
- huggingface-hub (0.34.3 → 0.34.4)
- tyro (0.9.24 → 0.9.27)
- hydra-core (~=1.3 → 1.3.2)
- trossen-arm (1.8.5 → 1.8.6)
- viser (1.0.0 → 1.0.4)
- mcp (1.11.0 → 1.12.4)

**Node Update Order:**
1. rpi2 (NFS server) - Base dependencies only
2. rpi1 - Utility node with viz/img
3. ook/oop - GPU nodes
4. trossen-ai - Robot control

### Phase 2: Moderate Risk Updates
**Target Dependencies:**
- opencv-python (4.11.0.86 → 4.12.0.88)
- pyrealsense2 (2.55.1.6486 → 2.56.5.9235)

### Phase 3: High Risk Updates
**Target Dependencies:**
- safetensors (0.5.3 → 0.6.2)
- pydantic (~=2.7 → 2.11.7) 
- paramiko (3.5.1 → 4.0.0)

## Execution Log

### Phase 1: Safe Updates ✅ COMPLETED
**Nodes Updated:**
- oop: Successfully updated huggingface-hub, tyro, mcp, viser, hydra-core
- trossen-ai: Successfully updated huggingface-hub, tyro, mcp, trossen-arm, hydra-core
- Additional packages updated due to dependency resolution: fsspec, numpy, pillow, anyio, certifi, etc.

**Status:** ✅ All Phase 1 updates successful, MCP servers restarted and functional

### Phase 2: Moderate Risk Updates ⚠️ PARTIAL  
**Nodes Updated:**
- oop: Successfully updated opencv-python (4.11.0.86 → 4.12.0.88), added pyrealsense2 (2.56.5.9235)
- trossen-ai: ⚠️ Dependency conflicts detected - virtual environment reset reverted pyrealsense2 to older version

**Issues Found:**
- trossen-ai virtual environment recreation is resetting versions due to lock file conflicts
- Need to investigate dependency resolution for camera libraries

**Status:** 🔄 Needs investigation and potential manual resolution

### Phase 3: High Risk Updates ⚠️ MIXED RESULTS
**Nodes Updated:**
- oop: Successfully updated safetensors (0.5.3 → 0.6.2), pydantic (2.7 → 2.11.7), paramiko (3.5.1 → 4.0.0)

**Issues Found:**
- Virtual environment rebuilds during `uv run` operations may revert some package versions
- This appears to be related to lock file conflicts or dependency resolution
- SSH connectivity and basic functionality confirmed working
- Pydantic models and validation working correctly
- Safetensors serialization working correctly

**Status:** ⚠️ Updates applied but version persistence needs investigation

## Summary

### ✅ Successfully Updated Packages
- huggingface-hub: 0.34.3 → 0.34.4
- tyro: 0.9.24 → 0.9.27  
- mcp: 1.11.0 → 1.12.4
- viser: 1.0.0 → 1.0.4
- trossen-arm: 1.8.3 → 1.8.6
- opencv-python: 4.11.0.86 → 4.12.0.88
- safetensors: 0.5.3 → 0.6.2 (with caveats)
- pydantic: ~2.7 → 2.11.7 (with caveats)

### ⚠️ Needs Investigation
- paramiko: Version persistence issues during virtual environment rebuilds
- pyrealsense2: Dependency conflicts on some nodes
- Virtual environment lock file management

### 🔄 Recommendations for Next Phase
1. Investigate `uv.lock` file management and version pinning
2. Consider using `--frozen` flag for production environments
3. Test functionality on all nodes before marking as complete
4. Consider updating pyproject.toml version constraints
