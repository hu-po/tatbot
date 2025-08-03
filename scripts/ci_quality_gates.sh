#!/usr/bin/env bash

# CI Quality Gates for MCP Refactor
# This script runs all quality checks for the MCP layer

set -euo pipefail

cd "$(dirname "$0")/.."

echo "🔍 Running MCP Quality Gates..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall success
OVERALL_SUCCESS=true

run_check() {
    local name="$1"
    local command="$2"
    
    echo -e "\n📋 Running: ${YELLOW}$name${NC}"
    echo "Command: $command"
    
    if eval "$command"; then
        echo -e "✅ ${GREEN}$name PASSED${NC}"
        return 0
    else
        echo -e "❌ ${RED}$name FAILED${NC}"
        OVERALL_SUCCESS=false
        return 1
    fi
}

# 1. Unit Tests
run_check "Unit Tests" "pytest tests/ -v --tb=short"

# 2. Linting with ruff
run_check "Ruff Linting" "ruff check src/tatbot"
run_check "Ruff Formatting" "ruff format --check src/tatbot"

# 3. Type checking with mypy
run_check "MyPy Type Checking" "mypy src/tatbot/mcp --ignore-missing-imports"

# 4. MCP Schema Validation
run_check "MCP Schema Validation" "python -c '
import sys
sys.path.insert(0, \"src\")
from tatbot.mcp.models import RunOpInput, RunOpResult, PingNodesInput, PingNodesResponse
print(\"RunOpInput schema:\", RunOpInput.model_json_schema())
print(\"RunOpResult schema:\", RunOpResult.model_json_schema())
print(\"PingNodesInput schema:\", PingNodesInput.model_json_schema())
print(\"PingNodesResponse schema:\", PingNodesResponse.model_json_schema())
print(\"✅ All schemas generated successfully\")
'"

# 5. Hydra Configuration Validation
run_check "Hydra Config Validation" "python -c '
import sys
sys.path.insert(0, \"src\")
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Test loading each node config
nodes = [\"ook\", \"oop\", \"rpi1\", \"rpi2\", \"ojo\", \"trossen-ai\"]
for node in nodes:
    config_path = Path(f\"conf/mcp/{node}.yaml\")
    if config_path.exists():
        config = OmegaConf.load(config_path)
        print(f\"✅ {node}.yaml loaded successfully\")
        # Validate required fields
        assert \"host\" in config or \"defaults\" in config, f\"{node}: missing host\"
        assert \"port\" in config or \"defaults\" in config, f\"{node}: missing port\"
    else:
        print(f\"⚠️  {node}.yaml not found\")

print(\"✅ All Hydra configs validated\")
'"

# 6. Import Test - Ensure all modules can be imported
run_check "Import Validation" "python -c '
import sys
sys.path.insert(0, \"src\")

# Test importing main modules
from tatbot.mcp import models, handlers, server
from tatbot.mcp.models import MCPSettings, RunOpInput, RunOpResult
from tatbot.mcp.handlers import get_available_tools

print(\"Available tools:\", list(get_available_tools().keys()))
print(\"✅ All MCP modules imported successfully\")
'"

# 7. Configuration Composition Test
run_check "Hydra Composition Test" "python -c '
import sys
sys.path.insert(0, \"src\")
from hydra import compose, initialize
from omegaconf import OmegaConf

try:
    with initialize(version_base=None, config_path=\"conf\"):
        # Test composing with ook node
        cfg = compose(config_name=\"config\", overrides=[\"mcp=ook\"])
        assert \"mcp\" in cfg, \"MCP config not found\"
        assert \"host\" in cfg.mcp, \"Host not found in MCP config\"
        assert \"port\" in cfg.mcp, \"Port not found in MCP config\"
        print(\"✅ Hydra composition working correctly\")
except Exception as e:
    print(f\"❌ Hydra composition failed: {e}\")
    raise
'"

# 8. FastMCP Integration Test (Quick)
run_check "FastMCP Integration" "python -c '
import sys
sys.path.insert(0, \"src\")
from tatbot.mcp.handlers import get_available_tools
from tatbot.mcp.models import MCPSettings

# Verify handlers are registered
tools = get_available_tools()
assert len(tools) > 0, \"No tools registered\"
assert \"ping_nodes\" in tools, \"ping_nodes tool not found\"
assert \"list_scenes\" in tools, \"list_scenes tool not found\"

# Verify settings model
settings = MCPSettings()
assert hasattr(settings, \"host\"), \"Settings missing host\"
assert hasattr(settings, \"port\"), \"Settings missing port\"

print(f\"✅ {len(tools)} tools registered successfully\")
print(f\"✅ MCPSettings model working correctly\")
'"

echo -e "\n🏁 Quality Gate Summary"
echo "========================"

if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "✅ ${GREEN}ALL QUALITY GATES PASSED${NC}"
    echo "🚀 MCP refactor is ready for deployment!"
    exit 0
else
    echo -e "❌ ${RED}SOME QUALITY GATES FAILED${NC}"
    echo "🔧 Please fix the issues above before deploying."
    exit 1
fi