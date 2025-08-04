"""Tests for MCP Pydantic models and validation."""

import pytest
from pydantic import ValidationError

from tatbot.mcp.models import MCPSettings, RunOpInput
from tatbot.mcp.ojo import ManagePolicyInput
from tatbot.mcp.ook import PingNodesInput
from tatbot.mcp.rpi1 import RunVizInput


class TestMCPSettings:
    """Test MCPSettings Pydantic model."""

    def test_mcp_settings_defaults(self):
        """Test MCPSettings with default values."""
        settings = MCPSettings()
        assert settings.debug is False
        assert settings.transport == "streamable-http"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_mcp_settings_custom_values(self):
        """Test MCPSettings with custom values."""
        settings = MCPSettings(debug=True, transport="custom-transport", port=9000)
        assert settings.debug is True
        assert settings.transport == "custom-transport"
        assert settings.port == 9000


class TestRunOpInput:
    """Test RunOpInput Pydantic model."""

    def test_run_op_input_valid(self):
        """Test RunOpInput with valid values."""
        input_data = RunOpInput(op_name="align", scene_name="default")
        assert input_data.op_name == "align"
        assert input_data.scene_name == "default"
        assert input_data.debug is False

    def test_run_op_input_invalid_op_name(self):
        """Test RunOpInput with invalid operation name."""
        with pytest.raises(ValidationError) as exc_info:
            RunOpInput(op_name="invalid_op")
        
        error = exc_info.value.errors()[0]
        assert "Invalid op_name: invalid_op" in error["msg"]
        assert "Available:" in error["msg"]

    def test_run_op_input_valid_ops(self):
        """Test RunOpInput with all valid operation names."""
        valid_ops = ["align", "stroke", "reset", "sense"]  # From NODE_AVAILABLE_OPS
        
        for op in valid_ops:
            input_data = RunOpInput(op_name=op)
            assert input_data.op_name == op

    def test_run_op_input_scene_validation(self):
        """Test scene name validation with valid scene names."""
        # Use a valid scene name from the test output: ['flower', 'default', 'zorya', 'tatbotlogo']
        input_data = RunOpInput(op_name="align", scene_name="default")
        assert input_data.scene_name == "default"

    def test_run_op_input_invalid_scene_name(self):
        """Test RunOpInput with invalid scene name."""
        with pytest.raises(ValidationError) as exc_info:
            RunOpInput(op_name="align", scene_name="invalid_scene")
        
        error = exc_info.value.errors()[0]
        assert "Invalid scene_name: invalid_scene" in error["msg"]
        assert "Available scenes:" in error["msg"]

    def test_run_op_input_debug_flag(self):
        """Test debug flag in RunOpInput."""
        input_data = RunOpInput(op_name="align", debug=True)
        assert input_data.debug is True


class TestPingNodesInput:
    """Test PingNodesInput Pydantic model."""

    def test_ping_nodes_input_none(self):
        """Test PingNodesInput with None (ping all nodes)."""
        input_data = PingNodesInput()
        assert input_data.nodes is None

    def test_ping_nodes_input_empty_list(self):
        """Test PingNodesInput with empty list."""
        input_data = PingNodesInput(nodes=[])
        assert input_data.nodes == []

    def test_ping_nodes_input_validation_skipped_for_none(self):
        """Test that validation is skipped when nodes is None."""
        # This should not raise an error even if NetworkManager is not available
        input_data = PingNodesInput(nodes=None)
        assert input_data.nodes is None

    def test_ping_nodes_input_valid_nodes(self):
        """Test PingNodesInput with valid node names."""
        # From test output: ['ojo', 'trossen-ai', 'rpi1', 'rpi2', 'ook', 'oop']
        valid_nodes = ["ook", "oop"]
        input_data = PingNodesInput(nodes=valid_nodes)
        assert input_data.nodes == valid_nodes

    def test_ping_nodes_input_invalid_nodes(self):
        """Test PingNodesInput with invalid node names."""
        with pytest.raises(ValidationError) as exc_info:
            PingNodesInput(nodes=["invalid_node"])
        
        error = exc_info.value.errors()[0]
        assert "Invalid nodes: ['invalid_node']" in error["msg"]
        assert "Available nodes:" in error["msg"]


class TestManagePolicyInput:
    """Test ManagePolicyInput Pydantic model."""

    def test_manage_policy_input_valid_actions(self):
        """Test ManagePolicyInput with valid actions."""
        for action in ["start", "stop"]:
            input_data = ManagePolicyInput(action=action)
            assert input_data.action == action
            assert input_data.policy_type == "gr00t"  # default

    def test_manage_policy_input_invalid_action(self):
        """Test ManagePolicyInput with invalid action."""
        with pytest.raises(ValidationError) as exc_info:
            ManagePolicyInput(action="invalid")
        
        error = exc_info.value.errors()[0]
        assert "Action must be 'start' or 'stop'" in error["msg"]

    def test_manage_policy_input_valid_policy_types(self):
        """Test ManagePolicyInput with valid policy types."""
        for policy_type in ["gr00t", "smolvla"]:
            input_data = ManagePolicyInput(action="start", policy_type=policy_type)
            assert input_data.policy_type == policy_type

    def test_manage_policy_input_invalid_policy_type(self):
        """Test ManagePolicyInput with invalid policy type."""
        with pytest.raises(ValidationError) as exc_info:
            ManagePolicyInput(action="start", policy_type="invalid")
        
        error = exc_info.value.errors()[0]
        assert "Policy type must be 'gr00t' or 'smolvla'" in error["msg"]


class TestRunVizInput:
    """Test RunVizInput Pydantic model."""

    def test_run_viz_input_valid(self):
        """Test RunVizInput with valid values."""
        input_data = RunVizInput(viz_type="stream", name="test_viz")
        assert input_data.viz_type == "stream"
        assert input_data.name == "test_viz"

    def test_run_viz_input_valid_types(self):
        """Test RunVizInput with all valid viz types."""
        valid_types = ["stream", "record", "plot", "mesh"]
        
        for viz_type in valid_types:
            input_data = RunVizInput(viz_type=viz_type, name="test")
            assert input_data.viz_type == viz_type

    def test_run_viz_input_invalid_type(self):
        """Test RunVizInput with invalid viz type."""
        with pytest.raises(ValidationError) as exc_info:
            RunVizInput(viz_type="invalid", name="test")
        
        error = exc_info.value.errors()[0]
        assert "Invalid viz_type: invalid" in error["msg"]
        assert "Valid types:" in error["msg"]


class TestIntegration:
    """Integration tests for MCP models."""

    def test_models_json_serializable(self):
        """Test that all models can be serialized to JSON."""
        models = [
            MCPSettings(debug=True),
            RunOpInput(op_name="align", scene_name="default"),
            PingNodesInput(nodes=["ook"]),  # Use a valid node name
            ManagePolicyInput(action="start"),
            RunVizInput(viz_type="stream", name="test"),
        ]
        
        for model in models:
            # Should not raise an exception
            json_data = model.model_dump_json()
            assert isinstance(json_data, str)
            
            # Should be able to parse back
            data = model.model_validate_json(json_data)
            assert data == model

    def test_model_pydantic_consistency(self):
        """Test that MCPSettings properly works as a Pydantic model."""
        settings = MCPSettings(debug=True, port=9000)
        
        # Should be able to convert to dict
        data = settings.model_dump()
        assert isinstance(data, dict)
        assert data['debug'] is True
        assert data['port'] == 9000