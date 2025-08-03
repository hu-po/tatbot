"""OpenAPI schema generation for MCP endpoints."""

from typing import Any, Dict

from tatbot.mcp import models


def generate_openapi_schema() -> Dict[str, Any]:
    """Generate OpenAPI schema for MCP endpoints."""
    
    # Collect all Pydantic models
    request_models = {
        "RunOpInput": models.RunOpInput,
        "PingNodesInput": models.PingNodesInput,
        "ListScenesInput": models.ListScenesInput,
        "ListNodesInput": models.ListNodesInput,
        "GetNfsInfoInput": models.GetNfsInfoInput,
        "GetLatestRecordingInput": models.GetLatestRecordingInput,
    }
    
    response_models = {
        "RunOpResult": models.RunOpResult,
        "PingNodesResponse": models.PingNodesResponse,
        "ListScenesResponse": models.ListScenesResponse,
        "ListNodesResponse": models.ListNodesResponse,
        "NodesStatusResponse": models.NodesStatusResponse,
        "GetNfsInfoResponse": models.GetNfsInfoResponse,
        "GetLatestRecordingResponse": models.GetLatestRecordingResponse,
    }
    
    # Generate schemas
    schemas = {}
    for name, model in {**request_models, **response_models}.items():
        schemas[name] = model.model_json_schema()
    
    # OpenAPI specification
    openapi_spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "Tatbot MCP API",
            "description": "Model Context Protocol API for Tatbot robotics system",
            "version": "1.0.0",
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Local MCP server"
            }
        ],
        "paths": {
            "/tools/run_op": {
                "post": {
                    "summary": "Run robot operation",
                    "description": "Execute a robot operation with specified scene configuration",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RunOpInput"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Operation result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RunOpResult"}
                                }
                            }
                        }
                    }
                }
            },
            "/tools/ping_nodes": {
                "post": {
                    "summary": "Ping network nodes",
                    "description": "Check connectivity status of network nodes",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PingNodesInput"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Ping results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PingNodesResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/tools/list_scenes": {
                "post": {
                    "summary": "List available scenes",
                    "description": "Get list of available scene configurations",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ListScenesInput"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Available scenes",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ListScenesResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/tools/list_nodes": {
                "post": {
                    "summary": "List network nodes",
                    "description": "Get list of available network nodes",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ListNodesInput"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Available nodes",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ListNodesResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Bearer token authentication"
                }
            }
        },
        "security": [
            {"BearerAuth": []}
        ]
    }
    
    return openapi_spec