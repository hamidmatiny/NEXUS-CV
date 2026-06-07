"""NEXUS-CV Ray Serve inference cluster and FastAPI gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serving.deployments import (
        DetectionDeployment,
        FusionDeployment,
        IntelligenceDeployment,
        LocalPipeline,
    )
    from serving.schemas import InferenceRequest, InferenceResponse

__all__ = [
    "DetectionDeployment",
    "FusionDeployment",
    "IntelligenceDeployment",
    "InferenceRequest",
    "InferenceResponse",
    "LocalPipeline",
    "app",
    "build_pipeline",
    "configure_pipeline",
    "create_app",
    "get_shared_pipeline",
]

_DEPLOYMENT_NAMES = frozenset(
    {
        "DetectionDeployment",
        "FusionDeployment",
        "IntelligenceDeployment",
        "LocalPipeline",
        "build_pipeline",
        "get_shared_pipeline",
    }
)
_GATEWAY_NAMES = frozenset({"app", "configure_pipeline", "create_app"})
_SCHEMA_NAMES = frozenset({"InferenceRequest", "InferenceResponse"})


def __getattr__(name: str) -> object:
    """Lazy-load serving submodules to avoid Ray decoration at import time."""
    if name in _DEPLOYMENT_NAMES:
        from serving import deployments as _deployments

        return getattr(_deployments, name)
    if name in _GATEWAY_NAMES:
        from serving import gateway as _gateway

        return getattr(_gateway, name)
    if name in _SCHEMA_NAMES:
        from serving import schemas as _schemas

        return getattr(_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
