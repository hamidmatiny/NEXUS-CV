"""Async gRPC server for NEXUS-CV bidirectional streaming inference."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import grpc
import structlog

from proto import nexus_cv_pb2, nexus_cv_pb2_grpc
from serving.deployments import LocalPipeline, get_shared_pipeline
from serving.gateway import _run_pipeline, configure_pipeline
from serving.grpc_codec import inference_request_from_proto, inference_response_to_proto
from serving.schemas import InferenceRequest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)

SLA_THRESHOLD_MS = 30.0


class NexusCVServicer(nexus_cv_pb2_grpc.NexusCVServiceServicer):
    """gRPC servicer implementing bidirectional StreamInference."""

    def __init__(self, pipeline: LocalPipeline | None = None) -> None:
        """Initialize the servicer with a shared pipeline handle.

        Args:
            pipeline: Optional pipeline override (defaults to shared singleton).
        """
        self._pipeline = pipeline or get_shared_pipeline()
        configure_pipeline(self._pipeline.remote)

    async def StreamInference(
        self,
        request_iterator: AsyncIterator[nexus_cv_pb2.InferenceRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[nexus_cv_pb2.InferenceResponse]:
        """Process a bidirectional stream of inference requests and responses.

        Args:
            request_iterator: Incoming stream of frame requests.
            context: gRPC servicer context.

        Yields:
            InferenceResponse protobuf messages.
        """
        async for request in request_iterator:
            camera_id, frame_b64, timestamp_ns = inference_request_from_proto(request)
            inf_request = InferenceRequest(
                camera_id=camera_id,
                frame_b64=frame_b64,
                timestamp_ns=timestamp_ns,
            )
            start = time.perf_counter()
            pydantic_response = await _run_pipeline(inf_request)
            serving_ms = (time.perf_counter() - start) * 1000.0
            pydantic_response = pydantic_response.model_copy(update={"serving_ms": serving_ms})

            yield inference_response_to_proto(pydantic_response)


async def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    pipeline: LocalPipeline | None = None,
) -> grpc.aio.Server:
    """Start the async gRPC inference server.

    Args:
        host: Bind address.
        port: Bind port.
        pipeline: Optional shared pipeline instance.

    Returns:
        Running gRPC aio server (already started).
    """
    server = grpc.aio.server()
    nexus_cv_pb2_grpc.add_NexusCVServiceServicer_to_server(
        NexusCVServicer(pipeline=pipeline),
        server,
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("grpc_server_started", address=listen_addr)
    return server


async def run_forever(host: str = "0.0.0.0", port: int = 50051) -> None:
    """Run the gRPC server until interrupted.

    Args:
        host: Bind address.
        port: Bind port.
    """
    server = await serve(host=host, port=port)
    await server.wait_for_termination()


def main() -> None:
    """CLI entrypoint for the gRPC server."""
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
