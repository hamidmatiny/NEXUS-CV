#!/usr/bin/env python3
"""Load test for the NEXUS-CV WebSocket inference stream."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import structlog
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = structlog.get_logger(__name__)

SLA_THRESHOLD_MS = 30.0


def _synthetic_jpeg(seed: int = 0) -> bytes:
    """Generate a small synthetic JPEG payload.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        JPEG-encoded bytes.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode synthetic frame")
    return buf.tobytes()


async def _stream_client(
    host: str,
    camera_id: str,
    fps: float,
    duration_s: float,
    latencies_ms: list[float],
    errors: list[str],
    sla_breaches: list[int],
) -> None:
    """Run a single WebSocket streaming client.

    Args:
        host: Gateway host (host:port).
        camera_id: Camera identifier for the stream path.
        fps: Target frames per second.
        duration_s: Test duration in seconds.
        latencies_ms: Shared list to append serving latencies.
        errors: Shared list to append error messages.
        sla_breaches: Shared counter list (single element).
    """
    url = f"ws://{host}/ws/stream/{camera_id}"
    interval = 1.0 / fps
    deadline = time.monotonic() + duration_s
    frame_idx = 0

    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            while time.monotonic() < deadline:
                start = time.perf_counter()
                payload = _synthetic_jpeg(seed=frame_idx)
                await ws.send(payload)
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)

                import json

                data = json.loads(raw)
                inference_ms = float(data.get("inference_ms", 0.0))
                if inference_ms > SLA_THRESHOLD_MS:
                    sla_breaches[0] += 1

                frame_idx += 1
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - start)))
    except Exception as exc:
        errors.append(str(exc))


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile from sorted values.

    Args:
        values: Sample values.
        pct: Percentile in [0, 100].

    Returns:
        Percentile value or 0.0 when empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


async def run_load_test(
    host: str,
    num_clients: int,
    fps: float,
    duration_s: float,
) -> None:
    """Execute concurrent WebSocket load test and print report.

    Args:
        host: Gateway host (host:port).
        num_clients: Number of concurrent WebSocket clients.
        fps: Target frames per second per client.
        duration_s: Test duration in seconds.
    """
    latencies_ms: list[float] = []
    errors: list[str] = []
    sla_breaches = [0]

    tasks = [
        _stream_client(
            host=host,
            camera_id=f"cam_{i:02d}",
            fps=fps,
            duration_s=duration_s,
            latencies_ms=latencies_ms,
            errors=errors,
            sla_breaches=sla_breaches,
        )
        for i in range(num_clients)
    ]
    await asyncio.gather(*tasks)

    total_requests = len(latencies_ms)
    error_rate = len(errors) / max(num_clients, 1)
    sla_rate = sla_breaches[0] / max(total_requests, 1)

    print("\n=== NEXUS-CV Load Test Report ===")
    print(f"Host:           {host}")
    print(f"Clients:        {num_clients}")
    print(f"FPS/client:     {fps}")
    print(f"Duration:       {duration_s}s")
    print(f"Total frames:   {total_requests}")
    print(f"p50 latency:    {_percentile(latencies_ms, 50):.2f} ms")
    print(f"p95 latency:    {_percentile(latencies_ms, 95):.2f} ms")
    print(f"p99 latency:    {_percentile(latencies_ms, 99):.2f} ms")
    print(f"SLA breach rate:{sla_rate * 100:.2f}%")
    print(f"Error rate:     {error_rate * 100:.2f}% ({len(errors)} client errors)")
    if latencies_ms:
        print(f"Mean latency:   {statistics.mean(latencies_ms):.2f} ms")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="NEXUS-CV WebSocket load test")
    parser.add_argument("--host", default="localhost:8000", help="Gateway host:port")
    parser.add_argument("--num-clients", type=int, default=10, help="Concurrent WebSocket clients")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second per client")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    args = parser.parse_args()
    asyncio.run(run_load_test(args.host, args.num_clients, args.fps, args.duration))


if __name__ == "__main__":
    main()
