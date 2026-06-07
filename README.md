# NEXUS-CV

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Ray](https://img.shields.io/badge/Ray-2.40-green.svg)
![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Production-grade real-time multi-modal computer vision intelligence platform.

## Quick Start

```bash
git clone https://github.com/your-org/nexus-cv.git && cd nexus-cv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python scripts/generate_synthetic_streams.py --num-cameras 2 --duration-seconds 5
pytest tests/ -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, data flow diagram, and component responsibilities.

Architecture decisions are documented in [ADR.md](ADR.md).

## Project Structure

```
nexus-cv/
├── ingestion/          # Stream capture, YOLO detection, Ray frame buffer
├── fusion/             # Multi-modal sensor fusion engine (Phase 2)
├── config/             # pydantic-settings configuration
├── tests/              # Unit tests (CPU-only, mocked YOLO)
├── scripts/            # Synthetic stream generator for CI/dev
├── docker/             # Container definitions
└── .github/workflows/  # CI pipeline
```

## Configuration

All settings are loaded from environment variables or `.env`. Copy `.env.example` to get started:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_CAMERAS` | 4 | Camera streams to ingest |
| `FRAME_BUFFER_SIZE` | 30 | Max buffered frames per camera |
| `YOLO_MODEL_PATH` | yolo11n.pt | YOLO weights or TensorRT engine |
| `YOLO_CONFIDENCE_THRESHOLD` | 0.45 | Detection confidence cutoff |
| `YOLO_IOU_THRESHOLD` | 0.5 | NMS IoU threshold |
| `QUARANTINE_DIR` | ./data/quarantine | Invalid detection storage |
| `LOG_LEVEL` | INFO | structlog level |
| `RAY_NUM_CPUS` | 4 | Ray cluster CPUs |
| `RAY_NUM_GPUS` | 0.0 | Ray cluster GPUs |

## Docker

```bash
docker compose up --build
```

Services: ingestion (port 8265), MLflow (5000), Prometheus (9090), Grafana (3000).

## Development

```bash
# Lint and format
ruff check config ingestion tests scripts
black config ingestion tests scripts
mypy config ingestion

# Run unit tests only
pytest tests/ -m unit -v

# Generate synthetic video streams
python scripts/generate_synthetic_streams.py --num-cameras 4 --fps 15 --duration-seconds 10
```

## Phase Completion

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Ingestion pipeline: stream capture, YOLO detection, frame buffer, schema validation | ✅ Complete |
| **2** | Multi-modal fusion: Kalman tracking, sensor alignment, LiDAR/radar simulators, FusionActor | ✅ Complete |
| 3 | Real-time alerting and event bus | 🔲 Planned |
| 4 | Edge deployment and TensorRT optimization | 🔲 Planned |
| 5 | Production hardening (auth, multi-tenancy) | 🔲 Planned |

## License

MIT
