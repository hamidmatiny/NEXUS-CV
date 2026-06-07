# NEXUS-CV Architecture

> **Canonical reference:** [docs/architecture.md](docs/architecture.md) — full system design, Docker topology, dashboard/replay engine, and SLA monitoring.

This file provides a quick-reference summary. For Phase 6 complete documentation, ADRs, and business value, see the links below.

## Quick Links

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Full technical architecture (Phase 6) |
| [ADR.md](ADR.md) | Architecture Decision Records (001–005) |
| [BENCHMARKS.md](BENCHMARKS.md) | Performance profiles and SLA analysis |
| [PHASE_REPORT.md](PHASE_REPORT.md) | Phase-by-phase engineering log |
| [docs/business_case.md](docs/business_case.md) | Business value and ROI |

## Docker Compose Services (nexus-net)

| Service | Host Port | Role |
|---------|-----------|------|
| `serving` | 8000 | Unified inference gateway + dashboard WebSocket |
| `ingestion` | 8001 → :8000 | Ray ingestion pipeline + metrics |
| `mlflow` | 5001 | Experiment tracking |
| `prometheus` | 9090 | Telemetry |
| `grafana` | 3000 | Dashboards |

## Phase Status

All six phases are **100% complete**. See [PHASE_REPORT.md](PHASE_REPORT.md).
