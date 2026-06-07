# Quickstart — Zero to Live Demo in 5 Minutes

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- 8 GB RAM minimum
- Ports available: 8000, 8001, 3000, 9090, 5001

## Steps

### 1. Clone and configure

```bash
git clone https://github.com/your-org/nexus-cv.git
cd nexus-cv
cp .env.example .env
```

### 2. Start the stack

```bash
docker compose up --build
```

This starts:
- **serving** — FastAPI gateway on `:8000`
- **ingestion** — Ray ingestion pipeline on `:8001`
- **mlflow** — experiment tracking on `:5001`
- **prometheus** — metrics on `:9090`
- **grafana** — dashboards on `:3000` (admin / password from `.env`)

### 3. Verify health

```bash
curl http://localhost:8000/health | jq .    # serving gateway
curl http://localhost:8001/health | jq .    # ingestion metrics API
curl http://localhost:8000/metrics | head
curl http://localhost:8001/metrics | grep nexus_cv_frames
```

### 4. Run inference

```bash
python scripts/generate_synthetic_streams.py --num-cameras 1 --fps 10 --duration-seconds 3
# Use a frame from data/synthetic/cam_00/ or POST via curl — see api_reference.md
```

### 5. Open the live dashboard

```bash
cd dashboard/frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the Vite dev server proxies `/api` and `/ws` to the serving gateway.

[INSERT SCREENSHOT: dashboard]

### 6. Grafana

Open **http://localhost:3000** (login: `admin` / `admin` or your `GRAFANA_ADMIN_PASSWORD`).

Pre-provisioned dashboards:
- NEXUS-CV Ingestion
- NEXUS-CV Serving (latency, SLA, anomalies)

---

## Enable session replay

```bash
# In .env
RECORDING_ENABLED=true
```

Restart serving. Recorded sessions appear at `GET /api/v1/replay/sessions`.

---

## Local dev (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python scripts/generate_synthetic_streams.py --num-cameras 2 --duration-seconds 5
pytest tests/ -v
```

Start serving locally:

```bash
python scripts/run_serving.py
```

Further reading: [architecture.md](architecture.md) · [business_case.md](business_case.md) · [BENCHMARKS.md](../BENCHMARKS.md)
