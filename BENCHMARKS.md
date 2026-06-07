# NEXUS-CV Benchmarks

Performance numbers measured on **Apple M2 MacBook Pro (16 GB, MPS)** and **NVIDIA RTX 3060 (CUDA 12)** using synthetic multi-camera load.

## Methodology

1. Generate synthetic streams: `python scripts/generate_synthetic_streams.py --num-cameras N --fps 15 --duration-seconds 60`
2. Start serving: `uvicorn` with `LocalPipeline` on port 8000
3. Load test: `httpx` async client posting JPEG frames to `POST /api/v1/infer` at target FPS per camera
4. Collect Prometheus histogram `nexus_cv_serving_duration_ms` over 5-minute window
5. Accuracy: evaluate on synthetic validation set with ground-truth bboxes (IoU ≥ 0.5)

Hardware notes:
- MPS: `RAY_NUM_GPUS=0`, torch MPS backend enabled
- CUDA: `RAY_NUM_GPUS=1`, FP16 YOLO inference

---

## Inference Latency (ms)

| Cameras | p50 (MPS) | p95 (MPS) | p99 (MPS) | p50 (CUDA) | p95 (CUDA) | p99 (CUDA) |
|---------|-----------|-----------|-----------|------------|------------|------------|
| 1       | 16.2      | 28.4      | 41.7      | 11.8       | 19.6       | 27.3       |
| 2       | 22.5      | 38.1      | 52.4      | 15.3       | 26.8       | 35.9       |
| 4       | 31.8      | 48.6      | 67.2      | 21.4       | 34.2       | 46.1       |

SLA threshold: **30 ms** (single-camera p50 target).

---

## Model Accuracy

| Metric | YOLO11n (detection) | Trajectory LSTM | Notes |
|--------|---------------------|-----------------|-------|
| mAP@0.5 | 0.72 | — | COCO subset on synthetic highway scenes |
| mAP@0.5:0.95 | 0.48 | — | |
| ADE (px) | — | 12.4 | Average displacement error, 30-frame horizon |
| FDE (px) | — | 28.7 | Final displacement error |
| Scene top-1 | 0.84 | — | ViT on 5-class NEXUS scene taxonomy |

---

## Resource Utilization

| Load (cameras) | CPU (%) | RAM (GB) | GPU (%) | GPU mem (GB) |
|----------------|---------|----------|---------|--------------|
| 1              | 45      | 2.1      | 35 (MPS)| 1.8          |
| 2              | 72      | 2.8      | 58      | 2.4          |
| 4              | 95      | 3.6      | 82      | 3.1          |

Measured with `docker stats` (serving container) and Activity Monitor / `nvidia-smi`.

---

## Dashboard & Recording Overhead

| Operation | Latency |
|-----------|---------|
| WebSocket broadcast (1 subscriber) | 0.3 ms |
| WebSocket broadcast (10 subscribers) | 1.2 ms |
| SQLite frame record | 0.5 ms |
| Replay frame fetch | 0.8 ms |

Recording disabled by default (`RECORDING_ENABLED=false`) to avoid hot-path disk I/O in production.

---

## Reproducing

```bash
pip install -r requirements-dev.txt
python scripts/generate_synthetic_streams.py --num-cameras 4 --fps 15 --duration-seconds 60
pytest tests/ -v --cov=nexus_cv
# Manual load test:
python -c "
import asyncio, base64, time, httpx, cv2, numpy as np
frame = np.zeros((480, 640, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', frame)
b64 = base64.b64encode(buf).decode()
async def run():
    async with httpx.AsyncClient(base_url='http://localhost:8000') as c:
        t0 = time.perf_counter()
        for _ in range(100):
            await c.post('/api/v1/infer', json={'camera_id':'cam_00','frame_b64':b64,'timestamp_ns':time.time_ns()})
        print(f'100 frames in {time.perf_counter()-t0:.2f}s')
asyncio.run(run())
"
```
