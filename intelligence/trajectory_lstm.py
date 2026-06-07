"""LSTM-based trajectory prediction model and batched predictor."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import structlog
from numpy.typing import NDArray

from config.settings import get_settings
from fusion.data_types import Track
from intelligence.data_types import TrajectoryPrediction
from intelligence.model_utils import denormalize_coords, normalize_coords, select_device

logger = structlog.get_logger(__name__)

HORIZON = 15
INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

try:
    import torch
    import torch.nn as nn

    class TrajectoryLSTM(nn.Module):
        """PyTorch LSTM module for multi-step trajectory forecasting."""

        def __init__(
            self,
            input_size: int = INPUT_SIZE,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            dropout: float = DROPOUT,
            horizon: int = HORIZON,
        ) -> None:
            """Initialize the LSTM trajectory model.

            Args:
                input_size: Number of input features per timestep.
                hidden_size: LSTM hidden state dimension.
                num_layers: Number of stacked LSTM layers.
                dropout: Dropout rate between LSTM layers.
                horizon: Number of future frames to predict.
            """
            super().__init__()
            self.horizon = horizon
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 2 * horizon)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run forward pass.

            Args:
                x: Input tensor of shape (B, seq_len, 6).

            Returns:
                Output tensor of shape (B, horizon, 2).
            """
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            flat = self.head(last)
            return flat.view(-1, self.horizon, 2)  # type: ignore[no-any-return]

except ImportError:

    class TrajectoryLSTM:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("PyTorch is required for TrajectoryLSTM")


def _bbox_features(
    bbox: tuple[float, float, float, float], velocity: tuple[float, float]
) -> tuple[float, ...]:
    """Extract normalized features from a bbox and velocity.

    Args:
        bbox: Bounding box (x1, y1, x2, y2).
        velocity: Velocity (vx, vy).

    Returns:
        Normalized 6-feature tuple.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return normalize_coords(cx, cy, w, h, velocity[0], velocity[1])


class TrajectoryPredictor:
    """Batched trajectory predictor wrapping TrajectoryLSTM."""

    def __init__(
        self,
        model_path: str | None = None,
        seq_len: int = 20,
        horizon: int = HORIZON,
    ) -> None:
        """Initialize the trajectory predictor.

        Args:
            model_path: Path to trained weights. Falls back to untrained model.
            seq_len: Required observation sequence length.
            horizon: Prediction horizon in frames.
        """
        settings = get_settings()
        self._model_path = model_path or settings.TRAJECTORY_LSTM_PATH
        self._seq_len = seq_len
        self._horizon = horizon
        self._device = select_device()
        self._model = TrajectoryLSTM(horizon=horizon)
        self._model.to(self._device)
        self._model.eval()
        self._history: dict[str, deque[tuple[float, ...]]] = {}
        self._load_weights()

    def _load_weights(self) -> None:
        """Load trained weights if available."""
        path = Path(self._model_path)
        if not path.exists():
            logger.warning("trajectory_model_fallback", path=str(path), msg="Using untrained model")
            return
        try:
            state = torch.load(path, map_location=self._device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                self._model.load_state_dict(state["model_state_dict"])
            else:
                self._model.load_state_dict(state)
            logger.info("trajectory_model_loaded", path=str(path))
        except Exception as exc:
            logger.warning("trajectory_model_load_failed", path=str(path), error=str(exc))

    def _update_history(self, track: Track) -> None:
        """Append current track state to rolling history.

        Args:
            track: Current track observation.
        """
        if track.last_bbox_2d is None:
            return
        features = _bbox_features(track.last_bbox_2d, track.velocity_2d)
        if track.track_id not in self._history:
            self._history[track.track_id] = deque(maxlen=self._seq_len)
        self._history[track.track_id].append(features)

    def predict_batch(
        self, tracks: list[Track], seq_len: int | None = None
    ) -> list[TrajectoryPrediction]:
        """Predict future trajectories for a batch of tracks.

        Args:
            tracks: Tracks to predict trajectories for.
            seq_len: Override minimum sequence length requirement.

        Returns:
            Trajectory predictions for tracks with sufficient history.
        """
        required_len = seq_len or self._seq_len
        eligible: list[Track] = []

        for track in tracks:
            self._update_history(track)
            hist = self._history.get(track.track_id, deque())
            if len(hist) >= required_len:
                eligible.append(track)

        if not eligible:
            return []

        batch_arrays: list[NDArray[np.float32]] = []
        for track in eligible:
            history = list(self._history[track.track_id])[-required_len:]
            batch_arrays.append(np.array(history, dtype=np.float32))

        tensor = torch.tensor(np.stack(batch_arrays), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            output = self._model.forward(tensor)
        output_np = output.cpu().numpy()

        predictions: list[TrajectoryPrediction] = []
        for i, track in enumerate(eligible):
            positions: list[tuple[float, float]] = []
            for step in range(self._horizon):
                cx_norm = float(output_np[i, step, 0])
                cy_norm = float(output_np[i, step, 1])
                positions.append(denormalize_coords(cx_norm, cy_norm))
            predictions.append(
                TrajectoryPrediction(
                    track_id=track.track_id,
                    predicted_positions=positions,
                    horizon_frames=self._horizon,
                    confidence=0.75,
                )
            )
        return predictions
