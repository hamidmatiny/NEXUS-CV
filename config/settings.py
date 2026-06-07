"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the NEXUS-CV ingestion pipeline.

    All values are loaded from environment variables or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    NUM_CAMERAS: int = Field(default=4, description="Number of camera streams to ingest")
    FRAME_BUFFER_SIZE: int = Field(default=30, description="Max frames per camera in buffer")
    YOLO_MODEL_PATH: str = Field(
        default="yolo11n.pt",
        description="Path to YOLO weights (.pt) or TensorRT engine (.engine)",
    )
    YOLO_CONFIDENCE_THRESHOLD: float = Field(
        default=0.45,
        description="Minimum detection confidence score",
    )
    YOLO_IOU_THRESHOLD: float = Field(
        default=0.5,
        description="IoU threshold for non-max suppression",
    )
    QUARANTINE_DIR: Path = Field(
        default=Path("./data/quarantine"),
        description="Directory for quarantined invalid detection records",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level for structlog")
    RAY_NUM_CPUS: int = Field(default=4, description="CPUs allocated to Ray cluster")
    RAY_NUM_GPUS: float = Field(default=0.0, description="GPUs allocated to Ray cluster")
    FUSION_ALIGNMENT_MAX_OFFSET_MS: float = Field(
        default=50.0,
        description="Max inter-sensor timestamp gap for temporal alignment",
    )
    FUSION_ALIGNMENT_BUFFER_SIZE: int = Field(
        default=50,
        description="Sliding buffer size per sensor modality",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Returns:
        Settings: Application configuration.
    """
    return Settings()
