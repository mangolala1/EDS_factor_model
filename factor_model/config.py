from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout for the project."""
    base_dir: Path
    data_dir: Path
    outputs_dir: Path

    @property
    def prices_file(self) -> Path:
        return self.data_dir / "prices.parquet"

    @property
    def fundamentals_file(self) -> Path:
        return self.data_dir / "fundamentals.parquet"

    @property
    def universe_file(self) -> Path:
        return self.data_dir / "universe.parquet"

    @property
    def returns_file(self) -> Path:
        return self.data_dir / "returns.parquet"


@dataclass(frozen=True)
class ModelConfig:
    client: str = "EDS"
    model: str = "BERKELEY"
    cov_window: int = 60
    var_window: int = 60
    lookback_days: int = 400  # must cover momentum (252) + rolling windows + buffer
    winsor_low: float = 0.01
    winsor_high: float = 0.99

    @property
    def model_id(self) -> str:
        return f"{self.client}_{self.model}"


@dataclass(frozen=True)
class SnowflakeConfig:
    """Snowflake connection details loaded from env/.env."""
    account: str | None = None
    user: str | None = None
    password: str | None = None
    warehouse: str | None = None
    role: str | None = None
    database: str = "EDS_DEV_BERKELEY"
    schema: str = "BERKELEY"


def repo_root() -> Path:
    # factor_model/ is one level under repo root
    return Path(__file__).resolve().parents[1]


def load_snowflake_config(env_path: str | None = None) -> SnowflakeConfig:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)

    return SnowflakeConfig(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "EDS_DEV_BERKELEY"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "BERKELEY"),
    )


def default_paths() -> PathsConfig:
    base = repo_root()
    return PathsConfig(
        base_dir=base,
        data_dir=base / "data",
        outputs_dir=base / "outputs",
    )
