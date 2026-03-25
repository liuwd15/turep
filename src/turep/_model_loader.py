"""Utilities for loading pre-trained turep models."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from platformdirs import user_cache_dir

from ._fanvi import FANVI

MODEL_URL = "https://zenodo.org/records/19005962/files/turep_cctr.zip"
MODEL_DIR_NAME = "turep_cctr"
MODEL_ZIP_NAME = "turep_cctr.zip"


def _get_cache_dir(app_name: str = "turep") -> Path:
    """Return a cross-platform cache directory for turep assets.

    The path is resolved via ``platformdirs.user_cache_dir``.
    """
    return Path(user_cache_dir(app_name))


def load_model(local_dir: str | Path | None = None) -> FANVI:
    """Load the pre-trained FANVI model, downloading it if needed.

    Parameters
    ----------
    local_dir
        Directory where model files should be stored. If None, uses the
        platform cache directory resolved by ``platformdirs``.

    Returns
    -------
    FANVI
        Loaded pre-trained FANVI model.

    Notes
    -----
    The downloaded archive is stored as ``turep_cctr.zip`` and extracted into
    ``turep_cctr`` within the selected base directory.
    """
    base_dir = Path(local_dir) if local_dir is not None else _get_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    model_dir = base_dir / MODEL_DIR_NAME
    zip_path = base_dir / MODEL_ZIP_NAME

    if not model_dir.exists():
        urlretrieve(MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(base_dir)

    return FANVI.load(model_dir)
