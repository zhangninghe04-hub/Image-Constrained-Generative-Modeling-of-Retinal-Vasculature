"""
I/O utilities for loading images, saving results, and managing
experiment configurations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_image(path: str) -> np.ndarray:
    """
    Load a retinal image from disk.

    Parameters
    ----------
    path : str
        Path to the image file (jpg, png, tif).

    Returns
    -------
    np.ndarray
        Image array (H x W x C) or (H x W) for grayscale.
    """
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from skimage import io as skio
        return skio.imread(path)


def save_config(config_dict: Dict[str, Any], path: str) -> None:
    """
    Save a configuration dictionary to a JSON file.

    Parameters
    ----------
    config_dict : dict
        Configuration parameters.
    path : str
        Output file path.
    """
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON config file.

    Returns
    -------
    dict
        Configuration parameters.
    """
    with open(path) as f:
        return json.load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir(subdir: str = "raw") -> Path:
    """Return the path to a data subdirectory."""
    return get_project_root() / "data" / subdir


def list_retinal_images(data_dir: Optional[str] = None) -> list:
    """
    List all retinal image files in the given directory.

    Parameters
    ----------
    data_dir : str, optional
        Directory to search. Defaults to data/raw/healthy/.

    Returns
    -------
    list
        Sorted list of image file paths.
    """
    if data_dir is None:
        data_dir = get_data_dir("raw") / "healthy"
    else:
        data_dir = Path(data_dir)

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)
