"""hfung4_dbr_ml_utils package."""

import importlib.metadata
import importlib.resources
from pathlib import Path

from .common import (
    MLOpsParser,
    get_dbr_token,
    get_workspace_url,
    is_databricks,
)
from .delta import get_delta_table_version

THIS_DIR = Path(__file__).parent
PROJECT_DIR = (THIS_DIR / "../..").resolve()


def get_version() -> str:
    """Retrieve the version of the package.

    This function attempts to read the version from the installed package,
    and if not found, reads it from a version.txt file.

    Returns
    -------
        (str) The version string of the package.

    """
    try:
        # Try to read from the installed package
        return importlib.metadata.version(__package__)
    except importlib.metadata.PackageNotFoundError:
        # If not installed, read from the version.txt file
        with importlib.resources.files(__package__).joinpath("../../version.txt").open("r", encoding="utf-8") as file:
            return file.read().strip()


__version__ = get_version()

__all__ = [
    "is_databricks",
    "get_dbr_token",
    "get_workspace_url",
    "MLOpsParser",
    "get_delta_table_version",
    "__version__",
]
