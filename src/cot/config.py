"""
Configuration file

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import logging
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"

CHECK_DIR = Path(__file__).parent.parent.parent / "models"

# -----------------------------------------------------------------------------
# Random seed
# -----------------------------------------------------------------------------

RNG = np.random.default_rng(0)

# -----------------------------------------------------------------------------
# Logging information
# -----------------------------------------------------------------------------

logging_datefmt = "%m-%d %H:%M:%S"
logging_format = "{asctime} {levelname} [{filename}:{lineno}] {message}"
logging_level = logging.INFO
