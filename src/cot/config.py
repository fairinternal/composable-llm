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
# Tokenizer
# -----------------------------------------------------------------------------

TOKEN_DICT = {
    0: 0,
    1: 1,
    "EoI": 2,
    "EoS": 3,
    "parity": 4,
    "binary_copy": 5,
    "BoS": 6,
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECK_DIR = Path(__file__).parent.parent.parent / "models"

# DATA_DIR = Path("/checkpoint/vivc/data/")
# CHECK_DIR = Path("/checkpoint/vivc/models/")

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
