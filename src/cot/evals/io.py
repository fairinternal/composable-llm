"""
Abstract Evaluation class

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import numpy as np


class EvaluationIO:

    def __init__(self, nb_evals, eval_dim, meaning=None, past_evals=None, past_timestamps=None):
        """
        Initialize tensor to save evalutions.

        Parameters
        ----------
        nb_evals: int
            Number of future evals.
        eval_dim: int
            Evaluation vector dimension.
        meaning: dict
            Meaning of the different eval coordinates.
        past_eval: np.ndarray, optional
            Evals already made (e.g., due to previous checkpointing).
        past_timestamps: np.ndarray, optional
            Tiemstamps of previous evals. Mandatory when using `past_evals`.
        """

        self.meaning = meaning

        if past_evals is None:
            self.evals = np.empty((nb_evals, eval_dim), dtype=float)
            self.timestamps = np.full(nb_evals, -1, dtype=int)
            self.ind = 0
        else:
            assert past_timestamps is not None
            self.ind = past_timestamps.argmax() + 1
            self.evals = np.empty((nb_evals + self.ind, eval_dim), dtype=float)
            self.timestamps = np.full(nb_evals + self.ind, -1, dtype=int)
            self.evals[: self.ind] = past_evals[: self.ind]
            self.timestamps[: self.ind] = past_timestamps[: self.ind]

    def __call__(self, timestamp, evals):
        """
        Report eval at a given timestamp.

        Parameters
        ----------
        timestamp: int
            Timestamp of the evaluation.
        evals: np.ndarray
            Evaluation vector.
        """
        self.evals[self.ind] = evals
        self.timestamps[self.ind] = timestamp
        self.ind += 1
