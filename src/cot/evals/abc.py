"""
Abstract Evaluation class

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import torch


class Evaluation:
    eval_dim = None
    eval_meaning = []
    eval_timestamps = torch.empty(0)

    def __init__(self, nb_evals, current_eval=None):
        """
        Initialize tensor to save evalutions.

        Parameters
        ----------
        nb_evals: int
            Number of future evals.
        current_eval: torch.Tensor
            Evals already made (e.g., due to previous checkpointing).
        """
        if current_eval is None:
            self.eval_tensor = torch.empty((nb_evals, self.eval_dim))
            self.ind = 0
        else:
            self.ind = len(current_eval)
            nb_evals = nb_evals + self.ind
            self.eval_tensor = torch.empty((nb_evals, self.eval_dim))
            self.eval_tensor[: self.ind] = current_eval
        pass

    def compress_eval(self, *args):
        """
        Concatenate all the metrics in a single vector
        """
        pass

    def decompress_eval(self, compressed_vector):
        """
        Parse a single vector into different metrics
        """
        pass
