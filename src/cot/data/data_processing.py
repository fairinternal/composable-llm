"""
Generate synthetic data to study LLM behaviors in controlled settings.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from cot.config import DATA_DIR, RNG, TOKEN_DICT

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Generic class
# -----------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """
    Attributes
    ----------
    data: tensor of size (n_data, seq_len)
        Tensor with data ordered by sequence length.
    indices: tensor of int of size (len)
        Indices to delimitate difference sequence lengths.

    Parameters
    ----------
    save_dir: str
        Path of the directory where to save the data.
    """

    prefix = None

    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = DATA_DIR
            if self.prefix is not None:
                save_dir = DATA_DIR / self.prefix
        self.save_dir = save_dir

    @classmethod
    def generate_fixed_len_data(cls, seq_len, n_data, rng=None):
        """Generate sequence with fixed sequence length."""
        raise NotImplementedError

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        raise NotImplementedError

    def change_save_dir(self, save_dir):
        """
        Change saving directory.

        Parameters
        ----------
        save_dir: str
            Path of the directory where to save the data.
        """
        self.save_dir = save_dir

    def generate_datafiles(self, n_data_per_len, split_probas_by_len, rng=None):
        """
        Test/train split.

        Parameters
        ----------
        n_data_per_len : int
            Maximum number of data points to generate for each sequence length.
        split_probas_by_len : list of float
            Proportion of data to put in the training set for each sequence length.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
        """

        logger.info(f"Generating data. Saving in {self.save_dir}")

        if rng is None:
            rng = np.random.default_rng()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for seq_len, split_proba in enumerate(split_probas_by_len):
            seq_len += 1
            data = self.generate_fixed_len_data(seq_len=seq_len, n_data=n_data_per_len, rng=rng)
            rng.shuffle(data)
            n_train = int(split_proba * len(data))
            np.save(self.save_dir / f"train_{seq_len}.npy", data[:n_train])
            np.save(self.save_dir / f"test_{seq_len}.npy", data[n_train:])
            logger.debug(f"Sequences of length {seq_len} done. Saved in {self.save_dir} ({n_train}/{len(data)} split).")

    def load_data(self, lengths, data_type=None):
        """
        Get data (load from data directory).

        Parameters
        ----------
        lengths : list of int
            List of sequence lengths.
        data_type : str, optional
            Type of data to load. Whether 'train' or 'test'.

        Returns
        -------
        data : numpy.ndarray
            Data containing sequence of tokens specified by TOKEN_DICT.
        indices : numpy.ndarray
            Indices to split the data by sequence length.

        Notes
        -----
        Should be called after `generate_datafiles`.
        """
        assert isinstance(lengths, list), "`lenghts` must be an a list of int."
        assert data_type in ["train", "test"], "`data_type` must be 'train' or 'test'."

        logging.info(f"Loading data from {self.save_dir}.")

        # memory preallocation
        # ... compute the data size by lenghts
        n_data_by_lens = np.empty(len(lengths))
        for i, seq_len in enumerate(lengths):
            filename = self.save_dir / f"{data_type}_{seq_len}.npy"
            with open(filename, "rb") as f:
                version = np.lib.format.read_magic(f)
                header = np.lib.format._read_array_header(f, version)
            n_data_by_lens[i] = header[0][0]

        # ... deduce memory allocation
        indices = np.cumsum(n_data_by_lens, dtype=int)
        indices = np.insert(indices, 0, 0)
        data = np.full((indices[-1], self.get_len(max(lengths)) + 2), TOKEN_DICT["EoS"], dtype=np.int32)

        # add spectial token at begining of sentence
        if self.prefix in TOKEN_DICT:
            data[:, 0] = TOKEN_DICT[self.prefix]
        else:
            logger.info(f"Prefix {self.prefix} not in TOKEN_DICT, falling back to generic BoS.")
            data[:, 0] = TOKEN_DICT["BoS"]

        # load the data in the allocated memory
        for i, seq_len in enumerate(lengths):
            data[indices[i] : indices[i + 1], 1 : self.get_len(seq_len) + 1] = np.load(
                self.save_dir / f"{data_type}_{seq_len}.npy"
            )

        return data, indices

    def set_data(self, lengths, data_type):
        """
        Load training data as a class attribute.

        Endows `self` with attributes `data` and `indices`.

        Parameters
        ----------
        lengths : list of int
            List of sequence lengths.
        data_type : str
            Type of data to load. Whether 'train' or 'test'.

        Notes
        -----
        Should be called after `generate_datafiles`.
        """
        data, indices = self.load_data(lengths, data_type=data_type)
        self.data = torch.from_numpy(data)
        self.indices = torch.from_numpy(indices)

    def get_sampler_by_len(self, probas_by_len):
        """
        Set the probability of each data point.

        Endows `self` with attributes `proba_by_data`.

        Parameters
        ----------
        probas_by_len : list of numpy.ndarray
            Probability vector to sample of sequence of a given length.

        Notes
        -----
        Should be called after `set_train_data`.
        """

        assert abs(sum(probas_by_len) - 1) < 1e-6, "The sum of the probabilities must be equal to 1."

        probas = torch.empty(self.indices[-1])
        for i in range(len(probas_by_len)):
            start, end = self.indices[i], self.indices[i + 1]
            if start != end:
                probas[start:end] = probas_by_len[i] / (end - start)

        return WeightedRandomSampler(probas, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------------------------------------------------------
# Copy problem
# -----------------------------------------------------------------------------


class BinaryCopy(SequenceDataset):
    prefix = "binary_copy"

    def __init__(self, save_dir=None):
        super().__init__(save_dir=save_dir)

    @classmethod
    def generate_fixed_len_data(cls, seq_len, n_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if n_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens specified by TOKEN_DICT.
        """
        if rng is None:
            rng = np.random.default_rng()

        # allocate memory
        if 2**seq_len < n_data:
            n_data = 2**seq_len
        length = cls.get_len(seq_len)
        data = np.empty((n_data, length), dtype=np.int32)

        # input data
        # ... exhaustive case
        if 2**seq_len == n_data:
            powers_of_two = 2 ** np.arange(seq_len)[::-1]
            data[:, :seq_len] = (np.arange(n_data).reshape(-1, 1) & powers_of_two != 0).astype(np.int32)
        # ... non-exhaustive case
        else:
            data[:, :seq_len] = (rng.random((n_data, seq_len)) > 0.5).astype(np.int32)
        ind_neg = data == 0
        ind_pos = data == 1
        data[ind_neg] = TOKEN_DICT[0]
        data[ind_pos] = TOKEN_DICT[1]

        # end of input
        data[:, seq_len] = TOKEN_DICT["EoI"]

        # copying the data
        data[:, seq_len + 1 :] = data[:, :seq_len]
        return data

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        return 2 * seq_len + 1


# -----------------------------------------------------------------------------
# Parity problem
# -----------------------------------------------------------------------------


class Parity(SequenceDataset):

    def __init__(self, cot=True, save_dir=None):
        self.cot = cot
        if self.cot:
            self.prefix = "parity"
        else:
            self.prefix = "no_cot"
        super().__init__(save_dir=save_dir)

    def generate_fixed_len_data(self, seq_len, n_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if n_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens with
                0: begining of sentence,
                1: end of input,
                2: end of sentence,
                3: negative bit,
                4: positive bit.
        """
        if rng is None:
            rng = np.random.default_rng()

        # allocate memory
        if 2**seq_len < n_data:
            n_data = 2**seq_len
        length = self.get_len(seq_len)
        data = np.empty((n_data, length), dtype=np.int32)

        # input data
        # ... exhaustive case
        if 2**seq_len == n_data:
            powers_of_two = 2 ** np.arange(seq_len)[::-1]
            data[:, :seq_len] = (np.arange(n_data).reshape(-1, 1) & powers_of_two != 0).astype(np.int32)
        # ... non-exhaustive case
        else:
            data[:, :seq_len] = (rng.random((n_data, seq_len)) > 0.5).astype(np.int32)

        if self.cot:
            # CoT data
            data[:, seq_len + 1 :] = np.cumsum(data[:, :seq_len], axis=1) % 2
        else:
            data[:, seq_len + 1] = np.sum(data[:, :seq_len], axis=1) % 2
            assert TOKEN_DICT["EoS"] not in [0, 1]
            data[:, seq_len + 2 :] = TOKEN_DICT["EoS"]

        ind_neg = data == 0
        ind_pos = data == 1
        data[ind_neg] = TOKEN_DICT[0]
        data[ind_pos] = TOKEN_DICT[1]

        # end of input
        data[:, seq_len] = TOKEN_DICT["EoI"]

        return data

    def get_len(self, seq_len):
        """Full sequence length."""
        return 2 * seq_len + 1


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------


def data_processing(
    problem="binary-copy",
    n_len=8,
    split_probas=0.5,
    n_data_per_len=2048,
    save_dir=None,
):
    """
    Training a Transformer model on a specified problem.

    Paramters
    ---------
    problem: str
        Problem to be solved. Currently supported are "binary-copy", "parity", and "no-cot".
    n_len: int
        Maximum number of lenghts for sequences.
    split_probas: float or list of float
        Percentage of train/test split, eventually specified by length.
    n_data_per_len: int
        Maximum number of data to generate for a given length.
    save_dir: str
        Path of the directory where to save the data.
    """
    match problem:
        case "binary-copy":
            problem = BinaryCopy(save_dir=save_dir)
        case "parity":
            problem = Parity(save_dir=save_dir)
        case "no-cot":
            problem = Parity(cot=False, save_dir=save_dir)
        case _:
            raise ValueError(f"Problem {problem} not recognized.")

    lengths = list(np.arange(n_len) + 1)

    if isinstance(split_probas, float):
        split_probas_by_len = split_probas * np.ones(len(lengths))
    else:
        split_probas_by_len = np.array(split_probas)
        assert len(split_probas_by_len) == n_len, "`split_probas` should be of size `n_len`"

    problem.generate_datafiles(n_data_per_len, split_probas_by_len, rng=RNG)


if __name__ == "__main__":
    import fire

    from cot.config import logging_datefmt, logging_format, logging_level

    logging.basicConfig(
        format=logging_format,
        datefmt=logging_datefmt,
        style="{",
        level=logging_level,
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(data_processing)
