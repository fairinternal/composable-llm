"""
Generate synthetic data to study LLM behaviors in controlled settings.

The sequences contain the following special tokens:
    token 0: begining of sentence,
    token 1: end of input,
    token 2: end of sentence.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import logging

import fire
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from cot.config import RAW_DIR, RNG

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Generic class
# -----------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """
    Attributes
    ----------
    data: tensor of size (nb_data, seq_len)
        Tensor with data ordered by sequence length.
    indices: tensor of int of size (len)
        Indices to delimitate difference sequence lengths.
    """

    data_dir = RAW_DIR
    prefix = None

    def __init__(self):
        pass

    @classmethod
    def generate_fixed_len_data(cls, seq_len, nb_data, rng=None):
        """Generate sequence with fixed sequence length."""
        raise NotImplementedError

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        raise NotImplementedError

    @classmethod
    def generate_datafiles(cls, max_data_per_len, split_probas_by_len, rng=None):
        """
        Test/train split.

        Parameters
        ----------
        max_data_per_len : int
            Maximum number of data points to generate for each sequence length.
        split_probas_by_len : list of float
            Proportion of data to put in the training set for each sequence length.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
        """

        logger.info(f"Generating data. Saving in {cls.data_dir}")

        if rng is None:
            rng = np.random.default_rng()

        cls.data_dir.mkdir(parents=True, exist_ok=True)
        for seq_len, split_proba in enumerate(split_probas_by_len):
            seq_len += 1
            data = cls.generate_fixed_len_data(seq_len=seq_len, nb_data=max_data_per_len, rng=rng)
            rng.shuffle(data)
            nb_train = int(split_proba * len(data))
            np.save(cls.data_dir / f"train_{seq_len}.npy", data[:nb_train])
            np.save(cls.data_dir / f"test_{seq_len}.npy", data[nb_train:])
            logger.debug(f"Sequences of length {seq_len} done. Saved in {cls.data_dir} ({nb_train}/{len(data)} split).")

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
            Data containing sequence of tokens with
                0: begining of sentence,
                1: end of input,
                2: end of sentence,
                x: other tokens generated by `generate_fixed_length_data`.
        indices : numpy.ndarray
            Indices to split the data by sequence length.

        Notes
        -----
        Should be called after `generate_datafiles`.
        """
        assert isinstance(lengths, list), "`lenghts` must be an a list of int."
        assert data_type in ["train", "test"], "`data_type` must be 'train' or 'test'."

        # memory preallocation
        # ... compute the data size by lenghts
        nb_data_by_lens = np.empty(len(lengths))
        for i, seq_len in enumerate(lengths):
            filename = self.data_dir / f"{data_type}_{seq_len}.npy"
            with open(filename, "rb") as f:
                version = np.lib.format.read_magic(f)
                header = np.lib.format._read_array_header(f, version)
            nb_data_by_lens[i] = header[0][0]

        # ... deduce memory allocation
        indices = np.cumsum(nb_data_by_lens, dtype=int)
        indices = np.insert(indices, 0, 0)
        data = np.full((indices[-1], self.get_len(max(lengths)) + 2), 2, dtype=np.int32)
        data[:, 0] = 0

        # load the data in the allocated memory
        for i, seq_len in enumerate(lengths):
            data[indices[i] : indices[i + 1], 1 : self.get_len(seq_len) + 1] = np.load(
                self.data_dir / f"{data_type}_{seq_len}.npy"
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
        train_data, indices = self.load_data(lengths, data_type=data_type)
        self.data = torch.from_numpy(train_data)
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
    data_dir = SequenceDataset.data_dir / prefix

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_fixed_len_data(cls, seq_len, nb_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        nb_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if nb_data is too small compared to all the potential sequences.

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
        if 2**seq_len < nb_data:
            nb_data = 2**seq_len
        length = cls.get_len(seq_len)
        data = np.empty((nb_data, length), dtype=np.int32)

        # input data
        # ... exhaustive case
        if 2**seq_len == nb_data:
            powers_of_two = 2 ** np.arange(seq_len)[::-1]
            data[:, :seq_len] = (np.arange(nb_data).reshape(-1, 1) & powers_of_two != 0).astype(np.int32)
        # ... non-exhaustive case
        else:
            data[:, :seq_len] = (rng.random((nb_data, seq_len)) > 0.5).astype(np.int32)
        data += 3

        # end of input
        data[:, seq_len] = 1

        # copying the data
        data[:, seq_len + 1 :] = data[:, :seq_len]
        return data

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        return 2 * seq_len + 1


class Copy(SequenceDataset):
    prefix = "copy"
    data_dir = SequenceDataset.data_dir / prefix
    vocab_size = 10

    def __init__(self, vocab_size=None):
        super().__init__()
        if vocab_size is not None:
            Copy.vocab_size = vocab_size

    @classmethod
    def generate_fixed_len_data(cls, seq_len, nb_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        nb_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if nb_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens with
                0: begining of sentence,
                1: end of input,
                2: end of sentence,
                x between 3 and 2 + `vocab_size`: some tokens.
        """
        logger.info(f"Generating data with vocabulary of size {cls.vocab_size}.")

        if rng is None:
            rng = np.random.default_rng()

        # input
        length = cls.get_len(seq_len)
        data = np.empty((nb_data, length), dtype=np.int32)
        data[:, :seq_len] = (rng.random((nb_data, seq_len)) * cls.vocab_size).astype(np.int32)
        data += 3

        # end of input
        data[:, seq_len] = 1

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
    prefix = "parity"
    data_dir = SequenceDataset.data_dir / prefix

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_fixed_len_data(cls, seq_len, nb_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        nb_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if nb_data is too small compared to all the potential sequences.

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
        if 2**seq_len < nb_data:
            nb_data = 2**seq_len
        length = cls.get_len(seq_len)
        data = np.empty((nb_data, length), dtype=np.int32)

        # input data
        # ... exhaustive case
        if 2**seq_len == nb_data:
            powers_of_two = 2 ** np.arange(seq_len)[::-1]
            data[:, :seq_len] = (np.arange(nb_data).reshape(-1, 1) & powers_of_two != 0).astype(np.int32)
        # ... non-exhaustive case
        else:
            data[:, :seq_len] = (rng.random((nb_data, seq_len)) > 0.5).astype(np.int32)

        # end of input
        data[:, seq_len] = -2

        # CoT data
        data[:, seq_len + 1 :] = np.cumsum(data[:, :seq_len], axis=1) % 2
        data += 3

        return data

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        return 2 * seq_len + 1


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------


def main(
    problem="binary-copy",
    n_len=8,
    split_probas=0.5,
    max_nb_data_per_len=2048,
):
    """
    Training a Transformer model on a specified problem.

    Paramters
    ---------
    problem: str
        Problem to be solved. Currently supported are "binary-copy" and "parity".
    n_len: int
        Maximum number of lenghts for sequences.
    split_probas: float or list of float
        Percentage of train/test split, eventually specified by length.
    max_nb_data_per_len: int
        Maximum number of data to generate for a given length.
    """
    match problem:
        case "binary-copy":
            Problem = BinaryCopy
        case "parity":
            Problem = Parity
        case _:
            raise ValueError(f"Problem {problem} not recognized.")

    lengths = list(np.arange(n_len) + 1)

    if isinstance(split_probas, float):
        split_probas_by_len = split_probas * np.ones(len(lengths))
    else:
        split_probas_by_len = np.array(split_probas)
        assert len(split_probas_by_len) == n_len, "`split_probas` should be of size `n_len`"

    Problem.generate_datafiles(max_nb_data_per_len, split_probas_by_len, RNG)


if __name__ == "__main__":
    logging.basicConfig(
        # format="{asctime} {levelname} [{filename}:{lineno}] {message}",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(main)
