"""
Generate synthetic data to study LLM behaviors in controlled settings.
"""

import logging

import numpy as np

from cot.config import RAW_DIR

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Copy problem
# -----------------------------------------------------------------------------

# Be exhaustive with 0/1 sequences.
# Be exhaustive with all the sequences over a vocabulary of size K.
# Take random sequences over a larger vocabulary.


# -----------------------------------------------------------------------------
# Parity problem data generation
# -----------------------------------------------------------------------------


class Parity:
    def generate_fixed_length_data(cls, seq_len, nb_data=None, random=True, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        nb_data : int
            Number of data points to generate.
        random : bool, optional
            If True, generate random data. If False, generate all possible sequences.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens with
                0: zero bit,
                1: one bit,
                2: end of input.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Input data
        if random:
            assert nb_data is not None, "nb_data must be provided if random is True."
            data = np.empty((nb_data, 2 * seq_len + 1), dtype=np.int32)
            data[:, :seq_len] = (rng.random((nb_data, seq_len)) > 0.5).astype(np.int32)
        else:
            assert nb_data is None, "nb_data must be None if random is False."
            nb_data = 2**seq_len
            data = np.empty((nb_data, 2 * seq_len + 1), dtype=np.int32)
            all_seq = np.arange(nb_data).reshape(-1, 1)
            powers_of_two = 2 ** np.arange(seq_len)
            data[:, :seq_len] = (all_seq & powers_of_two != 0).astype(np.int32)

        # End of input
        data[:, seq_len] = 2

        # CoT data
        data[:, seq_len + 1 :] = np.cumsum(data[:, :seq_len], axis=1) % 2
        return data

    def test_train_split(cls, split_probas_by_len, rng=None):
        """
        Test/train split

        Parameters
        ----------
        split_probas_by_len : list of float
            Proportion of data to put in the training set for each sequence length.
        data_dir : pathlib.Path
            Directory where to save the data.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
        """
        data_dir = RAW_DIR / "parity"
        data_dir.mkdir(parents=True, exist_ok=True)

        if rng is None:
            rng = np.random.default_rng()

        for seq_len, split_proba in enumerate(split_probas_by_len):
            seq_len += 1
            data = cls.generate_fixed_length_data(seq_len, nb_data=None, random=False, rng=rng)
            np.save(data_dir / f"data_{seq_len}.npy", data)
            rng.shuffle(data)
            nb_train = int(split_proba * len(data))
            np.save(data_dir / f"train_{seq_len}.npy", data[:nb_train])
            np.save(data_dir / f"test_{seq_len}.npy", data[nb_train:])
            logger.info(f"Sequences of length {seq_len} done")

    def load_test_data(cls, lengths, data_type=None):
        """
        Load test data.

        Parameters
        ----------
        lengths : list of int
            List of sequence lengths.
        data_type : str, optional
            Type of data to load. Whether 'train', 'test' or 'all'.

        Returns
        -------
        data : numpy.ndarray
            Data containing sequence of tokens with
                0: zero bit,
                1: one bit,
                2: end of input,
                3: end of sequence.
        indices : numpy.ndarray
            Indices to split the data by sequence length.
        """
        assert isinstance(lengths, list), "`lenghts` must be an a list of int."
        assert data_type in ["train", "test", "all"], "`data_type` must be 'train', 'test' or 'all'."

        prefix = data_type
        if data_type == "all":
            prefix = "data"

        # memory preallocation
        # ... compute the data size by lenghts
        nb_data_by_lens = np.empty(len(lengths))
        for i, seq_len in enumerate(lengths):
            filename = RAW_DIR / f"parity/{prefix}_{seq_len}.npy"
            with open(filename, "rb") as f:
                version = np.lib.format.read_magic(f)
                header = np.lib.format._read_array_header(f, version)
            nb_data_by_lens[i] = header[0][0]

        # ... deduce memory allocation
        indices = np.cumsum(nb_data_by_lens, dtype=int)
        indices = np.insert(indices, 0, 0)
        data = np.full((indices[-1], 2 * max(lengths) + 2), 3, dtype=np.int32)

        # load the data in the allocated memory
        for i, seq_len in enumerate(lengths):
            data[indices[i] : indices[i + 1], : 2 * seq_len + 1] = np.load(RAW_DIR / f"parity/{prefix}_{seq_len}.npy")
        return data, indices

    def generate_train_data(train_data, indices, length_proba, nb_data, rng=None):
        pass

    def former_generate_train_data(cls, length_proba=None, nb_data=None, random=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        nb_seq_by_len = rng.multinomial(nb_data, length_proba)
        max_len = 2 * len(nb_seq_by_len) + 2
        data = np.full((nb_data, max_len), 3, dtype=np.int32)
        # generate data for each length
        ind_data = 0
        for seq_len, nb_seq in enumerate(nb_seq_by_len):
            seq_len += 1
            data[ind_data : ind_data + nb_seq, : 2 * seq_len + 1] = cls.generate_fixed_length_data(
                seq_len, nb_data=nb_seq, random=True, rng=rng
            )
            ind_data += nb_seq
        # shuffle data
        rng.shuffle(data)
        return data
