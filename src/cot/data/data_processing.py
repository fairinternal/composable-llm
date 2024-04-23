"""
Generate synthetic data to study LLM behaviors in controlled settings.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------
# Parity problem data generation
# --------------------------------


def fixed_length_parity_data(seq_len, nb_data=None, random=True, rng=None):
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


def test_train_split(split_probas_by_len, data_dir, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    for seq_len, split_proba in enumerate(split_probas_by_len):
        seq_len += 1
        data = fixed_length_parity_data(seq_len, nb_data=None, random=False, rng=rng)
        np.save(data_dir / f"data_{seq_len}.npy", data)
        rng.shuffle(data)
        nb_train = int(split_proba * len(data))
        np.save(data_dir / f"train_{seq_len}.npy", data[:nb_train])
        np.save(data_dir / f"test_{seq_len}.npy", data[nb_train:])
        logger.info(f"Sequences of length {seq_len} done")
