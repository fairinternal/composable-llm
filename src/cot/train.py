"""
Training script

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import logging
import sys

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cot.config import CHECKPOINT_DIR
from cot.data import BinaryCopy, Parity
from cot.models import Transformer, TransformerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    # format="{asctime} {levelname} [{filename}:{lineno}] {message}",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

# -----------------------------------------------------------------------------
# Reproducibility and Device
# -----------------------------------------------------------------------------

rng = np.random.default_rng(0)

torch.manual_seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")


def main(
    problem="binary-copy",
    nb_len=8,
    split_probas=0.5,
    max_nb_data_per_len=10_000,
    zipf_offset=0,
    zipf_coef=0,
    emb_dim=128,
    emb_dropout=0.1,
    n_head=2,
    n_layer=2,
    nb_epochs=1000,
    learning_rate=1e-3,
    checkpoint_freq=100,
    overwrite_checkpoint=True,
    load_checkpoint=False,
    eval_freq=10,
    flash=None,
):
    """
    Training a Transformer model on a specified problem.

    Paramters
    ---------
    problem: str
        Problem to be solved. Currently supported are "binary-copy" and "parity".
    nb_len: int
        Maximum number of lenghts for sequences.
    split_probas: float or list of float
        Percentage of train/test split, eventually specified by length.
    max_nb_data_per_len: int
        Maximum number of data to generate for a given length.
    zipf_offset: float
        Index offset to the Zipf law generating sentence lengths.
    zipf_coef: float
        Decaying coefficient to the Zipf law generating sentence lengths.
    emb_dim: int
        Embedding dimension size.
    emb_dropout: float
        Dropout rate for the embeddings.
    n_head: int
        Number of attention heads.
    n_layer: int
        Number of layers.
    nb_epochs: int
        Total number of training epochs.
    learning_rate: float
        Learning rate.
    checkpoint_freq: int
        Checkpoint saving frequency.
    overwrite_checkpoint: bool
        Whether to overwrite existing checkpoints or not.
    load_checkpoint: bool
        Whether to load a previous checkpoint for continuing training.
    eval_freq: int
        Evaluation frequency.
    flash: bool
        Use flash attention or not.
    """

    # -----------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------

    match problem:
        case "binary-copy":
            Problem = BinaryCopy
        case "parity":
            Problem = Parity
        case _:
            raise ValueError(f"Problem {problem} not recognized.")

    # hyperparameters
    lengths = list(np.arange(nb_len) + 1)

    if isinstance(split_probas, float):
        split_probas_by_len = split_probas * np.ones(len(lengths))
    else:
        split_probas_by_len = np.array(split_probas)
        assert len(split_probas_by_len) == nb_len, "`split_probas` should be of size `nb_len`"

    probas_by_len = (np.arange(len(lengths), dtype=float) + zipf_offset) ** (-zipf_coef)
    probas_by_len /= probas_by_len.sum()

    # main objects
    if Problem.prefix == "copy":
        Problem(vocab_size=20)

    Problem.generate_datafiles(max_nb_data_per_len, split_probas_by_len, rng)

    trainset = Problem()
    trainset.set_as_trainset(lengths, probas_by_len)

    testset = Problem()
    testset.set_as_testset(lengths)

    loader = DataLoader(trainset, batch_size=len(trainset), sampler=trainset.sampler)
    logger.info(f"Number of training data: {len(trainset)}.")

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------

    config = TransformerConfig(
        vocab_size=torch.max(trainset.data).item() + 1,
        emb_dim=emb_dim,
        pos_emb=True,
        seq_len=len(trainset[0]),
        emb_dropout=emb_dropout,
        n_head=n_head,
        n_layer=n_layer,
        flash=flash,
    )

    losses = np.empty(nb_epochs)

    check_dir = CHECKPOINT_DIR / Problem.prefix
    check_dir.mkdir(parents=True, exist_ok=True)

    model = Transformer(config)
    logger.info(f"Model: {model}.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nb_epochs // 3, gamma=0.5)

    logger.info(f"Device used: {device}.")
    model.to(device)
    probas_by_len = torch.from_numpy(probas_by_len).to(device=device)

    if load_checkpoint:
        path = check_dir / "model.pth"
        logger.info(f"Loading from checkpoint {path}.")
        checkpoint = torch.load(path)

        epoch = checkpoint["epoch"]

        if epoch > nb_epochs:
            logger.error(f"Model has been trained for {epoch} epochs, which is higher than {nb_epochs}")
            sys.exit()
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        losses[:epoch] = checkpoint["losses"][:epoch]
    else:
        epoch = 0

    # --------------------------------------------------------------------------
    # Evaluation Placeholder
    # --------------------------------------------------------------------------

    if load_checkpoint:
        nb_eval = (nb_epochs - epoch) // eval_freq + 1
        eval = checkpoint["evals"].argmax() + 1

        acc_by_len = np.empty((nb_eval + eval, len(lengths)))
        test_acc_by_len = np.empty((nb_eval + eval, len(lengths)))
        spe_acc = np.empty((nb_eval + eval, 3))
        test_spe_acc = np.empty((nb_eval + eval, 3))
        evals = np.full(nb_eval + eval, -1, dtype=int)

        acc_by_len[:eval] = checkpoint["acc_by_len"][:eval]
        test_acc_by_len[:eval] = checkpoint["test_acc_by_len"][:eval]
        spe_acc[:eval] = checkpoint["spe_acc"][:eval]
        test_spe_acc[:eval] = checkpoint["test_spe_acc"][:eval]
        evals[:eval] = checkpoint["evals"][:eval]

        epoch = checkpoint["epoch"]
    else:
        nb_eval = nb_epochs // eval_freq + 1
        eval = 0

        acc_by_len = np.empty((nb_eval, len(lengths)))
        test_acc_by_len = np.empty((nb_eval, len(lengths)))
        spe_acc = np.empty((nb_eval, 3))
        test_spe_acc = np.empty((nb_eval, 3))
        evals = np.full(nb_eval, -1, dtype=int)

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------

    logger.info(f"Starting Training from epoch {epoch}.")
    while True:

        # evaluation
        if not epoch % eval_freq:
            with torch.no_grad():
                model.eval()
                _, seq_err, spe_err = trainset.eval_model(model, special=True)
                _, test_seq_err, test_spe_err = testset.eval_model(model, special=True)
                accuracy = 1 - (seq_err * probas_by_len).sum().item()
                test_accuracy = 1 - (test_seq_err * probas_by_len).sum().item()

            logger.info(f"Epoch {epoch:5d}, Accuracy: {accuracy:.4f}, {test_accuracy:.4f}")
            s = epoch // eval_freq
            acc_by_len[s] = 1 - seq_err.cpu()
            test_acc_by_len[s] = 1 - test_seq_err.cpu()
            spe_acc[s] = 1 - spe_err.cpu()
            test_spe_acc[s] = 1 - test_spe_err.cpu()
            evals[s] = epoch
            eval += 1

        if epoch >= nb_epochs:
            break

        epoch = epoch + 1

        # training
        model.train()
        running_loss = 0
        accuracy = 0
        for sequence in loader:
            sequence = sequence.to(device=device, dtype=torch.long)

            inputs = sequence[:, :-1]
            targets = sequence[:, 1:]

            # only train on the chain-of-thoughts process, EoI is represented by 1 in our case
            ind = targets == 1
            cot_mask = ind.cumsum(axis=1)
            cot_mask[ind] = 0
            cot_mask = cot_mask.to(dtype=bool)

            logits = model(inputs)
            loss = F.cross_entropy(logits[cot_mask].view(-1, logits.size(-1)), targets[cot_mask].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                running_loss += loss.item()

        losses[epoch - 1] = loss

        logger.info(f"Epoch {epoch:5d}, Loss: {running_loss:.4f}")

        # checkpointing
        if not epoch % checkpoint_freq or epoch == nb_epochs:
            if overwrite_checkpoint:
                path = check_dir / "model.pth"
            else:
                path = check_dir / f"model_{epoch}.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "losses": losses,
                    "acc_by_len": acc_by_len,
                    "test_acc_by_len": test_acc_by_len,
                    "spe_acc": spe_acc,
                    "test_spe_acc": test_spe_acc,
                    "evals": evals,
                },
                path,
            )

            logger.info(f"Checkpointing model at {path}.")


if __name__ == "__main__":
    fire.Fire(main)
