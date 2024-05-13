"""
Example of grid run (past code on Vivien side).

To be modified to fit the current framework.
"""

import logging
import time
from dataclasses import dataclass

import torch

from cot.config import CHECKPOINT_DIR, DATA_DIR
from cot.data import data_processing
from cot.train import train

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Reproducibility and Device
# -----------------------------------------------------------------------------

torch.manual_seed(100)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")


@dataclass
class MainConfig:
    # Data
    data_dir: str = None
    problem: str = "binary-copy"
    n_len: int = 16
    split_probas: float = 0.5
    n_data_per_len: int = 2048
    zipf_offset: int = 0
    zipf_coef: float = 0

    # Model
    emb_dim: int = 128
    emb_dropout: float = 0.1
    n_head: int = 1
    n_layer: int = 2

    # Optimization
    n_epochs: int = 1000
    batch_size: int = None
    learning_rate: float = 1e-3

    # Checkpointing
    checkpoint_freq: int = 100
    overwrite_checkpoint: bool = True
    load_checkpoint: bool = False
    check_dir: str = None
    eval_freq: int = 10

    def __post_init__(self):
        self.unique_id = str(int(time.time()))
        if self.data_dir == "special":
            self.data_dir = DATA_DIR / self.unique_id

        if self.check_dir is None:
            self.check_dir = CHECKPOINT_DIR / self.unique_id


def run_experiment(
    config,
):
    """
    Main script

    Parameters
    ----------
    data_dir: str
        Set `data_dir="special", to get a unique saving directory for your data mix.
    OTHER ARGUMENT TO DETAIL
    """

    data_processing(
        problem=config.problem,
        n_len=config.n_len,
        split_probas=config.split_probas,
        n_data_per_len=config.n_data_per_len,
        save_dir=config.data_dir,
    )

    train(
        problem=config.problem,
        data_dir=config.data_dir,
        n_len=config.n_len,
        zipf_offset=config.zipf_offset,
        zipf_coef=config.zipf_coef,
        emb_dim=config.emb_dim,
        emb_dropout=config.emb_dropout,
        n_head=config.n_head,
        n_layer=config.n_layer,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        checkpoint_freq=config.checkpoint_freq,
        overwrite_checkpoint=config.overwrite_checkpoint,
        load_checkpoint=config.load_checkpoint,
        check_dir=config.check_dir,
        eval_freq=config.eval_freq,
    )


def main(
    data_dir=None,
    problem="binary-copy",
    n_len=16,
    split_probas=0.5,
    n_data_per_len=2048,
    zipf_offset=0,
    zipf_coef=0,
    emb_dim=128,
    emb_dropout=0.1,
    n_head=1,
    n_layer=2,
    n_epochs=1000,
    batch_size=None,
    learning_rate=1e-3,
    checkpoint_freq=100,
    overwrite_checkpoint=True,
    load_checkpoint=False,
    check_dir=None,
    eval_freq=10,
):
    config = MainConfig(
        data_dir=data_dir,
        problem=problem,
        n_len=n_len,
        split_probas=split_probas,
        n_data_per_len=n_data_per_len,
        zipf_offset=zipf_offset,
        zipf_coef=zipf_coef,
        emb_dim=emb_dim,
        emb_dropout=emb_dropout,
        n_head=n_head,
        n_layer=n_layer,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_freq=checkpoint_freq,
        overwrite_checkpoint=overwrite_checkpoint,
        load_checkpoint=load_checkpoint,
        check_dir=check_dir,
        eval_freq=eval_freq,
    )

    run_experiment(config)


if __name__ == "__main__":
    debug = True

    import argparse
    from itertools import product

    from cot.config import logging_datefmt, logging_format, logging_level

    logging.basicConfig(
        format=logging_format,
        datefmt=logging_datefmt,
        style="{",
        level=logging_level,
        handlers=[logging.StreamHandler()],
    )

    if debug:
        import sys

        import fire

        fire.Fire(main)
        sys.exit()

    parser = argparse.ArgumentParser(description="CoT grid experiments")
    parser.add_argument("--num-tasks", default=100, type=int, help="Number of tasks to split the grid into.")
    parser.add_argument("--task-id", default=1, type=int, help="Task id, from 1 to `num_task`.")
    args = parser.parse_args()

    config = MainConfig()

    grid = {
        "batch_size": [64, 256, 1024, 2048, 4096],
        "learning_rate": [3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1e0],
    }

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % args.num_tasks != (args.task_id - 1):
            continue

        for k, v in zip(grid.keys(), values):
            setattr(config, k, v)

        try:
            run_experiment(config)
        except Exception:
            logger.warning(f"Error for configuration: {config}.")
            continue
