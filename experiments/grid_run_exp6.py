"""
Example of grid run (past code on Vivien side).

To be modified to fit the current framework.
"""

import json
import logging
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from uuid import uuid4

import torch

from cot.config import CHECK_DIR, DATA_DIR
from cot.data import data_processing
from cot.train import train
from cot.utils import JsonEncoder

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
    n_len: int = 32
    split_probas: float = 0.5
    n_data_per_len: int = 2048
    zipf_offset: int = 0
    zipf_coef: float = 0

    # Model
    emb_dim: int = 128
    emb_dropout: float = 0.1
    pos_dim: int = None
    freeze_pos: bool = False
    n_head: int = 1
    n_layer: int = 2

    # Optimization
    n_epochs: int = 5000
    batch_size: int = 256
    learning_rate: float = 3e-4

    # Checkpointing
    checkpoint_freq: int = 100
    overwrite_checkpoint: bool = True
    load_checkpoint: bool = False
    check_dir: str = None
    full_eval: bool = False
    eval_freq: int = 10

    def __post_init__(self):
        self.unique_id = str(uuid4())
        if self.data_dir == "special":
            self.data_dir = DATA_DIR / self.unique_id

        if self.check_dir == "special":
            self.check_dir = CHECK_DIR / self.unique_id


def run_experiment(
    config,
):
    """
    Run one experiments associated with one config file

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
        pos_dim=config.pos_dim,
        freeze_pos=config.freeze_pos,
        n_head=config.n_head,
        n_layer=config.n_layer,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        checkpoint_freq=config.checkpoint_freq,
        overwrite_checkpoint=config.overwrite_checkpoint,
        load_checkpoint=config.load_checkpoint,
        check_dir=config.check_dir,
        full_eval=config.full_eval,
        eval_freq=config.eval_freq,
    )


def run_grid(
    num_tasks=1,
    task_id=1,
    config_filename=None,
):
    """
    Run a grid of experiments

    Parameters
    ----------
    num_tasks: int
        Number of threads to split the grid run into.
    task_id: int
        Id of the current thread.
    config_filename: str
        Where to save the configuration that generate the runs.
    """

    grid = {
        "emb_dim": range(1, 32),
        "n_len": range(4, 32),
        "freeze_pos": [True, False],
        "full_eval": [True],
    }

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    if config_filename is None:
        config_filename = "config"

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        config = MainConfig(
            check_dir="special",
            data_dir="special",
        )

        for k, v in zip(grid.keys(), values):
            setattr(config, k, v)

        config_dict = asdict(config)
        with open(CHECK_DIR / f"{config_filename}.jsonl", "a") as f:
            json.dump(config_dict, f, cls=JsonEncoder, indent=4)
            f.write("\n")

        try:
            logger.info(f"Running with {config}")
            run_experiment(config)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(e)
            logger.warning(traceback.format_exc())
            continue


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

    fire.Fire(run_grid)
