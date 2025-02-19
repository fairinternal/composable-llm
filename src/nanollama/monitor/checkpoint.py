"""
Checkpoint manager

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta

#### Notes
Checkpointing does not support `DCP` yet.
See https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

Checkpointing is done `synchronously` in the main process, this may slow down the training process.
See https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html.
"""

import re
import shutil
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path, PosixPath
from types import TracebackType

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ..distributed import get_rank, is_master_process
from .monitor import Monitor

logger = getLogger("nanollama")


@dataclass
class CheckpointConfig:
    period: int = 0
    keep_only: int = 0
    sync_step: bool = True  # whether profiler step should be sync with optimizer step
    path: str = field(init=False, default="")

    def __check_init__(self):
        """Check validity of arguments."""
        assert self.path, "path was not set"


class Checkpointer(Monitor):
    """
    Checkpoint manager

    ### Attributes
    stateful_objects: various objects to checkpoints
    period: number of updates between each checkpoint
    keep_only: number of checkpoints to keep
    path: path to the checkpoint directory
    saved: whether the latest model has been saved

    ### Params
    config: configuration object
    stateful_objects: various objects to checkpoint
    model: model to checkpoint
    optimizer: optimizer to checkpoint

    #### Notes
    `model` and `optimizer` are not passed in `stateful_objects` as they require special treatment.
    """

    name = "{name}_{rank:05d}.pth"
    folder_name = "{:010d}"
    re_folder = r"\d{10}"
    re_digits = re.compile(r"\d+")

    def __init__(
        self,
        config: CheckpointConfig,
        stateful_objects: dict[str, Stateful],
        model: nn.Module,
        optimizer: Optimizer,
    ):
        super().__init__(config)

        self.period = config.period
        self.keep_only = config.keep_only
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.sync = config.sync_step

        # Create alias for the objects to monitor.
        self.model = model
        self.stateful_objects = stateful_objects

        self.stateful_objects.update({"model": model, "optimizer": optimizer})

        # self.states.update({"model": DCPModelWrapper(model)})
        # if optimizer is not None:
        #     self.states.update({"optimizer": DCPOptimizerWrapper(model, optimizer)})

        self.device_rank = get_rank()
        self.saved_step = 0
        self.step = 0

    @torch.no_grad()
    def __enter__(self) -> "Checkpointer":
        """
        Loading checkpoint if any
        """
        path = self.get_last_checkpoint_path(self.path)
        if not path:
            return self

        for name, object in self.stateful_objects.items():
            logger.info(f"Reloading {name} state")
            if name in ["model", "optimizer"]:
                file_path = path / self.name.format(name=name, rank=0)
            else:
                file_path = path / self.name.format(name=name, rank=self.device_rank)
            state_dict = torch.load(file_path)
            object.load_state_dict(state_dict)
            logger.info(f"{name} reloaded")

        self.saved_step = self.step

        return self

    def update(self, eval: bool = False) -> None:
        """
        Checkpoint model, optimizer, scheduler and training state

        ### Parameters
        eval: Whether to save the checkpoint for evaluation
        """
        # do not checkpoint twice
        if self.saved_step == self.step:
            return

        path = self.path / self.folder_name.format(self.step)
        path.mkdir(parents=False, exist_ok=True)

        # add evaluation flag, if needed
        if eval:
            eval_flag = path / "eval"
            eval_flag.touch()

        logger.info(f"Saving checkpoint at step {self.step} to {str(path)}")

        for name, object in self.stateful_objects.items():
            # only save the model and optimizer once
            if name in ["model", "optimizer"] and not is_master_process():
                continue
            logger.info(f"Saving {name} state")
            file_path = path / self.name.format(name=name, rank=self.device_rank)
            torch.save(object.state_dict(), file_path)

        self._cleaning()
        self.saved_step = self.step

    @classmethod
    def get_last_checkpoint_path(cls, path: str) -> str:
        """
        Get last existing checkpoint
        """
        folders = cls._list_checkpoints(path)
        if folders:
            return max(folders, key=lambda p: cls._get_key_step(p.name))
        return ""

    def _cleaning(self) -> None:
        """
        Clean up old checkpoints
        """
        if self.keep_only <= 0:
            return
        all_checkpoints = self._list_checkpoints(self.path)
        all_checkpoints.sort(key=lambda p: self._get_key_step(p.name))
        for prefix in all_checkpoints[: -self.keep_only]:
            if not (prefix / "eval").exists():
                logger.info(f"Removing: {str(prefix)}")
                shutil.rmtree(prefix)

    @classmethod
    def _list_checkpoints(cls, path: str) -> list[PosixPath]:
        """
        List all existing checkpoints
        """
        return [p for p in path.iterdir() if p.is_dir() and re.match(cls.re_folder, p.name)]

    @classmethod
    def _get_key_step(cls, name: str) -> int:
        return int(re.findall(cls.re_digits, name)[-1])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit checkpoint context by saving checkpoint if needed
        """
        self.update()


# ------------------------------------------------------------------------------
# Evaluation logic
# ------------------------------------------------------------------------------


class EvalCheckpointer:
    def __init__(self, model: nn.Module, path: str, train_step: int = None) -> None:
        self.model = model
        if train_step is None:
            path = Path(path)
            self.save_dir = Path(Checkpointer.get_last_checkpoint_path(path))
        else:
            self.save_dir = Path(path) / Checkpointer.folder_name.format(train_step)

    def __enter__(self):
        logger.info(f"Loading model from: {str(self.save_dir)}")
        state_dict = torch.load(self.save_dir / "checkpoint.pth", weights_only=True)
        self.model.load_state_dict(state_dict["model"])

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """
        Exit checkpoint context by remove `eval` flag

        #### See Also
        Checkpointer.update(eval=True)
        """
        eval_flag = self.save_dir / "eval"
        if eval_flag.exists():
            eval_flag.unlink()
