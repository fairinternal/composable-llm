"""
Example of grid run (past code on Vivien side).

To be modified to fit the current framework.
"""

import time

from cot.config import CHECKPOINT_DIR, DATA_DIR
from cot.data import data_processing
from cot.train import train


def main(
    save_dir=DATA_DIR,
    problem="binary-copy",
    n_len=8,
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

    if check_dir is None:
        check_dir = CHECKPOINT_DIR / str(time.time())

    data_processing(
        problem=problem,
        n_len=n_len,
        split_probas=split_probas,
        n_data_per_len=n_data_per_len,
        save_dir=save_dir,
    )

    train(
        problem=problem,
        n_len=n_len,
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


if __name__ == "__main__":
    import fire

    fire.Fire(main)
