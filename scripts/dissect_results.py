"""
Dissect results script

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch

from cot.config import (
    CHECK_DIR,
    SAVE_DIR,
    logging_datefmt,
    logging_format,
    logging_level,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=logging_format,
    datefmt=logging_datefmt,
    style="{",
    level=logging_level,
    handlers=[logging.StreamHandler()],
)

exp = 2
attention_eval = False

save_dir = SAVE_DIR / f"res-cot-exp{exp}"
save_dir.mkdir(parents=True, exist_ok=True)

all_configs = []
with open(CHECK_DIR / f"exp{exp}.jsonl", "r") as f:
    json_str = f.read()
    json_objs = json_str.split("}\n")
    for json_obj in json_objs:
        if json_obj:
            try:
                all_configs.append(json.loads(json_obj + "}"))
            except Exception as e:
                logger.info(e)
                logger.info(json_obj)
                continue


X = np.arange(4, 32)
Y = np.arange(8, 128)
Z = np.arange(0, 5001, 10)


Z1_parity = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
Z2_parity = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
Z1_nocot = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
Z2_nocot = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
Z1_copy = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
Z2_copy = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
if attention_eval:
    Z3_parity = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z4_parity = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z3_nocot = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z4_nocot = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z3_copy = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z4_copy = np.full((len(X), len(Y), len(Z)), -1, dtype=float)


for config in all_configs:
    data_dir = Path(config["data_dir"])
    problem = config["problem"]
    n_len = config["n_len"]
    emb_dim = config["emb_dim"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    check_dir = Path(config["check_dir"])

    try:
        checkpoint = torch.load(check_dir / "model.pth")
    except Exception as e:
        logger.warning(e)
        logger.warning("Problem with", problem, emb_dim, n_len)
        continue

    timestamps = checkpoint["timestamps"]
    ind = timestamps != -1
    timestamps = timestamps[ind]

    meaning = checkpoint["meaning"]
    evals = checkpoint["evals"][ind]

    eval_dim = evals.shape[1] // 2
    train_evals = evals[:, :eval_dim]
    test_evals = evals[:, eval_dim:]

    min_len = 4
    train_acc = train_evals[:, min_len - 1 : n_len].mean(axis=1)
    test_acc = test_evals[:, min_len - 1 : n_len].mean(axis=1)

    nd_meaning = np.array(meaning)
    if attention_eval:
        res = np.empty((2, n_len + 1 - min_len), dtype=float)
        for i, eval_prefix in enumerate(["attn0_peaky_thres", "attn1_peaky_thres"]):
            for j, length in enumerate(range(min_len, n_len + 1)):
                eval_name = f"{eval_prefix}_{length}"

                ind = np.argmax(np.array(meaning) == eval_name)

                train_res = train_evals[-1, ind]
                test_res = test_evals[-1, ind]
                res[i, j] = test_res

        res = res.mean(axis=1)

    x = np.argmax(X == n_len)
    y = np.argmax(Y == emb_dim)
    if problem == "parity":
        Z1_parity[x, y, : len(train_acc)] = train_acc
        Z2_parity[x, y, : len(test_acc)] = test_acc
        if attention_eval:
            Z3_parity[x, y] = res[0]
            Z4_parity[x, y] = res[1]
    elif problem == "no-cot":
        Z1_nocot[x, y, : len(train_acc)] = train_acc
        Z2_nocot[x, y, : len(test_acc)] = test_acc
        if attention_eval:
            Z3_nocot[x, y] = res[0]
            Z4_nocot[x, y] = res[1]
    elif problem == "binary-copy":
        Z1_copy[x, y, : len(train_acc)] = train_acc
        Z2_copy[x, y, : len(test_acc)] = test_acc
        if attention_eval:
            Z3_copy[x, y] = res[0]
            Z4_copy[x, y] = res[1]
    else:
        logger.error("Problem '{problem}' does not match excepted values.")

    logger.info(f"done with {problem}, {emb_dim}, {n_len}")


logging.info("Saving results.")
np.save(save_dir / "train_acc_parity.npy", Z1_parity)
np.save(save_dir / "test_acc_parity.npy", Z2_parity)
np.save(save_dir / "train_acc_nocot.npy", Z1_nocot)
np.save(save_dir / "test_acc_nocot.npy", Z2_nocot)
np.save(save_dir / "train_acc_copy.npy", Z1_copy)
np.save(save_dir / "test_acc_copy.npy", Z2_copy)

if attention_eval:
    np.save(save_dir / "attn0_parity.npy", Z3_parity)
    np.save(save_dir / "attn1_parity.npy", Z4_parity)
    np.save(save_dir / "attn0_nocot.npy", Z3_nocot)
    np.save(save_dir / "attn1_nocot.npy", Z4_nocot)
    np.save(save_dir / "attn0_copy.npy", Z3_copy)
    np.save(save_dir / "attn1_copy.npy", Z4_copy)

logger.info("Deleting checkpoints.")

for config in all_configs:
    data_dir = Path(config["data_dir"])
    check_dir = Path(config["check_dir"])

    try:
        subprocess.run(["rm", "-rf", data_dir])
    except Exception as e:
        logger.info(e)

    try:
        subprocess.run(["rm", "-rf", check_dir])
    except Exception as e:
        logger.info(e)
