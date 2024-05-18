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

exp = 1
attention_eval = False
problems = ["binary-copy", "no-cot", "parity"]

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

Z1, Z2, Z3, Z4 = {}, {}, {}, {}

for problem in problems:
    Z1[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z2[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)

    if attention_eval:
        Z3[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
        Z4[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)


logger.info("Parsing results.")
for config in all_configs:
    data_dir = Path(config["data_dir"])
    problem = config["problem"]
    n_len = config["n_len"]
    emb_dim = config["emb_dim"]
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
    try:
        Z1[problem][x, y, : len(train_acc)] = train_acc
        Z2[problem][x, y, : len(test_acc)] = test_acc
        if attention_eval:
            Z3[problem][x, y] = res[0]
            Z4[problem][x, y] = res[1]
    except Exception as e:
        logger.error(e)
        logger.error("Problem '{problem}' does not match excepted values.")

    logger.info(f"done with {problem}, {emb_dim}, {n_len}")


logging.info("Saving results.")
for problem in problems:
    np.save(save_dir / f"train_acc_{problem}.npy", Z1[problem])
    np.save(save_dir / f"test_acc_{problem}.npy", Z2[problem])
    if attention_eval:
        np.save(save_dir / f"attn0_{problem}.npy", Z3[problem])
        np.save(save_dir / f"attn1_{problem}.npy", Z4[problem])


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
