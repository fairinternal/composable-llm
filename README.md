# Compositionality 

## TODOS

Middle term goals:
- [ ] Write training loop
- [ ] Write cluster scripts
- [ ] Make sure that everything run smoothly and fast on the cluster
- [ ] Launch first grid experiments

## Objective

Push a learning perspective linked with compositionality.

- Stay grounded by prioritizing experiments first.
- Stay focus by reasoning with and on LLMs first, notably through math tasks.
- Ultimatively we would like to shed lights on mechanisms that allow to learn and compose concepts, so to build better reasoning systems.

TODO:
- we will keep several sub-repo for different projects, with the aim to have some transversal tools for all of them (experimental setup, models, progress measures...)

## Installation

The code requires Python 3.10+ (for `case` matching).
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch 2 (e.g., [PyTorch 2.0.1 requires python 3.10-](https://github.com/pytorch/pytorch/blob/2_0_fix_docs/torch/_dynamo/eval_frame.py#L377), and PyTorch 2.1 requires python 3.11- to use `torch.compile`).
```bash
$ conda create -n llm
$ conda activate llm
$ conda install pip
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)) (use `nvidia-smi` to check your CUDA driver)
```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cu118
$ python -c "import torch; print(torch.cuda.is_available())"
True
```
- Install this repo
```bash
$ git clone <repo url>
$ cd <repo path>
$ pip install -e .
```

## Development
For formatting, I recommand using `black`, `flake8`, and `isort`.
To configure `flake8` through the `pyproject.toml` file, you can use the `flake8-pyproject` library.
Consider automatic formatting when saving files (easy to setup in VSCode, ask ChatGPT to get set up if not confortable with VSCode configuration).

## Organization
The main code is in the `src` folder.
Other folders include:
- `data`: contains data used in the experiments.
- `launchers`: contains bash scripts to launch experiments
- `models`: saves models' weights.
- `notebooks`: used for exploration and visualization.
- `scripts`: contains python scripts to run experiments.
- `tests`: contains tests for the code.
- `tutorial`: contains tutorial notebooks to get started with LLMs' training.