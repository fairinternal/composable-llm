# Compositionality 

## TODOS

- make data processing and evaluation scripts more modular.
- how to use Fire with DAP (debugger adapter protocol)?

- integrate better attention map metrics
- write scripts to launch on the cluster

- find a better implementation for the dataloader.
Put it outside of the training loop, and just call a dataset name in the training loop.

Clear TODO:
- [ ] Find data with clean the attention maps, or change the training to make it cleaner.
    - implement automatic metric to find those clean maps (option: do it for all the matrix we predicted).
    - maybe run stuffs on the cluster.
- [ ] Look at skill transfert between binary problem and copying problem / look a bit into curriculum...
- [ ] Look at how the substraction of position embeddings is performed by the first MLP.
- [ ] Look at how the state updates is performed by the second MLP when we solve for the parity problem.
- [ ] Look at how the work get shared across heads when we have too much capacity.
- [ ] Implement a baseline without CoT

- Nice things to have
    - slurm launcher (with and without signal handling / resubmitting).
    - wandb logging and good referential for experiments.

Longer term research goals:
- [ ] Have a special first token that indicates the problem that has generated the sentence. Check how the transformer reuses circuits for one tasks to the others (it will learn useful generic skills such as copying previous tokens, but will also need to go through specific circuits).


Longer term implementation TODO:
- Unit tests
- Move EvaluationIO to a CSV file system
- Be more coherent between `n`, `nb` or `num` (always use `n`).

## Objective

Understand efficient machine learning, first by focusing on "linguistic rules" found by LLMs, later by studying how math could be solved with AI.

- Ultimatively we would like to shed lights on mechanisms that allow to learn and compose concepts, so to build better reasoning systems.
- We stay grounded by prioritizing experiments first.
- We stay focused by reasoning with and on LLMs first, notably through math tasks.

Code organization:
- we will keep several sub-repo for different projects, with the aim to have some transversal tools for all of them (experimental setup, models, progress measures...)

TODo:
- we need to write a small notes on what are the linguistic rules that we believe LLMs implement: compositionality, Zeillig Harris, ...

## Installation

The code requires Python 3.10+ (for `case` matching).
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch 2 (e.g., [PyTorch 2.0.1 requires python 3.10-](https://github.com/pytorch/pytorch/blob/2_0_fix_docs/torch/_dynamo/eval_frame.py#L377), and PyTorch 2.1 requires python 3.11- to use `torch.compile`).
```bash
conda create -n llm
conda activate llm
conda install pip
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)) (use `nvidia-smi` to check your CUDA driver)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
True
```
- Install this repo
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```

## Development
For formatting, I recommand using `black`, `flake8`, and `isort`.
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
