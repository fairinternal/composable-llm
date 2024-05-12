"""
Example of grid run (past code on Vivien side).

To be modified to fit the current framework.
"""

# Grid runs
grid = {
    "n": np.unique(np.logspace(2, 4, num=10).astype(int)),
    "d": np.arange(3, 20, 2),
    "p": np.unique(np.logspace(1.5, 3, num=5).astype(int)),
    "seed": list(range(100)),
}

if config.kernel == "polynomial":
    grid.update(
        {
            "graph_laplacian": [0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["polynomial"],
            "kernel_param": [2, 3, 4, 5, 6],
        }
    )
elif config.kernel == "exponential":
    grid.update(
        {
            "graph_laplacian": [0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["exponential"],
            "kernel_param": [1, 10, 100, 1000],
        }
    )
elif config.kernel == "gaussian":
    grid.update(
        {
            "graph_laplacian": [0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["gaussian"],
            "kernel_param": [0.1, 1, 10, 100],
        }
    )

# Output file
outdir = Path(config.save_dir) / config.name / config.kernel
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / f"task_{config.task_id}.jsonl"

# Clean file
with open(outfile, "w") as f:
    pass

logger.info(f"Number of experiments: {len(list(product(*grid.values()))) // config.num_tasks}")

# Run grids
for i, vals in enumerate(product(*grid.values())):
    # Splitting the grid into tasks
    if i % config.num_tasks != (config.task_id - 1):
        continue
    # Setting configuration
    for k, v in zip(grid.keys(), vals):
        setattr(config, k, v)
    # Running experiment
    try:
        res = run_exp(config)
    except Exception:
        logger.warning(f"Error for configuration: {config}")
        continue
    # Saving results
    with open(outdir / f"task_{config.task_id}.jsonl", "a") as f:
        print(json.dumps(res, cls=NpEncoder), file=f)
