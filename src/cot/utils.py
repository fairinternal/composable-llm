"""
Utils functions
"""

import os
import sys
from json import JSONEncoder
from pathlib import PosixPath

# -----------------------------------------------------------------------------
# Json Serializer
# -----------------------------------------------------------------------------


class JsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PosixPath):
            return str(obj)
        return super().default(obj)


# -----------------------------------------------------------------------------
# Slurm signal handling
# -----------------------------------------------------------------------------


def handle_sig(signum, frame):
    print(f"Requeuing after {signum}...", flush=True)
    os.system(f'scontrol requeue {os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}')
    sys.exit(-1)


def handle_term(signum, frame):
    print("Received TERM.", flush=True)
