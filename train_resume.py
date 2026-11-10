"""Continue a run in place from its last completed checkpoint.

Launch from the repository root with the original config and GPU count:
    torchrun --standalone --nproc-per-node=2 train_resume.py config/example.py

To change the number of ranks while keeping the original global batch:
    torchrun --standalone --nproc-per-node=8 train_resume.py config/example.py --allow-world-size-change

For a single-GPU run, use python train_resume.py instead. Do not run this
while another job is still writing to the same out_dir.
"""

from pathlib import Path
from runpy import run_path
import sys


if __name__ == "__main__":
    # This is a resume option, not an override of the experiment configuration.
    flag = "--allow-world-size-change"
    allow_world_size_change = flag in sys.argv[1:]
    sys.argv[1:] = [arg for arg in sys.argv[1:] if arg != flag]
    # Use the same training loop, data loader, and evaluation code as a fresh run.
    run_path(
        str(Path(__file__).with_name("train.py")),
        init_globals={"_resume_requested": True, "_allow_world_size_change": allow_world_size_change},
        run_name="__main__",
    )
