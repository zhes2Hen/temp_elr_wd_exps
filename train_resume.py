"""Continue a run in place from its last completed checkpoint.

Launch from the repository root with the original config and GPU count:
    torchrun --standalone --nproc-per-node=2 train_resume.py config/example.py

For a single-GPU run, use python train_resume.py instead. Do not run this
while another job is still writing to the same out_dir.
"""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    # Use the same training loop, data loader, and evaluation code as a fresh run.
    run_path(
        str(Path(__file__).with_name("train.py")),
        init_globals={"_resume_requested": True},
        run_name="__main__",
    )
