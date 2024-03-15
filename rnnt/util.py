import torch
import os
from omegaconf import DictConfig

def save_model(model, optimizer, completed_steps, output_dir):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": completed_steps,
    }, os.path.join(output_dir, f"checkpoint_step_{completed_steps}.pt"))


def get_output_dir(cfg: DictConfig) -> str:
    output_dir = os.path.join("experiments", cfg.model_name)
    next_run = get_next_run_number(output_dir)
    output_dir = os.path.join(output_dir, f"run-{next_run}")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_next_run_number(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    run_nums = [int(run.split('-')[-1]) for run in existing_runs if 'run-' in run]
    if run_nums:
        return max(run_nums) + 1
    else:
        return 1