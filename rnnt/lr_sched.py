import torch
import math

import torch
import math

class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler that combines linear warmup with cosine decay, with a minimum learning rate ratio.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to use.
        warmup_steps (int): Number of steps for the linear warmup phase.
        total_steps (int): Total number of scheduler steps, including warmup.
        min_lr_ratio (float, optional): Minimum learning rate ratio. Default: 0.05.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        verbose (bool, optional): If `True`, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio=0.05,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            warmup_factor = self._step_count / max(1, self.warmup_steps)
        else:
            progress = (self._step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed_factor = (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio  # Adjust decay for min_lr_ratio
            warmup_factor = decayed_factor
        
        return [base_lr * warmup_factor for base_lr in self.base_lrs]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Setup for demonstration
    model = torch.nn.Linear(1, 1)  # A simple model for demonstration purposes
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer with the initial learning rate
    scheduler = WarmupCosineDecayLR(optimizer, warmup_steps=2000, total_steps=100000)

    # Record learning rates
    lrs = []
    for step in range(100000):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label="Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule: Warmup and Cosine Decay")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save to file
    plt.savefig("learning_rate_schedule.png")