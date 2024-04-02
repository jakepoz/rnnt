import torch
import torch.nn.functional as F
import torchaudio
import hydra
import os
import json
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="config", config_name="basic_sp_convjs.yaml")
def calc_global_features(cfg: DictConfig) -> None:
    device = torch.device("cuda")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    train_ds = hydra.utils.instantiate(cfg.data.train.dataset, _convert_="object")
    if isinstance(train_ds, list):
        train_ds = concatenate_datasets(train_ds)
    eval_ds = hydra.utils.instantiate(cfg.data.eval.dataset, _convert_="object")
    if isinstance(eval_ds, list):
        eval_ds = concatenate_datasets(eval_ds)

    # Override the mean and invstddev so the normalization is not applied
    #featurizer = hydra.utils.instantiate(cfg.featurizer, mean=0.0, invstddev=1.0)
        
    featurizer = hydra.utils.instantiate(cfg.featurizer)
    

    # Wrap those in the processor class, which can provide augmentations, tokenization, etc.
    ds_processor = hydra.utils.get_class(cfg.data.processor_class)
    train_ds = ds_processor(train_ds, tokenizer, featurizer, torch.device("cpu"))
    eval_ds = ds_processor(eval_ds, tokenizer, featurizer, torch.device("cpu"))

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Eval dataset size : {len(eval_ds)}")

    # Force batch size to 1 so there is no weird padding that messes up the count
    train_dataloader = hydra.utils.instantiate(cfg.data.train.dataloader, train_ds, batch_size=1)

    total_mel_features_sum = torch.zeros(cfg.encoder.input_features, device=device)
    total_mel_features_sq_sum = torch.zeros(cfg.encoder.input_features, device=device)
    total_mel_features_count = 0

    # Initialize histogram variables
    histogram_bins = 500  # You can adjust the number of bins
    min_value, max_value = 0.000000001, 10000  # Adjust these based on your expected feature value range
    # Prepare logarithmically spaced bin edges
    log_bin_edges = np.logspace(np.log10(min_value), np.log10(max_value), histogram_bins + 1)
    histogram = torch.zeros(histogram_bins)

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        mel_features = batch["mel_features"].to(device)  # [batch_size, feature_size, sequence_length]

        flat_features = mel_features.sum(dim=0)  # Sum over the batch dimension

        total_mel_features_sum += flat_features.sum(dim=1)
        total_mel_features_sq_sum += (flat_features ** 2).sum(dim=1)
        total_mel_features_count += flat_features.size(1)

        # Update histogram for channel 0 values
        channel_0_values = flat_features[2, :]  # Get channel 0
        hist, _ = np.histogram(channel_0_values.cpu().numpy(), bins=log_bin_edges)
        histogram += hist

        if step > 1000:
            break

    global_average_mel_features = total_mel_features_sum / total_mel_features_count
    global_variance_mel_features = (total_mel_features_sq_sum / total_mel_features_count) - (global_average_mel_features ** 2)

    with open("global_features.json", "w") as f:
        json.dump({
            "means": global_average_mel_features.tolist(),
            "invstddev": (1 / global_variance_mel_features.sqrt()).tolist()
        }, f, indent=4)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    bin_centers = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2
    plt.bar(bin_centers, histogram, align='center', width=np.diff(log_bin_edges), edgecolor='black')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Logarithmic Histogram of Channel 0 Values')
    plt.xlabel('Value (log scale)')
    plt.ylabel('Count')
    plt.savefig('log_channel_0_histogram.png')

if __name__ == "__main__":
    calc_global_features()
