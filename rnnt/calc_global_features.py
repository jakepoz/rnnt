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
    featurizer = hydra.utils.instantiate(cfg.featurizer, mean=0.0, invstddev=1.0)
    

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


    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        mel_features = batch["mel_features"].to(device)  # Shape: [batch_size, feature_size, sequence_length]

        flat_features = mel_features.sum(dim=0)  # Sum over the batch dimension first

        total_mel_features_sum += flat_features.sum(dim=1)  # Sum over all instances, retain feature dimension
        total_mel_features_sq_sum += (flat_features ** 2).sum(dim=1)  # Sum the squares of the features
        total_mel_features_count += flat_features.size(1)  # Count the total number of feature instances


    # Step 3: Compute global average and variance after the loop
    global_average_mel_features = total_mel_features_sum / total_mel_features_count
    global_variance_mel_features = (total_mel_features_sq_sum / total_mel_features_count) - (global_average_mel_features ** 2)


    # Save to a json file
    with open("global_features.json", "w") as f:
        json.dump({
            "means": global_average_mel_features.tolist(),
            "invstddev": (1 / global_variance_mel_features.sqrt()).tolist()
        }, f, indent=4)




if __name__ == "__main__":
    calc_global_features()