import torch
import torch.nn.functional as F
import torchaudio
import os
import yaml
import hydra
import time
import jiwer
import argparse
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets
from torch.utils.tensorboard import SummaryWriter

from rnnt.model import RNNTModel
from rnnt.util import save_tensor_json



def eval(checkpoint, config_path=None) -> None:
    device = torch.device("cuda")

    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")

    with open(config_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    checkpoint = torch.load(checkpoint, map_location="cpu")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    eval_ds = hydra.utils.instantiate(cfg.data.eval.dataset, _convert_="object")
    if isinstance(eval_ds, list):
        eval_ds = concatenate_datasets(eval_ds)

    featurizer = hydra.utils.instantiate(cfg.featurizer)
    
    model = RNNTModel(hydra.utils.instantiate(cfg.predictor),
                      hydra.utils.instantiate(cfg.encoder),
                      hydra.utils.instantiate(cfg.joint))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)

    # Wrap those in the processor class, which can provide augmentations, tokenization, etc.
    ds_processor = hydra.utils.get_class(cfg.data.processor_class)
    eval_ds = ds_processor(eval_ds, tokenizer, featurizer, torch.device("cpu"))

    print(f"Eval dataset size : {len(eval_ds)}")

    eval_dataloader = hydra.utils.instantiate(cfg.data.eval.dataloader, eval_ds)

    params = model.parameters()

    print(f"Number of predictor parameters: {sum(p.numel() for p in model.predictor.parameters()):,}")
    print(f"Number of encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Number of joint parameters: {sum(p.numel() for p in model.joint.parameters()):,}")

    model.eval()

    print("Model loaded...")
    sample_mel_features = torch.zeros(1, cfg.featurizer.n_mels, 1000).to(device)
    sample_audio_features = model.encoder(sample_mel_features).permute(0, 2, 1)

    sample_tokens = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
    sample_text_features = model.predictor(sample_tokens)

    sample_joint1 = torch.zeros(1, 1, cfg.encoder.output_features).to(device)
    sample_joint2 = torch.zeros(1, 1, cfg.encoder.output_features).to(device)

    sample_joint = model.joint.forward(sample_joint1, sample_joint2)
    print(sample_joint)

    print("Starting eval...")
    originals, decoded = [], []

    start_time = time.perf_counter()

    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        mel_features = batch["mel_features"].to(device)
        mel_feature_lens = batch["mel_feature_lens"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        decoded_tokens = model.greedy_decode(mel_features, mel_feature_lens, max_length=batch["input_ids"].shape[1] * 2)
        decoded_text = tokenizer.decode(decoded_tokens)
        original_text = tokenizer.decode(input_ids[0].cpu().tolist())

        print(f"\nOriginal: {original_text}\nDecoded : {decoded_text}")

        # Grab some sample mels to test with on the tfjs side
        # if step == 0:
        #     with open("samplemels.json", "w") as f:
        #         f.write(save_tensor_json(mel_features[0]))

        originals.append(original_text)
        decoded.append(decoded_text)

        if step > cfg.data.eval.max_elements:
            break

    # Calculate overall wer using jiwer
    wer = jiwer.wer(originals, decoded)

    print(f"Done with eval... WER = {wer:.3f}")
    print(f"Time per sample: {(time.perf_counter() - start_time) / len(originals):.3f} s")
    model.train()

      
    # Returns last wer
    return wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from a model and run tests on it")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.yaml file (optional)")
    args = parser.parse_args()

    eval(args.checkpoint, args.config)