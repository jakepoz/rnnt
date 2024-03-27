import torch
import torch.nn.functional as F
import torchaudio
import os
import yaml
import hydra
import time
import argparse
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf


from rnnt.model import RNNTModel
from rnnt.util import save_tensor_json



def infer(checkpoint, audio, config_path=None) -> None:
    device = torch.device("cpu")

    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")

    with open(config_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    checkpoint = torch.load(checkpoint, map_location="cpu")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)


    featurizer = hydra.utils.instantiate(cfg.featurizer)
    
    model = RNNTModel(hydra.utils.instantiate(cfg.predictor),
                      hydra.utils.instantiate(cfg.encoder),
                      hydra.utils.instantiate(cfg.joint))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()

    print("Model loaded...")
   
    waveform, sample_rate = torchaudio.load(audio)

    assert sample_rate == 16000, "Sample rate must be 16kHz"

    with torch.no_grad():
        feats = featurizer(waveform)
        feats = feats.to(device)
  
        decoded_tokens = model.greedy_decode(feats, torch.tensor([feats.shape[2]]), max_length=100)
        decoded_text = tokenizer.decode(decoded_tokens)

        print("Decoded text: ", decoded_text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from a model and run tests on it")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("audio", type=str, help="Path to the wav file")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.yaml file (optional)")

    args = parser.parse_args()

    infer(args.checkpoint, args.audio, args.config)