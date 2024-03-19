import torch
import hydra
import numpy as np
import os

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets

from rnnt.model import RNNTModel

import onnx
from onnxsim import simplify



@hydra.main(version_base=None, config_path="config", config_name="basic_sp.yaml")
def convert(cfg: DictConfig) -> None:
    export_dir = "export"

    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    featurizer = hydra.utils.instantiate(cfg.featurizer)
    
    model = RNNTModel(hydra.utils.instantiate(cfg.predictor),
                      hydra.utils.instantiate(cfg.encoder),
                      hydra.utils.instantiate(cfg.joint))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    example_mel_features = torch.randn(1, 80, 1000)
    encoder_scripted = torch.jit.script(model.encoder)
    torch.onnx.export(encoder_scripted, example_mel_features, os.path.join(export_dir, "encoder.onnx"), verbose=True)

    encoder_onnx = onnx.load(os.path.join(export_dir, "encoder.onnx"))
    encoder_onnx, check = simplify(encoder_onnx)

    assert check, "Simplified ONNX model could not be validated"
    onnx.save(encoder_onnx, os.path.join(export_dir, "encoder_simplified.onnx"))





if __name__ == "__main__":
    convert()