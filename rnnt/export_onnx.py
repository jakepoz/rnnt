import torch
import argparse
import numpy as np
import os
import hydra
import yaml
import json
import sentencepiece as spm

from omegaconf import DictConfig, OmegaConf

from datasets import concatenate_datasets

from rnnt.model import RNNTModel


def convert(checkpoint, config_path=None, export_dir="export"):
    os.makedirs(export_dir, exist_ok=True)

    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")

    with open(config_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    checkpoint = torch.load(checkpoint, map_location="cpu")

    tokenizer = hydra.utils.instantiate(cfg["tokenizer"])
    featurizer = hydra.utils.instantiate(cfg["featurizer"])
    
    model = RNNTModel(hydra.utils.instantiate(cfg["predictor"]),
                      hydra.utils.instantiate(cfg["encoder"]),
                      hydra.utils.instantiate(cfg["joint"]))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    example_mel_features = torch.randn(1, cfg.featurizer.n_mels, 1000)
    encoder_scripted = torch.jit.script(model.encoder)
    torch.onnx.export(encoder_scripted, example_mel_features, os.path.join(export_dir, "encoder.onnx"), verbose=True,
                      dynamic_axes={"mel_features": {2: "frames"}},
                      input_names=["mel_features"],
                      output_names=["audio_features"])

    example_tokens = torch.ones(1, 10, dtype=torch.long)

    predictor_scripted = torch.jit.script(model.predictor)
    torch.onnx.export(predictor_scripted, example_tokens, os.path.join(export_dir, "predictor.onnx"), verbose=True,
                      dynamic_axes={"tokens": {1: "num_tokens"}},
                      input_names=["tokens"],
                      output_names=["text_features"])


    joint_scripted = torch.jit.script(model.joint)
    example_audio_frame = torch.randn(1, 1, cfg.encoder.output_features)
    example_text_frame = torch.randn(1, 1, cfg.predictor.output_dim)
    torch.onnx.export(joint_scripted, (example_audio_frame, example_text_frame), os.path.join(export_dir, "joint.onnx"), verbose=True,
                      input_names=["audio_frame", "text_frame"],
                      output_names=["logits"])

    # Output sentence piece data in a simple json format for now
    with open(os.path.join(export_dir, "tokenizer.json"), "w") as f:
        data = {}
        for i in range(tokenizer.get_piece_size()):
            data[i] = tokenizer.id_to_piece(i)
        json.dump(data, f)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.yaml file (optional)")
    parser.add_argument("--export_dir", type=str, default="export", help="Directory to save the ONNX models")
    args = parser.parse_args()

    convert(args.checkpoint, args.config)