import torch
import torch.nn.functional as F
import torchaudio
import hydra
import time
import jiwer
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets
from torch.utils.tensorboard import SummaryWriter

from rnnt.model import RNNTModel
from rnnt.util import save_model, get_output_dir


@hydra.main(version_base=None, config_path="config", config_name="basic_sp.yaml")
def eval(cfg: DictConfig) -> None:
    device = torch.device("cuda")

    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")

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
    eval()