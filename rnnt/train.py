import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils
import torchaudio
import hydra
import os
import jiwer
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets
from torch.utils.tensorboard import SummaryWriter

from rnnt.model import RNNTModel
from rnnt.util import save_model, get_output_dir, EmptyWriter


@hydra.main(version_base=None, config_path="config", config_name="basic_sp_convjs.yaml")
def train(cfg: DictConfig) -> None:
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        world_size = dist.get_world_size()
        is_main_process = rank == 0
        print(f"Starting Rank: {rank}, World size: {world_size}")
    else:
        device = torch.device("cuda")
        is_main_process = True

    if is_main_process:
        print(OmegaConf.to_yaml(cfg))
        output_dir = get_output_dir(cfg)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"Output directory  : {output_dir}")

        # TensorBoard writer setup
        writer = SummaryWriter(log_dir=output_dir)
    else:
        writer = EmptyWriter()

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    train_ds = hydra.utils.instantiate(cfg.data.train.dataset, _convert_="object")
    if isinstance(train_ds, list):
        train_ds = concatenate_datasets(train_ds)
    eval_ds = hydra.utils.instantiate(cfg.data.eval.dataset, _convert_="object")
    if isinstance(eval_ds, list):
        eval_ds = concatenate_datasets(eval_ds)

    featurizer = hydra.utils.instantiate(cfg.featurizer)
    
    model = RNNTModel(hydra.utils.instantiate(cfg.predictor),
                      hydra.utils.instantiate(cfg.encoder),
                      hydra.utils.instantiate(cfg.joint))
    
    model = model.to(device)

    if ddp:
        _ddp_model = DDP(model, device_ids=[rank])
    else:
        _ddp_model = model

    # Wrap those in the processor class, which can provide augmentations, tokenization, etc.
    ds_processor = hydra.utils.get_class(cfg.data.processor_class)

    if hasattr(cfg.data, 'audio_augmentation'):
        audio_augmentation = hydra.utils.instantiate(cfg.data.audio_augmentation)
    else:
        audio_augmentation = None

    train_ds = ds_processor(train_ds, tokenizer, featurizer, torch.device("cpu"), audio_augmentation=audio_augmentation)
    eval_ds = ds_processor(eval_ds, tokenizer, featurizer, torch.device("cpu"))

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Eval dataset size : {len(eval_ds)}")

    if ddp:
        train_dataloader = hydra.utils.instantiate(cfg.data.train.dataloader, train_ds, 
                                                   shuffle=None, 
                                                   sampler=torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True))
    else:
        train_dataloader = hydra.utils.instantiate(cfg.data.train.dataloader, train_ds)

    eval_dataloader = hydra.utils.instantiate(cfg.data.eval.dataloader, eval_ds)

    params = model.parameters()

    print(f"Number of predictor parameters: {sum(p.numel() for p in model.predictor.parameters()):,}")
    print(f"Number of encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Number of joint parameters: {sum(p.numel() for p in model.joint.parameters()):,}")

    total_steps = len(train_dataloader) * cfg.training.num_epochs
    completed_steps = 0

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params)
    lr_scheduler = hydra.utils.instantiate(cfg.training.lr_scheduler, optimizer, total_steps=total_steps)


    _ddp_model.train()


    for epoch in range(cfg.training.num_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader),
        ):
            mel_features = batch["mel_features"].to(device)
            mel_feature_lens = batch["mel_feature_lens"].to(device)
            input_ids = batch["input_ids"].to(device)
            input_id_lens = batch["input_id_lens"].to(device)

            # Calculate the total size of the joint feature vector, which will probably cause OOMs if it exceed an amount
            max_joint_size = torch.max(input_id_lens).item() * torch.max(mel_feature_lens).item()
            print(f"Max joint feature size: {max_joint_size:,}")

            if max_joint_size > cfg.training.max_joint_size:
                print("Cutting batch in half, it's probably too large otherwise")
                new_batch_size = mel_features.shape[0] // 2
                mel_feature_lens = mel_feature_lens[:new_batch_size]
                mel_features = mel_features[:new_batch_size, :, :torch.max(mel_feature_lens).item()]
                input_id_lens = input_id_lens[:new_batch_size]
                input_ids = input_ids[:new_batch_size, :torch.max(input_id_lens).item()]
                

            loss = _ddp_model(mel_features, mel_feature_lens, input_ids, input_id_lens, cfg.blank_idx)
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(params, cfg.training.clip_grad_norm)

            completed_steps += 1

            # TensorBoard logging
            writer.add_scalar("input_length/train", sum(input_id_lens.tolist()), completed_steps)
            writer.add_scalar("loss/train", loss.item(), completed_steps)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], completed_steps)
            writer.add_scalar("total_norm/train", total_norm, completed_steps)
            writer.add_scalar("epoch", epoch, completed_steps)

            if completed_steps % cfg.training.log_steps == 0:
                with torch.no_grad():
                    # Log gradient histograms for each layer
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            writer.add_histogram(f"debug/{name}/gradients", parameter.grad, completed_steps)
                            writer.add_histogram(f"debug/{name}/weights", parameter, completed_steps)

                            grad_norm = parameter.grad.norm(2).item()
                            writer.add_scalar(f"debug/{name}/grad_norm", grad_norm, completed_steps)

                    writer.add_scalar("model_grad_norm/train", np.sqrt(sum([torch.norm(p.grad).cpu().numpy()**2 for p in model.parameters()])), completed_steps)
                    writer.add_scalar("joint_grad_norm/train", np.sqrt(sum([torch.norm(p.grad).cpu().numpy()**2 for p in model.joint.parameters()])), completed_steps)
                    writer.add_scalar("encoder_grad_norm/train", np.sqrt(sum([torch.norm(p.grad).cpu().numpy()**2 for p in model.encoder.parameters()])), completed_steps)
                    writer.add_scalar("predictor_grad_norm/train", np.sqrt(sum([torch.norm(p.grad).cpu().numpy()**2 for p in model.predictor.parameters()])), completed_steps)

            # Actually do the step and zero out the gradients
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            # Do an eval step periodically
            if completed_steps % cfg.training.eval_steps == 0 or completed_steps == total_steps:
                _ddp_model.eval()

                print("Starting eval...")
                originals, decoded = [], []

                for step, batch in enumerate(eval_dataloader):
                    mel_features = batch["mel_features"].to(device)
                    mel_feature_lens = batch["mel_feature_lens"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    
                    decoded_tokens = model.greedy_decode(mel_features, mel_feature_lens, max_length=batch["input_ids"].shape[1] * 2)
                    decoded_text = tokenizer.decode(decoded_tokens)
                    original_text = tokenizer.decode(input_ids[0].cpu().tolist())

                    print(f"\nOriginal: {original_text}\nDecoded : {decoded_text}")

                    writer.add_text(f"original_text_{step}/eval", original_text, completed_steps)
                    writer.add_text(f"decoded_text_{step}/eval", decoded_text, completed_steps)

                    originals.append(original_text)
                    decoded.append(decoded_text)

                    if step > cfg.data.eval.max_elements:
                        break

                # Calculate overall wer using jiwer
                wer = jiwer.wer(originals, decoded)
                writer.add_scalar("wer/eval", wer, completed_steps)
                
                print("Done with eval...")
                _ddp_model.train()


            if completed_steps % cfg.training.checkpoint_steps == 0 and is_main_process:
                save_model(model, optimizer, completed_steps, output_dir)
            

    # Be sure to do one final save at the end
    if is_main_process:
        save_model(model, optimizer, completed_steps, output_dir)
        
    writer.close()

    if ddp:
        dist.destroy_process_group()

    # Returns last wer
    return wer

if __name__ == "__main__":
    train()