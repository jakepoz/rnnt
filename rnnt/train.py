import torch
import torch.nn.functional as F
import torchaudio
import hydra
import os
import jiwer
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from rnnt.model import RNNTModel

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets
from torch.utils.tensorboard import SummaryWriter

def save_model(model, optimizer, completed_steps, output_dir):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'completed_steps': completed_steps,
    }, os.path.join(output_dir, f"checkpoint_step_{completed_steps}.pt"))

@hydra.main(version_base=None, config_path="config", config_name="basic_sp.yaml")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    device = torch.device("cuda")

    # TensorBoard writer setup
    writer = SummaryWriter(log_dir=output_dir)

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

    # Wrap those in the processor class, which can provide augmentations, tokenization, etc.
    ds_processor = hydra.utils.get_class(cfg.data.processor_class)
    train_ds = ds_processor(train_ds, tokenizer, featurizer)
    eval_ds = ds_processor(eval_ds, tokenizer, featurizer)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Eval dataset size : {len(eval_ds)}")

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

    model = model.to(device)
    model.train()


    for epoch in range(cfg.training.num_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader),
        ):
            mel_features = batch["mel_features"].to(device)
            mel_feature_lens = batch["mel_feature_lens"].to(device)
            input_ids = batch["input_ids"].to(device)
            input_id_lens = batch["input_id_lens"].to(device)

            prepended_input_ids = torch.cat([torch.zeros(input_ids.shape[0], 1, dtype=input_ids.dtype, device=device), input_ids], dim=1)
            prepended_input_ids[:, 0] = cfg.blank_idx

            # Use traditional predictor for decoder features
            decoder_features, decoder_lengths, decoder_state = model.predictor(prepended_input_ids, input_id_lens + 1)
         
            
            # Generate the audio features, with gradients this time
            audio_features = model.encoder(mel_features) # (N, C, L)
            audio_features = audio_features.permute(0, 2, 1) # (N, L, C)
            audio_feature_lens = model.encoder.calc_output_lens(mel_feature_lens)

            # Now, apply the joint model to each combination
            joint_features = model.joint(audio_features, decoder_features)

            # Calculate the loss
            loss = torchaudio.functional.rnnt_loss(logits=joint_features, 
                                                   targets=input_ids.int(),
                                                   logit_lengths=audio_feature_lens,
                                                   target_lengths=input_id_lens.int(), 
                                                   blank=-1,
                                                   clamp=-1,
                                                   reduction="mean")

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(params, cfg.training.gradient_clipping)

            completed_steps += 1

            # TensorBoard logging
            writer.add_scalar("input_length/train", sum(input_id_lens.tolist()), completed_steps)
            writer.add_scalar("loss/train", loss.item(), completed_steps)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], completed_steps)
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
            if completed_steps % cfg.training.eval_steps == 0:
                model.eval()

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

                    if step > 5:
                        break

                # Calculate overall wer using jiwer
                wer = jiwer.wer(originals, decoded)
                writer.add_scalar("wer/eval", wer, completed_steps)
                
                print("Done with eval...")
                model.train()


            if completed_steps % cfg.training.checkpoint_steps == 0:
                save_model(model, optimizer, completed_steps, output_dir)
            

    # Be sure to do one final save at the end
    save_model(model, optimizer, completed_steps, output_dir)
    writer.close()

if __name__ == "__main__":
    train()