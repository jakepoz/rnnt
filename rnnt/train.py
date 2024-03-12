import os
import torch
import torch.nn.functional as F
import torchaudio
import hydra
import wandb
import sentencepiece as spm
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets



@hydra.main(version_base=None, config_path="config", config_name="basic_sp.yaml")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    train_ds = hydra.utils.instantiate(cfg.data.train.dataset, _convert_="object")
    if isinstance(train_ds, list):
        train_ds = concatenate_datasets(train_ds)
    eval_ds = hydra.utils.instantiate(cfg.data.eval.dataset, _convert_="object")
    if isinstance(eval_ds, list):
        eval_ds = concatenate_datasets(eval_ds)

   
    predictor = hydra.utils.instantiate(cfg.predictor)
    featurizer = hydra.utils.instantiate(cfg.featurizer)
    encoder = hydra.utils.instantiate(cfg.encoder)
    joint = hydra.utils.instantiate(cfg.joint)

    # Wrap those in the processor class, which can provide augmentations, tokenization, etc.
    ds_processor = hydra.utils.get_class(cfg.data.processor_class)
    train_ds = ds_processor(train_ds, tokenizer, featurizer)
    eval_ds = ds_processor(eval_ds, tokenizer, featurizer)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Eval dataset size : {len(eval_ds)}")

    train_dataloader = hydra.utils.instantiate(cfg.data.train.dataloader, train_ds)
    eval_dataloader = hydra.utils.instantiate(cfg.data.eval.dataloader, eval_ds)

  
   

    params = [*predictor.parameters(), *encoder.parameters(), *joint.parameters()]

    print(f"Number of predictor parameters: {sum(p.numel() for p in predictor.parameters())}")
    print(f"Number of encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Number of joint parameters: {sum(p.numel() for p in joint.parameters())}")

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params)
    lr_scheduler = hydra.utils.instantiate(cfg.training.lr_scheduler, optimizer)

    predictor.train()
    encoder.train()
    joint.train()



    completed_steps = 0

    for epoch in range(cfg.training.num_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader),
        ):
            mel_features = batch["mel_features"]
            mel_feature_lens = batch["mel_feature_lens"]
            input_ids = batch["input_ids"]
            input_id_lens = batch["input_id_lens"]
           
            # Use traditional predictor for decoder features
            # TODO: This is WRONG because the first token needs to be the blank token to initialize it properly???
            decoder_features, decoder_lengths, decoder_state = predictor(input_ids, input_id_lens)
         
            
            # Generate the audio features, with gradients this time
            audio_features = encoder(mel_features) # (N, C, L)
            audio_features = audio_features.permute(0, 2, 1) # (N, L, C)
            audio_feature_lens = encoder.calc_output_lens(mel_feature_lens)

            # Now, apply the joint model to each combination
            joint_features = joint(audio_features, decoder_features)

            # Calculate the loss
            loss = torchaudio.functional.rnnt_loss(logits=joint_features, 
                                                    targets=input_ids.int(),
                                                    logit_lengths=audio_feature_lens,
                                                    target_lengths=input_id_lens.int() - 1, # -1 because we removed the start token
                                                    blank=-1,
                                                    clamp=-1,
                                                    reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.training.gradient_clipping)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            completed_steps += 1

            #     accelerator.log({
            #         "input_length/train": input_ids.shape[1],
            #         "loss/train": loss,
            #         "lr": lr_scheduler.get_last_lr()[0],
            #         "epoch": epoch,
            #     })

            # # Do an eval step periodically
            # if completed_steps % cfg.training.eval_steps == 0:
            #     encoder.eval()
            #     joint.eval()
            #     predictor.eval()

            #     print("Starting eval...")

            #     text_table = wandb.Table(columns=["step", "original_text", "decoded_text"])

            #     for step, batch in enumerate(eval_dataloader):
            #         mel_features = batch["mel_features"][0]
            #         input_ids = batch["input_ids"][0]
                    
            #         decoded_text = greedy_decode_audio(mel_features, llm, accelerator.unwrap_model(encoder), accelerator.unwrap_model(joint), tokenmap, tokenizer)
                    
            #         text_table.add_data(step, tokenizer.decode(input_ids), decoded_text)

            #         print(f"S: {step} Original: {tokenizer.decode(input_ids)} Decoded: {decoded_text}")

            #         if step > 5:
            #             break

            #     wandb_tracker.log({"text_examples/eval": text_table}, step=completed_steps)

            #     encoder.train()
            #     joint.train()
            #     predictor.train()


            # if completed_steps % cfg.training.checkpoint_steps == 0:
            #     accelerator.save_state(output_dir=os.path.join(output_dir, f"checkpoint_step_{completed_steps}"))
            

    # # Be sure to do one final save at the end
    # accelerator.save_state(output_dir=os.path.join(output_dir, f"checkpoint_step_{completed_steps}"))
    # accelerator.end_training()

if __name__ == "__main__":
    train()