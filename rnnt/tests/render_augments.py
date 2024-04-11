import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader
from datasets import concatenate_datasets
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rnnt.augment import TimeDomainAugmentor

def generate_audio_samples(audio_file: str, audio_augmentation_config: dict, output_dir: str, num_individual_samples: int, num_combined_samples: int) -> tuple[dict, List[str]]:
    waveform, sample_rate = torchaudio.load(audio_file, channels_first=False)
    augmentor = instantiate(audio_augmentation_config)
    
    individual_samples = {}
    for a, aug in enumerate(augmentor.ff):
        aug_samples = []

        orig_p = aug.p
        aug.p = 1.0

        for i in range(num_individual_samples):
            single_augmentor = TimeDomainAugmentor(ffmpeg_augmentations=[aug], time_domain_augmentations=[])
            augmented_waveform = single_augmentor(waveform, sample_rate)
            output_file = os.path.join(output_dir, f"{aug.__class__.__name__}_{i}.wav")
            torchaudio.save(output_file, augmented_waveform, sample_rate, channels_first=False)
            aug_samples.append(os.path.basename(output_file))

        aug.p = orig_p
        individual_samples[f"{aug.__class__.__name__}_{a}"] = aug_samples

    for a, aug in enumerate(augmentor.td):
        aug_samples = []

        orig_p = aug.p
        aug.p = 1.0

        for i in range(num_individual_samples):
            single_augmentor = TimeDomainAugmentor(ffmpeg_augmentations=[], time_domain_augmentations=[aug])
            augmented_waveform = single_augmentor(waveform, sample_rate)
            output_file = os.path.join(output_dir, f"{aug.__class__.__name__}_{i}.wav")
            torchaudio.save(output_file, augmented_waveform, sample_rate, channels_first=False)
            aug_samples.append(os.path.basename(output_file))

        aug.p = orig_p
        individual_samples[f"{aug.__class__.__name__}_{a}"] = aug_samples
    
    combined_samples = []
    for i in range(num_combined_samples):
        augmented_waveform = augmentor(waveform, sample_rate)
        output_file = os.path.join(output_dir, f"combined_{i}.wav")
        torchaudio.save(output_file, augmented_waveform, sample_rate, channels_first=False)
        combined_samples.append(os.path.basename(output_file))
    
    return individual_samples, combined_samples

def render_html(template_env, output_dir, original_audio, individual_samples, audio_augmentation_config, combined_samples):
    template = template_env.get_template('render_augments.html')

    augmentation_params = {}

    for a, aug in enumerate(audio_augmentation_config.ffmpeg_augmentations):
        augmentation_params[aug._target_.split(".")[-1] + f"_{a}"] = dict(aug)

    for a, aug in enumerate(audio_augmentation_config.time_domain_augmentations):
        augmentation_params[aug._target_.split(".")[-1] + f"_{a}"] = dict(aug)

    html_content = template.render(
        original_audio=original_audio,
        individual_samples=individual_samples,
        augmentation_params=augmentation_params,
        combined_samples=combined_samples
    )
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(html_content)

def main(audio_file: Optional[str], config_file: str, output_dir: str, num_individual_samples: int, num_combined_samples: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(config_file, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    audio_augmentation_config = config.data.audio_augmentation
    
    if audio_file is None:
        # Randomly select an audio file from the training dataset
        train_ds = hydra.utils.instantiate(config.data.train.dataset, _convert_="object")
        if isinstance(train_ds, list):
            train_ds = concatenate_datasets(train_ds)
        
        idx = np.random.randint(len(train_ds))
        audio_data = train_ds[idx]
        audio_file = os.path.join(output_dir, f"audio_{idx}.wav")
        audio_array = torch.from_numpy(audio_data['audio']['array']).unsqueeze(0)
        torchaudio.save(audio_file, audio_array, audio_data['audio']['sampling_rate'])
    
    individual_samples, combined_samples = generate_audio_samples(audio_file, audio_augmentation_config, output_dir, num_individual_samples, num_combined_samples)
    
    # Set up Jinja2 for HTML rendering
    file_loader = FileSystemLoader(os.path.dirname(__file__))
    env = Environment(loader=file_loader)
    
    render_html(env, output_dir, os.path.basename(audio_file), individual_samples, audio_augmentation_config, combined_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the config YAML file')
    parser.add_argument('--audio_file', type=str, help='Path to the input audio file')
    parser.add_argument('--output_dir', type=str, default="augmented_audio", help='Path to the output directory')
    parser.add_argument('--num_individual_samples', type=int, default=3, help='Number of samples for each individual augmentation')
    parser.add_argument('--num_combined_samples', type=int, default=10, help='Number of samples for combined augmentations')
    
    args = parser.parse_args()
    
    main(args.audio_file, args.config_file, args.output_dir, args.num_individual_samples, args.num_combined_samples)
