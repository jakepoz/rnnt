import os
import argparse
import yaml
from typing import List
from jinja2 import Environment, FileSystemLoader
import torch
import torchaudio
from hydra.utils import instantiate
from rnnt.augment import TimeDomainAugmentor

def generate_audio_samples(audio_file: str, audio_augmentation_config: dict, output_dir: str, num_individual_samples: int, num_combined_samples: int) -> tuple[dict, List[str]]:
    waveform, sample_rate = torchaudio.load(audio_file, channels_first=False)
    
    augmentor = instantiate(audio_augmentation_config)
    
    individual_samples = {}
    for aug in augmentor.augmentations:
        aug_samples = []
        for i in range(num_individual_samples):
            single_augmentor = TimeDomainAugmentor([aug])
            augmented_waveform = single_augmentor(waveform, sample_rate)
            output_file = os.path.join(output_dir, f"{aug.__class__.__name__}_{i}.wav")
            torchaudio.save(output_file, augmented_waveform, sample_rate, channels_first=False)
            aug_samples.append(os.path.basename(output_file))
        individual_samples[aug.__class__.__name__] = aug_samples
    
    combined_samples = []
    for i in range(num_combined_samples):
        augmented_waveform = augmentor(waveform, sample_rate)
        output_file = os.path.join(output_dir, f"combined_{i}.wav")
        torchaudio.save(output_file, augmented_waveform, sample_rate, channels_first=False)
        combined_samples.append(os.path.basename(output_file))
    
    return individual_samples, combined_samples

def render_html(individual_samples: dict, combined_samples: List[str], output_dir: str, augmentation_params: dict):
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('render_augments.html')
    
    output = template.render(individual_samples=individual_samples, combined_samples=combined_samples, augmentation_params=augmentation_params)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(output)

def main(audio_file: str, config_file: str, output_dir: str, num_individual_samples: int, num_combined_samples: int):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    audio_augmentation_config = config['data']['audio_augmentation']
    
    individual_samples, combined_samples = generate_audio_samples(audio_file, audio_augmentation_config, output_dir, num_individual_samples, num_combined_samples)
    
    augmentation_params = {
        'ATempoAugmentation': {
            'min_tempo_rate': audio_augmentation_config['augmentations'][0]['min_tempo_rate'],
            'max_tempo_rate': audio_augmentation_config['augmentations'][0]['max_tempo_rate']
        },
        'PitchShiftAugmentation': {
            'min_semitones': audio_augmentation_config['augmentations'][1]['min_semitones'],
            'max_semitones': audio_augmentation_config['augmentations'][1]['max_semitones']
        },
        'TrimAugmentation': {
            'max_trim': audio_augmentation_config['augmentations'][2]['max_trim']
        }
    }
    
    render_html(individual_samples, combined_samples, output_dir, augmentation_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', type=str, help='Path to the input audio file')
    parser.add_argument('config_file', type=str, help='Path to the config YAML file')
    parser.add_argument('--output_dir', type=str, default="augmented_audio", help='Path to the output directory')
    parser.add_argument('--num_individual_samples', type=int, default=3, help='Number of samples for each individual augmentation')
    parser.add_argument('--num_combined_samples', type=int, default=10, help='Number of samples for combined augmentations')
    
    args = parser.parse_args()
    
    main(args.audio_file, args.config_file, args.output_dir, args.num_individual_samples, args.num_combined_samples)