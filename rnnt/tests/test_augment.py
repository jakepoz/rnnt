import unittest
import os
import torch
import torchaudio

from rnnt.augment import TimeDomainAugmentor, ATempoAugmentation, PitchShiftAugmentation, TrimAugmentation


class TestTimeDomainAugmentation(unittest.TestCase):
    def setUp(self):
        self.filepath = 'jake4.wav'
        self.output_folder = 'augmented_audio'
        os.makedirs(self.output_folder, exist_ok=True)

    def test_atempo_augmentation(self):
        # Setup test environment
        test_augmentations = [
            ATempoAugmentation(p=1.0, min_tempo_rate=0.8, max_tempo_rate=1.2)
        ]
        augmentor = TimeDomainAugmentor(test_augmentations)

        # Load test audio file
        waveform, sample_rate = torchaudio.load(self.filepath, channels_first=False)

        # Apply augmentation at fixed rates for testing purposes
        # For slow tempo
        test_augmentations[0].min_tempo_rate = 0.8
        test_augmentations[0].max_tempo_rate = 0.8
        augmented_waveform_slow = augmentor(waveform, sample_rate)
        torchaudio.save(os.path.join(self.output_folder, 'atempo_slow.wav'), augmented_waveform_slow, sample_rate, channels_first=False)

        # For fast tempo
        test_augmentations[0].min_tempo_rate = 1.2
        test_augmentations[0].max_tempo_rate = 1.2
        augmented_waveform_fast = augmentor(waveform, sample_rate)
        torchaudio.save(os.path.join(self.output_folder, 'atempo_fast.wav'), augmented_waveform_fast, sample_rate, channels_first=False)

        self.assertLess(augmented_waveform_fast.shape[0], waveform.shape[0])
        self.assertGreater(augmented_waveform_slow.shape[0], waveform.shape[0])

    def test_pitch_shift_augmentation(self):
        test_augmentations = [
            PitchShiftAugmentation(p=1.0, min_semitones=-4, max_semitones=4)
        ]
        augmentor = TimeDomainAugmentor(test_augmentations)

        # Load test audio file
        waveform, sample_rate = torchaudio.load(self.filepath, channels_first=False)

        # Apply augmentation at fixed rates for testing purposes
        test_augmentations[0].min_semitones = -4
        test_augmentations[0].max_semitones = -3
        augmented_waveform_down = augmentor(waveform, sample_rate)
        torchaudio.save(os.path.join(self.output_folder, 'pitch_down.wav'), augmented_waveform_down, sample_rate, channels_first=False)

        test_augmentations[0].min_semitones = 4
        test_augmentations[0].max_semitones = 5
        augmented_waveform_up = augmentor(waveform, sample_rate)
        torchaudio.save(os.path.join(self.output_folder, 'pitch_up.wav'), augmented_waveform_up, sample_rate, channels_first=False)

        # self.assertEqual(augmented_waveform_up.shape, waveform.shape)
        # self.assertEqual(augmented_waveform_down.shape, waveform.shape)

    def test_trim_augmentation(self):
        test_augmentations = [
            TrimAugmentation(p=1.0, max_trim=0.5)
        ]
        augmentor = TimeDomainAugmentor(test_augmentations)

        # Load test audio file
        waveform, sample_rate = torchaudio.load(self.filepath, channels_first=False)

        # Apply augmentation at fixed rates for testing purposes
        augmented_waveform = augmentor(waveform, sample_rate)
        torchaudio.save(os.path.join(self.output_folder, 'trim.wav'), augmented_waveform, sample_rate, channels_first=False)

        self.assertLessEqual(augmented_waveform.shape[0], waveform.shape[0])

     
if __name__ == '__main__':
    unittest.main()
