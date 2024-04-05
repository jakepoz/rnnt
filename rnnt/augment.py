import torch
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig

class Augmentation():
    def __init__(self, p: float):
        self.p = p


# Time-domain augmentations for audio data, inspired by https://github.com/asteroid-team/torch-audiomentations
class TimeDomainAugmentation(Augmentation):
    def __init__(self, p: float):
        super(TimeDomainAugmentation, self).__init__(p)

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise NotImplementedError()


class FFMpegAugmentation(Augmentation):
    def __init__(self, p: float):
        super(FFMpegAugmentation, self).__init__(p)

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        raise NotImplementedError()


class TimeDomainAugmentor():
    def __init__(self, ffmpeg_augmentations: list[FFMpegAugmentation], time_domain_augmentations: list[TimeDomainAugmentation]):
        self.ff = ffmpeg_augmentations
        self.td = time_domain_augmentations

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if len(waveform.shape) != 2:
            raise ValueError("Input waveform must be 2D")
    
        if waveform.shape[1] != 1:
            raise ValueError("Input waveform must have a single channel in the second dimension")

        # First all ffmpeg augmentations are applied
        filters = []

        for aug in self.ff:
            if torch.rand(1).item() < aug.p:
                filters.append(aug.filter_string(waveform, sample_rate))

        filters = [f for f in filters if f != None]
        
        if len(filters) > 0:
            filter_string = ",".join(filters)

            effector = AudioEffector(effect=filter_string, pad_end=False)
            waveform = effector.apply(waveform, sample_rate)

        # Then all time-domain augmentations are applied
        for aug in self.td:
            if torch.rand(1).item() < aug.p:
                waveform = aug.forward(waveform, sample_rate)

        return waveform


class PeakLevelAugmentation(TimeDomainAugmentation):
    def __init__(self, p: float, min_peak_level: float=0.5, max_peak_level: float=1.0):
        super(PeakLevelAugmentation, self).__init__(p)
        self.min_peak_level = min_peak_level
        self.max_peak_level = max_peak_level

        assert self.min_peak_level <= self.max_peak_level
        assert self.min_peak_level >= 0.0
        assert self.max_peak_level <= 1.0

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        peak_level = torch.rand(1).item() * (self.max_peak_level - self.min_peak_level) + self.min_peak_level
        return (waveform / waveform.abs().max()) * peak_level


class WhiteNoiseAugmentation(TimeDomainAugmentation):
    def __init__(self, p: float, min_noise_level: float=0.01, max_noise_level: float=0.1):
        super(WhiteNoiseAugmentation, self).__init__(p)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level

        assert self.min_noise_level <= self.max_noise_level
        assert self.min_noise_level >= 0.0
        assert self.max_noise_level <= 1.0

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        noise_level = torch.rand(1).item() * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
        noise = torch.rand_like(waveform) * noise_level * 2 - noise_level
        return waveform + noise


class ATempoAugmentation(FFMpegAugmentation):
    def __init__(self, p: float, min_tempo_rate: float=0.8, max_tempo_rate: float=1.2):
        super(ATempoAugmentation, self).__init__(p)
        self.min_tempo_rate = min_tempo_rate
        self.max_tempo_rate = max_tempo_rate

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        tempo_rate = torch.rand(1).item() * (self.max_tempo_rate - self.min_tempo_rate) + self.min_tempo_rate
        return f"atempo={tempo_rate:.2f}"


class PitchShiftAugmentation(FFMpegAugmentation):
    def __init__(self, p: float, min_semitones: int=-4, max_semitones: int=4):
        super(PitchShiftAugmentation, self).__init__(p)
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        semitones = torch.randint(self.min_semitones, self.max_semitones, (1,)).item()
        
        return f"asetrate={sample_rate}*2^(1/12*{semitones})"
    

class TrimAugmentation(FFMpegAugmentation):
    def __init__(self, p: float, max_trim: float):
        super(TrimAugmentation, self).__init__(p)
        self.max_trim = max_trim

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        trim = torch.rand(1).item() * self.max_trim

        duration = waveform.shape[0] / sample_rate
        if trim < duration:
            return f"atrim=start={trim:.4f}"
        else:
            return None