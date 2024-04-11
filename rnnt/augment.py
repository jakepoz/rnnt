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


class PeakLevel(TimeDomainAugmentation):
    def __init__(self, p: float, min_peak_level: float=0.5, max_peak_level: float=1.0):
        super(PeakLevel, self).__init__(p)
        self.min_peak_level = min_peak_level
        self.max_peak_level = max_peak_level

        assert self.min_peak_level <= self.max_peak_level
        assert self.min_peak_level >= 0.0
        assert self.max_peak_level <= 1.0

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        peak_level = torch.rand(1).item() * (self.max_peak_level - self.min_peak_level) + self.min_peak_level
        return (waveform / waveform.abs().max()) * peak_level


class WhiteNoise(TimeDomainAugmentation):
    def __init__(self, p: float, min_noise_level: float=0.01, max_noise_level: float=0.1):
        super(WhiteNoise, self).__init__(p)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level

        assert self.min_noise_level <= self.max_noise_level
        assert self.min_noise_level >= 0.0
        assert self.max_noise_level <= 1.0

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Logarithmic sampling of the noise level
        log_min = torch.log10(torch.tensor(self.min_noise_level))
        log_max = torch.log10(torch.tensor(self.max_noise_level))
        random_log = torch.rand(1).item() * (log_max - log_min) + log_min
        noise_level = 10 ** random_log

        noise = torch.rand_like(waveform) * noise_level * 2 - noise_level
        return waveform + noise


class ShapedNoise(TimeDomainAugmentation):
    def __init__(self, p: float, min_noise_level: float = 0.01, max_noise_level: float = 0.1, num_buckets: int = 256):
        super(ShapedNoise, self).__init__(p)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.num_buckets = num_buckets

        # Validate input values
        assert self.min_noise_level > 0.0  # Log scale cannot handle zero
        assert self.max_noise_level <= 1.0
        assert self.min_noise_level <= self.max_noise_level

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Logarithmic sampling of the noise level
        log_min = torch.log10(torch.tensor(self.min_noise_level))
        log_max = torch.log10(torch.tensor(self.max_noise_level))
        random_log = torch.rand(1).item() * (log_max - log_min) + log_min
        noise_level = 10 ** random_log

        # Generate white noise
        noise = torch.rand_like(waveform)
        noise = noise[:, 0]

        # Compute the frequency domain representation of the noise
        noise_fft = torch.fft.rfft(noise)
        
        energy_ratios = torch.rand(self.num_buckets)
        energy_ratios = energy_ratios / energy_ratios.sum()

        band_size = len(noise_fft) // len(energy_ratios)
        desired_fft = torch.zeros_like(noise_fft)

        for i in range(len(energy_ratios)):
            start = i * band_size
            end = (i + 1) * band_size

            ratio_start = energy_ratios[i]
            ratio_end = energy_ratios[i + 1] if i < len(energy_ratios) - 1 else 0.0

            desired_fft[start:end] = noise_fft[start:end] * (torch.linspace(ratio_start, ratio_end, band_size) ** 0.5)

        # Remove any DC component
        desired_fft[0] = 0
        
        # Convert the desired frequency domain representation back to the time domain
        shaped_noise = torch.fft.irfft(desired_fft)

        shaped_noise = (shaped_noise / shaped_noise.abs().max()) * noise_level

        # Expand shaped noise to the same shape as the input waveform, padding with zeros
        shaped_noise = torch.nn.functional.pad(shaped_noise, (0, waveform.shape[0] - shaped_noise.shape[0]))

        return waveform + shaped_noise.unsqueeze(1)
    

class ATempo(FFMpegAugmentation):
    def __init__(self, p: float, min_tempo_rate: float=0.8, max_tempo_rate: float=1.2):
        super(ATempo, self).__init__(p)
        self.min_tempo_rate = min_tempo_rate
        self.max_tempo_rate = max_tempo_rate

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        tempo_rate = torch.rand(1).item() * (self.max_tempo_rate - self.min_tempo_rate) + self.min_tempo_rate
        return f"atempo={tempo_rate:.2f}"


class PitchShift(FFMpegAugmentation):
    def __init__(self, p: float, min_semitones: int=-4, max_semitones: int=4):
        super(PitchShift, self).__init__(p)
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        semitones = torch.randint(self.min_semitones, self.max_semitones, (1,)).item()
        
        return f"asetrate={sample_rate}*2^(1/12*{semitones})"
    

class Trim(FFMpegAugmentation):
    def __init__(self, p: float, max_trim: float):
        super(Trim, self).__init__(p)
        self.max_trim = max_trim

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        trim = torch.rand(1).item() * self.max_trim

        duration = waveform.shape[0] / sample_rate
        if trim < duration:
            return f"atrim=start={trim:.4f}"
        else:
            return None
        
class ChooseAFilter(FFMpegAugmentation):
    def __init__(self, p: float, filters: list[str]):
        super(ChooseAFilter, self).__init__(p)
        self.filters = filters

    def filter_string(self, waveform: torch.Tensor, sample_rate: int) -> str:
        return self.filters[torch.randint(len(self.filters), (1,)).item()]