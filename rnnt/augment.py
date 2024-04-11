import torch
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig

class Augmentation():
    def __init__(self, p: float):
        self.p = p


def create_band_pass_filter(sample_rate, num_bands=256):
    # Generate random coefficients between 0.1 and 1.0
    #coeffs = torch.rand(num_bands) * 0.9 + 0.1

    coeffs = [0.9, 0.9, 0.01, 0.9]
    
    # Calculate frequency bins
    freq_bins = torch.linspace(0, sample_rate / 2, num_bands + 1)
    
    # Design the filter: Here, we will use the random coefficients as the filter's frequency response
    # The simplest way to apply this would be using an FIR filter designed with these frequency responses
    # However, directly designing a time-domain filter from arbitrary frequency responses is non-trivial in PyTorch
    # We will use an inverse FFT to approximate this (note: this is a heuristic approach and may not create a perfect filter)
    freq_response = torch.zeros(sample_rate)
    mid_point = len(freq_response) // 2
    for i in range(1, num_bands):
        start_idx = int(freq_bins[i-1])
        end_idx = int(freq_bins[i])
        freq_response[start_idx:end_idx] = coeffs[i-1]
    
    # Mirror the frequency response for negative frequencies to maintain real-valued time signal
    freq_response[mid_point+1:] = torch.flip(freq_response[1:mid_point], dims=[0])

    # Convert frequency response to time-domain filter using inverse FFT
    time_filter = torch.fft.irfft(freq_response, n=sample_rate)

    return time_filter, freq_response


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
        # Logarithmic sampling of the noise level
        log_min = torch.log10(torch.tensor(self.min_noise_level))
        log_max = torch.log10(torch.tensor(self.max_noise_level))
        random_log = torch.rand(1).item() * (log_max - log_min) + log_min
        noise_level = 10 ** random_log

        noise = torch.rand_like(waveform) * noise_level * 2 - noise_level
        return waveform + noise


class ShapedNoiseAugmentation(TimeDomainAugmentation):
    def __init__(self, p: float, min_noise_level: float = 0.01, max_noise_level: float = 0.1, num_buckets: int = 256):
        super(ShapedNoiseAugmentation, self).__init__(p)
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