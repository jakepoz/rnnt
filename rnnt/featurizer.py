import torch
import torchaudio
import math

_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)


def _piecewise_linear_log(x):
    x = x * _gain
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class NormalizedSpectrogram(torchaudio.transforms.Spectrogram):
    def __init__(self, apply_linear_log: bool=True, mean: float=15.0, invstddev: float=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_linear_log = apply_linear_log
        self.mean = mean
        self.invstddev = invstddev

    @torch.no_grad()
    def forward(self, waveform):
        mel_spec = super().forward(waveform)

        if self.apply_linear_log:
            mel_spec = _piecewise_linear_log(mel_spec + 1e-6)

        mel_spec = (mel_spec - self.mean) * self.invstddev

        return mel_spec


class NormalizedMelSpectrogram(torchaudio.transforms.MelSpectrogram):
    def __init__(self, apply_linear_log: bool=True, mean: float=15.0, invstddev: float=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_linear_log = apply_linear_log
        self.mean = mean
        self.invstddev = invstddev

    @torch.no_grad()
    def forward(self, waveform):
        mel_spec = super().forward(waveform)

        if self.apply_linear_log:
            mel_spec = _piecewise_linear_log(mel_spec + 1e-6)

        mel_spec = (mel_spec - self.mean) * self.invstddev

        return mel_spec
    

class TFJSSpectrogram(torch.nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length:int, apply_linear_log: bool=True, mean: float=15.0, invstddev: float=0.2) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.apply_linear_log = apply_linear_log

        if isinstance(mean, float):
            self.mean = mean
        else:
            self.mean = torch.tensor(mean)

        if isinstance(invstddev, float):
            self.invstddev = invstddev
        else:
            self.invstddev = torch.tensor(invstddev)

    @torch.no_grad()
    def forward(self, waveform):
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, 
                          window=torch.hann_window(self.win_length),
                          center=True,
                          onesided=True,
                          normalized=False,
                          return_complex=True)
        
        spec = spec.abs().pow(2.0)

        if self.apply_linear_log:
            spec = _piecewise_linear_log(spec + 1e-6)

        if isinstance(self.mean, float):
            spec = (spec - self.mean) * self.invstddev
        else:
            spec = (spec - self.mean.unsqueeze(-1)) * self.invstddev.unsqueeze(-1)


        return spec