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