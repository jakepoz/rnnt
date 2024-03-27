import torch
import unittest

from rnnt.featurizer import TFJSSpectrogram

class JavascriptFeaturizerTest(unittest.TestCase):
    def test_stft(self):
        input = torch.ones(1000, dtype=torch.float32);

        n_fft = 400;
        hop_length = 160;
        win_length = 400;
        result = torch.stft(input, n_fft, hop_length, win_length, 
                            window=torch.hann_window(win_length),
                            center=False,
                            onesided=True,
                            normalized=False,
                            return_complex=True)
        
        result = result.abs().pow(2.0)

        match = result.T

        print(match.shape)
        print(match)

    def test_tfjs_spectrogram(self):
        input = torch.ones(10544, dtype=torch.float32);

        n_fft = 400;
        hop_length = 160;
        win_length = 400;
        apply_linear_log = True;
        mean = 15.0;
        invstddev = 0.25;

        featurizer = TFJSSpectrogram(n_fft, hop_length, win_length, apply_linear_log, mean, invstddev)
        result = featurizer(input)

        print("A")
        print(result.shape)

        print(result.T)