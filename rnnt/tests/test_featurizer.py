import torch
import unittest


class JavascriptFeaturizerTest(unittest.TestCase):
    def test_stft(self):
        input = torch.ones(1000, dtype=torch.float32);

        n_fft = 400;
        hop_length = 160;
        win_length = 400;
        result = torch.stft(input, n_fft, hop_length, win_length, 
                            window=torch.hann_window(win_length),
                            center=True,
                            onesided=True,
                            normalized=False,
                            return_complex=True)
        
        result = result.abs().pow(2.0)

        match = result.T

        print(match.shape)
        print(match)