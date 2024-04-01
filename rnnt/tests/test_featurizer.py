import torch
import unittest
import math
import matplotlib.pyplot as plt

from rnnt.featurizer import TFJSSpectrogram, _piecewise_linear_log

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

    def test_plot_linearlog(self):
        # Generate a range of x values
        x_values = torch.linspace(0.1, 10, 1000) # Avoid starting at 0 to prevent log(0)

        # Apply the piecewise function to these x values
        y_values_piecewise = _piecewise_linear_log(x_values)

        # Calculate the regular log values
        y_values_log = torch.log(x_values)

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(x_values.numpy(), y_values_piecewise.numpy(), label='Piecewise Linear-Log Function')
        plt.plot(x_values.numpy(), y_values_log.numpy(), label='Log Function', linestyle='--')
        plt.axvline(x=math.e, color='r', linestyle='--', label='x = e')
        plt.axhline(y=1, color='g', linestyle='--', label='y = 1')
        plt.title('Piecewise Linear-Log Function vs Regular Log Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plt.savefig('piecewise_linear_log.png')

        # Display the plot
        plt.show()