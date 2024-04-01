import torch
import torchaudio
import unittest

from rnnt.featurizer import TFJSSpectrogram

# The idea here is to demo some code that produces the same results whether the audio has been run all at once
# through the audio encoder, or has been run through in smaller chunks

class FeaturizerStreamingTest(unittest.TestCase):
    def test_tfjs_feature_consistency(self):
        featurizer = TFJSSpectrogram(n_fft=400, hop_length=160, win_length=400, apply_linear_log=False, mean=15.0, invstddev=0.25)

        # Make a 4khz sine wave for 16000 samples
        audio = torch.sin(torch.linspace(0, 2 * 3.14159 * 4000, 16000))
    
        # Run the audio all at once
        full_output = featurizer(audio)
        full_output = full_output.T

        print(full_output)

        # you'd expect all the rows to be roughly equivalent since it's a constant signal
        self.assertTrue(torch.allclose(full_output[0], full_output[1]))
        self.assertTrue(torch.allclose(full_output[0], full_output[full_output.shape[0] // 2]))
        self.assertTrue(torch.allclose(full_output[0], full_output[-1]))

    def test_tfjs_streaming(self):
        featurizer = TFJSSpectrogram(n_fft=400, hop_length=160, win_length=400, apply_linear_log=False, mean=15.0, invstddev=0.25)
    
        # Run the audio in chunks
        chunk_size = 1280

        # Load a real audio file
        audio, sr = torchaudio.load("jake4.wav")
        
        # Trim it to the nearest chunk size multiple, to simplify the test
        audio = audio[:, :audio.shape[1] - (audio.shape[1] % chunk_size)]

        # Run the audio all at once
        full_output = featurizer(audio)

        print(full_output)


        chunked_output = torch.zeros((full_output.shape[1], 0))


        # The way this loop works is that on the first chunk, you want it to start at sample 0, and then produce
        # an integer number of featurizer frames.
        # The formula for the number of produced frames is: (audio_length - win_length) // hop_length + 1
        # To be consistent, each frame needs to overlap the previous one by (win_length - hop_length) samples
        # And then to have the first sample start at sample 0, this is what the formula needs to be

        for i in range(featurizer.win_length-featurizer.hop_length, audio.shape[1], chunk_size):
            # You need to include some previous samples to make sure the windowing is correct
            chunk = audio[0, i-(featurizer.win_length-featurizer.hop_length):i+chunk_size]

            output = featurizer(chunk)
            chunked_output = torch.cat((chunked_output, output), dim=1)

            print(output.shape)

        print(chunked_output.shape)

        diff = full_output - chunked_output
        diff = diff[0] # Drop batch dimension

        print(diff)
        print(torch.max(torch.abs(diff)))
        
