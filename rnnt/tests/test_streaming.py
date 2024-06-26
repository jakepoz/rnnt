import torch
import torchaudio
import unittest
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from rnnt.featurizer import TFJSSpectrogram
from rnnt.jasper import AudioEncoder, JasperBlock

# The idea here is to demo some code that produces the same results whether the audio has been run all at once
# through the audio encoder, or has been run through in smaller chunks

class FeaturizerStreamingTest(unittest.TestCase):
    @unittest.skip("Not implemented yet, need to fix linearization / log scaling issues first")
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
        self.assertLess(torch.max(torch.abs(diff)), 1e-5)


class JasperStreamingTest(unittest.TestCase):
    def test_jasper_streaming_full_causal(self):
        # This code sets things up to run the audio encoder in a streaming fashion

        # In this first test, we will run just the prologue and epilogue with no JasperBlocks
        # and no additional context
        featurizer = TFJSSpectrogram(n_fft=400, hop_length=160, win_length=400, apply_linear_log=False, mean=15.0, invstddev=0.25)
    
        input_stride = 2
        for input_kernel, input_stride, input_dilation, output_kernel, output_stride, output_dilation in [
            (3, 2, 3,  5, 1, 5),
            (3, 1, 1, 29, 1, 1),
            (3, 2, 1, 29, 1, 2),
            (10, 2, 1, 29, 1, 2),
            (11, 2, 1, 29, 1, 2),
            (11, 2, 1, 30, 1, 2),
            (3, 1, 3,  5, 1, 5),
        ]:
            with self.subTest("Testing prologue and epilogue", input_kernel=input_kernel, input_stride=input_stride, input_dilation=input_dilation, output_kernel=output_kernel, output_stride=output_stride, output_dilation=output_dilation):
                encoder = AudioEncoder(
                    input_features=201,
                    prologue_kernel_size=input_kernel,
                    prologue_stride=input_stride,
                    prologue_dilation=input_dilation,
                    blocks = [],
                    epilogue_features=512,
                    epilogue_kernel_size=output_kernel,
                    epilogue_stride=output_stride,
                    epilogue_dilation=output_dilation,
                )
                encoder.eval() # Turn off batch norm since we will have size 1 basically in a lot of places when streaming

                # Load a real audio file    
                audio, sr = torchaudio.load("jake4.wav")

                # Trim it to the nearest chunk size multiple, to simplify the test
                chunk_size = 1280
                audio = audio[:, :audio.shape[1] - (audio.shape[1] % 1280)]

                # Run it all at once as a reference
                features = featurizer(audio)
                full_output = encoder(features)

                print(full_output.shape)

                streaming_output = torch.zeros((1, full_output.shape[1], 0))

                state = encoder.streaming_init_state(batch_size=1)

                for i in range(0, features.shape[2], input_stride):
                    result, state = encoder.streaming_forward(features[:, :, i: i + input_stride], state)
                    streaming_output = torch.cat((streaming_output, result), dim=2)

                self.assertTrue(torch.allclose(full_output, streaming_output, atol=1e-5))

    def test_jasper_streaming_with_blocks(self):
        featurizer = TFJSSpectrogram(n_fft=400, hop_length=160, win_length=400, apply_linear_log=False, mean=15.0, invstddev=0.25)
    
        input_stride = 2

        encoder = AudioEncoder(
                input_features=201,
                prologue_kernel_size=11,
                prologue_stride=input_stride,
                prologue_dilation=1,
                blocks = [
                    JasperBlock(kernel_size=11, in_channels=256, out_channels=256, dropout=0.0, num_sub_blocks=4),
                ],
                epilogue_features=512,
                epilogue_kernel_size=29,
                epilogue_stride=1,
                epilogue_dilation=2,
            )
        encoder.eval()

        # Load a real audio file    
        audio, sr = torchaudio.load("jake4.wav")

        # Trim it to the nearest chunk size multiple, to simplify the test
        chunk_size = 1280
        audio = audio[:, :audio.shape[1] - (audio.shape[1] % 1280)]

        # Run it all at once as a reference
        features = featurizer(audio)
        full_output = encoder(features)

        print(full_output.shape)

        streaming_output = torch.zeros((1, full_output.shape[1], 0))

        state = encoder.streaming_init_state(batch_size=1)

        for i in range(0, features.shape[2], input_stride):
            result, state = encoder.streaming_forward(features[:, :, i: i + input_stride], state)
            streaming_output = torch.cat((streaming_output, result), dim=2)

        self.assertTrue(torch.allclose(full_output, streaming_output, atol=1e-5))

    def test_with_real_yaml_config(self):
        config_path = "rnnt/config/basic_sp_convjs_fullcausal.yaml"

        with open(config_path, "r") as f:
            cfg = OmegaConf.create(yaml.safe_load(f))

        featurizer = hydra.utils.instantiate(cfg.featurizer)
        encoder = hydra.utils.instantiate(cfg.encoder)
        encoder.eval()

        input_stride = 2

        # Load a real audio file    
        audio, sr = torchaudio.load("jake4.wav")

        # Trim it to the nearest chunk size multiple, to simplify the test
        chunk_size = 1280
        audio = audio[:, :audio.shape[1] - (audio.shape[1] % 1280)]

        # Run it all at once as a reference
        features = featurizer(audio)
        full_output = encoder(features)

        print(full_output.shape)

        streaming_output = torch.zeros((1, full_output.shape[1], 0))

        state = encoder.streaming_init_state(batch_size=1)

        for i in range(0, features.shape[2], input_stride):
            result, state = encoder.streaming_forward(features[:, :, i: i + input_stride], state)
            streaming_output = torch.cat((streaming_output, result), dim=2)

        self.assertTrue(torch.allclose(full_output, streaming_output, atol=1e-5))