//import * as tf from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

import { featurizer, FeatureStreamer } from './featurizer';

describe('featurizer function', () => {
    it('should correctly compute the STFT of the input waveform', () => {
        // Create a mock waveform as a 1D Tensor. This is a simplistic example.
        const mockWaveform = tf.ones([1600], 'float32');

        // Call the featurizer function
        const output = featurizer(mockWaveform);

        console.log(output.shape);
        output.print();
    });

    it('should correctly compute for a reference sine wave', () => {
        // Make a 4khz sine wave for 16000 samples
        // audio = torch.sin(torch.linspace(0, 2 * 3.14159 * 4000, 16000))
        const audio = tf.linspace(0, 2 * 3.14159 * 4000, 16000);
        const audioSin = tf.sin(audio);

        // Call the featurizer function
        const output = featurizer(audioSin);

        console.log(output.shape);
        output.print();
    });

    it('should match performance in streaming and non-streaming mode', () => {
        // Create a mock waveform as a 1D Tensor. This is a simplistic example.
        const mockWaveformSize = 32000;
        const mockWaveform = new Float32Array(mockWaveformSize);
        for (let i = 0; i < mockWaveformSize; i++) {
            mockWaveform[i] = Math.random();
        }
 
        // Call the featurizer function on the whole output
        const output = featurizer(tf.tensor1d(mockWaveform));

        console.log("non-streaming output shape: ", output.shape);

        // Call the streamer once, it should match on the first N frames
        let streamer = new FeatureStreamer(featurizer);
        let outputStreamer = streamer.process(mockWaveform);

        console.log("streaming output shape: ", outputStreamer.shape);  

        let diff = tf.sub(output.slice(0, outputStreamer.shape[0]), outputStreamer).abs().sum().arraySync();
        console.log("Difference between streaming and non-streaming: ", diff);
        expect(diff).toBeLessThan(1e-6);

        // Now make a new streamer, and feed in the mock waveform in chunks and accumulate all the features
        streamer = new FeatureStreamer(featurizer);
        let outputAccum = tf.zeros([0, 201]);
        const streamerInputSize = 320;

        for (let i = 0; i < mockWaveformSize; i += streamerInputSize) {
            outputStreamer = streamer.process(mockWaveform.slice(i, i + streamerInputSize));
            if (outputStreamer != null) {
                outputAccum = tf.concat([outputAccum, outputStreamer], 0);
            }
        }

        console.log("streaming output shape: ", outputAccum.shape);

        diff = tf.sub(output.slice(0, outputAccum.shape[0]), outputAccum).abs().sum().arraySync();
        console.log("Difference between streaming and non-streaming: ", diff);
        expect(diff).toBeLessThan(1e-6);
    });
});
