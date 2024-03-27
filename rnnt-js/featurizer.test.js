//import * as tf from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

import { featurizer } from './featurizer';

describe('featurizer function', () => {
    it('should correctly compute the STFT of the input waveform', () => {
        // Create a mock waveform as a 1D Tensor. This is a simplistic example.
        const mockWaveform = tf.ones([10544], 'float32');

        // Call the featurizer function
        const output = featurizer(mockWaveform);

        console.log(output.shape);
        output.print();

        // // Basic checks to ensure the output is of expected shape and type
        // expect(output instanceof tf.Tensor).toBeTruthy();
        // expect(output.shape.length).toBeGreaterThanOrEqual(2); // STFT output should be at least a 2D tensor

        // // Optionally, check the specifics of the shape, which depends on your featurizer's settings
        // // This is just an example and might need to be adjusted based on actual output
        // const expectedRows = Math.floor((1000 - 400) / 160) + 1;
        // const expectedCols = 512 / 2 + 1;
        // expect(output.shape).toEqual([expectedRows, expectedCols]);

   
    });
});
