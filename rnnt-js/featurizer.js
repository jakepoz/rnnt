import * as tf from '@tensorflow/tfjs';

const _decibel = 2 * 20 * Math.log10(Math.pow(2, 15) - 1); // torch.iinfo(torch.int16).max is 2^15 - 1
const _gain = Math.pow(10, 0.05 * _decibel);


function classicPiecewiseLinearLog(x) {
    // Apply gain
    x = x.mul(_gain);

    // Apply piecewise function
    let mask = x.greater(tf.scalar(Math.E)); // mask for values where x > e

    return tf.where(mask, x.log(), x.div(tf.scalar(Math.E)));
}

export function classicFeaturizer(waveform) {
    const n_fft = 400;
    const hop_length = 160;
    const win_length = 400;

    const mean = 15.0;
    const invstddev = 0.25;
  
    const stft_f = tf.signal.stft(
        waveform,
        win_length,
        hop_length,
        n_fft
    )

    let spec_f = tf.pow(tf.abs(stft_f), 2.0);

    spec_f = classicPiecewiseLinearLog(tf.add(spec_f, 1e-6));

    // Hardcoded mean/stddev
    spec_f = tf.mul(tf.sub(spec_f, mean), invstddev);

    return spec_f;
}

function piecewiseLinearLog(x, xCutoff = 1e-2, slope = 50) {  
    const yTransition = Math.log(xCutoff);
    const interceptC = yTransition - slope * xCutoff;
  
    const logPart = tf.log(x);
    const linearPart = x.mul(slope).add(interceptC);
  
    return tf.where(x.greater(xCutoff), logPart, linearPart);
}


export function featurizer(waveform) {
    const n_fft = 400;
    const hop_length = 160;
    const win_length = 400;

    const mean =  tf.tensor1d([-3.43, -3.10, -2.75, -2.16, -1.64, -1.45, -1.69, -2.04, -2.16, -2.15, -2.16, -2.24, -2.38, -2.54, -2.74, -2.95, -3.15, -3.33, -3.47, -3.58, -3.68, -3.77, -3.85, -3.91, -3.97, -4.01, -4.05, -4.08, -4.11, -4.14, -4.16, -4.18, -4.19, -4.19, -4.19, -4.19, -4.18, -4.17, -4.17, -4.17, -4.18, -4.19, -4.20, -4.21, -4.23, -4.25, -4.27, -4.29, -4.31, -4.33, -4.35, -4.36, -4.38, -4.39, -4.40, -4.40, -4.40, -4.40, -4.40, -4.40, -4.40, -4.39, -4.39, -4.39, -4.40, -4.41, -4.42, -4.43, -4.44, -4.45, -4.46, -4.47, -4.48, -4.49, -4.50, -4.50, -4.51, -4.51, -4.52, -4.54, -4.55, -4.56, -4.57, -4.57, -4.58, -4.58, -4.59, -4.60, -4.61, -4.61, -4.62, -4.63, -4.63, -4.64, -4.64, -4.65, -4.65, -4.66, -4.66, -4.67, -4.67, -4.68, -4.69, -4.70, -4.70, -4.71, -4.72, -4.72, -4.73, -4.73, -4.74, -4.74, -4.75, -4.76, -4.76, -4.77, -4.77, -4.78, -4.78, -4.79, -4.79, -4.80, -4.80, -4.81, -4.81, -4.81, -4.82, -4.82, -4.82, -4.83, -4.83, -4.83, -4.83, -4.84, -4.84, -4.84, -4.85, -4.85, -4.85, -4.86, -4.86, -4.86, -4.86, -4.87, -4.87, -4.87, -4.88, -4.88, -4.88, -4.88, -4.88, -4.89, -4.89, -4.89, -4.89, -4.89, -4.90, -4.90, -4.90, -4.90, -4.91, -4.91, -4.91, -4.91, -4.92, -4.92, -4.92, -4.92, -4.92, -4.93, -4.93, -4.93, -4.93, -4.93, -4.94, -4.94, -4.94, -4.94, -4.94, -4.95, -4.95, -4.95, -4.95, -4.95, -4.95, -4.96, -4.96, -4.96, -4.96, -4.96, -4.96, -4.96, -4.96, -4.96, -4.96, -4.97, -4.97, -4.97, -4.97, -4.97, -4.99]);
    const invstddev = tf.tensor1d([0.42, 0.44, 0.39, 0.33, 0.31, 0.30, 0.31, 0.33, 0.33, 0.33, 0.33, 0.33, 0.34, 0.35, 0.36, 0.37, 0.39, 0.41, 0.43, 0.44, 0.46, 0.48, 0.50, 0.51, 0.53, 0.54, 0.55, 0.56, 0.57, 0.57, 0.58, 0.59, 0.59, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61, 0.61, 0.62, 0.63, 0.63, 0.64, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.73, 0.73, 0.74, 0.74, 0.75, 0.76, 0.77, 0.78, 0.78, 0.79, 0.80, 0.80, 0.81, 0.81, 0.82, 0.82, 0.83, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.90, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.98, 0.98, 0.99, 0.99, 1.00, 1.00, 1.01, 1.01, 1.02, 1.03, 1.03, 1.04, 1.05, 1.05, 1.06, 1.06, 1.07, 1.08, 1.09, 1.10, 1.10, 1.11, 1.11, 1.11, 1.12, 1.13, 1.13, 1.14, 1.15, 1.15, 1.15, 1.16, 1.17, 1.17, 1.18, 1.18, 1.19, 1.20, 1.21, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.26, 1.27, 1.27, 1.28, 1.29, 1.29, 1.30, 1.31, 1.32, 1.33, 1.33, 1.35, 1.36, 1.37, 1.38, 1.39, 1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.47, 1.48, 1.49, 1.50, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.57, 1.58, 1.59, 1.59, 1.60, 1.60, 1.61, 1.61, 1.62, 1.62, 1.63, 1.63, 1.63, 1.64, 1.64, 1.64, 1.65, 1.74]);
  
    const stft_f = tf.signal.stft(
        waveform,
        win_length,
        hop_length,
        n_fft
    )

    let spec_f = tf.pow(tf.abs(stft_f), 2.0);

    spec_f = piecewiseLinearLog(spec_f);

    // Hardcoded mean/stddev
    spec_f = tf.mul(tf.sub(spec_f, mean), invstddev);

    return spec_f;
}

//Applies the featurizer as new samples come in and returns the newly generated features
export class FeatureStreamer {
    constructor(featurizerFunc) {
        this.buffer = new Float32Array();

        this.n_fft = 400;
        this.hop_length = 160;
        this.win_length = 400;

        this.chunk_size = 3200;
        this.overlap = this.win_length - this.hop_length;

        this.featurizer = featurizerFunc;
    }

    process(waveform) {
        if (waveform && waveform.length > 0) {
            this.buffer = new Float32Array([...this.buffer, ...waveform]);
        }

        if (this.buffer.length < this.chunk_size + this.overlap) {
            return null;
        }

        let waveform_tensor = tf.tensor1d(this.buffer.slice(0, this.chunk_size + this.overlap));
        this.buffer = this.buffer.slice(this.chunk_size);

        return this.featurizer(waveform_tensor);
    }
}
