import * as tf from '@tensorflow/tfjs';

const _decibel = 2 * 20 * Math.log10(Math.pow(2, 15) - 1); // torch.iinfo(torch.int16).max is 2^15 - 1
const _gain = Math.pow(10, 0.05 * _decibel);

function _piecewise_linear_log(x) {
    // Apply gain
    x = x.mul(_gain);
    
    // Apply piecewise function
    let mask = x.greater(tf.scalar(Math.E)); // mask for values where x > e

    return tf.where(mask, x.log(), x.div(tf.scalar(Math.E)));
}


export function featurizer(waveform) {
    const n_fft = 400;
    const hop_length = 160;
    const win_length = 400;

    const stft_f = tf.signal.stft(
        waveform,
        win_length,
        hop_length,
        n_fft
    )

    let spec_f = tf.pow(tf.abs(stft_f), 2.0);

    spec_f = _piecewise_linear_log(tf.add(spec_f, 1e-6));

    // Hardcoded mean/stddev
    spec_f = tf.mul(tf.add(spec_f, -15.0), 0.25);

    return spec_f;
}