import * as tf from '@tensorflow/tfjs';

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

    const spec_f = tf.pow(tf.abs(stft_f), 2.0);

    return spec_f;
}