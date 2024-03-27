class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        // Assuming the input contains data and we only process channel 0
        if (input && input.length > 0) {
            // We are running the main input audio at 48khz, which we just take every 3rd sample to downsample easily to 16khz
            const channelData = input[0];
            const downsampledData = new Float32Array(Math.ceil(channelData.length / 3));
            for (let i = 0, j = 0; i < channelData.length; i += 3, j++) {
                downsampledData[j] = channelData[i];
            }

            this.port.postMessage(downsampledData);
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
