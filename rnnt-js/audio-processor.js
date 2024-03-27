class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // You can initialize your variables here
    }



    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        // Assuming the input contains data and we only process channel 0
        if (input && input.length > 0) {
            const channelData = input[0];
            this.port.postMessage(channelData);
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
