import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';

import { setThreadsCount, getThreadsCount } from "@tensorflow/tfjs-backend-wasm";

import { classicFeaturizer, FeatureStreamer } from './featurizer';
import { loadTensor } from './tensor';
import { greedyDecode, decodeTokens, incrementalGreedyDecode } from './decoder';

import workletProcessorUrl from "./audio-processor.js?url";


function updateLog(...args) {
    const logElement = document.getElementById('log');
    // Convert all arguments to string and concatenate them with spaces
    const message = args.map(arg => {
        // Attempt to stringify objects, fall back to toString for others
        try {
            return (typeof arg === 'object') ? JSON.stringify(arg) : arg.toString();
        } catch (error) {
            return arg.toString(); // Fallback for objects that cannot be stringified
        }
    }).join(' ');

    logElement.innerHTML += message + '<br/>';
    logElement.scrollTop = logElement.scrollHeight; // Auto-scroll to the bottom
}


async function doSampleInference(encoder, encoderStreaming, predictor, joint, tokenizer) {
    // Load some sample audio directly, and attempt to decode it
    let audioData = await loadTensor('sampleaudio.json');

    console.log("audioData.shape: ", audioData.shape);

    const spec_features = tf.expandDims(featurizer(audioData), 0);
    //const spec_features = tf.zeros([1, 657, 201]);
    console.log("Featurizing!");
    console.log(spec_features.shape);
    spec_features.print();

    console.log(spec_features.slice([0, 100, 0], [1, 1, 201]).shape);
    spec_features.slice([0, 100, 0], [1, 1, 201]).print();

    // Convert the mel data to audio features
    const audioFeatures = encoder.execute(spec_features);
    console.log("audioFeatures.shape: ", audioFeatures.shape);

    console.time('greedyDecode');
    let result = greedyDecode(audioFeatures, predictor, joint);
    console.log(decodeTokens(result, tokenizer));
    console.timeEnd('greedyDecode');

    updateLog("Full Greedy Decoded Result: ", decodeTokens(result, tokenizer));

    let state = {
        input_state_0: tf.zeros([1, 9, 201]),
        input_state_1: tf.zeros([1, 10, 256]),
        input_state_2: tf.zeros([1, 10, 256]),
        input_state_3: tf.zeros([1, 10, 256]),
        input_state_4: tf.zeros([1, 10, 256]),
        input_state_5: tf.zeros([1, 12, 256]),
        input_state_6: tf.zeros([1, 12, 384]),
        input_state_7: tf.zeros([1, 12, 384]),
        input_state_8: tf.zeros([1, 12, 384]),
        input_state_9: tf.zeros([1, 24, 384]),
        input_state_10: tf.zeros([1, 24, 512]),
        input_state_11: tf.zeros([1, 24, 512]),
        input_state_12: tf.zeros([1, 24, 512]),
        input_state_13: tf.zeros([1, 56, 512]),
    }

    let decoderState = null;

    let startTime = performance.now();

    const streaming_feature_step = 20;

    for (let i = 0; i < spec_features.shape[1] - (spec_features.shape[1] % streaming_feature_step); i+=streaming_feature_step) {
        let firstSpecFeatures = spec_features.slice([0, i, 0], [1, streaming_feature_step, 201]);

        let streamingResult = encoderStreaming.execute({
            mel_features: firstSpecFeatures,
            ...state
        });


        let newAudioFeatures = streamingResult[1];

        // This is a horrible destructuring, because the model outputs get scrambled during the conversion process
        // So you need to just generate this table by hand for now
        state = {
            input_state_0: streamingResult[7],
            input_state_1: streamingResult[0],
            input_state_2: streamingResult[10],
            input_state_3: streamingResult[9],
            input_state_4: streamingResult[2],
            input_state_5: streamingResult[4],
            input_state_6: streamingResult[11],
            input_state_7: streamingResult[12],
            input_state_8: streamingResult[6],
            input_state_9: streamingResult[8],
            input_state_10: streamingResult[5],
            input_state_11: streamingResult[14],
            input_state_12: streamingResult[3],
            input_state_13: streamingResult[13],
        }

        decoderState = incrementalGreedyDecode(newAudioFeatures, predictor, joint, decoderState);
        console.log("Streaming tokens: ", decodeTokens(decoderState.tokens.slice(1), tokenizer));
    }

    let endTime = performance.now();
    console.log("Time taken for streaming inference: ", endTime - startTime);

}


async function startListening(encoderStreaming, predictor, joint, tokenizer) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Browser API navigator.mediaDevices.getUserMedia not available");
        updateLog("Error: Browser does not support required media devices.");
        return;
    }

    const vuLevel = document.getElementById('vuLevel');
    const perfDiv = document.getElementById('perf');
    const transcriptDiv = document.getElementById('transcript');
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        sampleRate: 48000,
        channelCount: 1,
    }, video: false });


    const audioContext = new AudioContext({
        sampleRate: 48000,
    });
    await audioContext.audioWorklet.addModule(workletProcessorUrl);
    const audioSourceNode = audioContext.createMediaStreamSource(stream);

    const audioProcessorNode = new AudioWorkletNode(audioContext, 'audio-processor');

    audioSourceNode.connect(audioProcessorNode);

    console.log("Connected audio source to processor node", audioContext.sampleRate);

    let decoderState = null;
    let featureStreamer = new FeatureStreamer(classicFeaturizer);

    let encoderState = {
        input_state_0: tf.zeros([1, 9, 201]),
        input_state_1: tf.zeros([1, 10, 256]),
        input_state_2: tf.zeros([1, 10, 256]),
        input_state_3: tf.zeros([1, 10, 256]),
        input_state_4: tf.zeros([1, 10, 256]),
        input_state_5: tf.zeros([1, 12, 256]),
        input_state_6: tf.zeros([1, 12, 384]),
        input_state_7: tf.zeros([1, 12, 384]),
        input_state_8: tf.zeros([1, 12, 384]),
        input_state_9: tf.zeros([1, 24, 384]),
        input_state_10: tf.zeros([1, 24, 512]),
        input_state_11: tf.zeros([1, 24, 512]),
        input_state_12: tf.zeros([1, 24, 512]),
        input_state_13: tf.zeros([1, 56, 512]),
    }

    let rmsFiltered = 0.0;
    let portMessages = 0, predictions = 0;
    let startTime = null;
  
    audioProcessorNode.port.onmessage = (event) => {
        const incomingSamples = new Float32Array(event.data);

        // Calculate the RMS of the samples for the VU meter
        const rms = Math.sqrt(incomingSamples.reduce((sum, val) => sum + val * val, 0) / incomingSamples.length);
        rmsFiltered = 0.90 * rmsFiltered + 0.1 * rms;
        const rmsPercentage = Math.min(rmsFiltered * 100, 100); 
        vuLevel.style.width = `${rmsPercentage}%`;

        const audioFeatures = featureStreamer.process(incomingSamples);
        portMessages++;

        if (!startTime) {
            startTime = performance.now();
        }

        if (audioFeatures) {
            let streamingResult = encoderStreaming.execute({
                mel_features: audioFeatures.expandDims(0),
                ...encoderState
            });

            let newAudioFeatures = streamingResult[1];

            // This is a horrible destructuring, because the model outputs get scrambled during the conversion process
            // So you need to just generate this table by hand for now
            encoderState = {
                input_state_0: streamingResult[7],
                input_state_1: streamingResult[0],
                input_state_2: streamingResult[10],
                input_state_3: streamingResult[9],
                input_state_4: streamingResult[2],
                input_state_5: streamingResult[4],
                input_state_6: streamingResult[11],
                input_state_7: streamingResult[12],
                input_state_8: streamingResult[6],
                input_state_9: streamingResult[8],
                input_state_10: streamingResult[5],
                input_state_11: streamingResult[14],
                input_state_12: streamingResult[3],
                input_state_13: streamingResult[13],
            }

            decoderState = incrementalGreedyDecode(newAudioFeatures, predictor, joint, decoderState);
            predictions++;
            updateLog("Predicted tokens: ", decodeTokens(decoderState.tokens.slice(1), tokenizer));

            transcriptDiv.innerText = decodeTokens(decoderState.tokens.slice(1), tokenizer);

            perfDiv.innerText = `Predictions / sec: ${(predictions / ((performance.now() - startTime) / 1000)).toFixed(2)}`;
        } 

    };
    
    updateLog("Listening...");

    return [audioContext, audioProcessorNode];
}

async function setupListeningButton(encoderStreaming, predictor, joint, tokenizer) {
    const button = document.getElementById('start-listening');

    let isListening = false;
    let audioProcessorNode = null;
    let audioContext = null;

    button.disabled = false;
    button.addEventListener('click', async () => {
        if (!isListening) {
            isListening = true;
            [audioContext, audioProcessorNode] = await startListening(encoderStreaming, predictor, joint, tokenizer);
            button.innerText = "Stop Listening";
        }
        else {
            isListening = false;
            audioProcessorNode.disconnect();
            audioContext.close();
            button.innerText = "Start Listening";
        }
    });
}

async function loadModelAndWarmup() {
    updateLog("Initializing TensorFlow.js...");

    document.getElementById('backend-info').innerText = tf.getBackend() + " " + tf.version.tfjs.toString();
    document.getElementById('start-listening').disabled = true;

    updateLog(JSON.stringify(tf.version));

    // tf.env().reset();
    //tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);    
    tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
    //tf.env().set('WEBGL_EXP_CONV', true);
    //tf.env().set("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE", 0);

    updateLog(JSON.stringify(tf.env().flags, null, 2));

    tf.registerOp("_MklLayerNorm", (node) => {
        const [ x, scale, offset ] = node.inputs;
        const { epsilon } = node.attrs;
        
        // Compute mean and variance
        const mean = x.mean(-1, true);
        const variance = x.sub(mean).square().mean(-1, true);
        // Normalize
        const normed = x.sub(mean).div(variance.add(epsilon).sqrt());
        // Apply scale and offset (gamma and beta)
        const scaled = normed.mul(scale);
        const shifted = scaled.add(offset);

        return shifted;
    });
  
    const encoder = await tf.loadGraphModel('models/encoder/model.json', {
        onProgress: (fraction) => {
            const percentComplete = Math.round(fraction * 100);
            updateLog(`Encoder model loading progress: ${percentComplete}%`);
        }
    });

    const encoderStreaming = await tf.loadGraphModel('models/encoder_streaming/model.json', {
        onProgress: (fraction) => {
            const percentComplete = Math.round(fraction * 100);
            updateLog(`Encoder streaming model loading progress: ${percentComplete}%`);
        }
    });

    const predictor = await tf.loadGraphModel('models/predictor/model.json', {
        onProgress: (fraction) => {
            const percentComplete = Math.round(fraction * 100);
            updateLog(`Predictor model loading progress: ${percentComplete}%`);
        }
    });

    const joint = await tf.loadGraphModel('models/joint/model.json', {
        onProgress: (fraction) => {
            const percentComplete = Math.round(fraction * 100);
            updateLog(`Joint model loading progress: ${percentComplete}%`);
        }
    });

    const tokenizer = await fetch('models/tokenizer.json').then(response => response.json());



    //Warmup all the networks once 
    const testMelFeatures = tf.zeros([1, 1000, 201]);
    const testTextTokens = tf.tensor2d([[0, 1, 2, 3, 4]], [1, 5], 'int32');

    const testJoinerInput1 = tf.zeros([1, 1, 1024]);
    const testJoinerInput2 = tf.zeros([1, 1, 1024]);

    await encoder.predictAsync(testMelFeatures);
    updateLog("Warmed up encoder");

    await predictor.predictAsync(testTextTokens);
    updateLog("Warmed up predictor");

    await joint.predictAsync([testJoinerInput1, testJoinerInput2]);
    updateLog("Warmed up joint");

    // Now time them
    console.time('encoder');
    let testAudioFeatures = encoder.execute(testMelFeatures);
    console.log(testAudioFeatures);
    console.log(testAudioFeatures.shape)
    console.timeEnd('encoder');

    console.time('predictor');
    let testTextFeatures = predictor.execute(testTextTokens);
    console.log(testTextFeatures);
    console.log(testTextFeatures.shape);
    console.timeEnd('predictor');

    console.time('joint');
    for (let i = 0; i < 350; i++) {
        let testLogits = joint.execute({
            audio_frame: testJoinerInput1, 
            text_frame: testJoinerInput2
        });
    }
    console.timeEnd('joint');

    //doSampleInference(encoder, encoderStreaming, predictor, joint, tokenizer);

    setupListeningButton(encoderStreaming, predictor, joint, tokenizer);

    updateLog("Ready, press 'Start Listening' to begin.");
}

setThreadsCount(4);
// console.log(tf.env().flags);
//tf.setBackend('wasm').then(() => loadModelAndWarmup());
//tf.setBackend('webgpu').then(() => loadModelAndWarmup());
tf.setBackend('webgl').then(() => loadModelAndWarmup());
