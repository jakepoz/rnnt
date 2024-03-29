import { base64 } from "rfc4648";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';

import { setThreadsCount, getThreadsCount } from "@tensorflow/tfjs-backend-wasm";

import { featurizer } from './featurizer';
import { downloadAudio } from "./wav";

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


async function loadTensor(url) {
    try {
        const response = await fetch(url);
        const json = await response.json();

        // Decode the base64 data to a byte array
        const byteArray = base64.parse(json.data);

        // Determine the appropriate dtype and create the tensor
        let tensor;
        switch (json.dtype) {
            case 'float32':
                tensor = tf.tensor(new Float32Array(byteArray.buffer), json.shape);
                break;
            case 'int32':
                tensor = tf.tensor(new Int32Array(byteArray.buffer), json.shape);
                break;
            default:
                throw new Error('Unsupported tensor data type');
        }

        return tensor;
    } catch (error) {
        console.error('Error loading tensor from JSON:', error);
        throw error;
    }
}

function greedyDecode(audioFeatures, encoder, predictor, joint, existingTokens = [], max_length = 200) {
    const blank_idx = 1023;

    // Initialize tokens array with the blank index, assuming 0 is the blank index
    let tokens = [blank_idx]; // Update this if your model has a different blank index
    let cur_audio_time = 0;
    const max_audio_time = audioFeatures.shape[1];
    let cur_outputs_per_step = 0;
    const max_outputs_per_step = 10;

    // If there are existing tokens, add them to the tokens array
    if (existingTokens.length > 0) {
        tokens = tokens.concat(existingTokens);
    }

    // Convert tokens to a tensor
    let input_ids = tf.tensor2d([tokens], [1, tokens.length], 'int32');

    let allAudioSlices = tf.split(audioFeatures, audioFeatures.shape[1], 1);

    // Get the latest token for prediction
    let predictor_feature_slice = predictor.predict(input_ids);
    predictor_feature_slice = predictor_feature_slice.slice([0, predictor_feature_slice.shape[1] - 1, 0], [1, 1, predictor_feature_slice.shape[2]]);


    while (cur_audio_time < max_audio_time && tokens.length < max_length) {
        // Extract a slice of audio features for the current time step
        //const audio_feature_slice = audioFeatures.slice([0, cur_audio_time, 0], [1, 1, audioFeatures.shape[2]]);
        const audio_feature_slice = allAudioSlices[cur_audio_time];
  
        // Compute joint features for the current audio and predictor features
        //const joint_features = joint.predict([audio_feature_slice, predictor_feature_slice]);
        const joint_features = joint.predict({text_frame: predictor_feature_slice, audio_frame: audio_feature_slice});
        const joint_logits = joint_features.squeeze([0]);

        // Get the most likely token index
        const token_idx = joint_logits.argMax(-1).dataSync()[0];
        //const token_idx = tf.argMax(joint_logits, -1).bufferSync().values[0];
        //const token_idx = joint_logits.argMax(-1)[0];

        if (token_idx === blank_idx || cur_outputs_per_step >= max_outputs_per_step) { // Assuming 0 is the blank token index
            cur_audio_time += 1;
            cur_outputs_per_step = 0;
        } else {
            tokens.push(token_idx);

            // Update input_ids for the next prediction
            input_ids = tf.tensor2d([tokens], [1, tokens.length], 'int32');

            predictor_feature_slice = predictor.predict(input_ids);
            predictor_feature_slice = predictor_feature_slice.slice([0, predictor_feature_slice.shape[1] - 1, 0], [1, 1, predictor_feature_slice.shape[2]]);
        
            cur_outputs_per_step += 1;
        }
    }

    return tokens.slice(1); // Skip the initial blank token for the result
}

function decodeTokens(tokenIds, tokenizer) {
    let decodedString = '';
  
    for (let i = 0; i < tokenIds.length; i++) {
      let token = tokenizer[tokenIds[i]];
  
      // If the token starts with the special space symbol, add it without the leading space
      // character and ensure there's a space before it unless it's the first token.
      if (token && token.startsWith('\u2581')) {
        decodedString += (i > 0 ? ' ' : '') + token.substring(1);
      } else {
        // For other tokens, just concatenate them.
        decodedString += token;
      }
    }
  
    return decodedString;
}

function updateMemoryInfo() {
    const memoryInfo = JSON.stringify(tf.memory(), null, 2);
    document.getElementById('memory-info').innerText = memoryInfo;
}

async function doSampleInference(encoder, predictor, joint, tokenizer) {
    // Now load some sample mel data and attempt to decode it
    let melData = await loadTensor('samplemels.json');
    melData = melData.reshape([1, melData.shape[0], melData.shape[1]]);
    melData = melData.transpose([0, 2, 1]);
    console.log("melData.shape: ", melData.shape);




    // Load some sample audio directly, and attempt to decode it
    let audioData = await loadTensor('sampleaudio.json');

    console.log("audioData.shape: ", audioData.shape);

    const spec_features = tf.expandDims(featurizer(audioData), 0);
    console.log("Featurizing!");
    console.log(spec_features.shape);
    spec_features.print();

    // Convert the mel data to audio features
    const audioFeatures = encoder.predict(spec_features);
    console.log("audioFeatures.shape: ", audioFeatures.shape);

    console.time('greedyDecode');
    let result = greedyDecode(audioFeatures, encoder, predictor, joint);
    console.log(decodeTokens(result, tokenizer));
    console.timeEnd('greedyDecode');

    updateLog("Result: ", decodeTokens(result, tokenizer));
}


async function startListening(encoder, predictor, joint, tokenizer) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Browser API navigator.mediaDevices.getUserMedia not available");
        updateLog("Error: Browser does not support required media devices.");
        return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        sampleRate: 48000,
        channelCount: 1,
    }, video: false });


    const audioContext = new AudioContext({
        sampleRate: 48000,
    });
    await audioContext.audioWorklet.addModule('audio-processor.js'); // Ensure this path is correct!
    const audioSourceNode = audioContext.createMediaStreamSource(stream);

    const audioProcessorNode = new AudioWorkletNode(audioContext, 'audio-processor');

    audioSourceNode.connect(audioProcessorNode);

    console.log("Connected audio source to processor node", audioContext.sampleRate);

    let allTokens = [];
    
    const hopSize = 160, windowSize = 400;
    const sampleBufferSize = 1280;
    const largeBufferSize = 16000;
    let largeBuffer = new Float32Array(largeBufferSize);
    let sampleBuffer = new Float32Array(0);

    let start = null;
    let totalSamples = 0, totalRuns = 0;

    audioProcessorNode.port.onmessage = (event) => {
        const incomingSamples = new Float32Array(event.data);
        
        if (!start) {
            start = performance.now();
        }

        sampleBuffer = new Float32Array([...sampleBuffer, ...incomingSamples]);
        totalSamples += incomingSamples.length;
        //console.log("Sample buffer length: ", sampleBuffer.length);
        
        if (sampleBuffer.length >= sampleBufferSize) {
            const localSampleBuffer = sampleBuffer.subarray(0, sampleBufferSize);
            sampleBuffer = sampleBuffer.subarray(sampleBufferSize);
    
            // Update largeBuffer by simulating a rolling buffer
            largeBuffer.copyWithin(0, sampleBufferSize);
            largeBuffer.set(localSampleBuffer, largeBufferSize - sampleBufferSize);
    
            const newSpecFeatures = tf.expandDims(featurizer(tf.tensor1d(largeBuffer)), 0);
            const newAudioFeatures = encoder.predict(newSpecFeatures);
    
            const numNewFeatures = sampleBufferSize / hopSize / 2;
            const audioFeatureBuffer = newAudioFeatures.slice([0, newAudioFeatures.shape[1] - numNewFeatures, 0], [1, numNewFeatures, newAudioFeatures.shape[2]]);
    
            allTokens = greedyDecode(audioFeatureBuffer, encoder, predictor, joint, allTokens);
            updateLog(decodeTokens(allTokens, tokenizer));
            //console.log(decodeTokens(allTokens, tokenizer));

            // if (++totalRuns % 10 === 0) {
            //     downloadAudio(largeBuffer, 16000);
            // }
        }
    };
    
    console.log("Listening...");
}

async function loadModelAndPredict() {
    updateLog("Initializing TensorFlow.js...");

    document.getElementById('backend-info').innerText = tf.getBackend() + " " + tf.version.tfjs.toString();
 
    updateLog(JSON.stringify(tf.version));

    // tf.env().reset();
    //tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);    
    tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
    //tf.env().set('WEBGL_EXP_CONV', true);
    //tf.env().set("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE", 0);

    updateLog(JSON.stringify(tf.env().flags, null, 2));
    updateMemoryInfo();

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

    updateMemoryInfo();

    // Now time them
    console.time('encoder');
    let testAudioFeatures = encoder.predict(testMelFeatures);
    console.log(testAudioFeatures);
    console.log(testAudioFeatures.shape)
    console.timeEnd('encoder');

    console.time('predictor');
    let testTextFeatures = predictor.predict(testTextTokens);
    console.log(testTextFeatures);
    console.log(testTextFeatures.shape);
    console.timeEnd('predictor');

    console.time('joint');
    for (let i = 0; i < 350; i++) {
        let testLogits = joint.predict({
            audio_frame: testJoinerInput1, 
            text_frame: testJoinerInput2
        });
    }
    console.timeEnd('joint');

    doSampleInference(encoder, predictor, joint, tokenizer);

    document.getElementById('start-listening').disabled = false;
    document.getElementById('start-listening').addEventListener('click', () => {
        startListening(encoder, predictor, joint, tokenizer);
    });
}

setThreadsCount(4);
// console.log(tf.env().flags);
//tf.setBackend('wasm').then(() => loadModelAndPredict());
//tf.setBackend('webgpu').then(() => loadModelAndPredict());
tf.setBackend('webgl').then(() => loadModelAndPredict());
