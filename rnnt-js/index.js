import { base64 } from "rfc4648";
import * as tf from '@tensorflow/tfjs';

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

function greedyDecode(audioFeatures, encoder, predictor, joint, max_length = 200) {
    const blank_idx = 1023;

    // Initialize tokens array with the blank index, assuming 0 is the blank index
    let tokens = [blank_idx]; // Update this if your model has a different blank index
    let cur_audio_time = 0;
    const max_audio_time = audioFeatures.shape[1];
    let cur_outputs_per_step = 0;
    const max_outputs_per_step = 10;

    // Convert tokens to a tensor
    let input_ids = tf.tensor2d([tokens], [1, tokens.length], 'int32');

    while (cur_audio_time < max_audio_time && tokens.length < max_length) {
        // Extract a slice of audio features for the current time step
        const audio_feature_slice = audioFeatures.slice([0, cur_audio_time, 0], [1, 1, audioFeatures.shape[2]]);
        audio_feature_slice.print();

        // Get the latest token for prediction
        let predictor_feature_slice = predictor.predict(input_ids);
        predictor_feature_slice = predictor_feature_slice.slice([0, predictor_feature_slice.shape[1] - 1, 0], [1, 1, predictor_feature_slice.shape[2]]);
        predictor_feature_slice.print();

        // Compute joint features for the current audio and predictor features
        //const joint_features = joint.predict([audio_feature_slice, predictor_feature_slice]);
        const joint_features = joint.predict([predictor_feature_slice, audio_feature_slice]);
        const joint_logits = joint_features.squeeze([0]);
        joint_logits.print();

        // Get the most likely token index
        const token_idx = joint_logits.argMax(-1).dataSync()[0];

        if (token_idx === blank_idx || cur_outputs_per_step >= max_outputs_per_step) { // Assuming 0 is the blank token index
            cur_audio_time += 1;
            cur_outputs_per_step = 0;
        } else {
            tokens.push(token_idx);
            // Update input_ids for the next prediction
            input_ids = tf.tensor2d([tokens], [1, tokens.length], 'int32');
            cur_outputs_per_step += 1;
        }
    }

    // Log the final tokens
    console.log("Decoded tokens:", tokens);
    return tokens.slice(1); // Skip the initial blank token for the result
}


async function loadModelAndPredict() {
    tf.setBackend('webgl');
    console.log("tf backend: ", tf.getBackend());

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
  
    const encoder = await tf.loadGraphModel('models/encoder/model.json');
    const predictor = await tf.loadGraphModel('models/predictor/model.json');
    const joint = await tf.loadGraphModel('models/joint/model.json');

    const testMelFeatures = tf.zeros([1, 1000, 80]);
    const testTextTokens = tf.tensor2d([[0, 1, 2, 3, 4]], [1, 5], 'int32');

    const testJoinerInput1 = tf.zeros([1, 1, 1024]);
    const testJoinerInput2 = tf.zeros([1, 1, 1024]);


    //Warmup all the networks once 
    console.log("warming up")
    encoder.predict(testMelFeatures);
    predictor.predict(testTextTokens);
    joint.predict([testJoinerInput1, testJoinerInput2]);
    console.log("done warming up");

    // Now time them
    console.time('encoder');
    let testAudioFeatures = encoder.predict(testMelFeatures);
    testAudioFeatures.print();
    console.log(testAudioFeatures);
    console.log(testAudioFeatures.shape)
    console.timeEnd('encoder');

    console.time('predictor');
    let testTextFeatures = predictor.predict(testTextTokens);
    testTextFeatures.print();
    console.log(testTextFeatures);
    console.log(testTextFeatures.shape);
    console.timeEnd('predictor');

    console.time('joint');
    let testLogits = joint.predict([testJoinerInput1, testJoinerInput2]);
    testLogits.print();
    console.log(testLogits);
    console.log(testLogits.shape);
    console.timeEnd('joint');

    // Now load some sample mel data and attempt to decode it
    let melData = await loadTensor('samplemels.json');
    melData = melData.reshape([1, melData.shape[0], melData.shape[1]]);
    melData = melData.transpose([0, 2, 1]);
    console.log("melData.shape: ", melData.shape);
    melData.print();

    // Convert the mel data to audio features
    const audioFeatures = encoder.predict(melData);
    console.log("audioFeatures.shape: ", audioFeatures.shape);

    result = greedyDecode(audioFeatures, encoder, predictor, joint);
    console.log(result);
}

// Call the function to load the model and make a prediction
loadModelAndPredict();
