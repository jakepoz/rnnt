import * as tf from '@tensorflow/tfjs';

export function greedyDecode(audioFeatures, predictor, joint, max_length = 200) {
    const blank_idx = 1023;

    // Initialize tokens array with the blank index, assuming 0 is the blank index
    let tokens = [blank_idx]; // Update this if your model has a different blank index
    let cur_audio_time = 0;
    const max_audio_time = audioFeatures.shape[1];
    let cur_outputs_per_step = 0;
    const max_outputs_per_step = 10;

    // Convert tokens to a tensor
    let input_ids = tf.tensor2d([tokens], [1, tokens.length], 'int32');

    let allAudioSlices = tf.split(audioFeatures, audioFeatures.shape[1], 1);

    // Get the latest token for prediction
    let predictor_feature_slice = predictor.execute(input_ids);
    predictor_feature_slice = predictor_feature_slice.slice([0, predictor_feature_slice.shape[1] - 1, 0], [1, 1, predictor_feature_slice.shape[2]]);


    while (cur_audio_time < max_audio_time && tokens.length < max_length) {
        // Extract a slice of audio features for the current time step
        //const audio_feature_slice = audioFeatures.slice([0, cur_audio_time, 0], [1, 1, audioFeatures.shape[2]]);
        const audio_feature_slice = allAudioSlices[cur_audio_time];
  
        // Compute joint features for the current audio and predictor features
        //const joint_features = joint.execute([audio_feature_slice, predictor_feature_slice]);
        const joint_features = joint.execute({text_frame: predictor_feature_slice, audio_frame: audio_feature_slice});
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

            predictor_feature_slice = predictor.execute(input_ids);
            predictor_feature_slice = predictor_feature_slice.slice([0, predictor_feature_slice.shape[1] - 1, 0], [1, 1, predictor_feature_slice.shape[2]]);
        
            cur_outputs_per_step += 1;
        }
    }

    return tokens.slice(1); // Skip the initial blank token for the result
}

export function incrementalGreedyDecode(audioFeatures, predictor, joint, state) {
    const blank_idx = 1023;
    const max_audio_time = audioFeatures.shape[1];
    const max_outputs_per_step = 10;

    if (!state) {
        state = {
            tokens: [blank_idx], 
            predictor_feature_slice: predictor.execute(tf.tensor2d([[blank_idx]], [1, 1], 'int32')),
        }
    }

    let cur_outputs_per_step = 0;
    let cur_audio_time = 0;
    let allAudioSlices = tf.split(audioFeatures, audioFeatures.shape[1], 1);

    while (cur_audio_time < max_audio_time) {
        // Extract a slice of audio features for the current time step
        const audio_feature_slice = allAudioSlices[cur_audio_time];
  
        // Compute joint features for the current audio and predictor features
        const joint_features = joint.execute({text_frame: state.predictor_feature_slice, audio_frame: audio_feature_slice});
        const joint_logits = joint_features.squeeze([0]);

        // Get the most likely token index
        const token_idx = joint_logits.argMax(-1).dataSync()[0];

        if (token_idx === blank_idx || cur_outputs_per_step >= max_outputs_per_step) { // Assuming 0 is the blank token index
            cur_audio_time += 1;
            cur_outputs_per_step = 0;
        } else {
            state.tokens.push(token_idx);

            // Update input_ids for the next prediction, TODO you can optimize this for long lengths by figuring out what the receptive field is
            let input_ids = tf.tensor2d([state.tokens], [1, state.tokens.length], 'int32');
            let new_prediction = predictor.execute(input_ids);
            state.predictor_feature_slice = new_prediction.slice([0, new_prediction.shape[1] - 1, 0], [1, 1, new_prediction.shape[2]]);
        
            cur_outputs_per_step += 1;
        }
    }

    return state;
}

export function decodeTokens(tokenIds, tokenizer) {
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