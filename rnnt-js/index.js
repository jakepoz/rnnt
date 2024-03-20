import * as tf from '@tensorflow/tfjs';

async function loadModelAndPredict() {
    tf.setBackend('webgl');
    console.log("tf backend: ", tf.getBackend());

    tf.registerOp("_MklLayerNorm", (node) => {
        const [ x, scale, offset ] = node.inputs;
        const { epsilon } = node.attrs;

        // console.log("x.shape: ", x.shape);
        // console.log("scale.shape: ", scale.shape);
        // console.log("offset.shape: ", offset.shape);
        // console.log("epsilon: ", epsilon);
        
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
    const testTextTokens = tf.tensor2d([[0]], [1, 1], 'int32');



    // Time the prediction in the second run
    console.time('encoder');

    //tf.enableDebugMode();
    let testAudioFeatures = encoder.predict(testMelFeatures);
    console.log(testAudioFeatures);
    console.log(testAudioFeatures.shape)
    console.timeEnd('encoder');


    console.time('predictor');
    let testTextFeatures = predictor.predict(testTextTokens);
    console.log(testTextFeatures);
    console.log(testTextFeatures.shape);
    console.timeEnd('predictor');
}

// Call the function to load the model and make a prediction
loadModelAndPredict();
