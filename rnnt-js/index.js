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
    const testTextTokens = tf.tensor2d([[0, 1, 2, 3, 4]], [1, 5], 'int32');

    const testJoinerInput1 = tf.zeros([1, 1024, 1]);
    const testJoinerInput2 = tf.zeros([1, 1024, 1]);

     let prof = await tf.profile(async () => {
        result =  encoder.predict(testMelFeatures);
    });

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
    console.log(testLogits);
    console.log(testLogits.shape);
    console.timeEnd('joint');
}

// Call the function to load the model and make a prediction
loadModelAndPredict();
