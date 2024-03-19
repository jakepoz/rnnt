import * as tf from '@tensorflow/tfjs';

async function loadModelAndPredict() {
    tf.setBackend('webgl');
    console.log("tf backend: ", tf.getBackend());
    //tf.ENV.set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 0);

    const model = await tf.loadGraphModel('tfjs_model/model.json');

    const inputTensor = tf.randomNormal([1, 1000, 80]);

    // Use the model to make a prediction
    let result = model.predict(inputTensor);

    // Process the result here (e.g., displaying it in the console or on the page)
    result.print(); // For demonstration, this will print the result tensor to the console

    // Time the prediction in the second run
    console.time('prediction');

    // let prof = await tf.profile(async () => {
    //     result = model.predict(inputTensor);
    // });

    //tf.enableDebugMode();
    result = model.predict(inputTensor);
    console.log(result);
    console.timeEnd('prediction');


}

// Call the function to load the model and make a prediction
loadModelAndPredict();
