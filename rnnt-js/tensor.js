import * as tf from '@tensorflow/tfjs';
import { base64 } from "rfc4648";

export async function loadTensor(url) {
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