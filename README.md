This is a basic clean PyTorch reimplementation of RNN-T transducers for speech recognition.

The coolest part is that it can run in the browser with [TFJS](https://github.com/tensorflow/tfjs)! This code is available in the [rnnt-js](rnnt-js) folder.
 In order to accomplish this, the architecture was simplified to use only convolutional layers for both the audio and text encoding. These are easy to export to ONNX and then to TFJS, 
 and are also easy to stream.

You can see a demo [here](https://rnnt.jakepoz.com/). But be warned, the network has not been fully trained, and only gets ~15% WER on Librispeech.

---

Some other nice things about this repo:
 - Basic RNN-T architecture: https://arxiv.org/abs/1211.3711
 - Jasper-like convolutional encoder: https://arxiv.org/abs/1904.03288
 - Options for trying different configurations
 - Simple streaming featurizer that works the same in [PyTorch](https://github.com/jakepoz/rnnt/blob/master/rnnt/featurizer.py) and [TFJS](https://github.com/jakepoz/rnnt/blob/master/rnnt-js/featurizer.js).
 - [Audio Augmentations](https://github.com/jakepoz/rnnt/blob/master/rnnt/augment.py) for training

The idea is to be fast and simple, and to support training something in just a few hours on a single consumer GPU so you can quickly do experiments and try things out.

You can achieve around 30% WER on librispeech test-clean, which is horrible. However, this is with just 4 hours of training on a single RTX 3090, one epoch through librispeech. (And greedy decoding) With more training and GPUs, you can push this down a bit further.

Ideas for improvement:
 - [ ] Narrow down to a better featurizer that stays identical between Python and JS
 - [ ] More dataset augmentation, ex. SpecAugment, room impulse response, etc.
 - [ ] Train for 100+ epochs
 - [ ] Train with more data
 - [ ] Performance improvement on the TFJS side.
 - [ ] Pretrain the text encoder with lots of text


