This is a basic clean PyTorch reimplementation of RNN-T transducers for speech recognition.

Original RNN-T Paper: https://arxiv.org/abs/1211.3711

It has many modifications:
 - Jasper-like convolutional encoder: https://arxiv.org/abs/1904.03288
 - Custom LSTM predictor from: https://github.com/pytorch/audio/blob/main/src/torchaudio/models/rnnt.py#L296

The idea is to be fast and simple, and to support training something in just a few hours on a single consumer GPU so you can quickly do experiments and try things out.

Current best WER is 28% on librispeech test-clean, which is horrible. However, this is with just 4 hours of training on a single RTX 3090, one epoch through librispeech. (And greedy decoding)

Ideas worth exploring:
 - Try with a Conformer style audio encoder
 - More tricks for packing in denser batches of audio
 - CUDA Graphs
 - Dataset augmentation

