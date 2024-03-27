# Quit on errors
set -e

checkpoint="/home/jake/rnnt/experiments/basic_char_convjs/run-4/checkpoint_step_20000.pt"

# Export to ONNX
python -m rnnt.export_onnx $checkpoint


# Onnx to TF using https://github.com/PINTO0309/onnx2tf
# Needs to be setup correctly in its own conda environment, depends on TF 2.15.0
pushd export
source activate onnx2tf

onnx2tf -i encoder.onnx -b1 -osd -o saved_model_encoder

onnx2tf -i predictor.onnx -b1 -osd -o saved_model_predictor

# Have to keep the input shapes, because otherwise it thinks it's doing a NCW -> NWC conversion
# Also passing -coion to keep input output names the same, otherwise they can get reordered and it's a pain to debug why the joiner doesn't work anymore
onnx2tf -i joint.onnx -b1 -osd -kat "audio_frame" "text_frame" -coion -o saved_model_joint
popd

# Convert to TFJS
mkdir -p rnnt-js/public/models

pushd rnnt-js/public/models
cp ../../../export/tokenizer.json .

source activate tfjs
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../../export/saved_model_encoder encoder

# Skipping op check because LayerNorm is not supported, but maybe there is a better way to handle this
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve --skip_op_check ../../../export/saved_model_predictor predictor

tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve  ../../../export/saved_model_joint joint
