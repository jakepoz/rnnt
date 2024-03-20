# Quit on errors
set -e

checkpoint="/home/jake/rnnt/experiments/basic_char/run-64/checkpoint_step_40000.pt"

# Export to ONNX
python -m rnnt.export_onnx $checkpoint


# Onnx to TF using https://github.com/PINTO0309/onnx2tf
# Needs to be setup correctly in its own conda environment, depends on TF 2.15.0
pushd export
source activate onnx2tf

onnx2tf -i encoder.onnx -b1 -osd -o saved_model_encoder

onnx2tf -i predictor.onnx -b1 -osd -o saved_model_predictor

onnx2tf -i joint.onnx -b1 -osd -o saved_model_joint
popd

# # Convert to TFJS
pushd rnnt-js/models
cp ../../export/tokenizer.json .

source activate tfjs
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../export/saved_model_encoder encoder

# Skipping op check because LayerNorm is not supported, but maybe there is a better way to handle this
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve --skip_op_check ../../export/saved_model_predictor predictor
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../export/saved_model_joint joint
