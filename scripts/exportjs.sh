# Quit on errors
set -e

checkpoint="/home/jake/rnnt/experiments/basic_char/run-7/checkpoint_step_140621.pt"

# Export to ONNX
python -m rnnt.export_onnx $checkpoint


# Onnx to TF using https://github.com/PINTO0309/onnx2tf
# Needs to be setup correctly in its own conda environment, depends on TF 2.15.0
pushd export
source activate onnx2tf

# TODO Uncomment its just slow
#onnx2tf -i encoder.onnx -b1 -osd -o saved_model_encoder

onnx2tf -i predictor.onnx -b1 -osd -o saved_model_predictor

onnx2tf -i joint.onnx -b1 -osd -o saved_model_joint
popd

# # Convert to TFJS
pushd rnnt-js/models
cp ../../export/tokenizer.json .

source activate tfjs
#tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../export/saved_model_encoder encoder
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../export/saved_model_predictor predictor
#tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ../../export/saved_model_joint joint
