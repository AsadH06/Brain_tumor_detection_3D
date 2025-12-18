IMAGES_DATA_DIR = "BrainTumorData/imagesTest"
LABELS_DATA_DIR = "BrainTumorData/labelsTest"

# Primary TensorFlow/Keras checkpoint used for inference.
TF_MODEL_PATH = "finalvalaug.h5"

# Optional ONNX export for faster inference when using onnxruntime.
ONNX_MODEL_PATH = "finalvalaug.onnx"

# Default backend toggle for CLI inference (options: "tf", "onnx").
INFERENCE_BACKEND = "tf"

# Batch processing defaults
BATCH_INPUT_DIR = "BrainTumorData/imagesTest"
BATCH_OUTPUT_DIR = "outputs"
