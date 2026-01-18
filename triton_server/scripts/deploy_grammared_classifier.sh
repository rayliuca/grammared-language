#!/bin/bash
# Deployment script for Grammared Classifier Triton model

set -e

MODEL_NAME="grammared_classifier"
MODEL_REPO="/path/to/model_repository"
HF_MODEL="rayliuca/grammared-classifier-deberta-v3-base"

echo "Deploying Grammared Classifier model to Triton..."

# Check if model directory exists
if [ ! -d "${MODEL_REPO}/${MODEL_NAME}" ]; then
    echo "Error: Model directory ${MODEL_REPO}/${MODEL_NAME} not found"
    exit 1
fi

# Check if model files exist
if [ ! -f "${MODEL_REPO}/${MODEL_NAME}/1/model.py" ]; then
    echo "Error: model.py not found"
    exit 1
fi

if [ ! -f "${MODEL_REPO}/${MODEL_NAME}/1/pipeline.py" ]; then
    echo "Error: pipeline.py not found"
    exit 1
fi

if [ ! -f "${MODEL_REPO}/${MODEL_NAME}/config.pbtxt" ]; then
    echo "Error: config.pbtxt not found"
    exit 1
fi

echo "Model files found. Checking dependencies..."

# Create Python backend environment if it doesn't exist
if [ ! -d "/opt/tritonserver/python_backend_env" ]; then
    echo "Creating Python backend environment..."
    python3 -m venv /opt/tritonserver/python_backend_env
fi

# Install dependencies
echo "Installing dependencies..."
/opt/tritonserver/python_backend_env/bin/pip install --upgrade pip
/opt/tritonserver/python_backend_env/bin/pip install transformers torch numpy scikit-learn

# Download model from HuggingFace (optional - will be done on first run)
echo "Pre-downloading model from HuggingFace..."
python3 << EOF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("${HF_MODEL}")
print("Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained("${HF_MODEL}")
print("Model downloaded successfully!")
EOF

echo ""
echo "Deployment complete!"
echo ""
echo "To start the Triton server, run:"
echo "  tritonserver --model-repository=${MODEL_REPO}"
echo ""
echo "To test the model, run:"
echo "  python triton_server/scripts/test_grammared_classifier.py"
echo ""
