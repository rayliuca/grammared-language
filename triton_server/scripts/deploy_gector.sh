#!/bin/bash
# Script to prepare the GECToR HuggingFace model for Triton deployment
# This script downloads the model and prepares it for serving

set -e

# Determine script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

MODEL_NAME="gector_roberta"
MODEL_VERSION="1"
MODEL_REPOSITORY="$REPO_ROOT/triton_server/model_repository"
MODEL_PATH="$MODEL_REPOSITORY/$MODEL_NAME/$MODEL_VERSION"

echo "=========================================="
echo "GECToR Model Deployment Script"
echo "=========================================="
echo ""
echo "Model: gotutiyan/gector-roberta-base-5k"
echo "Destination: $MODEL_PATH"
echo ""

# Ensure Python dependencies are installed
echo "Checking Python dependencies..."
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing transformers..."
    pip install transformers torch --quiet
fi

# Create a Python script to download the model
echo "Creating model download script..."
cat > /tmp/download_gector.py << 'EOF'
"""Download and cache the GECToR model from HuggingFace."""
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "gotutiyan/gector-roberta-base-5k"

print(f"Downloading model: {MODEL_NAME}")
print("This may take a few minutes on first run...")

try:
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer downloaded successfully")
    
    # Download model
    print("Downloading model...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    print(f"Model downloaded successfully")
    
    # Print model info
    print(f"\nModel Information:")
    print(f"  - Architecture: {model.config.architectures}")
    print(f"  - Number of labels: {model.config.num_labels}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Number of layers: {model.config.num_hidden_layers}")
    
    print("\nModel is cached and ready for use!")
    print("Triton will load the model from cache on startup.")
    
except Exception as e:
    print(f"Error downloading model: {e}", file=sys.stderr)
    sys.exit(1)
EOF

# Run the download script
echo ""
echo "Downloading model from HuggingFace..."
python3 /tmp/download_gector.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Model deployment successful!"
    echo "=========================================="
    echo ""
    echo "Model Location: $MODEL_PATH"
    echo "Configuration: $MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt"
    echo ""
    echo "The model is now ready to be served by Triton."
    echo "The actual model files are cached by HuggingFace and will be"
    echo "loaded automatically when Triton starts."
    echo ""
    echo "To start Triton server:"
    echo "  docker-compose up triton-server"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Model deployment failed!"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    exit 1
fi

# Clean up
rm -f /tmp/download_gector.py

echo "Deployment script completed."
