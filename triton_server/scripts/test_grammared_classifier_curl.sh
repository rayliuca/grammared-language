#!/bin/bash
# Test script for Grammared Classifier using curl
# This script tests the Triton server with the example input format

# Configuration
SERVER_URL="${TRITON_SERVER_URL:-localhost:8000}"
MODEL_NAME="${MODEL_NAME:-grammared_classifier}"
MODEL_VERSION="${MODEL_VERSION:-1}"

echo "================================================================================"
echo "Testing Grammared Classifier on Triton Server (HTTP REST API)"
echo "================================================================================"
echo "Server URL: $SERVER_URL"
echo "Model: $MODEL_NAME, Version: $MODEL_VERSION"
echo ""

# Check if server is live
echo "Checking if Triton server is live..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${SERVER_URL}/v2/health/live")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Triton server is live"
else
    echo "✗ Triton server is not responding (HTTP code: $HTTP_CODE)"
    exit 1
fi

# Check if model is ready
echo "Checking if model is ready..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${SERVER_URL}/v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}/ready")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Model ${MODEL_NAME} is ready"
else
    echo "✗ Model ${MODEL_NAME} is not ready (HTTP code: $HTTP_CODE)"
    echo ""
    echo "Available models:"
    curl -s "http://${SERVER_URL}/v2/models" | jq '.' 2>/dev/null || echo "Could not fetch model list"
    exit 1
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "Running inference with example text..."
echo "--------------------------------------------------------------------------------"
echo ""

# Test text matching the pipeline example
TEST_TEXT='I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'

echo "Input text:"
echo "  '${TEST_TEXT}'"
echo ""

# Create JSON request
# Note: The model has max_batch_size configured, so we need to include batch dimension
# Shape should be [batch_size, num_elements] = [1, 1]
REQUEST_JSON=$(cat <<EOF
{
  "inputs": [
    {
      "name": "TEXT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["${TEST_TEXT}"]
    }
  ]
}
EOF
)

echo "Sending inference request..."
echo ""

# Send request and capture response
RESPONSE=$(curl -s -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}/infer" \
    -H "Content-Type: application/json" \
    -d "${REQUEST_JSON}")

echo "Raw response:"
echo "${RESPONSE}" | jq '.' 2>/dev/null || echo "${RESPONSE}"
echo ""

# Extract the output data (it's a JSON string)
OUTPUT=$(echo "${RESPONSE}" | jq -r '.outputs[0].data[0]' 2>/dev/null || echo "")

if [ -n "$OUTPUT" ] && [ "$OUTPUT" != "null" ]; then
    echo "✓ Inference successful!"
    echo ""
    echo "Result:"
    echo "  ${OUTPUT}" | jq '.' 2>/dev/null || echo "  ${OUTPUT}"
    echo ""

    # Parse label and score
    LABEL=$(echo "${OUTPUT}" | jq -r '.label' 2>/dev/null || echo "")
    SCORE=$(echo "${OUTPUT}" | jq -r '.score' 2>/dev/null || echo "")

    if [ -n "$LABEL" ] && [ -n "$SCORE" ]; then
        echo "Details:"
        echo "  Label: ${LABEL}"
        echo "  Score: ${SCORE}"
        echo ""
    fi

    echo "--------------------------------------------------------------------------------"
    echo "Expected output (from local pipeline):"
    echo "  {'label': 'none', 'score': 0.9268353033472629}"
    echo ""
    echo "Note: Scores may differ slightly due to:"
    echo "  - Different hardware (CPU vs GPU)"
    echo "  - Floating point precision"
    echo "  - Model calibration"
else
    echo "✗ Failed to get valid output from model"
    exit 1
fi

echo "================================================================================"
