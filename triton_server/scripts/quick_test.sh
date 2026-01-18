#!/bin/bash
# Quick Test - Single Command
# Usage: ./quick_test.sh [SERVER_URL]

SERVER_URL="${1:-localhost:8000}"
TEST_TEXT='I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'

echo "Testing Triton server at: $SERVER_URL"
echo "Input: $TEST_TEXT"
echo ""

curl -s -X POST "http://${SERVER_URL}/v2/models/grammared_classifier/versions/1/infer" \
    -H "Content-Type: application/json" \
    -d "{
  \"inputs\": [
    {
      \"name\": \"TEXT\",
      \"shape\": [1, 1],
      \"datatype\": \"BYTES\",
      \"data\": [\"${TEST_TEXT}\"]
    }
  ]
}" | jq -r '.outputs[0].data[0]' | jq '.'

echo ""
echo "Expected: {\"label\": \"none\", \"score\": ~0.92-0.97}"
