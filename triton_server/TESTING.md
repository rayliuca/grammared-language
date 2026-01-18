# Testing Triton Server Models

This guide explains how to test the Grammared Classifier model deployed on Triton Inference Server.

## Prerequisites

### Start the Triton Server

Make sure the Triton server is running via Docker Compose:

```bash
docker-compose up triton-server
```

Check that the server is healthy:

```bash
docker ps | grep grammared-triton
```

You should see the container running with status `(healthy)`.

## Testing Methods

### Method 1: Using curl (Recommended - No Dependencies)

The easiest way to test is using the provided bash script that only requires `curl` and `jq`:

```bash
cd /media/auser/WD4TB/github/grammared_language
./triton_server/scripts/test_grammared_classifier_curl.sh
```

**Example Output:**
```
================================================================================
Testing Grammared Classifier on Triton Server (HTTP REST API)
================================================================================
Server URL: localhost:8000
Model: grammared_classifier, Version: 1

✓ Triton server is live
Checking if model is ready...
✓ Model grammared_classifier is ready

--------------------------------------------------------------------------------
Running inference with example text...
--------------------------------------------------------------------------------

Input text:
  'I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'

✓ Inference successful!

Result:
{
  "label": "none",
  "score": 0.9689850294275044
}

Details:
  Label: none
  Score: 0.9689850294275044

--------------------------------------------------------------------------------
Expected output (from local pipeline):
  {'label': 'none', 'score': 0.9268353033472629}
================================================================================
```

### Method 2: Direct curl Command

You can also send requests directly using curl:

```bash
curl -X POST "http://localhost:8000/v2/models/grammared_classifier/versions/1/infer" \
    -H "Content-Type: application/json" \
    -d '{
  "inputs": [
    {
      "name": "TEXT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>"]
    }
  ]
}' | jq '.'
```

**Note:** The shape `[1, 1]` includes the batch dimension because the model has `max_batch_size: 8` configured.

### Method 3: Using Python with tritonclient

First, install the Triton client library:

```bash
pip install tritonclient[http]
```

Then run the Python test script:

```bash
python triton_server/scripts/test_grammared_classifier_example.py
```

For multiple examples:

```bash
python triton_server/scripts/test_grammared_classifier_example.py --multiple
```

## Understanding the Input Format

The model expects text in the following format:

```
<sentence> [CLS]<|start_of_replace|><original_word>[SEP]<replacement_word><|end_of_replace|>
```

Example:
```
I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>
```

This format is used to evaluate whether a grammar correction candidate (replacing "job." with "job") is valid or not.

## Model Output

The model returns a JSON object with:
- `label`: The predicted class (`"none"` or other label)
- `score`: Confidence score (0.0 to 1.0)

### Labels

- `"none"`: The correction is not needed or is invalid
- Other labels may indicate specific types of valid corrections

## Important Notes

### Batch Dimension

The model configuration has `max_batch_size: 8`, which means:
- Input shape must be `[batch_size, 1]` not just `[1]`
- For single inference: use shape `[1, 1]`
- For batch inference: use shape `[N, 1]` where N is the batch size

**Incorrect (will fail):**
```json
{
  "name": "TEXT",
  "shape": [1],
  "data": ["some text"]
}
```

**Correct:**
```json
{
  "name": "TEXT",
  "shape": [1, 1],
  "data": ["some text"]
}
```

### Score Differences

The scores from the Triton server may differ slightly from local pipeline predictions due to:
- Different hardware (CPU vs GPU)
- Floating point precision differences
- Model calibration settings

As long as the **label** matches and the **score is similar** (within ~5-10%), the deployment is working correctly.

## API Endpoints

### Health Check Endpoints

Check if server is alive:
```bash
curl http://localhost:8000/v2/health/live
```

Check if server is ready:
```bash
curl http://localhost:8000/v2/health/ready
```

Check if specific model is ready:
```bash
curl http://localhost:8000/v2/models/grammared_classifier/versions/1/ready
```

### Model Information

Get model metadata:
```bash
curl http://localhost:8000/v2/models/grammared_classifier
```

Get model configuration:
```bash
curl http://localhost:8000/v2/models/grammared_classifier/config
```

List all models:
```bash
curl http://localhost:8000/v2/models
```

## Troubleshooting

### Error: "unexpected shape for input 'TEXT'"

This means you forgot the batch dimension. Use shape `[1, 1]` instead of `[1]`.

### Error: "Model not ready"

The model might still be loading. Check the Docker logs:

```bash
docker logs grammared-triton --tail 50
```

Wait for the model to finish loading (usually 30-60 seconds on first start).

### Error: "Connection refused"

Make sure the Triton server container is running:

```bash
docker ps | grep grammared-triton
```

Check port mappings are correct (should be `0.0.0.0:8000-8002->8000-8002/tcp`).

### Server responds but model inference fails

Check the Triton server logs for detailed error messages:

```bash
docker logs grammared-triton --follow
```

## Comparison with Local Pipeline

### Local Pipeline Usage:
```python
from grammared_language.grammared_classifier import classifier_pipeline

pipeline = classifier_pipeline.GrammaredClassifierPipeline()
result = pipeline.predict('I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>')
print(result)
# Output: {'label': 'none', 'score': 0.9268353033472629}
```

### Triton Server Usage:
```bash
curl -X POST "http://localhost:8000/v2/models/grammared_classifier/versions/1/infer" \
    -H "Content-Type: application/json" \
    -d '{
  "inputs": [
    {
      "name": "TEXT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>"]
    }
  ]
}'
# Output: {"label": "none", "score": 0.9689850294275044}
```

Both methods should produce the same label with similar scores.

## Performance Testing

For load testing and performance benchmarking, you can use `perf_analyzer`:

```bash
docker exec -it grammared-triton perf_analyzer \
    -m grammared_classifier \
    -v \
    --shape TEXT:1,1 \
    --string-data "I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>"
```

## Next Steps

- See [API.md](../docs/API.md) for REST API integration
- See [DEPLOYMENT.md](../docs/DEPLOYMENT.md) for production deployment
- See [MODELS.md](../docs/MODELS.md) for model configuration details
