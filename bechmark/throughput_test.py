import time
import requests
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Settings
DATASET_NAME = "jhu-clsp/jfleg"
SPLIT = "test"
LANGUAGETOOL_SERVER = "http://localhost:8081/v2/check"  # Change if needed
LANGUAGETOOL_SERVER = "https://grammared-language-demo.rayliu.ca/v2/check"  # Change if needed

TEXT_FIELD = "sentence"
CONCURRENT_WORKERS = 4  # Tune for your machine/network/server capacity
MAX_SAMPLES = 100  # Set to an integer to limit number of samples

def check_with_languagetool(text):
    """Send text to LanguageTool server and return response."""
    data = {
        "language": "en-US",
        "text": text,
    }
    response = requests.post(LANGUAGETOOL_SERVER, data=data)
    response.raise_for_status()
    # print(response.json())
    return response.json()

def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    texts = dataset[TEXT_FIELD][:MAX_SAMPLES]
    num_samples = len(texts)

    print(f"Loaded {num_samples} examples from {SPLIT} split.")
    print(f"Benchmarking with {CONCURRENT_WORKERS} concurrent requests...")

    successes, failures = 0, 0
    start_time = time.time()
    pbar = tqdm(total=num_samples, desc="Processing samples")
    # Thread pool for concurrency
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        # Start all requests
        future_to_idx = {executor.submit(check_with_languagetool, text): idx for idx, text in enumerate(texts)}
        completed = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                future.result()
                successes += 1
            except Exception as e:
                print(f"Error on sample {idx}: {repr(e)}")
                failures += 1
            completed += 1
            elapsed = time.time() - start_time
            throughput = completed / elapsed
            pbar.set_description(f"Success: {successes}, Fail: {failures}, {throughput:.2f} samples/sec")
            pbar.update(1)
            if completed % 100 == 0 or completed == num_samples:
                print(f"Processed {completed}/{num_samples} samples in {elapsed:.2f}s ({throughput:.2f} samples/sec)")
    
    pbar.close()
    total_time = time.time() - start_time
    print("\n=== Benchmark Results ===")
    print(f"Processed {successes} samples successfully, {failures} failures.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {successes/total_time:.2f} samples/sec")

if __name__ == "__main__":
    main()


"""
gotutiyan/gector-deberta-large-5k
GPU:
CONCURRENT_WORKERS = 8
=== Benchmark Results ===
Processed 500 samples successfully, 0 failures.
Total time: 15.82 seconds
Throughput: 31.60 samples/sec


CPU:
CONCURRENT_WORKERS = 2
=== Benchmark Results ===
Processed 100 samples successfully, 0 failures.
Total time: 34.66 seconds
Throughput: 2.88 samples/sec

CPU:
CONCURRENT_WORKERS = 8

=== Benchmark Results ===
Processed 100 samples successfully, 0 failures.
Total time: 32.67 seconds
Throughput: 3.06 samples/sec



grammarly/coedit-large + gotutiyan/gector-deberta-large-5k

5090 GPU:
CONCURRENT_WORKERS = 8
=== Benchmark Results ===
Processed 100 samples successfully, 0 failures.
Total time: 20.80 seconds
Throughput: 4.81 samples/sec

9900x CPU:
CONCURRENT_WORKERS = 8
=== Benchmark Results ===
Processed 100 samples successfully, 0 failures.
Total time: 228.75 seconds
Throughput: 0.44 samples/sec

Orcal ARM CPU (demo server):
=== Benchmark Results ===
Processed 100 samples successfully, 0 failures.
Total time: 197.29 seconds
Throughput: 0.51 samples/sec
"""