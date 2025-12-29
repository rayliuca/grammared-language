
import time
from contextlib import asynccontextmanager
from transformers import AutoTokenizer
from gector import predict, load_verb_dict
from gector import GECToRTriton
from .util import GrammarCorrectionExtractor, SimpleCacheStore
from .output_models import LanguageToolRemoteResult

# FastAPI imports
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading

# Global resources
triton_model = None
tokenizer = None
encode = None
decode = None
cache_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global triton_model, tokenizer, encode, decode, cache_store, grammar_correction_extractor
    print(f"[STARTUP] Starting initialization at {time.time() * 1000}")

    cache_store = SimpleCacheStore()
    grammar_correction_extractor = GrammarCorrectionExtractor()

    model_id = "gotutiyan/gector-bert-base-cased-5k"
    print(f"[STARTUP] Loading model at {time.time() * 1000}")
    t0 = time.time()
    triton_model = GECToRTriton.from_pretrained(model_id, model_name="gector_bert")
    print(f"[STARTUP] Model loaded in {(time.time() - t0) * 1000:.0f}ms")

    print(f"[STARTUP] Loading tokenizer at {time.time() * 1000}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"[STARTUP] Tokenizer loaded in {(time.time() - t0) * 1000:.0f}ms")

    print(f"[STARTUP] Loading verb dict at {time.time() * 1000}")
    t0 = time.time()
    encode, decode = load_verb_dict('data/verb-form-vocab.txt')
    print(f"[STARTUP] Verb dict loaded in {(time.time() - t0) * 1000:.0f}ms")
    print(f"[STARTUP] all loaded at {time.time() * 1000}")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Cleaning up resources")

def pred_gector(src: str) -> LanguageToolRemoteResult:
    """
    Perform grammar error correction using GECToR model.
    Args:
        src: Source sentence (string)
    Returns:
        LanguageToolRemoteResult
    """
    corrected = predict(
        triton_model, tokenizer, [src],
        encode, decode,
        keep_confidence=0,
        min_error_prob=0,
        n_iteration=5,
        batch_size=2,
    )
    print(corrected)
    matches = grammar_correction_extractor.extract_replacements(src, corrected[0])
    return LanguageToolRemoteResult(
        language="English",
        languageCode="en-US",
        matches=matches
    )

# FastAPI app
app = FastAPI(lifespan=lifespan)

class CheckRequest(BaseModel):
    language: str
    text: str
    
@app.post("/v2/check")
async def check(request: CheckRequest):
    """
    LanguageTool-compatible endpoint for grammar checking.
    """
    print(f"check @ {time.time() * 1000}")

    text = request.text
    if cache_store.contains(text):
        result = cache_store.get(text)
    else:
        result = pred_gector(request.text)
        cache_store.add(text, result)
    # Convert result to dict for JSON serialization
    return JSONResponse(content=result.model_dump(exclude_none=True))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.src.server:app", host="0.0.0.0", port=8005, reload=True)


