"""
gRPC server implementation compatible with LanguageTool ML Server protocol (ml_server.proto).

Implements ProcessingServer and MLServer services.
"""

import logging
from concurrent import futures
import grpc
import time
import sys
import os

# Add parent directory to path for absolute imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from grammared_language.clients.async_multi_client import AsyncMultiClient
from grammared_language.clients.gector_client import GectorClient
from grammared_language.clients.coedit_client import CoEditClient
# from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor
from grammared_language.api.util import SimpleCacheStore
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.api.grpc_gen import ml_server_pb2, ml_server_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model initialization
model_id = "gotutiyan/gector-deberta-large-5k"
gector_client = GectorClient(
    model_id=model_id,
    triton_model_name="gector_deberta_large",
    verb_dict_path='data/verb-form-vocab.txt',
    keep_confidence=0,
    min_error_prob=0,
    n_iteration=5,
    batch_size=2
)

coedit_client = CoEditClient(
    model_name="coedit_large",
    triton_host="localhost",
    triton_port=8001
)

# Multi-client for running predictions across all configured clients
correction_multi_client = AsyncMultiClient(
    clients=[
        gector_client,
        coedit_client,
    ]
)

# Initialize ERRANT Grammar Correction Extractor
errant_extractor = ErrantGrammarCorrectionExtractor(language='en', min_length=1)

analyze_cache_store = SimpleCacheStore()
process_cache_store = SimpleCacheStore()


def enrich_matches_with_errant(sentence: str, corrected: str) -> list:
    """
    Extract matches with error types using ERRANT.
    
    Uses ErrantGrammarCorrectionExtractor to analyze differences between
    original and corrected text, providing detailed error types.
    
    Args:
        sentence: The original sentence
        corrected: The corrected sentence
        
    Returns:
        List of Match objects with error type information from ERRANT
    """
    if not corrected or sentence == corrected:
        return []
    
    try:
        # Use ERRANT to extract matches with error types
        errant_matches = errant_extractor.extract_replacements(sentence, corrected)
        return errant_matches
    except Exception as e:
        logger.warning(f"Error during ERRANT extraction: {str(e)}, returning empty matches")
        return []


def predict_enriched_result(text: str) -> LanguageToolRemoteResult:
    """
    Predict grammar corrections using all configured clients, then use ERRANT
    to extract matches with detailed error type information.
    """
    # Get corrected text from all clients and select best
    predictions = correction_multi_client.predict(text)
    
    # Use the first non-empty corrected text (could be improved with voting)
    corrected = None
    for pred in predictions:
        if pred.matches:
            # Try to reconstruct corrected text from first client
            # For now, just use ERRANT with original match-based approach
            break
    
    # If we have predictions with matches, try to get corrected text
    # Otherwise use ERRANT directly with the original merged approach
    merged_result = correction_multi_client.predict_with_merge(text)
    
    # For now, use the basic merged matches since we need corrected text
    # In a full implementation, clients should return corrected text
    enriched_matches = merged_result.matches
    
    print(f"Enriched matches: {enriched_matches}")
    merged_result.matches = enriched_matches
    return merged_result


def pydantic_match_to_ml_match(match, offset_adjustment: int = 0) -> ml_server_pb2.Match:
    """Convert Pydantic Match model to ml_server Match."""
    grpc_match = ml_server_pb2.Match(
        offset=match.offset + offset_adjustment,
        length=match.length,
        id="grammared_language",
        sub_id="",
        suggestions=match.suggestions or [],
        ruleDescription=match.rule.description if match.rule else None,
        matchDescription=match.message or "",
        matchShortDescription=match.shortMessage or match.message or "",
        url="",
        suggestedReplacements=[
            ml_server_pb2.SuggestedReplacement(
                replacement=r.replacement,
                description="",
                suffix="",
                confidence=0.8
            )
            for r in (match.suggested_replacements or [])
        ],
        autoCorrect=True,
        type=ml_server_pb2.Match.MatchType.UnknownWord,  # Grammar errors are "Other" type
        contextForSureMatch=0,
        rule=ml_server_pb2.Rule(
            sourceFile="grammared_language",
            issueType=match.rule.issueType or "style",
            tempOff=False,
            category=ml_server_pb2.RuleCategory(
                id=match.rule.id or "grammared_language",
                name=match.rule.description or "Style Suggestion"
            ) if match.rule.category else None,
            isPremium=False,
            tags=[]
        ) if match.rule else None
    )
    print(f"Converted gRPC match: {grpc_match}")
    print(f"Match Type: {grpc_match.type}")
    return grpc_match

class ProcessingServerServicer(ml_server_pb2_grpc.ProcessingServerServicer):
    """gRPC servicer for LanguageTool ProcessingServer."""

    def Analyze(self, request: ml_server_pb2.AnalyzeRequest, context: grpc.ServicerContext) -> ml_server_pb2.AnalyzeResponse:
        """
        Analyze text and return analyzed sentences (tokenization, POS tagging, etc).
        
        Args:
            request: AnalyzeRequest containing text and processing options
            context: gRPC context
            
        Returns:
            AnalyzeResponse with analyzed sentences
        """
        try:
            logger.info(f"Analyze request: language={request.options.language}, text_len={len(request.text)}")
            
            # Simple tokenization - just split by spaces for now
            # In production, use a proper NLP library
            text = request.text
            if not text:
                return ml_server_pb2.AnalyzeResponse(sentences=[])

            if analyze_cache_store.contains(text):
                analyzed_sentences = analyze_cache_store.get(text)
                return ml_server_pb2.AnalyzeResponse(sentences=analyzed_sentences)
            
            sentences = text.split('. ')
            analyzed_sentences = []
            
            for sentence in sentences:
                tokens = sentence.split()
                token_readings = []
                pos = 0
                
                for token in tokens:
                    token_reading = ml_server_pb2.AnalyzedTokenReadings(
                        readings=[
                            ml_server_pb2.AnalyzedToken(
                                token=token,
                                posTag="NN",  # Default POS tag
                                lemma=token.lower()
                            )
                        ],
                        chunkTags=[],
                        startPos=pos
                    )
                    token_readings.append(token_reading)
                    pos += len(token) + 1
                
                analyzed_sentences.append(
                    ml_server_pb2.AnalyzedSentence(
                        text=sentence,
                        tokens=token_readings
                    )
                )
            
            analyze_cache_store.add(text, analyzed_sentences)
            return ml_server_pb2.AnalyzeResponse(sentences=analyzed_sentences)
            
        except Exception as e:
            logger.error(f"Error in Analyze RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise

    def Process(self, request: ml_server_pb2.ProcessRequest, context: grpc.ServicerContext) -> ml_server_pb2.ProcessResponse:
        """
        Process analyzed sentences and return grammar matches.
        
        Args:
            request: ProcessRequest containing analyzed sentences and options
            context: gRPC context
            
        Returns:
            ProcessResponse with grammar error matches
        """
        try:
            logger.info(f"Process request: {len(request.sentences)} sentences, language={request.options.language}")
            
            raw_matches = []
            matches_by_sentence = []
            
            for sentence in request.sentences:
                text = sentence.text
                if process_cache_store.contains(text):
                    result = process_cache_store.get(text)
                else:
                    # Predict using both models, merge and enrich, then cache
                    result = predict_enriched_result(text)
                    process_cache_store.add(text, result)

                # Convert matches to ml_server format
                sentence_matches = ml_server_pb2.MatchList(
                    matches=[pydantic_match_to_ml_match(m) for m in result.matches]
                )
                matches_by_sentence.append(sentence_matches)
                raw_matches.extend(sentence_matches.matches)
            
            return ml_server_pb2.ProcessResponse(
                rawMatches=raw_matches,
                matches=raw_matches
            )
            
        except Exception as e:
            logger.error(f"Error in Process RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise


class MLServerServicer(ml_server_pb2_grpc.MLServerServicer):
    """gRPC servicer for LanguageTool MLServer."""

    def Match(self, request: ml_server_pb2.MatchRequest, context: grpc.ServicerContext) -> ml_server_pb2.MatchResponse:
        """
        Match grammar errors in sentences.
        
        Args:
            request: MatchRequest containing sentences to check
            context: gRPC context
            
        Returns:
            MatchResponse with matches for each sentence
        """
        try:
            logger.info(f"Match request: {len(request.sentences)} sentences")
            
            sentence_matches = []
            for sentence in request.sentences:
                enriched_result = predict_enriched_result(sentence)
                matches = ml_server_pb2.MatchList(
                    matches=[pydantic_match_to_ml_match(m) for m in enriched_result.matches]
                )
                sentence_matches.append(matches)
            
            return ml_server_pb2.MatchResponse(sentenceMatches=sentence_matches)
            
        except Exception as e:
            logger.error(f"Error in Match RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise

    def MatchAnalyzed(self, request: ml_server_pb2.AnalyzedMatchRequest, context: grpc.ServicerContext) -> ml_server_pb2.MatchResponse:
        """
        Match grammar errors in pre-analyzed sentences.
        
        Args:
            request: AnalyzedMatchRequest containing pre-analyzed sentences
            context: gRPC context
            
        Returns:
            MatchResponse with matches
        """
        try:
            logger.info(f"MatchAnalyzed request: {len(request.sentences)} analyzed sentences")
            
            sentence_matches = []
            for analyzed_sentence in request.sentences:
                enriched_result = predict_enriched_result(analyzed_sentence.text)
                matches = ml_server_pb2.MatchList(
                    matches=[pydantic_match_to_ml_match(m) for m in enriched_result.matches]
                )
                sentence_matches.append(matches)
            
            return ml_server_pb2.MatchResponse(sentenceMatches=sentence_matches)
            
        except Exception as e:
            logger.error(f"Error in MatchAnalyzed RPC: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise


def serve(host: str = "0.0.0.0", port: int = 50051):
    """
    Start the gRPC server with ProcessingServer and MLServer services.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
    
    # Add servicers
    ml_server_pb2_grpc.add_ProcessingServerServicer_to_server(
        ProcessingServerServicer(), server
    )
    ml_server_pb2_grpc.add_MLServerServicer_to_server(
        MLServerServicer(), server
    )
    
    # Bind to port
    server.add_insecure_port(f"{host}:{port}")
    
    logger.info(f"Starting gRPC server on {host}:{port}")
    logger.info("Services:")
    logger.info("  - lt_ml_server.ProcessingServer (Analyze, Process)")
    logger.info("  - lt_ml_server.MLServer (Match, MatchAnalyzed)")
    server.start()
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        server.stop(0)


if __name__ == "__main__":
    serve()
