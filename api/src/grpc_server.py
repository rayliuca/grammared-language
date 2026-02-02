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
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent directory to path for absolute imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from grammared_language.clients.async_multi_client import AsyncMultiClient
from grammared_language.clients.gector_client import GectorClient
from grammared_language.clients.coedit_client import CoEditClient
# from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor
# from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor
from grammared_language.api.util import SimpleCacheStore
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.api.grpc_gen import ml_server_pb2, ml_server_pb2_grpc
from grammared_language.utils.config_parser import create_clients_from_config, get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model initialization - load from config file
config_path = os.getenv('GRAMMARED_LANGUAGE__MODEL_CONFIG_PATH', 'model_config.yaml')
clients = create_clients_from_config(config_path=config_path)
logger.info(f"Successfully loaded {len(clients)} client(s)")

# Multi-client for running predictions across all configured clients
correction_multi_client = AsyncMultiClient(clients=clients)

# # Initialize ERRANT Grammar Correction Extractor
# errant_extractor = ErrantGrammarCorrectionExtractor(language='en', min_length=1)

cache_size = int(os.getenv('GRAMMARED_LANGUAGE__GRPC_SERVER_CACHE_SIZE', '100000'))
analyze_cache_store = SimpleCacheStore(cache_size)
process_cache_store = SimpleCacheStore(cache_size)
match_cache_store = SimpleCacheStore(cache_size)
match_anylized_cache_store = SimpleCacheStore(cache_size)

# Health check state
service_healthy = False

def pydantic_match_to_ml_match(match, offset_adjustment: int = 0) -> ml_server_pb2.Match:
    """Convert Pydantic Match model to ml_server Match."""
    grpc_match = ml_server_pb2.Match(
        offset=match.offset + offset_adjustment,
        length=match.length,
        id=match.id or "GrammaredLanguage",
        sub_id=match.sub_id or "",
        suggestions=match.suggestions or [],
        ruleDescription=match.rule.description if match.rule else None,
        matchDescription=match.message or "",
        matchShortDescription=match.shortMessage or match.message or "",
        url=match.url or "",
        suggestedReplacements=[
            ml_server_pb2.SuggestedReplacement(
                replacement=r.replacement,
                description=r.description or "",
                suffix=r.suffix or "",
                confidence=r.confidence if r.confidence is not None else -1,
            )
            for r in (match.suggested_replacements or [])
        ],
        autoCorrect=True,
        type=ml_server_pb2.Match.MatchType.UnknownWord,  # Grammar errors are "Other" type
        contextForSureMatch=0,
        rule=ml_server_pb2.Rule(
            sourceFile=match.rule.id or "grammared_language",
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

    return grpc_match


class HealthCheckHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Docker health checks."""
    
    def do_GET(self):
        """Handle GET requests for health check."""
        if self.path == '/health' or self.path == '/healthz':
            if service_healthy:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(503)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Service not ready')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass


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
                    result = correction_multi_client.predict_with_merge(text)
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
            
            results = {i:None for i in range(len(request.sentences))}
            match_tasks = [] # (idx, text)
            for i, analyzed_sentence in enumerate(request.sentences):
                if match_cache_store.contains(analyzed_sentence):
                    results[i] = match_cache_store.get(analyzed_sentence)
                else:
                    match_tasks.append((i, analyzed_sentence))

            task_results = correction_multi_client.predict_with_merge([s for _, s in match_tasks])
            for (idx, _), result in zip(match_tasks, task_results):
                results[idx] = result
                match_cache_store.add(request.sentences[idx], result)
            
            sentence_matches = []
            for i in range(len(request.sentences)):
                enriched_result = results[i]
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
            
            results = {i:None for i in range(len(request.sentences))}
            match_tasks = [] # (idx, text)
            for i, analyzed_sentence in enumerate(request.sentences):
                if match_anylized_cache_store.contains(analyzed_sentence):
                    results[i] = match_anylized_cache_store.get(analyzed_sentence)
                else:
                    match_tasks.append((i, analyzed_sentence))

            task_results = correction_multi_client.predict_with_merge([s for _, s in match_tasks])
            for (idx, _), result in zip(match_tasks, task_results):
                results[idx] = result
                match_anylized_cache_store.add(request.sentences[idx], result)
            
            sentence_matches = []
            for i in range(len(request.sentences)):
                enriched_result = results[i]
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


def serve(host: str = "0.0.0.0", port: int = 50051, health_port: int = 50052):
    """
    Start the gRPC server with ProcessingServer and MLServer services.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        health_port: Port for HTTP health check endpoint
    """
    global service_healthy
    
    # Create gRPC server
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
    
    # Start HTTP health check server in a background thread
    def run_http_health_server():
        health_server = HTTPServer((host, health_port), HealthCheckHTTPHandler)
        logger.info(f"Starting HTTP health check server on {host}:{health_port}")
        health_server.serve_forever()
    
    health_thread = Thread(target=run_http_health_server, daemon=True)
    health_thread.start()
    
    # Mark service as healthy after successful startup
    service_healthy = True
    logger.info("Service marked as healthy")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        service_healthy = False
        server.stop(0)


if __name__ == "__main__":
    api_port = int(os.getenv('GRAMMARED_LANGUAGE__API_PORT', '50051'))
    api_host = os.getenv('GRAMMARED_LANGUAGE__API_HOST', '0.0.0.0')
    serve(host=api_host, port=api_port)
