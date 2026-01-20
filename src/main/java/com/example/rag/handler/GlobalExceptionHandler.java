package com.example.rag.handler;

import com.example.rag.dto.ErrorResponse;
import com.example.rag.exception.EmbeddingException;
import com.example.rag.exception.LlmException;
import com.example.rag.exception.VectorStoreException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.time.LocalDateTime;

/**
 * Global exception handler for the RAG application
 * Handles exceptions from all controllers and converts them to appropriate HTTP responses
 * Provides centralized, consistent error handling across the entire application
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * Handles IllegalArgumentException - thrown for invalid input
     * Returns: 400 Bad Request
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        logger.error("Invalid argument: {}", ex.getMessage());
        ErrorResponse response = ErrorResponse.builder()
                .status("error")
                .message("Invalid input: " + ex.getMessage())
                .errorCode("INVALID_ARGUMENT")
                .timestamp(LocalDateTime.now())
                .build();
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
    }

    /**
     * Handles EmbeddingException - thrown when embedding generation fails
     * Returns: 503 Service Unavailable (external API failure)
     */
    @ExceptionHandler(EmbeddingException.class)
    public ResponseEntity<ErrorResponse> handleEmbeddingException(EmbeddingException ex) {
        logger.error("Embedding generation failed: {}", ex.getMessage());
        logger.info("Embedding exception details", ex);
        ErrorResponse response = ErrorResponse.builder()
                .status("error")
                .message("Failed to generate embeddings: " + ex.getMessage())
                .errorCode("EMBEDDING_FAILED")
                .timestamp(LocalDateTime.now())
                .build();
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(response);
    }

    /**
     * Handles LlmException - thrown when LLM operation fails
     * Returns: 503 Service Unavailable (external API failure)
     */
    @ExceptionHandler(LlmException.class)
    public ResponseEntity<ErrorResponse> handleLlmException(LlmException ex) {
        logger.error("LLM operation failed: {}", ex.getMessage());
        logger.info("LLM exception details", ex);
        ErrorResponse response = ErrorResponse.builder()
                .status("error")
                .message("Failed to generate answer: " + ex.getMessage())
                .errorCode("LLM_FAILED")
                .timestamp(LocalDateTime.now())
                .build();
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(response);
    }

    /**
     * Handles VectorStoreException - thrown when vector store operation fails
     * Returns: 500 Internal Server Error (internal operation failure)
     */
    @ExceptionHandler(VectorStoreException.class)
    public ResponseEntity<ErrorResponse> handleVectorStoreException(VectorStoreException ex) {
        logger.error("Vector store operation failed: {}", ex.getMessage());
        logger.info("Vector store exception details", ex);
        ErrorResponse response = ErrorResponse.builder()
                .status("error")
                .message("Vector store error: " + ex.getMessage())
                .errorCode("VECTOR_STORE_FAILED")
                .timestamp(LocalDateTime.now())
                .build();
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }

    /**
     * Handles generic exceptions - fallback for unexpected errors
     * Returns: 500 Internal Server Error (unexpected failure)
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(Exception ex) {
        logger.error("Unexpected error occurred: {}", ex.getMessage());
        logger.info("Exception details", ex);
        ErrorResponse response = ErrorResponse.builder()
                .status("error")
                .message("An unexpected error occurred: " + ex.getMessage())
                .errorCode("INTERNAL_ERROR")
                .timestamp(LocalDateTime.now())
                .build();
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }
}

