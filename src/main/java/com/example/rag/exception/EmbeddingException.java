package com.example.rag.exception;

/**
 * Exception thrown when embedding generation fails
 * Typically thrown when Azure OpenAI embedding API is unavailable or returns invalid data
 */
public class EmbeddingException extends RuntimeException {

    /**
     * Constructor with error message
     */
    public EmbeddingException(String message) {
        super(message);
    }

    /**
     * Constructor with error message and root cause
     */
    public EmbeddingException(String message, Throwable cause) {
        super(message, cause);
    }
}

