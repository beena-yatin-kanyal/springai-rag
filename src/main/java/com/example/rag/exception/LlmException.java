package com.example.rag.exception;

/**
 * Exception thrown when LLM chat operation fails
 * Typically thrown when Azure OpenAI LLM API is unavailable or returns errors
 */
public class LlmException extends RuntimeException {

    /**
     * Constructor with error message
     */
    public LlmException(String message) {
        super(message);
    }

    /**
     * Constructor with error message and root cause
     */
    public LlmException(String message, Throwable cause) {
        super(message, cause);
    }
}

