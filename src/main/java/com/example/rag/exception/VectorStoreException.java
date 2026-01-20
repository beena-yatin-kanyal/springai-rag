package com.example.rag.exception;

/**
 * Exception thrown when vector store operation fails
 * Typically thrown during document storage, retrieval, or similarity search operations
 */
public class VectorStoreException extends RuntimeException {

    /**
     * Constructor with error message
     */
    public VectorStoreException(String message) {
        super(message);
    }

    /**
     * Constructor with error message and root cause
     */
    public VectorStoreException(String message, Throwable cause) {
        super(message, cause);
    }
}

