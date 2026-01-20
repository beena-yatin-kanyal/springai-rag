package com.example.rag.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;

/**
 * Standard error response format for all API errors
 * Provides consistent structure for error responses across the API
 */
public record ErrorResponse(
        @JsonProperty("status")
        String status,

        @JsonProperty("message")
        String message,

        @JsonProperty("error_code")
        String errorCode,

        @JsonProperty("timestamp")
        LocalDateTime timestamp
) {
    /**
     * Builder for creating ErrorResponse instances
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class for ErrorResponse
     */
    public static class Builder {
        private String status = "error";
        private String message;
        private String errorCode;
        private LocalDateTime timestamp;

        public Builder status(String status) {
            this.status = status;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder errorCode(String errorCode) {
            this.errorCode = errorCode;
            return this;
        }

        public Builder timestamp(LocalDateTime timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public ErrorResponse build() {
            if (timestamp == null) {
                timestamp = LocalDateTime.now();
            }
            return new ErrorResponse(status, message, errorCode, timestamp);
        }
    }
}

