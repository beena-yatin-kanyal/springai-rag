package com.example.rag.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Response DTO for the /ask endpoint containing answer and relevant documents
 * Used as the standard API response for RAG queries
 */
public record RagResponse(
        @JsonProperty("question")
        String question,

        @JsonProperty("answer")
        String answer,

        @JsonProperty("relevant_documents")
        List<String> relevantDocuments
) {
}


