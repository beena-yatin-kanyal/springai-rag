package com.example.rag.service;

import com.example.rag.dto.RagResponse;
import com.example.rag.exception.EmbeddingException;
import com.example.rag.exception.LlmException;
import com.example.rag.exception.VectorStoreException;
import com.example.rag.repository.InMemoryVectorStore;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Spring service that implements RAG (Retrieval-Augmented Generation) functionality
 * Orchestrates the workflow of embedding generation, document retrieval, and LLM interaction
 */
@Service
public class RagService {

    private static final Logger logger = LoggerFactory.getLogger(RagService.class);
    // Sample documents loaded into the vector store during initialization
    private static final List<String> DOCUMENTS = List.of(
            "Spring AI is a framework that provides abstractions for AI integration in Spring applications. It offers ChatClient for LLM interactions and EmbeddingClient for vector embeddings.",
            "Spring Boot is an opinionated framework built on top of the Spring Framework. It simplifies the development of production-ready applications with embedded servers and auto-configuration.",
            "Large Language Models (LLMs) are neural networks trained on vast amounts of text data. They can generate human-like text, answer questions, and perform various NLP tasks."
    );
    // Client for interacting with Large Language Models (LLMs)
    private final ChatClient chatClient;
    // Model used to generate vector embeddings from text
    private final EmbeddingModel embeddingModel;
    // In-memory vector store for document storage and similarity search
    private final InMemoryVectorStore vectorStore;

    /**
     * Constructor to inject dependencies via Spring dependency injection
     */
    public RagService(ChatClient.Builder chatClientBuilder, EmbeddingModel embeddingModel, InMemoryVectorStore vectorStore) {
        this.chatClient = chatClientBuilder.build();
        this.embeddingModel = embeddingModel;
        this.vectorStore = vectorStore;
    }

    /**
     * Initializes the vector store by embedding and storing all sample documents
     * Called after bean construction via @PostConstruct
     */
    @PostConstruct
    public void init() {
        logger.info("Initializing RagService with {} documents", DOCUMENTS.size());
        int addedCount = 0;
        int skippedCount = 0;
        int errorCount = 0;

        // Iterate through each document and embed it
        for (String doc : DOCUMENTS) {
            try {
                // Validate document
                if (doc == null || doc.trim().isEmpty()) {
                    logger.warn("Skipping null or empty document");
                    errorCount++;
                    continue;
                }

                float[] embedding = generateEmbedding(doc);

                // Validate embedding
                if (embedding.length == 0) {
                    logger.warn("Skipping document with invalid embedding");
                    errorCount++;
                    continue;
                }

                // Store the document with its embedding in the vector store
                // Returns true if added, false if duplicate was skipped
                boolean added = vectorStore.add(doc, embedding);

                if (added) {
                    addedCount++;
                    logger.info("Added document to vector store: {}", doc.substring(0, Math.min(50, doc.length())) + "...");
                } else {
                    skippedCount++;
                    logger.info("Skipped duplicate document: {}", doc.substring(0, Math.min(50, doc.length())) + "...");
                }
            } catch (EmbeddingException e) {
                logger.error("Failed to embed document");
                logger.info("Embedding exception details", e);
                errorCount++;
            } catch (Exception e) {
                logger.error("Unexpected error processing document");
                logger.info("Exception details", e);
                errorCount++;
            }
        }
        logger.info("RagService initialization completed - Added: {}, Skipped duplicates: {}, Errors: {}",
                addedCount, skippedCount, errorCount);

        if (addedCount == 0) {
            logger.warn("No documents were successfully loaded into the vector store");
        }
    }

    /**
     * Main method that answers questions using RAG (Retrieval-Augmented Generation)
     * Orchestrates the complete workflow of embedding, retrieval, and LLM generation
     *
     * @param question The user's question
     * @return RagResponse containing the question, answer, and relevant documents
     * @throws IllegalArgumentException if question is null or empty
     * @throws EmbeddingException       if embedding generation fails
     * @throws VectorStoreException     if document retrieval fails
     * @throws LlmException             if LLM operation fails
     */
    public RagResponse askWithContext(String question) {
        // Validate input
        if (question == null || question.trim().isEmpty()) {
            logger.error("Question cannot be null or empty");
            throw new IllegalArgumentException("Question is required and cannot be empty");
        }

        try {
            logger.info("Processing question with context: {}", question);

            // Step 1: Generate question embedding
            float[] questionEmbedding = generateEmbedding(question);
            if (questionEmbedding.length == 0) {
                logger.error("Failed to generate valid embedding for question");
                throw new EmbeddingException("Generated embedding is invalid or empty");
            }

            // Step 2: Retrieve the 2 most relevant documents from the vector store
            List<String> relevantDocs = vectorStore.similaritySearch(questionEmbedding, 2);
            if (relevantDocs == null) {
                logger.error("Similarity search returned null");
                throw new VectorStoreException("Similarity search returned null result");
            }
            logger.info("Retrieved {} relevant documents", relevantDocs.size());

            if (relevantDocs.isEmpty()) {
                logger.warn("No relevant documents found for question: {}", question);
            }

            // Step 3: Combine relevant documents into a single context string
            String context = String.join("\n\n", relevantDocs);

            // Step 4: Build the prompt with context and question for the LLM
            String prompt = buildPrompt(context, question);
            if (prompt.isEmpty()) {
                logger.error("Failed to build prompt");
                throw new IllegalArgumentException("Prompt cannot be empty");
            }

            // Step 5: Send the prompt to the LLM and get the answer
            String answer = callLlm(prompt);
            if (answer.trim().isEmpty()) {
                logger.error("LLM returned empty answer");
                throw new LlmException("LLM returned empty response");
            }

            logger.info("Generated answer for question: {}", question);
            logger.info("Answer: {}", answer);

            // Create and return structured response
            return new RagResponse(question, answer, relevantDocs);

        } catch (IllegalArgumentException e) {
            logger.error("Invalid argument while processing question: {}", e.getMessage());
            throw e;
        } catch (EmbeddingException e) {
            logger.error("Embedding error while processing question: {}", e.getMessage());
            logger.info("Embedding exception details", e);
            throw e;
        } catch (LlmException e) {
            logger.error("LLM error while processing question: {}", e.getMessage());
            logger.info("LLM exception details", e);
            throw e;
        } catch (Exception e) {
            logger.error("Unexpected error while processing question");
            logger.info("Exception details for question: {}", question, e);
            throw new RuntimeException("Failed to process question: " + e.getMessage(), e);
        }
    }

    /**
     * Generates a vector embedding for the given text using the embedding model
     *
     * @param text The text to embed
     * @return float[] The embedding vector
     * @throws IllegalArgumentException if text is null or empty
     * @throws EmbeddingException       if embedding generation fails
     */
    private float[] generateEmbedding(String text) {
        // Validate input
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        try {
            String preview = text.substring(0, Math.min(50, text.length())) + "...";
            logger.info("Generating embedding for text: {}", preview);

            // Call embedding model
            EmbeddingResponse response = embeddingModel.embedForResponse(List.of(text));

            // Validate response
            if (response == null) {
                logger.error("Embedding model returned null response");
                throw new EmbeddingException("Embedding model returned null response");
            }

            // Validate results
            if (response.getResults() == null || response.getResults().isEmpty()) {
                logger.error("Embedding model returned empty results");
                throw new EmbeddingException("Embedding model returned no results");
            }

            // Extract embedding vector
            float[] embedding = response.getResults().get(0).getOutput();

            // Validate embedding
            if (embedding == null || embedding.length == 0) {
                logger.error("Embedding vector is null or empty");
                throw new EmbeddingException("Generated embedding is invalid");
            }

            logger.info("Embedding generated with dimension: {}", embedding.length);
            return embedding;

        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            logger.error("Failed to generate embedding: {}", e.getMessage());
            throw new EmbeddingException("Failed to generate embedding: " + e.getMessage(), e);
        }
    }

    /**
     * Constructs a prompt string that includes context and the user's question
     *
     * @param context  The relevant document context
     * @param question The user's question
     * @return String The formatted prompt for the LLM
     * @throws IllegalArgumentException if question is invalid
     */
    private String buildPrompt(String context, String question) {
        // Validate inputs
        if (context == null || context.trim().isEmpty()) {
            logger.warn("Context is empty for question: {}", question);
            context = "No relevant context available.";
        }

        if (question == null || question.trim().isEmpty()) {
            throw new IllegalArgumentException("Question cannot be null or empty");
        }

        try {
            String prompt = String.format(
                    "Use the following context to answer the question. If the answer cannot be found in the context, say so.\n\nContext:\n%s\n\nQuestion: %s\n\nAnswer:",
                    context, question
            );

            if (prompt.isEmpty()) {
                throw new IllegalArgumentException("Generated prompt is empty");
            }

            logger.info("Prompt built successfully with {} characters", prompt.length());
            return prompt;
        } catch (Exception e) {
            logger.error("Failed to build prompt: {}", e.getMessage());
            throw new IllegalArgumentException("Failed to build prompt: " + e.getMessage(), e);
        }
    }

    /**
     * Calls the LLM with the given prompt and returns the response
     *
     * @param prompt The formatted prompt for the LLM
     * @return String The LLM's response
     * @throws IllegalArgumentException if prompt is invalid
     * @throws LlmException             if LLM operation fails
     */
    private String callLlm(String prompt) {
        // Validate input
        if (prompt == null || prompt.trim().isEmpty()) {
            throw new IllegalArgumentException("Prompt cannot be null or empty");
        }

        try {
            logger.info("Calling LLM with prompt of {} characters", prompt.length());

            // Call the LLM
            String answer = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();

            // Validate response
            if (answer == null) {
                logger.error("LLM returned null response");
                throw new LlmException("LLM returned null response");
            }

            logger.info("LLM returned response of {} characters", answer.length());
            return answer;

        } catch (LlmException e) {
            throw e;
        } catch (Exception e) {
            logger.error("Failed to call LLM: {}", e.getMessage());
            throw new LlmException("Failed to call LLM: " + e.getMessage(), e);
        }
    }
}

