package com.example.rag.controller;

import com.example.rag.dto.RagResponse;
import com.example.rag.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST Controller for RAG (Retrieval-Augmented Generation) operations
 * Handles incoming HTTP requests for question answering
 */
@RestController
public class AskController {

    private static final Logger logger = LoggerFactory.getLogger(AskController.class);
    private final RagService ragService;

    /**
     * Constructor to inject RagService dependency
     */
    public AskController(RagService ragService) {
        this.ragService = ragService;
    }

    /**
     * POST endpoint to ask questions
     *
     * @param request QuestionRequest containing the question
     * @return RagResponse with answer and relevant documents
     * @throws IllegalArgumentException if request or question is invalid
     */
    @PostMapping("/ask")
    public RagResponse ask(@RequestBody QuestionRequest request) {
        // Validate request
        if (request == null) {
            logger.error("Request is null");
            throw new IllegalArgumentException("Request cannot be null");
        }

        // Validate question
        String question = request.question();
        if (question == null || question.trim().isEmpty()) {
            logger.error("Question in request is null or empty");
            throw new IllegalArgumentException("Question cannot be null or empty");
        }

        logger.info("Received request to process question");
        return ragService.askWithContext(question);
    }

    /**
     * Record class for incoming question request
     */
    // https://medium.com/@ksaquib/understanding-java-records-a-comprehensive-guide-448442d8cda9
    public record QuestionRequest(String question) {
    }
}

