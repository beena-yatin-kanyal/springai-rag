package com.example.rag.controller;

import com.example.rag.dto.RagResponse;
import com.example.rag.exception.EmbeddingException;
import com.example.rag.exception.LlmException;
import com.example.rag.service.RagService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(AskController.class)
class AskControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private RagService ragService;

    @Test
    void shouldReturnAnswerWhenQuestionIsValid() throws Exception {
        String question = "What is Spring Boot?";
        RagResponse expectedResponse = new RagResponse(
                question,
                "Spring Boot is a framework",
                List.of("Spring Boot doc")
        );

        when(ragService.askWithContext(question)).thenReturn(expectedResponse);

        AskController.QuestionRequest request = new AskController.QuestionRequest(question);

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.question").value(question))
                .andExpect(jsonPath("$.answer").value("Spring Boot is a framework"))
                .andExpect(jsonPath("$.relevant_documents[0]").value("Spring Boot doc"));
    }

    @Test
    void shouldReturn400WhenQuestionIsEmpty() throws Exception {
        AskController.QuestionRequest request = new AskController.QuestionRequest("");

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void shouldReturn400WhenQuestionIsNull() throws Exception {
        AskController.QuestionRequest request = new AskController.QuestionRequest(null);

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void shouldReturn503WhenEmbeddingServiceFails() throws Exception {
        String question = "What is Spring Boot?";
        when(ragService.askWithContext(any())).thenThrow(new EmbeddingException("Embedding failed"));

        AskController.QuestionRequest request = new AskController.QuestionRequest(question);

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error_code").value("EMBEDDING_ERROR"));
    }

    @Test
    void shouldReturn503WhenLlmServiceFails() throws Exception {
        String question = "What is Spring Boot?";
        when(ragService.askWithContext(any())).thenThrow(new LlmException("LLM failed"));

        AskController.QuestionRequest request = new AskController.QuestionRequest(question);

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error_code").value("LLM_ERROR"));
    }

    @Test
    void shouldHandleComplexQuestion() throws Exception {
        String question = "How does dependency injection work in Spring Framework with multiple beans?";
        RagResponse expectedResponse = new RagResponse(
                question,
                "Dependency injection in Spring works by...",
                List.of("Doc 1", "Doc 2")
        );

        when(ragService.askWithContext(question)).thenReturn(expectedResponse);

        AskController.QuestionRequest request = new AskController.QuestionRequest(question);

        mockMvc.perform(post("/ask")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.question").value(question))
                .andExpect(jsonPath("$.relevant_documents").isArray())
                .andExpect(jsonPath("$.relevant_documents.length()").value(2));
    }
}
