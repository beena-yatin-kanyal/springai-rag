package com.example.rag.service;

import com.example.rag.dto.RagResponse;
import com.example.rag.exception.EmbeddingException;
import com.example.rag.exception.LlmException;
import com.example.rag.repository.InMemoryVectorStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RagServiceTest {

    @Mock
    private ChatClient.Builder chatClientBuilder;

    @Mock
    private ChatClient chatClient;

    @Mock
    private ChatClient.ChatClientRequestSpec requestSpec;

    @Mock
    private ChatClient.CallResponseSpec callResponseSpec;

    @Mock
    private EmbeddingModel embeddingModel;

    @Mock
    private InMemoryVectorStore vectorStore;

    private RagService ragService;

    @BeforeEach
    void setUp() {
        when(chatClientBuilder.build()).thenReturn(chatClient);
        ragService = new RagService(chatClientBuilder, embeddingModel, vectorStore);
    }

    @Test
    void shouldReturnAnswerWithContext() {
        String question = "What is Spring Boot?";
        float[] embedding = {1.0f, 0.5f, 0.3f};
        List<String> relevantDocs = List.of("Spring Boot is a framework");

        EmbeddingResponse embeddingResponse = mock(EmbeddingResponse.class);
        Embedding embeddingResult = mock(Embedding.class);
        when(embeddingModel.embedForResponse(anyList())).thenReturn(embeddingResponse);
        when(embeddingResponse.getResults()).thenReturn(List.of(embeddingResult));
        when(embeddingResult.getOutput()).thenReturn(embedding);

        when(vectorStore.similaritySearch(embedding, 2)).thenReturn(relevantDocs);

        when(chatClient.prompt()).thenReturn(requestSpec);
        when(requestSpec.user(any(String.class))).thenReturn(requestSpec);
        when(requestSpec.call()).thenReturn(callResponseSpec);
        when(callResponseSpec.content()).thenReturn("Spring Boot simplifies development");

        RagResponse response = ragService.askWithContext(question);

        assertThat(response).isNotNull();
        assertThat(response.question()).isEqualTo(question);
        assertThat(response.answer()).isEqualTo("Spring Boot simplifies development");
        assertThat(response.relevantDocuments()).isEqualTo(relevantDocs);

        verify(embeddingModel).embedForResponse(anyList());
        verify(vectorStore).similaritySearch(embedding, 2);
        verify(chatClient).prompt();
    }

    @Test
    void shouldThrowExceptionWhenQuestionIsNull() {
        assertThatThrownBy(() -> ragService.askWithContext(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Question is required");
    }

    @Test
    void shouldThrowExceptionWhenQuestionIsEmpty() {
        assertThatThrownBy(() -> ragService.askWithContext(""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Question is required");
    }

    @Test
    void shouldThrowEmbeddingExceptionWhenEmbeddingFails() {
        String question = "What is Spring Boot?";

        when(embeddingModel.embedForResponse(anyList()))
                .thenThrow(new RuntimeException("Embedding service error"));

        assertThatThrownBy(() -> ragService.askWithContext(question))
                .isInstanceOf(EmbeddingException.class);
    }

    @Test
    void shouldHandleEmptyRelevantDocuments() {
        String question = "What is Spring Boot?";
        float[] embedding = {1.0f, 0.5f, 0.3f};

        EmbeddingResponse embeddingResponse = mock(EmbeddingResponse.class);
        Embedding embeddingResult = mock(Embedding.class);
        when(embeddingModel.embedForResponse(anyList())).thenReturn(embeddingResponse);
        when(embeddingResponse.getResults()).thenReturn(List.of(embeddingResult));
        when(embeddingResult.getOutput()).thenReturn(embedding);

        when(vectorStore.similaritySearch(embedding, 2)).thenReturn(List.of());

        when(chatClient.prompt()).thenReturn(requestSpec);
        when(requestSpec.user(any(String.class))).thenReturn(requestSpec);
        when(requestSpec.call()).thenReturn(callResponseSpec);
        when(callResponseSpec.content()).thenReturn("I don't have enough context");

        RagResponse response = ragService.askWithContext(question);

        assertThat(response).isNotNull();
        assertThat(response.relevantDocuments()).isEmpty();
    }

    @Test
    void shouldThrowLlmExceptionWhenLlmFails() {
        String question = "What is Spring Boot?";
        float[] embedding = {1.0f, 0.5f, 0.3f};

        EmbeddingResponse embeddingResponse = mock(EmbeddingResponse.class);
        Embedding embeddingResult = mock(Embedding.class);
        when(embeddingModel.embedForResponse(anyList())).thenReturn(embeddingResponse);
        when(embeddingResponse.getResults()).thenReturn(List.of(embeddingResult));
        when(embeddingResult.getOutput()).thenReturn(embedding);

        when(vectorStore.similaritySearch(embedding, 2)).thenReturn(List.of("doc"));

        when(chatClient.prompt()).thenReturn(requestSpec);
        when(requestSpec.user(any(String.class))).thenReturn(requestSpec);
        when(requestSpec.call()).thenReturn(callResponseSpec);
        when(callResponseSpec.content()).thenThrow(new RuntimeException("LLM error"));

        assertThatThrownBy(() -> ragService.askWithContext(question))
                .isInstanceOf(LlmException.class);
    }
}
