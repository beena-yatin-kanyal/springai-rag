package com.example.rag.repository;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class InMemoryVectorStoreTest {

    private InMemoryVectorStore vectorStore;

    @BeforeEach
    void setUp() {
        vectorStore = new InMemoryVectorStore();
    }

    @Test
    void shouldAddDocumentSuccessfully() {
        String content = "Spring Boot is a framework";
        float[] embedding = {1.0f, 0.5f, 0.3f};

        vectorStore.add(content, embedding);

        List<String> results = vectorStore.similaritySearch(embedding, 1);
        assertThat(results).hasSize(1);
        assertThat(results.get(0)).isEqualTo(content);
    }

    @Test
    void shouldReturnEmptyListWhenStoreIsEmpty() {
        float[] queryEmbedding = {1.0f, 0.5f, 0.3f};

        List<String> results = vectorStore.similaritySearch(queryEmbedding, 5);

        assertThat(results).isEmpty();
    }

    @Test
    void shouldReturnTopKSimilarDocuments() {
        vectorStore.add("Document 1", new float[]{1.0f, 0.0f, 0.0f});
        vectorStore.add("Document 2", new float[]{0.9f, 0.1f, 0.0f});
        vectorStore.add("Document 3", new float[]{0.0f, 1.0f, 0.0f});

        float[] queryEmbedding = {1.0f, 0.0f, 0.0f};
        List<String> results = vectorStore.similaritySearch(queryEmbedding, 2);

        assertThat(results).hasSize(2);
        assertThat(results.get(0)).isEqualTo("Document 1");
        assertThat(results.get(1)).isEqualTo("Document 2");
    }

    @Test
    void shouldNotAddDuplicateDocuments() {
        String content = "Duplicate content";
        float[] embedding1 = {1.0f, 0.5f, 0.3f};
        float[] embedding2 = {0.8f, 0.6f, 0.4f};

        vectorStore.add(content, embedding1);
        vectorStore.add(content, embedding2);

        List<String> results = vectorStore.similaritySearch(embedding1, 10);
        assertThat(results).hasSize(1);
    }

    @Test
    void shouldHandleTopKGreaterThanStoreSize() {
        vectorStore.add("Document 1", new float[]{1.0f, 0.0f, 0.0f});
        vectorStore.add("Document 2", new float[]{0.9f, 0.1f, 0.0f});

        float[] queryEmbedding = {1.0f, 0.0f, 0.0f};
        List<String> results = vectorStore.similaritySearch(queryEmbedding, 10);

        assertThat(results).hasSize(2);
    }

    @Test
    void shouldCalculateCosineSimilarityCorrectly() {
        float[] embedding1 = {1.0f, 0.0f, 0.0f};
        float[] embedding2 = {1.0f, 0.0f, 0.0f};

        vectorStore.add("Identical", embedding1);

        List<String> results = vectorStore.similaritySearch(embedding2, 1);
        assertThat(results.get(0)).isEqualTo("Identical");
    }
}
