package com.example.rag.repository;

import com.example.rag.exception.VectorStoreException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * In-memory vector store for document embeddings
 * Provides storage and retrieval of documents with their vector representations
 * Uses cosine similarity for semantic search
 */
@Component
public class InMemoryVectorStore {

    private static final Logger logger = LoggerFactory.getLogger(InMemoryVectorStore.class);

    // In-memory storage for document embeddings
    private final List<DocumentEmbedding> store = new ArrayList<>();

    /**
     * Adds a new document with its embedding vector to the store
     * Automatically skips if a duplicate document already exists
     *
     * @param content   The document content text
     * @param embedding The embedding vector for the document
     * @return true if document was added, false if it was a duplicate
     * @throws VectorStoreException if content or embedding is invalid
     */
    public boolean add(String content, float[] embedding) {
        // Validate inputs
        if (content == null || content.trim().isEmpty()) {
            logger.error("Cannot add document with null or empty content");
            throw new VectorStoreException("Document content cannot be null or empty");
        }

        if (embedding == null || embedding.length == 0) {
            logger.error("Cannot add document with null or empty embedding");
            throw new VectorStoreException("Document embedding cannot be null or empty");
        }

        try {
            // Check if document with same content already exists
            boolean isDuplicate = store.stream()
                    .anyMatch(doc -> doc.content.equals(content));

            // Only add if it's not a duplicate
            if (!isDuplicate) {
                store.add(new DocumentEmbedding(content, embedding));
                logger.info("Document added to store. Current store size: {}", store.size());
                return true;  // Successfully added
            }

            logger.info("Duplicate document detected, skipping");
            return false;  // Document was duplicate, skipped

        } catch (VectorStoreException e) {
            throw e;
        } catch (Exception e) {
            logger.error("Error adding document to store: {}", e.getMessage());
            throw new VectorStoreException("Error adding document to store: " + e.getMessage(), e);
        }
    }

    /**
     * Searches for top-K documents most similar to the query embedding using cosine similarity
     *
     * @param queryEmbedding The query vector to search for
     * @param topK           The number of most similar documents to return
     * @return List of document contents sorted by similarity (highest first)
     * @throws VectorStoreException if inputs are invalid or search fails
     */
    public List<String> similaritySearch(float[] queryEmbedding, int topK) {
        // Validate inputs
        if (queryEmbedding == null || queryEmbedding.length == 0) {
            logger.error("Query embedding cannot be null or empty");
            throw new VectorStoreException("Query embedding cannot be null or empty");
        }

        if (topK <= 0) {
            logger.error("topK must be greater than 0, got: {}", topK);
            throw new VectorStoreException("topK must be greater than 0");
        }

        try {
            // Check if store is empty
            if (store.isEmpty()) {
                logger.warn("Vector store is empty, returning empty results");
                return new ArrayList<>();
            }

            logger.info("Performing similarity search on {} documents with topK={}", store.size(), topK);

            // Perform similarity search
            List<String> results = store.stream()
                    // Map each document to its similarity score with the query
                    .map(doc -> new SimilarityScore(doc.content, cosineSimilarity(queryEmbedding, doc.embedding)))
                    // Sort documents by similarity score in descending order (highest first)
                    .sorted(Comparator.comparingDouble(SimilarityScore::score).reversed())
                    // Keep only the top K most similar documents
                    .limit(topK)
                    // Extract the document content from the similarity score objects
                    .map(SimilarityScore::content)
                    .collect(Collectors.toList());

            logger.info("Similarity search returned {} results", results.size());
            return results;

        } catch (VectorStoreException e) {
            throw e;
        } catch (Exception e) {
            logger.error("Error performing similarity search: {}", e.getMessage());
            throw new VectorStoreException("Error performing similarity search: " + e.getMessage(), e);
        }
    }

    /**
     * Calculates cosine similarity between two vectors
     * Formula: (A·B) / (||A|| × ||B||)
     * Range: -1 to 1 (1 = identical direction, 0 = orthogonal, -1 = opposite direction)
     *
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity score
     * @throws VectorStoreException if vectors are invalid or calculation fails
     */
    private double cosineSimilarity(float[] a, float[] b) {
        // Validate inputs
        if (a == null || b == null || a.length == 0 || b.length == 0) {
            logger.error("Cannot calculate cosine similarity with null or empty vectors");
            throw new VectorStoreException("Vectors cannot be null or empty");
        }

        if (a.length != b.length) {
            logger.error("Vector dimensions don't match: {} vs {}", a.length, b.length);
            throw new VectorStoreException("Vector dimensions must match");
        }

        try {
            // Initialize accumulators for dot product and vector magnitudes
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;

            // Compute dot product and squared magnitudes in a single pass
            for (int i = 0; i < a.length; i++) {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            // Check for zero vectors
            if (normA == 0.0 || normB == 0.0) {
                logger.warn("One or both vectors have zero magnitude");
                return 0.0;
            }

            // Return cosine similarity: (a · b) / (||a|| × ||b||)
            double similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

            // Validate result
            if (Double.isNaN(similarity) || Double.isInfinite(similarity)) {
                logger.error("Cosine similarity calculation resulted in NaN or Infinity");
                throw new VectorStoreException("Invalid cosine similarity result");
            }

            return similarity;
        } catch (VectorStoreException e) {
            throw e;
        } catch (Exception e) {
            logger.error("Error calculating cosine similarity: {}", e.getMessage());
            throw new VectorStoreException("Error calculating cosine similarity: " + e.getMessage(), e);
        }
    }

    /**
     * Record to store document content paired with its embedding vector
     */
    private record DocumentEmbedding(String content, float[] embedding) {
    }

    /**
     * Record to store document content paired with its similarity score
     */
    private record SimilarityScore(String content, double score) {
    }
}

