# RAG Application - Interview Questions & Answers

## Architecture & Design

**Q1: What design pattern does this application follow?**

Layered architecture with 3 layers: Controller (AskController), Service (RagService), and Repository (InMemoryVectorStore). Uses dependency injection via Spring Boot.

---

**Q2: What is RAG and how does this application implement it?**

RAG = Retrieval-Augmented Generation. The app:
1. Embeds the user's question into a vector
2. Retrieves top 2 similar documents using cosine similarity
3. Augments the LLM prompt with retrieved context
4. Generates answer based on context

---

**Q3: Why use an in-memory vector store instead of a database?**

Simple proof-of-concept. In production, you'd use persistent storage like PostgreSQL with pgvector, Pinecone, or Weaviate for scalability and persistence across restarts.

---

**Q4: What happens when the application restarts?**

All documents are lost since InMemoryVectorStore uses a `List` in memory. The `@PostConstruct` method reloads sample documents on startup.

---

## Spring Boot & Java

**Q5: What does `@PostConstruct` do in RagService?**

Runs the `init()` method after dependency injection completes. Used here to preload sample documents into the vector store when the application starts.

---

**Q6: Why use Java records for QuestionRequest, RagResponse, and ErrorResponse?**

Records provide immutable data carriers with automatic constructor, getters, equals(), hashCode(), and toString(). Perfect for DTOs and responses.

---

**Q7: What is `@RestControllerAdvice` and how is it used here?**

Centralized exception handler that intercepts exceptions from all controllers. Maps exceptions to HTTP status codes and returns structured ErrorResponse objects.

---

**Q8: How does Spring Boot know to inject ChatClient and EmbeddingModel?**

Spring AI auto-configuration creates these beans based on properties in `application.properties`. Constructor injection in RagService pulls them in.

---

## Vector Operations

**Q9: How does similarity search work?**

1. Computes cosine similarity between query embedding and all stored document embeddings
2. Sorts results by similarity score (descending)
3. Returns top K documents (K=2 in this case)

---

**Q10: What is cosine similarity and why use it?**

Measures angle between two vectors. Formula: `(A · B) / (||A|| * ||B||)`. Ranges from -1 to 1. Used because it measures semantic similarity regardless of vector magnitude.

---

**Q11: What are the dimensions of the embedding vectors?**

The code uses `float[]` but doesn't specify. Azure OpenAI embeddings are typically 1536 dimensions.

---

**Q12: How does the code prevent duplicate documents?**

In `InMemoryVectorStore.add()`, it checks if any existing document has the same content using `documents.stream().anyMatch()`. Logs a warning and skips if found.

---

## API & Request Flow

**Q13: Walk through what happens when POST /ask is called.**

1. AskController validates request
2. Calls RagService.askWithContext()
3. RagService generates embedding for question
4. Retrieves top 2 similar documents from vector store
5. Builds prompt: "Context: {docs}\n\nQuestion: {question}\n\nAnswer:"
6. Calls LLM via ChatClient
7. Returns RagResponse with question, answer, and relevant docs

---

**Q14: What HTTP status codes can this API return?**

- 200: Success
- 400: Invalid request (empty question)
- 500: Vector store error
- 503: Embedding or LLM service unavailable

---

**Q15: Why return relevant_documents in the response?**

Transparency and traceability. Users can see which sources the answer is based on, improving trust and allowing verification.

---

## Error Handling

**Q16: What happens if Azure OpenAI is down?**

Either `EmbeddingException` or `LlmException` is thrown, caught by GlobalExceptionHandler, and returns 503 Service Unavailable with error details.

---

**Q17: Why use custom exceptions instead of generic RuntimeException?**

Specific exceptions allow fine-grained error handling, better logging, and different HTTP status codes. Makes debugging easier.

---

**Q18: What validations are performed on the question?**

Checks if request is null, if question field exists, and if question is blank. Throws `IllegalArgumentException` for invalid input (returns 400).

---

## Configuration & Dependencies

**Q19: What environment variables are required?**

- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_CHAT_DEPLOYMENT
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT

---

**Q20: What is Spring AI and why use it?**

Framework that abstracts AI model interactions. Provides ChatClient and EmbeddingModel interfaces, making it easy to swap providers (Azure, OpenAI, AWS Bedrock, etc.) without changing code.

---

## Code Quality & Best Practices

**Q21: Why inject dependencies via constructor instead of @Autowired fields?**

Constructor injection enforces required dependencies, makes testing easier (can pass mocks), and supports immutability. Recommended by Spring.

---

**Q22: Why use SLF4J for logging?**

Logging facade that works with any logging implementation (Logback, Log4j, etc.). Spring Boot uses Logback by default.

---

**Q23: Is this code thread-safe?**

Partially. `InMemoryVectorStore.documents` is a `CopyOnWriteArrayList`, which is thread-safe for reads and writes. RagService methods are stateless, so safe. However, no synchronization on similarity search.

---

## Improvements & Scaling

**Q24: How would you improve this application for production?**

- Replace in-memory store with persistent vector database
- Add caching for frequent questions
- Implement rate limiting
- Add authentication/authorization
- Use connection pooling for Azure OpenAI
- Add metrics and monitoring
- Implement document chunking for large texts
- Add streaming responses

---

**Q25: How would you handle very large document collections?**

- Use persistent vector database with indexing (HNSW, IVF)
- Implement pagination for results
- Add metadata filtering to narrow search space
- Use approximate nearest neighbor (ANN) algorithms
- Consider distributed vector stores

---

**Q26: How would you add support for document upload?**

Add `POST /documents` endpoint that:
1. Accepts document content
2. Generates embedding
3. Stores in vector store
4. Returns document ID

---

**Q27: What if two questions have identical embeddings?**

They'll retrieve the same documents. The LLM might give different answers based on subtle prompt differences, but context will be identical.

---

**Q28: Why is topK hardcoded to 2?**

Simplicity for demo. In production, make it configurable via application.properties or request parameter based on use case.

---

## Testing

**Q29: How would you test the similarity search?**

Unit test:
- Add documents with known embeddings
- Query with specific embedding
- Assert correct documents returned in correct order
- Test edge cases (empty store, topK > size)

---

**Q30: How would you mock Azure OpenAI for testing?**

Create mock implementations of `EmbeddingModel` and `ChatClient` interfaces, inject them in tests, and verify interactions without actual API calls.

---

## Performance

**Q31: What is the time complexity of similarity search?**

O(n*m) where n = number of documents, m = embedding dimensions. Linear scan through all documents. Not efficient for large datasets.

---

**Q32: How could you optimize similarity search?**

- Use approximate nearest neighbor (ANN) algorithms like HNSW or IVF
- Pre-filter documents with metadata
- Use vector databases with built-in indexing
- Implement caching for common queries

---

**Q33: What is the memory footprint of storing 1000 documents?**

Assuming 1536-dim embeddings (4 bytes/float): 1000 * 1536 * 4 = ~6MB for vectors alone, plus document text content.

---

## Spring Boot Specifics

**Q34: What does `@SpringBootApplication` do?**

Combines `@Configuration`, `@EnableAutoConfiguration`, and `@ComponentScan`. Enables Spring Boot auto-configuration and component scanning.

---

**Q35: How does Spring Boot auto-configure the server port?**

Reads `server.port=8080` from application.properties. Default is 8080 if not specified.

---

## Advanced

**Q36: What is the difference between embeddings and tokens?**

Embeddings are dense vector representations of text (float arrays). Tokens are discrete units of text the model processes (words/subwords). You send tokens to get embeddings.

---

**Q37: Could you use a different similarity metric?**

Yes. Euclidean distance, dot product, or Manhattan distance. Cosine similarity is preferred for text embeddings because it normalizes for vector magnitude.

---

**Q38: What if the retrieved documents contradict each other?**

The LLM will try to synthesize or may indicate uncertainty. Better prompt engineering could instruct the model on how to handle conflicts.

---

**Q39: How does prompt formatting affect answer quality?**

Critical. The format "Context: {docs}\n\nQuestion: {question}\n\nAnswer:" clearly separates context from question, helping the LLM understand its task.

---

**Q40: What is the role of temperature in LLM calls?**

Controls randomness. Lower = more deterministic, higher = more creative. Not configured here, uses default from Azure deployment.

---
