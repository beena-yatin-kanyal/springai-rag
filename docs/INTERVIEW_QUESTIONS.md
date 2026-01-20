# Java Full Stack Architect - Agentic AI Interview Questions

**Role:** Java Full Stack Architect Agentic AI
**Prepared For:** Interview Preparation
**Based On:** RAG Application Codebase Analysis
**Date:** January 2026

---

## Table of Contents

1. [Understanding Current RAG Implementation](#1-understanding-current-rag-implementation)
2. [Spring AI & GenAI Deep Dive](#2-spring-ai--genai-deep-dive)
3. [Architecture & System Design](#3-architecture--system-design)
4. [Multi-Tenant Architecture](#4-multi-tenant-architecture)
5. [Agentic AI Concepts](#5-agentic-ai-concepts)
6. [RAG (Retrieval-Augmented Generation)](#6-rag-retrieval-augmented-generation)
7. [Full Stack Development](#7-full-stack-development)
8. [Database & Persistence](#8-database--persistence)
9. [Cloud Architecture](#9-cloud-architecture)
10. [Production Readiness & Scalability](#10-production-readiness--scalability)
11. [Hands-On Coding Scenarios](#11-hands-on-coding-scenarios)
12. [Spring Boot Deep Dive](#12-spring-boot-deep-dive)
13. [Microservices Architecture](#13-microservices-architecture)

---

## 1. Understanding Current RAG Implementation

### Q1.1: Walk me through the RAG workflow in this codebase
**Focus:** Understanding of end-to-end flow
**Expected Discussion Points:**
- User submits question via POST /ask endpoint (AskController:32-42)
- RagService.askQuestion() orchestrates the workflow
- Question embedding generation using EmbeddingModel
- Cosine similarity search in InMemoryVectorStore
- Retrieval of top-K relevant documents (K=2)
- Prompt construction with context and question
- LLM generation via ChatClient
- Return structured RagResponse with answer and sources

**Follow-up:** Why is the embedding dimension typically 1536 for Azure OpenAI? What are the implications?

### Q1.2: Explain the cosine similarity implementation
**Focus:** Vector mathematics understanding
**Reference:** InMemoryVectorStore.java:132-179
**Expected Discussion:**
- Dot product calculation: Σ(a[i] * b[i])
- Magnitude calculations: sqrt(Σ(a[i]²)) and sqrt(Σ(b[i]²))
- Similarity formula: dot product / (magnitude_a * magnitude_b)
- Result range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
- Why cosine vs Euclidean distance for semantic similarity?

**Coding Challenge:** Optimize the similarity calculation for 10,000+ vectors

### Q1.3: What are the limitations of the current in-memory vector store?
**Focus:** Production readiness awareness
**Expected Answers:**
- Data loss on application restart
- No persistence layer
- Memory constraints for large document sets
- No distributed search capabilities
- Single-node limitation
- No sharding or partitioning

**Follow-up:** How would you migrate to pgvector, Pinecone, or Weaviate?

### Q1.4: Analyze the prompt engineering strategy
**Focus:** LLM prompt design
**Reference:** RagService.java:249-276
**Discussion Points:**
- Context injection before question
- Instruction clarity: "Use the following context..."
- Fallback instruction: "If the answer cannot be found..."
- Trade-offs: specificity vs flexibility
- How to prevent hallucinations?
- How to handle context window limits?

**Challenge:** Improve the prompt to support multi-turn conversations

---

## 2. Spring AI & GenAI Deep Dive

### Q2.1: Explain Spring AI's abstraction layer
**Focus:** Framework architecture understanding
**Expected Discussion:**
- `ChatClient` abstraction for LLM providers
- `EmbeddingModel` abstraction for embedding services
- Provider-agnostic design (Azure OpenAI, OpenAI, AWS Bedrock, etc.)
- Builder pattern usage: `ChatClient.builder()`
- How to swap from Azure to AWS Bedrock with minimal code changes?

**Follow-up:** What are the trade-offs between vendor-specific SDKs vs Spring AI?

### Q2.2: How would you implement streaming responses?
**Focus:** Real-time LLM capabilities
**Current State:** RagService uses `.call().content()` for blocking calls
**Expected Solution:**
- Spring AI's `.stream()` method
- `Flux<String>` reactive streams
- Server-Sent Events (SSE) for HTTP streaming
- Frontend integration with EventSource API
- Error handling in streaming context

**Coding Task:** Modify `AskController` to support streaming responses

### Q2.3: Explain the lifecycle of EmbeddingModel bean
**Focus:** Spring Boot internals
**Discussion:**
- Auto-configuration via `spring-ai-azure-openai-spring-boot-starter`
- Environment-based configuration (application.properties)
- Bean initialization order
- `@PostConstruct` usage in RagService for document initialization
- Connection pooling and resource management

### Q2.4: What are embeddings and how do they work?
**Focus:** Fundamental GenAI knowledge
**Expected Explanation:**
- Vector representations of text in high-dimensional space
- Semantic similarity preservation: similar meanings → similar vectors
- Dimensionality reduction from token space to fixed-size vectors
- Training process: transformer models, contrastive learning
- Use cases: RAG, semantic search, clustering, recommendation systems

**Challenge:** How would you handle multilingual embeddings?

---

## 3. Architecture & System Design

### Q3.1: Design a scalable RAG system for 1 million users
**Focus:** System design & scalability
**Expected Architecture:**
- **API Gateway:** Rate limiting, authentication, routing
- **Compute Layer:** Multiple stateless Spring Boot instances (Kubernetes)
- **Vector Database:** Distributed (Pinecone, Weaviate, or pgvector with replicas)
- **Caching:** Redis for frequently asked questions
- **Message Queue:** Kafka/RabbitMQ for async processing
- **Monitoring:** Prometheus, Grafana, ELK stack
- **CDN:** Static content delivery
- **Load Balancer:** Distribute traffic across instances

**Follow-up:** How do you handle vector store consistency across replicas?

### Q3.2: Explain the layered architecture in this codebase
**Focus:** Clean architecture principles
**Layers:**
1. **Presentation:** AskController (REST API)
2. **Business Logic:** RagService (orchestration)
3. **Data Access:** InMemoryVectorStore (repository)

**Discussion:**
- Dependency flow: Controller → Service → Repository
- Dependency injection benefits
- Single Responsibility Principle
- How to add a caching layer without breaking existing code?

### Q3.3: Design error handling for Azure OpenAI failures
**Focus:** Resilience & fault tolerance
**Current State:** Custom exceptions (EmbeddingException, LlmException)
**Enhancements Needed:**
- **Retry Logic:** Exponential backoff with jitter
- **Circuit Breaker:** Resilience4j integration
- **Fallback Strategies:** Cached responses, degraded mode
- **Bulkhead Pattern:** Isolate embedding from LLM failures
- **Rate Limiting:** Handle 429 responses from Azure

**Coding Task:** Implement retry logic using Spring Retry or Resilience4j

### Q3.4: How would you version this API?
**Focus:** API design & backward compatibility
**Options:**
- URI versioning: `/v1/ask`, `/v2/ask`
- Header versioning: `Accept: application/vnd.ragapp.v1+json`
- Query parameter: `/ask?version=1`

**Discussion:**
- Pros/cons of each approach
- Deprecation strategies
- OpenAPI specification versioning (docs/openapi.yaml)
- Contract testing

---

## 4. Multi-Tenant Architecture

### Q4.1: Design a multi-tenant RAG system
**Focus:** Multi-tenancy patterns (CRITICAL for JD)
**Expected Design:**

**Option 1: Shared Database, Separate Schema**
- Single Spring Boot deployment
- Tenant ID in JWT token
- Hibernate multitenancy with `SCHEMA` strategy
- Vector store namespacing: `tenant_{id}_vectors`

**Option 2: Database per Tenant**
- Dynamic datasource routing
- AbstractRoutingDataSource implementation
- Separate vector databases per tenant

**Option 3: Application per Tenant**
- Isolated deployments
- Kubernetes namespace per tenant
- Complete data isolation

**Discussion:**
- Cost vs isolation trade-offs
- Data residency requirements
- Tenant-specific model customization

### Q4.2: Implement tenant isolation in the current codebase
**Focus:** Hands-on multi-tenancy
**Coding Task:**
1. Add `tenantId` to `DocumentEmbedding` record
2. Modify `InMemoryVectorStore` to filter by tenant
3. Extract tenant ID from request header (X-Tenant-ID)
4. Implement `TenantContext` ThreadLocal storage
5. Add tenant validation in `AskController`

**Security Considerations:**
- Prevent cross-tenant data leakage
- Tenant-aware error messages (no information disclosure)

### Q4.3: How do you handle tenant-specific LLM configurations?
**Focus:** Customization in multi-tenant systems
**Scenarios:**
- Tenant A uses GPT-4, Tenant B uses GPT-3.5-turbo (cost optimization)
- Different temperature settings per tenant
- Custom system prompts
- Tenant-specific document sets

**Implementation:**
- Configuration service: `TenantConfigService`
- Database-driven config: MongoDB collection `tenant_configs`
- Caching with eviction on config updates

### Q4.4: Design tenant onboarding workflow
**Focus:** Operational aspects
**Expected Process:**
1. Provision database schema/namespace
2. Initialize tenant configuration
3. Create vector store partition
4. Generate API keys/credentials
5. Run migration scripts
6. Health check validation

**Discuss:**
- Automation via REST API
- Rollback strategies
- Tenant offboarding (data deletion compliance)

---

## 5. Agentic AI Concepts

### Q5.1: What is Agentic AI and how does it differ from RAG?
**Focus:** Conceptual understanding (CRITICAL for JD)
**Expected Answer:**
- **RAG:** Retrieval + Generation (single-step, context-aware answering)
- **Agentic AI:** Multi-step reasoning, tool use, planning, self-correction

**Agent Characteristics:**
- Autonomy: Can decide next actions
- Tool Use: Can call APIs, search databases, execute code
- Memory: Maintains conversation state
- Planning: Breaks tasks into subtasks
- Reflection: Self-evaluates and corrects

**Examples:**
- Customer support agent: search KB → call API → update ticket → respond
- Data analysis agent: query DB → generate chart → interpret results

### Q5.2: Design an Agentic AI system on top of this RAG codebase
**Focus:** Architectural extension
**Expected Design:**

**Components:**
1. **Agent Orchestrator:** Coordinates multi-step workflows
2. **Tool Registry:** Available tools (RAG search, API calls, calculations)
3. **Planning Module:** LLM-based task decomposition
4. **Execution Engine:** Runs tool calls sequentially/parallel
5. **Memory Store:** Conversation history, intermediate results

**Architecture:**
```
User Query → Planner (LLM) → [Tool 1, Tool 2, Tool 3] → Executor → Response Generator
                ↓
         Memory/Context Store
```

**Example Workflow:**
- User: "What's the weather in the city where Spring Boot was created?"
- Plan: [1) RAG: Where was Spring Boot created? 2) API: Get weather for city]
- Execute: RAG returns "Spring Boot was created by Pivotal in San Francisco"
- Execute: Call weather API for San Francisco
- Generate: "Spring Boot was created in San Francisco. Current weather: 65°F, sunny."

### Q5.3: Implement tool calling with Spring AI
**Focus:** Hands-on Agentic AI
**Spring AI Feature:** Function calling support
**Coding Task:**
```java
@Bean
public FunctionCallback weatherTool() {
    return FunctionCallback.builder()
        .function("getWeather", (String city) -> {
            // Call weather API
            return new WeatherResponse(city, "Sunny", 72);
        })
        .description("Get current weather for a city")
        .inputType(String.class)
        .build();
}

// Use in ChatClient
String response = chatClient.prompt()
    .user("What's the weather in Seattle?")
    .functions(weatherTool())
    .call()
    .content();
```

**Discussion:**
- How does the LLM decide when to call tools?
- Error handling when tool fails
- Chaining multiple tool calls

### Q5.4: What are the challenges in building production Agentic AI?
**Focus:** Real-world considerations
**Expected Challenges:**
- **Cost:** Multiple LLM calls per query (exponential)
- **Latency:** Sequential tool calls increase response time
- **Reliability:** Cascading failures across tools
- **Security:** Agent-initiated API calls need validation
- **Controllability:** Agent may take unexpected paths
- **Observability:** Debugging multi-step reasoning

**Solutions:**
- Streaming for perceived performance
- Caching tool results
- Timeout and circuit breakers
- Tool usage policies and sandboxing
- Comprehensive logging and tracing

---

## 6. RAG (Retrieval-Augmented Generation)

### Q6.1: How would you implement document chunking?
**Focus:** Advanced RAG technique
**Current State:** Documents stored as-is (full text)
**Problem:** Large documents exceed context windows

**Chunking Strategies:**
1. **Fixed-size:** 512 tokens per chunk with 50-token overlap
2. **Semantic:** Split on paragraphs, sentences, or semantic boundaries
3. **Recursive:** Hierarchical splitting (sections → paragraphs → sentences)

**Implementation:**
```java
List<String> chunkDocument(String document, int chunkSize, int overlap) {
    // Tokenize, split, maintain overlap
}
```

**Considerations:**
- How to maintain context across chunks?
- Metadata preservation (source, page number)
- Chunk ID generation for traceability

### Q6.2: Design a document ingestion pipeline
**Focus:** Data engineering for RAG
**Pipeline Stages:**
1. **Upload:** REST API for PDF/DOCX/TXT/HTML
2. **Extraction:** Apache Tika for content extraction
3. **Preprocessing:** Cleaning, deduplication, language detection
4. **Chunking:** Split into optimal sizes
5. **Embedding:** Batch embedding generation
6. **Indexing:** Store in vector database
7. **Metadata Storage:** MongoDB for document metadata

**Technical Considerations:**
- Async processing with message queues
- Error handling and retry logic
- Progress tracking for large document sets
- Incremental updates (avoid re-embedding unchanged docs)

**Coding Task:** Create `DocumentIngestionService` with async processing

### Q6.3: Implement reranking for better retrieval quality
**Focus:** Advanced retrieval technique
**Current State:** Top-K retrieval by cosine similarity
**Problem:** Embedding similarity ≠ relevance to specific query

**Reranking Approach:**
1. Retrieve top-20 candidates (broad recall)
2. Use cross-encoder model to score each candidate against query
3. Return top-3 reranked results (high precision)

**Models:**
- Cohere Rerank API
- Cross-encoder/ms-marco-MiniLM-L-12-v2
- Custom fine-tuned reranker

**Implementation:**
```java
List<Document> rerank(String query, List<Document> candidates) {
    // Call reranking API/model
    // Sort by relevance score
    // Return top-K
}
```

### Q6.4: How do you evaluate RAG system performance?
**Focus:** MLOps and evaluation
**Metrics:**

**Retrieval Metrics:**
- **Precision@K:** Relevant docs in top-K / K
- **Recall@K:** Relevant docs retrieved / total relevant
- **Mean Reciprocal Rank (MRR):** 1 / rank of first relevant doc
- **NDCG:** Normalized Discounted Cumulative Gain

**Generation Metrics:**
- **Faithfulness:** Answer grounded in context (LLM-as-judge)
- **Relevance:** Answer addresses the question
- **BLEU/ROUGE:** Comparison with reference answers
- **Human Evaluation:** Quality, correctness, helpfulness

**Tooling:**
- Ragas framework
- LangSmith evaluation
- Custom evaluation harness

**Example Evaluation Pipeline:**
```java
EvaluationResult evaluate(List<TestCase> testCases) {
    for (TestCase tc : testCases) {
        RagResponse response = ragService.askQuestion(tc.question);
        double faithfulness = evaluateFaithfulness(response, tc.groundTruth);
        double relevance = evaluateRelevance(response, tc.question);
        // Aggregate metrics
    }
}
```

---

## 7. Full Stack Development

### Q7.1: Design a Next.js frontend for this RAG application
**Focus:** Full-stack integration (Next.js in JD)
**Expected Architecture:**

**Tech Stack:**
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- SWR for data fetching
- Server Components for SEO

**Key Pages:**
1. `/` - Chat interface
2. `/documents` - Document management
3. `/admin` - Tenant configuration (if multi-tenant)

**Chat Component:**
```typescript
'use client';

export default function ChatInterface() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState<RagResponse | null>(null);

  const handleAsk = async () => {
    const res = await fetch('/api/ask', {
      method: 'POST',
      body: JSON.stringify({ question }),
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    // Chat UI with question input, submit button, response display
  );
}
```

**API Route (Next.js):**
```typescript
// app/api/ask/route.ts
export async function POST(request: Request) {
  const { question } = await request.json();

  // Call Spring Boot backend
  const response = await fetch('http://localhost:8080/ask', {
    method: 'POST',
    body: JSON.stringify({ question }),
    headers: { 'Content-Type': 'application/json' }
  });

  return Response.json(await response.json());
}
```

**Discussion:**
- Authentication: NextAuth.js with JWT
- SSE for streaming responses
- Optimistic UI updates
- Error boundaries

### Q7.2: How do you handle CORS in this setup?
**Focus:** Frontend-backend integration
**Current Setup:** Spring Boot on :8080, Next.js on :3000

**Spring Boot Configuration:**
```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/ask")
                .allowedOrigins("http://localhost:3000", "https://app.example.com")
                .allowedMethods("POST", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true);
    }
}
```

**Production Considerations:**
- Same-origin deployment: Next.js API routes proxy to Spring Boot
- API Gateway for unified domain
- CORS only needed for development

### Q7.3: Implement real-time document upload progress
**Focus:** Full-stack real-time features
**Backend (Spring Boot):**
```java
@PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
public SseEmitter uploadDocument(@RequestParam("file") MultipartFile file) {
    SseEmitter emitter = new SseEmitter();

    CompletableFuture.runAsync(() -> {
        try {
            emitter.send("Extracting text...");
            String text = extractText(file);

            emitter.send("Generating embeddings...");
            List<float[]> embeddings = generateEmbeddings(text);

            emitter.send("Storing vectors...");
            store(embeddings);

            emitter.send("Complete!");
            emitter.complete();
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
    });

    return emitter;
}
```

**Frontend (Next.js):**
```typescript
const eventSource = new EventSource('/api/upload');
eventSource.onmessage = (event) => {
  setProgress(event.data);
};
```

---

## 8. Database & Persistence

### Q8.1: Design MongoDB schema for this RAG application
**Focus:** MongoDB integration (in JD)
**Collections:**

**1. `documents`**
```json
{
  "_id": "doc_123",
  "title": "Spring Boot Documentation",
  "content": "Full text...",
  "chunks": ["chunk_1", "chunk_2", "chunk_3"],
  "metadata": {
    "source": "docs.spring.io",
    "uploadedBy": "user@example.com",
    "uploadedAt": ISODate("2026-01-19T00:00:00Z"),
    "fileType": "text/html"
  },
  "tenantId": "tenant_abc"
}
```

**2. `embeddings`**
```json
{
  "_id": "emb_456",
  "documentId": "doc_123",
  "chunkId": "chunk_1",
  "vector": [0.123, -0.456, 0.789, ...],  // 1536 dimensions
  "text": "Spring Boot is a framework...",
  "tenantId": "tenant_abc"
}
```

**3. `queries`**
```json
{
  "_id": "query_789",
  "question": "What is Spring Boot?",
  "answer": "Spring Boot is...",
  "relevantDocs": ["doc_123", "doc_124"],
  "latency": 450,
  "timestamp": ISODate("2026-01-19T12:00:00Z"),
  "userId": "user@example.com",
  "tenantId": "tenant_abc"
}
```

**Indexes:**
- `documents`: `{ tenantId: 1, uploadedAt: -1 }`
- `embeddings`: `{ tenantId: 1, documentId: 1 }`
- `queries`: `{ tenantId: 1, timestamp: -1 }`

**Vector Search:**
- **MongoDB Atlas Vector Search** (knnBeta)
- Create vector index on `embeddings.vector`
- Query: `$vectorSearch` aggregation stage

### Q8.2: Implement MongoDB repository for document management
**Focus:** Spring Data MongoDB
**Coding Task:**

**Entity:**
```java
@Document(collection = "documents")
public record DocumentEntity(
    @Id String id,
    String title,
    String content,
    List<String> chunks,
    Map<String, Object> metadata,
    String tenantId,
    LocalDateTime createdAt
) {}
```

**Repository:**
```java
public interface DocumentRepository extends MongoRepository<DocumentEntity, String> {
    List<DocumentEntity> findByTenantIdOrderByCreatedAtDesc(String tenantId);

    @Query("{ 'tenantId': ?0, 'metadata.source': ?1 }")
    Optional<DocumentEntity> findByTenantAndSource(String tenantId, String source);
}
```

**Service:**
```java
@Service
public class DocumentService {
    private final DocumentRepository repository;

    public DocumentEntity saveDocument(String title, String content, String tenantId) {
        var doc = new DocumentEntity(
            null,
            title,
            content,
            chunkContent(content),
            Map.of("uploadedAt", LocalDateTime.now()),
            tenantId,
            LocalDateTime.now()
        );
        return repository.save(doc);
    }
}
```

### Q8.3: Migrate from in-memory to PostgreSQL with pgvector
**Focus:** Production database migration
**Steps:**

**1. Add Dependencies:**
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
</dependency>
<dependency>
    <groupId>com.pgvector</groupId>
    <artifactId>pgvector</artifactId>
    <version>0.1.4</version>
</dependency>
```

**2. Entity:**
```java
@Entity
@Table(name = "document_embeddings")
public class DocumentEmbeddingEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    private String documentId;

    @Column(columnDefinition = "text")
    private String text;

    @Type(type = "com.pgvector.PGvector")
    @Column(columnDefinition = "vector(1536)")
    private float[] embedding;

    private String tenantId;
}
```

**3. Repository with Vector Search:**
```java
public interface EmbeddingRepository extends JpaRepository<DocumentEmbeddingEntity, UUID> {

    @Query(value = """
        SELECT *, (embedding <=> CAST(:queryVector AS vector)) AS distance
        FROM document_embeddings
        WHERE tenant_id = :tenantId
        ORDER BY distance
        LIMIT :topK
        """, nativeQuery = true)
    List<DocumentEmbeddingEntity> findSimilar(
        @Param("queryVector") String queryVector,
        @Param("tenantId") String tenantId,
        @Param("topK") int topK
    );
}
```

**4. Migration Script:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL
);

CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX ON document_embeddings (tenant_id);
```

### Q8.4: Implement caching strategy
**Focus:** Performance optimization
**Caching Layers:**

**1. Application-Level (Caffeine):**
```java
@Configuration
@EnableCaching
public class CacheConfig {
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager manager = new CaffeineCacheManager("embeddings", "queries");
        manager.setCaffeine(Caffeine.newBuilder()
            .expireAfterWrite(1, TimeUnit.HOURS)
            .maximumSize(1000));
        return manager;
    }
}

@Service
public class RagService {
    @Cacheable(value = "embeddings", key = "#text")
    public float[] generateEmbedding(String text) {
        // Expensive Azure OpenAI call
    }

    @Cacheable(value = "queries", key = "#question")
    public RagResponse askQuestion(String question) {
        // Full RAG workflow
    }
}
```

**2. Distributed Cache (Redis):**
```java
@Bean
public RedisCacheManager cacheManager(RedisConnectionFactory factory) {
    return RedisCacheManager.builder(factory)
        .cacheDefaults(RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))
            .serializeValuesWith(/* Jackson serialization */))
        .build();
}
```

**Discussion:**
- Cache invalidation strategies
- Cache warming for common queries
- Cache hit rate monitoring

---

## 9. Cloud Architecture

### Q9.1: Deploy this application to AWS (not Azure)
**Focus:** AWS cloud services (JD mentions AWS)
**Architecture:**

**Compute:**
- **ECS Fargate** or **EKS** for containerized Spring Boot app
- Auto-scaling based on CPU/memory

**AI Services:**
- **AWS Bedrock** instead of Azure OpenAI
- Spring AI supports Bedrock with `spring-ai-bedrock-spring-boot-starter`
- Models: Claude 3, Titan Embeddings

**Vector Database:**
- **Amazon OpenSearch Service** with vector search
- **Amazon RDS PostgreSQL** with pgvector
- **Pinecone** (managed service)

**Storage:**
- **S3** for document storage
- **DocumentDB (MongoDB compatible)** for metadata

**Networking:**
- **ALB** (Application Load Balancer) for traffic distribution
- **Route 53** for DNS
- **CloudFront** for CDN

**Code Changes for Bedrock:**
```xml
<!-- Replace Azure dependency -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-bedrock-spring-boot-starter</artifactId>
</dependency>
```

```properties
# application.properties
spring.ai.bedrock.aws.region=us-west-2
spring.ai.bedrock.chat.model=anthropic.claude-3-sonnet-20240229-v1:0
spring.ai.bedrock.embedding.model=amazon.titan-embed-text-v1
```

### Q9.2: Design CI/CD pipeline
**Focus:** DevOps practices
**Pipeline:**

**1. Source → Build:**
- GitHub/GitLab trigger on push
- Maven build: `mvn clean package`
- Unit tests: `mvn test`
- Integration tests with Testcontainers

**2. Build → Container:**
- Docker build using multi-stage Dockerfile
- Security scanning: Trivy, Snyk
- Push to ECR (AWS) or ACR (Azure)

**3. Container → Deploy:**
- **Staging:** Deploy to ECS staging cluster
- Smoke tests, integration tests
- **Production:** Blue-green deployment to ECS production
- Health checks before traffic switch

**4. Monitoring:**
- CloudWatch logs and metrics
- Application Performance Monitoring (APM): Datadog, New Relic
- Alerting on error rates, latency

**Tools:**
- GitHub Actions, GitLab CI, Jenkins
- Terraform for infrastructure as code
- Helm for Kubernetes deployments

### Q9.3: Implement observability for LLM calls
**Focus:** Monitoring AI applications
**Challenges:**
- LLM calls are expensive and slow
- Need to track: latency, token usage, costs, errors

**Implementation:**

**1. Custom Metrics (Micrometer):**
```java
@Service
public class RagService {
    private final MeterRegistry registry;

    public RagResponse askQuestion(String question) {
        Timer.Sample sample = Timer.start(registry);

        try {
            RagResponse response = performRag(question);

            registry.counter("llm.calls", "status", "success").increment();
            registry.counter("llm.tokens", "type", "completion")
                .increment(response.getTokensUsed());

            return response;
        } catch (Exception e) {
            registry.counter("llm.calls", "status", "error", "type", e.getClass().getSimpleName()).increment();
            throw e;
        } finally {
            sample.stop(registry.timer("llm.latency"));
        }
    }
}
```

**2. Distributed Tracing (Spring Cloud Sleuth + Zipkin):**
```java
@Service
public class RagService {
    private final Tracer tracer;

    public RagResponse askQuestion(String question) {
        Span span = tracer.nextSpan().name("rag.workflow").start();
        try (Tracer.SpanInScope ws = tracer.withSpan(span)) {
            span.tag("question", question);

            // Embedding span
            Span embeddingSpan = tracer.nextSpan(span).name("embedding.generate");
            // ... generate embedding
            embeddingSpan.end();

            // Vector search span
            Span searchSpan = tracer.nextSpan(span).name("vector.search");
            // ... search
            searchSpan.end();

            // LLM span
            Span llmSpan = tracer.nextSpan(span).name("llm.generate");
            // ... call LLM
            llmSpan.tag("model", "gpt-4");
            llmSpan.tag("tokens", "1500");
            llmSpan.end();

            return response;
        } finally {
            span.end();
        }
    }
}
```

**3. LLM-Specific Observability:**
- **LangSmith:** Trace LLM calls, prompts, responses
- **Helicone:** OpenAI proxy with analytics
- **Custom dashboards:** Grafana with cost, latency, error rate

### Q9.4: Design disaster recovery strategy
**Focus:** Business continuity
**RTO/RPO:**
- **RTO (Recovery Time Objective):** 1 hour
- **RPO (Recovery Point Objective):** 15 minutes

**Strategy:**

**1. Data Backup:**
- **Vector Database:** Continuous replication to secondary region
- **MongoDB:** Replica set with cross-region nodes
- **S3 Documents:** Cross-region replication (CRR)

**2. Multi-Region Deployment:**
- Active-active in two AWS regions (us-west-2, us-east-1)
- Route 53 health checks and failover routing
- Database writes to primary, reads from nearest region

**3. Runbook:**
- Automated health checks every 1 minute
- Alert on consecutive failures (3+)
- Failover script: Update Route 53, promote secondary DB

**4. Testing:**
- Quarterly disaster recovery drills
- Chaos engineering (AWS Fault Injection Simulator)

---

## 10. Production Readiness & Scalability

### Q10.1: Identify security vulnerabilities in current codebase
**Focus:** Security awareness
**Vulnerabilities:**

**1. No Authentication/Authorization:**
- `/ask` endpoint is publicly accessible
- No tenant validation
- Solution: Spring Security with JWT

**2. No Input Validation:**
- `QuestionRequest.question` can be extremely long (DoS)
- No sanitization (potential prompt injection)
- Solution: `@NotBlank @Size(min=1, max=500)`

**3. No Rate Limiting:**
- Expensive LLM calls can be abused
- Solution: Bucket4j or API Gateway rate limiting

**4. Secrets in application.properties:**
- `AZURE_OPENAI_API_KEY` should be in secrets manager
- Solution: AWS Secrets Manager, HashiCorp Vault

**5. No HTTPS enforcement:**
- Production should enforce TLS
- Solution: Spring Security `requiresSecure()`

**Coding Task:** Implement JWT authentication for `/ask` endpoint

### Q10.2: How do you handle Azure OpenAI rate limits?
**Focus:** Resilience engineering
**Azure OpenAI Limits:**
- Tokens per minute (TPM): 240K for GPT-4
- Requests per minute (RPM): 720

**Strategies:**

**1. Client-Side Rate Limiting:**
```java
@Service
public class RateLimitedChatClient {
    private final ChatClient chatClient;
    private final RateLimiter rateLimiter = RateLimiter.create(10.0); // 10 req/sec

    public String call(String prompt) {
        rateLimiter.acquire();
        return chatClient.prompt().user(prompt).call().content();
    }
}
```

**2. Retry with Exponential Backoff:**
```java
@Retryable(
    value = {LlmException.class},
    maxAttempts = 3,
    backoff = @Backoff(delay = 1000, multiplier = 2)
)
public String callLlm(String prompt) {
    // Azure OpenAI call
}
```

**3. Request Batching:**
- Combine multiple embeddings into single API call
- Azure OpenAI supports batch embedding requests

**4. Caching:**
- Cache embeddings for duplicate documents
- Cache LLM responses for identical questions

**5. Queue-Based Processing:**
- Async processing with bounded queue
- Back-pressure when queue is full

### Q10.3: Design a monitoring dashboard
**Focus:** Observability
**Key Metrics:**

**Application Metrics:**
- Request rate (req/sec)
- Latency (p50, p95, p99)
- Error rate (% and count)
- Active connections

**AI-Specific Metrics:**
- LLM call latency
- Embedding generation time
- Vector search latency
- Token usage (total, per request)
- LLM costs ($)

**Infrastructure Metrics:**
- CPU, memory, disk usage
- JVM heap usage, GC pauses
- Container restarts
- Network I/O

**Business Metrics:**
- Questions per day
- Unique users
- Answer satisfaction (if feedback mechanism exists)
- Document upload volume

**Tools:**
- **Prometheus:** Metric collection
- **Grafana:** Visualization
- **ELK Stack:** Log aggregation
- **Sentry:** Error tracking

**Dashboard Panels:**
1. Overview: RPS, latency, error rate
2. AI Performance: LLM latency, token usage, costs
3. Infrastructure: CPU, memory, JVM
4. Alerts: Critical errors, high latency, cost spikes

### Q10.4: Load testing strategy
**Focus:** Performance validation
**Tools:** JMeter, Gatling, k6

**Test Scenarios:**

**1. Baseline Load:**
- 100 concurrent users
- 10 req/sec sustained
- Duration: 10 minutes
- Expected: p95 < 2s, error rate < 0.1%

**2. Stress Test:**
- Gradually increase to 1000 concurrent users
- Identify breaking point
- Observe: When does latency spike? When do errors start?

**3. Spike Test:**
- Sudden traffic surge (10x normal)
- Validate auto-scaling response
- Expected: Graceful degradation, no crashes

**4. Soak Test:**
- Normal load for 24 hours
- Identify memory leaks, resource exhaustion

**Gatling Script:**
```scala
val scn = scenario("RAG Ask")
  .exec(http("Ask Question")
    .post("/ask")
    .header("Content-Type", "application/json")
    .body(StringBody("""{"question":"What is Spring Boot?"}"""))
    .check(status.is(200))
    .check(jsonPath("$.answer").exists))

setUp(
  scn.inject(
    rampUsersPerSec(1) to 100 during (5.minutes),
    constantUsersPerSec(100) during (10.minutes)
  )
).protocols(http.baseUrl("http://localhost:8080"))
```

**Analysis:**
- Bottleneck identification (DB, LLM, vector search?)
- Cost projection at scale
- Auto-scaling configuration tuning

---

## 11. Hands-On Coding Scenarios

### Q11.1: Implement document deduplication
**Focus:** Algorithm design & coding
**Problem:** Prevent duplicate document indexing

**Approach 1: Hash-Based:**
```java
@Service
public class DocumentDeduplicationService {
    private final Set<String> documentHashes = new ConcurrentHashMap<>.newKeySet();

    public boolean isDuplicate(String content) {
        String hash = DigestUtils.sha256Hex(content);
        return !documentHashes.add(hash);
    }
}
```

**Approach 2: Embedding Similarity:**
```java
public boolean isDuplicate(String newDoc, float[] newEmbedding) {
    List<SimilarityScore> similar = vectorStore.findSimilar(newEmbedding, 1);

    if (similar.isEmpty()) return false;

    // If top match has similarity > 0.95, consider duplicate
    return similar.get(0).score() > 0.95;
}
```

**Discussion:**
- Hash approach: Exact duplicates only
- Embedding approach: Near-duplicates detected
- Trade-offs: Speed vs accuracy

### Q11.2: Implement streaming responses
**Focus:** Reactive programming
**Modify RagService:**
```java
public Flux<String> askQuestionStreaming(String question) {
    // Generate embedding
    float[] embedding = generateEmbedding(question);

    // Retrieve documents
    List<Document> docs = vectorStore.findSimilar(embedding, 2);

    // Build prompt
    String prompt = buildPrompt(question, docs);

    // Stream LLM response
    return chatClient.prompt()
        .user(prompt)
        .stream()
        .content();
}
```

**Modify Controller:**
```java
@GetMapping(value = "/ask-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> askStream(@RequestParam String question) {
    return ragService.askQuestionStreaming(question);
}
```

**Frontend (JavaScript):**
```javascript
const eventSource = new EventSource('/ask-stream?question=What+is+Spring+Boot');
eventSource.onmessage = (event) => {
  document.getElementById('answer').innerText += event.data;
};
```

### Q11.3: Add request tracing with correlation IDs
**Focus:** Distributed tracing
**Implementation:**

**Filter:**
```java
@Component
public class CorrelationIdFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain chain
    ) throws ServletException, IOException {
        String correlationId = request.getHeader("X-Correlation-ID");

        if (correlationId == null) {
            correlationId = UUID.randomUUID().toString();
        }

        MDC.put("correlationId", correlationId);
        response.setHeader("X-Correlation-ID", correlationId);

        try {
            chain.doFilter(request, response);
        } finally {
            MDC.clear();
        }
    }
}
```

**Logging Configuration (logback-spring.xml):**
```xml
<pattern>%d{ISO8601} [%X{correlationId}] %-5level [%thread] %logger{36} - %msg%n</pattern>
```

**Result:**
```
2026-01-19T12:00:00.123 [abc-123-def] INFO  [http-nio-8080-exec-1] c.e.r.service.RagService - Received question: What is Spring Boot?
2026-01-19T12:00:00.456 [abc-123-def] INFO  [http-nio-8080-exec-1] c.e.r.service.RagService - Generated embedding in 200ms
2026-01-19T12:00:00.789 [abc-123-def] INFO  [http-nio-8080-exec-1] c.e.r.service.RagService - Retrieved 2 relevant documents
```

### Q11.4: Write a custom Spring AI EmbeddingModel implementation
**Focus:** Framework extension
**Scenario:** Integrate a custom embedding service

**Implementation:**
```java
@Component
public class CustomEmbeddingModel implements EmbeddingModel {
    private final RestTemplate restTemplate;

    public CustomEmbeddingModel(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Override
    public float[] embed(String text) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> body = Map.of("text", text);
        HttpEntity<Map<String, String>> request = new HttpEntity<>(body, headers);

        ResponseEntity<float[]> response = restTemplate.postForEntity(
            "https://custom-embedding-api.com/embed",
            request,
            float[].class
        );

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new EmbeddingException("Failed to generate embedding");
        }

        return response.getBody();
    }

    @Override
    public List<float[]> embed(List<String> texts) {
        return texts.stream()
            .map(this::embed)
            .collect(Collectors.toList());
    }

    @Override
    public EmbeddingResponse embedForResponse(List<String> texts) {
        List<float[]> embeddings = embed(texts);

        List<Embedding> embeddingList = embeddings.stream()
            .map(arr -> new Embedding(arr, 0))
            .collect(Collectors.toList());

        return new EmbeddingResponse(embeddingList);
    }
}
```

**Configuration:**
```java
@Configuration
public class AIConfig {
    @Bean
    @Primary
    public EmbeddingModel customEmbeddingModel(RestTemplate restTemplate) {
        return new CustomEmbeddingModel(restTemplate);
    }
}
```

### Q11.5: Implement A/B testing for different prompts
**Focus:** Experimentation framework
**Goal:** Test which prompt generates better answers

**Implementation:**

**Prompt Variants:**
```java
public enum PromptVariant {
    CONTROL("""
        Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Answer:
        """),

    VARIANT_A("""
        You are a helpful assistant. Answer the question based on the context.
        Context: {context}
        Question: {question}
        Provide a concise answer:
        """),

    VARIANT_B("""
        Context information:
        {context}

        Based on the above context, answer this question: {question}

        Answer in 2-3 sentences:
        """);

    private final String template;

    PromptVariant(String template) {
        this.template = template;
    }

    public String format(String context, String question) {
        return template
            .replace("{context}", context)
            .replace("{question}", question);
    }
}
```

**A/B Test Service:**
```java
@Service
public class ABTestingService {
    private final Random random = new Random();

    public PromptVariant assignVariant(String userId) {
        // Hash-based assignment for consistency
        int hash = userId.hashCode();
        int bucket = Math.abs(hash % 100);

        if (bucket < 33) return PromptVariant.CONTROL;
        if (bucket < 66) return PromptVariant.VARIANT_A;
        return PromptVariant.VARIANT_B;
    }

    public void recordResult(String userId, PromptVariant variant,
                            String question, String answer,
                            Integer feedbackScore) {
        // Store in MongoDB for analysis
        ExperimentResult result = new ExperimentResult(
            userId, variant, question, answer, feedbackScore, LocalDateTime.now()
        );
        // repository.save(result);
    }
}
```

**Usage in RagService:**
```java
public RagResponse askQuestion(String question, String userId) {
    PromptVariant variant = abTestingService.assignVariant(userId);

    // ... generate embedding, retrieve docs

    String prompt = variant.format(context, question);
    String answer = chatClient.prompt().user(prompt).call().content();

    MDC.put("promptVariant", variant.name());
    log.info("Used prompt variant: {}", variant);

    return new RagResponse(question, answer, docs);
}
```

**Analysis:**
- Collect feedback scores per variant
- Statistical significance testing (t-test)
- Metrics: answer quality, latency, cost
- Rollout winning variant to 100%

---

## 12. Spring Boot Deep Dive

### Q12.1: Explain Spring Boot auto-configuration mechanism
**Focus:** Core Spring Boot internals (CRITICAL for JD)
**Expected Discussion:**

**Auto-Configuration Process:**
1. **@SpringBootApplication** combines three annotations:
   - `@Configuration`: Marks class as configuration source
   - `@EnableAutoConfiguration`: Enables auto-configuration
   - `@ComponentScan`: Scans for components in current package and below

2. **META-INF/spring.factories** (Spring Boot 2.x) or **META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports** (Spring Boot 3.x):
   - Lists all auto-configuration classes
   - Conditionally loaded based on classpath and configuration

3. **Conditional Annotations:**
   - `@ConditionalOnClass`: Load if class present on classpath
   - `@ConditionalOnMissingBean`: Load if bean not already defined
   - `@ConditionalOnProperty`: Load if property has specific value

**Example from this codebase:**
```java
// Spring AI auto-configuration activates when:
// 1. Azure OpenAI classes are on classpath
// 2. Properties spring.ai.azure.openai.* are configured
// 3. No custom EmbeddingModel bean defined
```

**Follow-up Questions:**
- How would you disable specific auto-configurations?
- How do you debug which auto-configurations are active?
- What's the difference between `@Component` and `@Configuration`?

**Coding Task:** Create a custom auto-configuration for a "RagMetrics" component that auto-configures only when Micrometer is on the classpath.

### Q12.2: Explain Spring Boot's application properties hierarchy
**Focus:** Configuration management
**Reference:** application.properties in this codebase
**Property Sources (in order of precedence):**
1. Command line arguments: `--server.port=9090`
2. Java System properties: `System.setProperty()`
3. OS environment variables: `AZURE_OPENAI_API_KEY`
4. `application-{profile}.properties` in JAR
5. `application.properties` in JAR
6. `@PropertySource` annotations
7. Default properties

**Current codebase configuration:**
```properties
# application.properties
spring.ai.azure.openai.api-key=${AZURE_OPENAI_API_KEY}
spring.ai.azure.openai.endpoint=${AZURE_OPENAI_ENDPOINT}
spring.ai.azure.openai.chat.options.deployment-name=${AZURE_OPENAI_CHAT_DEPLOYMENT}
spring.ai.azure.openai.embedding.options.deployment-name=${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}
```

**Discussion Points:**
- Environment-specific configuration (dev, staging, prod)
- Externalizing configuration for Kubernetes ConfigMaps/Secrets
- Using `@ConfigurationProperties` vs `@Value`
- Spring Cloud Config Server for centralized configuration

**Coding Task:**
1. Create a `RagConfigurationProperties` class with `@ConfigurationProperties(prefix="rag")`
2. Add properties: `vectorStore.topK`, `llm.temperature`, `llm.maxTokens`
3. Enable validation with `@Validated` and JSR-303 annotations

### Q12.3: Explain the Spring Boot startup lifecycle
**Focus:** Application lifecycle understanding
**Startup Sequence:**

**1. SpringApplication.run() invoked:**
```java
@SpringBootApplication
public class RagApplication {
    public static void main(String[] args) {
        SpringApplication.run(RagApplication.class, args);
    }
}
```

**2. Application Context Creation:**
- Load configuration classes
- Component scanning
- Bean definition registration

**3. Bean Instantiation Order:**
- `@DependsOn` annotations respected
- Dependency injection order (constructor → field → setter)
- `@PostConstruct` methods called after dependency injection

**4. Lifecycle Callbacks:**
```java
@Service
public class RagService {
    @PostConstruct
    public void init() {
        // Called after dependencies injected
        // Used in codebase to initialize documents (RagService.java:52-105)
    }

    @PreDestroy
    public void cleanup() {
        // Called before bean destruction
    }
}
```

**5. ApplicationRunner and CommandLineRunner:**
```java
@Component
public class DataLoader implements ApplicationRunner {
    @Override
    public void run(ApplicationArguments args) {
        // Runs after context fully initialized
        // Good for data loading, health checks
    }
}
```

**6. Server Startup:**
- Embedded Tomcat/Jetty/Undertow starts
- Application ready to serve requests

**Follow-up:**
- What's the difference between `@PostConstruct` and `ApplicationRunner`?
- How do you ensure bean A is created before bean B?
- What happens during graceful shutdown?

**Coding Task:** Add a startup health check that verifies Azure OpenAI connectivity before marking application as ready.

### Q12.4: Design custom Spring Boot Starter
**Focus:** Framework extension
**Scenario:** Create a `spring-boot-starter-rag` that auto-configures RAG components

**Starter Structure:**
```
spring-boot-starter-rag/
├── pom.xml
└── src/main/
    ├── java/com/example/rag/autoconfigure/
    │   ├── RagAutoConfiguration.java
    │   ├── RagProperties.java
    │   └── VectorStoreAutoConfiguration.java
    └── resources/
        └── META-INF/spring/
            └── org.springframework.boot.autoconfigure.AutoConfiguration.imports
```

**Auto-Configuration Class:**
```java
@AutoConfiguration
@ConditionalOnClass({EmbeddingModel.class, ChatClient.class})
@EnableConfigurationProperties(RagProperties.class)
public class RagAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public VectorStore vectorStore(RagProperties properties) {
        if (properties.getVectorStore().getType().equals("in-memory")) {
            return new InMemoryVectorStore();
        } else if (properties.getVectorStore().getType().equals("pgvector")) {
            return new PgVectorStore(properties.getVectorStore().getPgvector());
        }
        throw new IllegalStateException("Unknown vector store type");
    }

    @Bean
    @ConditionalOnMissingBean
    public RagService ragService(
        EmbeddingModel embeddingModel,
        ChatClient chatClient,
        VectorStore vectorStore
    ) {
        return new RagService(embeddingModel, chatClient, vectorStore);
    }
}
```

**Properties Class:**
```java
@ConfigurationProperties(prefix = "rag")
public class RagProperties {
    private VectorStoreProperties vectorStore = new VectorStoreProperties();
    private LlmProperties llm = new LlmProperties();

    public static class VectorStoreProperties {
        private String type = "in-memory";
        private int topK = 3;
        private PgVectorProperties pgvector;
    }

    public static class LlmProperties {
        private double temperature = 0.7;
        private int maxTokens = 500;
    }
}
```

**AutoConfiguration.imports:**
```
com.example.rag.autoconfigure.RagAutoConfiguration
com.example.rag.autoconfigure.VectorStoreAutoConfiguration
```

**Discussion:**
- When to create a starter vs library?
- Testing auto-configurations with `@SpringBootTest`
- Documentation and configuration metadata (spring-configuration-metadata.json)

### Q12.5: Explain Spring Boot Actuator endpoints
**Focus:** Production monitoring
**Add Dependency:**
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

**Key Endpoints:**

**1. /actuator/health:**
- Aggregates health indicators
- Returns UP/DOWN status
- Custom health indicator for Azure OpenAI:

```java
@Component
public class AzureOpenAIHealthIndicator implements HealthIndicator {
    private final ChatClient chatClient;

    @Override
    public Health health() {
        try {
            // Test ping to Azure OpenAI
            chatClient.prompt().user("ping").call().content();
            return Health.up()
                .withDetail("service", "Azure OpenAI")
                .withDetail("status", "Connected")
                .build();
        } catch (Exception e) {
            return Health.down()
                .withDetail("service", "Azure OpenAI")
                .withDetail("error", e.getMessage())
                .build();
        }
    }
}
```

**2. /actuator/metrics:**
- JVM metrics (heap, threads, GC)
- HTTP request metrics
- Custom metrics (LLM calls, token usage)

**3. /actuator/info:**
- Application metadata
- Git commit info
- Build version

**4. /actuator/prometheus:**
- Prometheus-formatted metrics
- Integrates with Prometheus/Grafana

**5. /actuator/env:**
- Environment properties
- Configuration sources

**Configuration:**
```properties
management.endpoints.web.exposure.include=health,metrics,info,prometheus
management.endpoint.health.show-details=always
management.metrics.export.prometheus.enabled=true
```

**Security Considerations:**
- Expose only necessary endpoints
- Use Spring Security to protect actuator endpoints
- Avoid exposing sensitive configuration in /env

**Coding Task:** Create custom metrics for:
- Number of RAG queries per minute
- Average LLM latency
- Token usage per tenant

### Q12.6: Implement custom exception handling with @ControllerAdvice
**Focus:** Error handling strategy
**Reference:** GlobalExceptionHandler.java in codebase

**Current Implementation Analysis:**
```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        ErrorResponse error = ErrorResponse.builder()
            .status(HttpStatus.BAD_REQUEST.value())
            .message(ex.getMessage())
            .errorCode("INVALID_ARGUMENT")
            .timestamp(LocalDateTime.now())
            .build();
        return ResponseEntity.badRequest().body(error);
    }

    @ExceptionHandler(VectorStoreException.class)
    public ResponseEntity<ErrorResponse> handleVectorStore(VectorStoreException ex) {
        // Returns 500 with structured error
    }
}
```

**Enhancement: Add request context:**
```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
        Exception ex,
        HttpServletRequest request
    ) {
        String correlationId = (String) request.getAttribute("correlationId");

        ErrorResponse error = ErrorResponse.builder()
            .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
            .message("An unexpected error occurred")
            .errorCode("INTERNAL_ERROR")
            .timestamp(LocalDateTime.now())
            .correlationId(correlationId)
            .path(request.getRequestURI())
            .build();

        log.error("[{}] Unhandled exception at {}: {}",
            correlationId, request.getRequestURI(), ex.getMessage(), ex);

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(error);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(
        MethodArgumentNotValidException ex
    ) {
        Map<String, String> fieldErrors = ex.getBindingResult()
            .getFieldErrors()
            .stream()
            .collect(Collectors.toMap(
                FieldError::getField,
                FieldError::getDefaultMessage
            ));

        ErrorResponse error = ErrorResponse.builder()
            .status(HttpStatus.BAD_REQUEST.value())
            .message("Validation failed")
            .errorCode("VALIDATION_ERROR")
            .timestamp(LocalDateTime.now())
            .details(fieldErrors)
            .build();

        return ResponseEntity.badRequest().body(error);
    }
}
```

**Best Practices:**
- Return consistent error structure
- Log errors with correlation IDs
- Don't expose internal stack traces to clients
- Use appropriate HTTP status codes
- Support internationalization (i18n) for error messages

### Q12.7: Explain Spring Boot testing strategies
**Focus:** Testing pyramid
**Reference:** Test files in codebase

**1. Unit Tests (Service Layer):**
```java
@ExtendWith(MockitoExtension.class)
class RagServiceTest {
    @Mock
    private EmbeddingModel embeddingModel;

    @Mock
    private ChatClient chatClient;

    @Mock
    private InMemoryVectorStore vectorStore;

    @InjectMocks
    private RagService ragService;

    @Test
    void testAskQuestion_Success() {
        // Arrange
        when(embeddingModel.embed(anyString())).thenReturn(new float[]{1.0f, 2.0f});
        when(vectorStore.findSimilar(any(), anyInt())).thenReturn(List.of(...));
        when(chatClient.prompt().user(anyString()).call().content())
            .thenReturn("Spring Boot is a framework...");

        // Act
        RagResponse response = ragService.askQuestion("What is Spring Boot?");

        // Assert
        assertThat(response.answer()).isNotBlank();
        verify(embeddingModel).embed(anyString());
    }
}
```

**2. Integration Tests (Controller Layer):**
```java
@WebMvcTest(AskController.class)
class AskControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RagService ragService;

    @Test
    void testAskEndpoint_ValidQuestion() throws Exception {
        when(ragService.askQuestion(anyString()))
            .thenReturn(new RagResponse("Question", "Answer", List.of()));

        mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"question\":\"What is Spring Boot?\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.answer").value("Answer"));
    }
}
```

**3. Spring Boot Test (Full Context):**
```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RagApplicationIntegrationTest {
    @Autowired
    private TestRestTemplate restTemplate;

    @MockBean
    private EmbeddingModel embeddingModel;

    @Test
    void testFullRagWorkflow() {
        // Mock external dependencies
        when(embeddingModel.embed(anyString())).thenReturn(new float[1536]);

        // Make real HTTP request
        ResponseEntity<RagResponse> response = restTemplate.postForEntity(
            "/ask",
            new QuestionRequest("What is Spring Boot?"),
            RagResponse.class
        );

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
    }
}
```

**4. Testcontainers (Database Integration):**
```java
@SpringBootTest
@Testcontainers
class PostgresVectorStoreIntegrationTest {
    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("pgvector/pgvector:pg16")
        .withDatabaseName("rag_test")
        .withUsername("test")
        .withPassword("test");

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
    }

    @Test
    void testVectorSearch() {
        // Test with real PostgreSQL + pgvector
    }
}
```

**5. Contract Testing (for Microservices):**
```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureStubRunner(
    ids = "com.example:embedding-service:+:stubs:8100",
    stubsMode = StubRunnerProperties.StubsMode.LOCAL
)
class EmbeddingServiceContractTest {
    // Verify contract with embedding service
}
```

**Test Annotations Summary:**
- `@SpringBootTest`: Full application context
- `@WebMvcTest`: Only web layer (controllers)
- `@DataJpaTest`: Only JPA layer
- `@MockBean`: Mock Spring beans
- `@Testcontainers`: Integration with Docker containers

### Q12.8: Implement application profiles for different environments
**Focus:** Environment management
**Profile Strategy:**

**1. application.properties (common):**
```properties
spring.application.name=rag-application
server.port=8080
logging.level.com.example.rag=INFO
```

**2. application-dev.properties:**
```properties
spring.ai.azure.openai.api-key=dev-key-from-local-env
logging.level.com.example.rag=DEBUG
management.endpoints.web.exposure.include=*
```

**3. application-prod.properties:**
```properties
spring.ai.azure.openai.api-key=${AZURE_OPENAI_API_KEY}
logging.level.com.example.rag=WARN
management.endpoints.web.exposure.include=health,metrics,prometheus
server.tomcat.max-threads=200
```

**Profile-Specific Beans:**
```java
@Configuration
@Profile("dev")
public class DevConfiguration {
    @Bean
    public VectorStore vectorStore() {
        return new InMemoryVectorStore(); // Fast for dev
    }
}

@Configuration
@Profile("prod")
public class ProdConfiguration {
    @Bean
    public VectorStore vectorStore(DataSource dataSource) {
        return new PgVectorStore(dataSource); // Persistent for prod
    }
}
```

**Activating Profiles:**
- Command line: `java -jar app.jar --spring.profiles.active=prod`
- Environment variable: `SPRING_PROFILES_ACTIVE=prod`
- application.properties: `spring.profiles.active=prod`
- Kubernetes ConfigMap: `SPRING_PROFILES_ACTIVE: prod`

**Multiple Profiles:**
```bash
java -jar app.jar --spring.profiles.active=prod,monitoring,eu-region
```

**Profile-based Testing:**
```java
@SpringBootTest
@ActiveProfiles("test")
class RagServiceTestWithProfile {
    // Uses application-test.properties
}
```

---

## 13. Microservices Architecture

### Q13.1: Break down this monolith into microservices
**Focus:** Microservices decomposition (CRITICAL for JD)
**Current Monolith:** Single Spring Boot app with all components

**Proposed Microservices:**

**1. Document Ingestion Service:**
- **Responsibilities:**
  - File upload and parsing
  - Document chunking
  - Metadata extraction
- **Tech Stack:** Spring Boot, Apache Tika, MinIO/S3
- **API:** POST /documents, GET /documents/{id}
- **Database:** MongoDB (document metadata)
- **Message:** Publish "DocumentUploaded" event to Kafka

**2. Embedding Service:**
- **Responsibilities:**
  - Generate embeddings for text chunks
  - Batch embedding processing
  - Embedding model management
- **Tech Stack:** Spring Boot, Spring AI, Azure OpenAI
- **API:** POST /embeddings (batch), GET /embeddings/{id}
- **Database:** Redis (embedding cache)
- **Message:** Subscribe to "DocumentUploaded", publish "EmbeddingGenerated"

**3. Vector Search Service:**
- **Responsibilities:**
  - Store and index embeddings
  - Similarity search
  - Vector operations
- **Tech Stack:** Spring Boot, pgvector/Pinecone/Weaviate
- **API:** POST /search, POST /index, DELETE /vectors/{id}
- **Database:** PostgreSQL with pgvector extension
- **Message:** Subscribe to "EmbeddingGenerated"

**4. RAG Query Service:**
- **Responsibilities:**
  - Orchestrate RAG workflow
  - Call embedding service for query embedding
  - Call vector search for retrieval
  - Call LLM service for generation
- **Tech Stack:** Spring Boot, Spring AI, Resilience4j
- **API:** POST /ask
- **Database:** Redis (query cache)

**5. LLM Service:**
- **Responsibilities:**
  - LLM inference
  - Prompt management
  - Response streaming
- **Tech Stack:** Spring Boot, Spring AI, Azure OpenAI
- **API:** POST /generate, POST /generate-stream
- **Database:** None (stateless)

**6. Tenant Management Service:**
- **Responsibilities:**
  - Tenant onboarding/offboarding
  - Tenant configuration
  - Usage tracking
- **Tech Stack:** Spring Boot, Spring Data JPA
- **API:** POST /tenants, GET /tenants/{id}, PUT /tenants/{id}/config
- **Database:** PostgreSQL

**Service Communication:**
```
User Request → API Gateway → RAG Query Service
                                 ↓ (REST)
                           Embedding Service → Azure OpenAI
                                 ↓
                           Vector Search Service → pgvector
                                 ↓ (REST)
                           LLM Service → Azure OpenAI
                                 ↓
                           Response → User
```

**Event-Driven Flow (Document Upload):**
```
Document Ingestion → Kafka: DocumentUploaded
                        ↓
                   Embedding Service → Kafka: EmbeddingGenerated
                        ↓
                   Vector Search Service (indexes)
```

**Discussion:**
- Bounded contexts and domain-driven design
- Sync vs async communication
- Data consistency (eventual consistency)
- Transaction management across services (Saga pattern)

### Q13.2: Implement service discovery with Spring Cloud Netflix Eureka
**Focus:** Service registry pattern
**Architecture:**

**1. Eureka Server:**
```java
@SpringBootApplication
@EnableEurekaServer
public class EurekaServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(EurekaServerApplication.class, args);
    }
}
```

**application.yml (Eureka Server):**
```yaml
server:
  port: 8761

eureka:
  client:
    register-with-eureka: false
    fetch-registry: false
```

**2. RAG Query Service (Eureka Client):**
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

```java
@SpringBootApplication
@EnableDiscoveryClient
public class RagQueryServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(RagQueryServiceApplication.class, args);
    }
}
```

**application.yml (RAG Query Service):**
```yaml
spring:
  application:
    name: rag-query-service

eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
  instance:
    prefer-ip-address: true
```

**3. Service-to-Service Communication with Feign:**
```java
@FeignClient(name = "embedding-service")
public interface EmbeddingServiceClient {
    @PostMapping("/embeddings")
    EmbeddingResponse generateEmbedding(@RequestBody EmbeddingRequest request);
}

@Service
public class RagService {
    private final EmbeddingServiceClient embeddingClient;

    public RagResponse askQuestion(String question) {
        // Feign automatically discovers "embedding-service" via Eureka
        EmbeddingResponse embedding = embeddingClient.generateEmbedding(
            new EmbeddingRequest(question)
        );

        // Continue RAG workflow...
    }
}
```

**4. Load Balancing with Ribbon (deprecated, use Spring Cloud LoadBalancer):**
```java
@Bean
@LoadBalanced
public RestTemplate restTemplate() {
    return new RestTemplate();
}

@Service
public class RagService {
    @Autowired
    private RestTemplate restTemplate;

    public void callEmbeddingService() {
        // Automatically load-balanced across "embedding-service" instances
        String response = restTemplate.postForObject(
            "http://embedding-service/embeddings",
            request,
            String.class
        );
    }
}
```

**Alternative: Kubernetes Service Discovery:**
- Use Kubernetes Services instead of Eureka
- Spring Cloud Kubernetes integration
- Service discovery via DNS (embedding-service.default.svc.cluster.local)

**Discussion:**
- Eureka vs Consul vs Kubernetes service discovery
- Health checks and heartbeats
- Service deregistration
- Multi-datacenter support

### Q13.3: Implement API Gateway pattern with Spring Cloud Gateway
**Focus:** Edge service
**Architecture:**

**API Gateway Responsibilities:**
- **Routing:** Forward requests to appropriate microservices
- **Authentication:** JWT validation
- **Rate Limiting:** Per-tenant rate limits
- **Request/Response Transformation:** Add correlation IDs
- **Circuit Breaking:** Fallback responses
- **Monitoring:** Centralized logging, metrics

**Implementation:**

**1. Dependencies:**
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

**2. Gateway Configuration:**
```java
@SpringBootApplication
@EnableDiscoveryClient
public class ApiGatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(ApiGatewayApplication.class, args);
    }
}
```

**application.yml:**
```yaml
spring:
  application:
    name: api-gateway
  cloud:
    gateway:
      routes:
        - id: rag-query-route
          uri: lb://rag-query-service
          predicates:
            - Path=/api/ask/**
          filters:
            - StripPrefix=1
            - name: RequestRateLimiter
              args:
                redis-rate-limiter:
                  replenishRate: 10
                  burstCapacity: 20
            - name: CircuitBreaker
              args:
                name: ragQueryCircuitBreaker
                fallbackUri: forward:/fallback/rag-query

        - id: document-ingestion-route
          uri: lb://document-ingestion-service
          predicates:
            - Path=/api/documents/**
          filters:
            - StripPrefix=1

        - id: tenant-management-route
          uri: lb://tenant-management-service
          predicates:
            - Path=/api/tenants/**
          filters:
            - StripPrefix=1
            - name: AuthFilter  # Custom filter for authentication

      discovery:
        locator:
          enabled: true
          lower-case-service-id: true

server:
  port: 8080
```

**3. Custom Filters:**

**Authentication Filter:**
```java
@Component
public class AuthenticationFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = exchange.getRequest().getHeaders().getFirst("Authorization");

        if (token == null || !token.startsWith("Bearer ")) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }

        try {
            // Validate JWT
            String jwt = token.substring(7);
            Claims claims = Jwts.parser()
                .setSigningKey("secret")
                .parseClaimsJws(jwt)
                .getBody();

            // Add tenant ID to request header
            String tenantId = claims.get("tenantId", String.class);
            ServerHttpRequest request = exchange.getRequest().mutate()
                .header("X-Tenant-ID", tenantId)
                .build();

            return chain.filter(exchange.mutate().request(request).build());
        } catch (Exception e) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }
    }

    @Override
    public int getOrder() {
        return -1; // Execute before other filters
    }
}
```

**Correlation ID Filter:**
```java
@Component
public class CorrelationIdFilter implements GlobalFilter {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String correlationId = exchange.getRequest().getHeaders()
            .getFirst("X-Correlation-ID");

        if (correlationId == null) {
            correlationId = UUID.randomUUID().toString();
        }

        ServerHttpRequest request = exchange.getRequest().mutate()
            .header("X-Correlation-ID", correlationId)
            .build();

        ServerHttpResponse response = exchange.getResponse();
        response.getHeaders().add("X-Correlation-ID", correlationId);

        return chain.filter(exchange.mutate().request(request).build());
    }
}
```

**4. Fallback Controller:**
```java
@RestController
@RequestMapping("/fallback")
public class FallbackController {

    @PostMapping("/rag-query")
    public ResponseEntity<ErrorResponse> ragQueryFallback() {
        ErrorResponse error = ErrorResponse.builder()
            .status(HttpStatus.SERVICE_UNAVAILABLE.value())
            .message("RAG Query Service is temporarily unavailable. Please try again later.")
            .errorCode("SERVICE_UNAVAILABLE")
            .timestamp(LocalDateTime.now())
            .build();

        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(error);
    }
}
```

**Discussion:**
- API Gateway vs Service Mesh (Istio, Linkerd)
- Performance implications of gateway
- Scaling the gateway
- Security at the edge

### Q13.4: Implement distributed tracing with Spring Cloud Sleuth and Zipkin
**Focus:** Observability in microservices
**Challenge:** Tracing requests across multiple services

**1. Dependencies (add to all services):**
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-sleuth</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-sleuth-zipkin</artifactId>
</dependency>
```

**2. Configuration:**
```yaml
spring:
  sleuth:
    sampler:
      probability: 1.0  # Sample 100% of requests (reduce in prod)
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: web
```

**3. Automatic Instrumentation:**
- Spring Cloud Sleuth automatically:
  - Adds trace ID and span ID to logs
  - Propagates trace context across HTTP calls (RestTemplate, Feign, WebClient)
  - Instruments database calls, messaging (Kafka, RabbitMQ)

**Log Output with Trace IDs:**
```
2026-01-19 12:00:00.123 INFO [rag-query-service,abc123,def456] c.e.r.RagService - Received question
2026-01-19 12:00:00.234 INFO [embedding-service,abc123,ghi789] c.e.e.EmbeddingService - Generating embedding
2026-01-19 12:00:00.345 INFO [vector-search-service,abc123,jkl012] c.e.v.VectorSearchService - Searching vectors
```

**Format:** `[service-name, trace-id, span-id]`

**4. Custom Spans:**
```java
@Service
public class RagService {
    private final Tracer tracer;

    public RagResponse askQuestion(String question) {
        Span customSpan = tracer.nextSpan().name("rag-workflow");

        try (Tracer.SpanInScope ws = tracer.withSpan(customSpan.start())) {
            customSpan.tag("question.length", String.valueOf(question.length()));
            customSpan.tag("tenant.id", getCurrentTenantId());

            // Business logic...

            customSpan.tag("relevant.docs", "3");
            return response;
        } finally {
            customSpan.end();
        }
    }
}
```

**5. Zipkin UI:**
- Access: http://localhost:9411
- Visualize trace timeline
- Identify latency bottlenecks
- Analyze service dependencies

**Example Trace:**
```
Trace ID: abc123
├─ Span: API Gateway (5ms)
├─ Span: RAG Query Service (1200ms)
│  ├─ Span: Call Embedding Service (300ms)
│  │  └─ Span: Azure OpenAI API (280ms)
│  ├─ Span: Call Vector Search Service (100ms)
│  │  └─ Span: pgvector query (80ms)
│  └─ Span: Call LLM Service (800ms)
│     └─ Span: Azure OpenAI API (780ms)
```

**Alternative: OpenTelemetry:**
```xml
<dependency>
    <groupId>io.opentelemetry</groupId>
    <artifactId>opentelemetry-spring-boot-starter</artifactId>
</dependency>
```

**Discussion:**
- Sampling strategies (probability vs rate-limiting)
- Storage backends (Zipkin vs Jaeger vs Tempo)
- Performance overhead of tracing
- Distributed tracing vs logging

### Q13.5: Implement circuit breaker pattern with Resilience4j
**Focus:** Fault tolerance (CRITICAL for microservices)
**Problem:** Cascading failures when embedding service is down

**1. Dependency:**
```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
</dependency>
```

**2. Configuration:**
```yaml
resilience4j:
  circuitbreaker:
    instances:
      embeddingService:
        register-health-indicator: true
        sliding-window-size: 10
        minimum-number-of-calls: 5
        failure-rate-threshold: 50
        wait-duration-in-open-state: 60000
        permitted-number-of-calls-in-half-open-state: 3
        automatic-transition-from-open-to-half-open-enabled: true

  retry:
    instances:
      embeddingService:
        max-attempts: 3
        wait-duration: 1000
        exponential-backoff-multiplier: 2

  bulkhead:
    instances:
      embeddingService:
        max-concurrent-calls: 10
        max-wait-duration: 500

  timelimiter:
    instances:
      embeddingService:
        timeout-duration: 5s
```

**3. Implementation:**

**Service Interface:**
```java
@FeignClient(name = "embedding-service")
public interface EmbeddingServiceClient {
    @PostMapping("/embeddings")
    EmbeddingResponse generateEmbedding(@RequestBody EmbeddingRequest request);
}
```

**Service with Resilience:**
```java
@Service
public class RagService {
    private final EmbeddingServiceClient embeddingClient;

    @CircuitBreaker(name = "embeddingService", fallbackMethod = "generateEmbeddingFallback")
    @Retry(name = "embeddingService")
    @Bulkhead(name = "embeddingService")
    @TimeLimiter(name = "embeddingService")
    public CompletableFuture<float[]> generateEmbedding(String text) {
        return CompletableFuture.supplyAsync(() ->
            embeddingClient.generateEmbedding(new EmbeddingRequest(text)).getEmbedding()
        );
    }

    // Fallback method
    private CompletableFuture<float[]> generateEmbeddingFallback(
        String text,
        Exception ex
    ) {
        log.warn("Embedding service unavailable, using cached embedding for: {}", text);

        // Try to get from cache
        float[] cached = embeddingCache.get(text);
        if (cached != null) {
            return CompletableFuture.completedFuture(cached);
        }

        // Return error if no fallback available
        throw new EmbeddingException("Embedding service unavailable and no cached value", ex);
    }
}
```

**4. Circuit Breaker States:**
- **CLOSED:** Normal operation, requests flow through
- **OPEN:** Too many failures, requests immediately fail (fallback invoked)
- **HALF_OPEN:** Testing if service recovered, allow limited requests

**5. Monitoring:**
```java
@RestController
@RequestMapping("/actuator/health")
public class CircuitBreakerHealthController {

    private final CircuitBreakerRegistry circuitBreakerRegistry;

    @GetMapping("/circuit-breakers")
    public Map<String, String> getCircuitBreakersStatus() {
        return circuitBreakerRegistry.getAllCircuitBreakers()
            .stream()
            .collect(Collectors.toMap(
                CircuitBreaker::getName,
                cb -> cb.getState().toString()
            ));
    }
}
```

**6. Metrics:**
```java
@Configuration
public class MetricsConfig {
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }

    @PostConstruct
    public void registerCircuitBreakerMetrics() {
        circuitBreakerRegistry.getAllCircuitBreakers().forEach(cb -> {
            Metrics.gauge("circuit_breaker_state", cb,
                c -> c.getState() == State.CLOSED ? 0 :
                     c.getState() == State.OPEN ? 1 : 2);
        });
    }
}
```

**Discussion:**
- Circuit breaker vs retry (when to use which?)
- Bulkhead pattern (isolate thread pools)
- Rate limiting
- Timeout strategies

### Q13.6: Implement event-driven architecture with Spring Cloud Stream and Kafka
**Focus:** Async communication
**Scenario:** Document upload triggers async embedding generation

**1. Dependencies:**
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-stream</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-stream-binder-kafka</artifactId>
</dependency>
```

**2. Event Classes:**
```java
public record DocumentUploadedEvent(
    String documentId,
    String title,
    String content,
    String tenantId,
    LocalDateTime uploadedAt
) {}

public record EmbeddingGeneratedEvent(
    String documentId,
    String chunkId,
    float[] embedding,
    String text,
    String tenantId,
    LocalDateTime generatedAt
) {}
```

**3. Document Ingestion Service (Producer):**
```java
@Configuration
public class KafkaConfig {
    @Bean
    public Supplier<Flux<DocumentUploadedEvent>> documentUploadPublisher() {
        return () -> Flux.empty(); // Actual events published via StreamBridge
    }
}

@Service
public class DocumentIngestionService {
    private final StreamBridge streamBridge;

    public void uploadDocument(MultipartFile file, String tenantId) {
        // Parse and store document
        String documentId = UUID.randomUUID().toString();
        String content = extractContent(file);

        // Save to MongoDB
        documentRepository.save(new Document(documentId, file.getOriginalFilename(), content, tenantId));

        // Publish event
        DocumentUploadedEvent event = new DocumentUploadedEvent(
            documentId,
            file.getOriginalFilename(),
            content,
            tenantId,
            LocalDateTime.now()
        );

        streamBridge.send("documentUploaded-out-0", event);
        log.info("Published DocumentUploadedEvent for documentId: {}", documentId);
    }
}
```

**4. Embedding Service (Consumer & Producer):**
```java
@Configuration
public class EmbeddingProcessorConfig {

    @Bean
    public Function<DocumentUploadedEvent, Flux<EmbeddingGeneratedEvent>> processDocument(
        EmbeddingModel embeddingModel
    ) {
        return event -> {
            log.info("Received DocumentUploadedEvent for documentId: {}", event.documentId());

            // Chunk the document
            List<String> chunks = chunkDocument(event.content());

            // Generate embeddings for each chunk
            return Flux.fromIterable(chunks)
                .map(chunk -> {
                    float[] embedding = embeddingModel.embed(chunk);

                    return new EmbeddingGeneratedEvent(
                        event.documentId(),
                        UUID.randomUUID().toString(),
                        embedding,
                        chunk,
                        event.tenantId(),
                        LocalDateTime.now()
                    );
                });
        };
    }
}
```

**5. Vector Search Service (Consumer):**
```java
@Configuration
public class VectorIndexerConfig {

    @Bean
    public Consumer<EmbeddingGeneratedEvent> indexEmbedding(
        VectorStore vectorStore
    ) {
        return event -> {
            log.info("Received EmbeddingGeneratedEvent for documentId: {}", event.documentId());

            // Index in vector database
            vectorStore.store(new DocumentEmbedding(
                event.documentId(),
                event.text(),
                event.embedding()
            ));

            log.info("Indexed embedding for chunk: {}", event.chunkId());
        };
    }
}
```

**6. Configuration (application.yml):**
```yaml
spring:
  cloud:
    stream:
      bindings:
        # Document Ingestion Service
        documentUploaded-out-0:
          destination: document-uploaded
          producer:
            partition-key-expression: headers['tenantId']

        # Embedding Service
        processDocument-in-0:
          destination: document-uploaded
          group: embedding-service
          consumer:
            max-attempts: 3
        processDocument-out-0:
          destination: embedding-generated
          producer:
            partition-key-expression: headers['tenantId']

        # Vector Search Service
        indexEmbedding-in-0:
          destination: embedding-generated
          group: vector-search-service

      kafka:
        binder:
          brokers: localhost:9092
        bindings:
          processDocument-in-0:
            consumer:
              enable-dlq: true
              dlq-name: document-uploaded-dlq
```

**7. Error Handling (Dead Letter Queue):**
```java
@Configuration
public class ErrorHandlerConfig {

    @Bean
    public Consumer<Message<DocumentUploadedEvent>> handleDlq() {
        return message -> {
            log.error("Message sent to DLQ: {}", message.getPayload());

            // Store in error table for manual retry
            ErrorEvent errorEvent = new ErrorEvent(
                message.getPayload().documentId(),
                message.getHeaders().toString(),
                LocalDateTime.now()
            );
            errorRepository.save(errorEvent);
        };
    }
}
```

**Discussion:**
- Event sourcing vs event streaming
- Exactly-once vs at-least-once delivery
- Kafka vs RabbitMQ vs AWS SNS/SQS
- Schema evolution (Avro, Protocol Buffers)
- Saga pattern for distributed transactions

### Q13.7: Implement centralized configuration with Spring Cloud Config
**Focus:** Configuration management
**Architecture:**

**1. Config Server:**
```java
@SpringBootApplication
@EnableConfigServer
public class ConfigServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(ConfigServerApplication.class, args);
    }
}
```

**application.yml (Config Server):**
```yaml
server:
  port: 8888

spring:
  cloud:
    config:
      server:
        git:
          uri: https://github.com/yourorg/config-repo
          default-label: main
          search-paths: '{application}'
        native:
          search-locations: classpath:/config
```

**2. Config Repository Structure:**
```
config-repo/
├── rag-query-service.yml
├── rag-query-service-dev.yml
├── rag-query-service-prod.yml
├── embedding-service.yml
└── application.yml  # Common config for all services
```

**rag-query-service-prod.yml:**
```yaml
spring:
  ai:
    azure:
      openai:
        api-key: ${AZURE_OPENAI_API_KEY}
        endpoint: ${AZURE_OPENAI_ENDPOINT}

rag:
  vector-store:
    type: pgvector
    top-k: 3
  llm:
    temperature: 0.7
    max-tokens: 500

logging:
  level:
    com.example.rag: INFO
```

**3. Config Client (RAG Query Service):**
```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-config</artifactId>
</dependency>
```

**bootstrap.yml (executes before application.yml):**
```yaml
spring:
  application:
    name: rag-query-service
  cloud:
    config:
      uri: http://localhost:8888
      fail-fast: true
  profiles:
    active: prod
```

**4. Refreshing Configuration without Restart:**
```java
@RestController
@RefreshScope  // Allows dynamic config refresh
public class AskController {

    @Value("${rag.vector-store.top-k}")
    private int topK;

    @PostMapping("/ask")
    public RagResponse ask(@RequestBody QuestionRequest request) {
        // Uses latest topK value after refresh
    }
}
```

**Trigger refresh:**
```bash
curl -X POST http://localhost:8080/actuator/refresh
```

**5. Encryption (for secrets):**
```yaml
# Encrypted value in config file
spring:
  ai:
    azure:
      openai:
        api-key: '{cipher}AQA1Fz...'  # Encrypted with Config Server key
```

**6. Alternatives:**
- **Kubernetes ConfigMaps/Secrets:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-query-config
data:
  application.yml: |
    rag:
      vector-store:
        top-k: 3
```

- **HashiCorp Vault:**
```yaml
spring:
  cloud:
    vault:
      uri: https://vault.example.com
      token: ${VAULT_TOKEN}
      kv:
        enabled: true
```

**Discussion:**
- Git vs database vs Vault for config storage
- Config versioning and rollback
- Security (encryption at rest, in transit)
- Config per environment vs per service

### Q13.8: Design a microservices deployment strategy on Kubernetes
**Focus:** Container orchestration
**Architecture:**

**1. Dockerfile (per service):**
```dockerfile
FROM eclipse-temurin:17-jre-alpine AS runtime
WORKDIR /app
COPY target/rag-query-service.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-XX:+UseContainerSupport", "-XX:MaxRAMPercentage=75.0", "-jar", "app.jar"]
```

**2. Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-query-service
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-query-service
  template:
    metadata:
      labels:
        app: rag-query-service
        version: v1
    spec:
      containers:
      - name: rag-query-service
        image: yourregistry.azurecr.io/rag-query-service:1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "prod"
        - name: AZURE_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: azure-openai-secret
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /actuator/health/liveness
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
```

**3. Service (for internal communication):**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-query-service
  namespace: rag-system
spec:
  selector:
    app: rag-query-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

**4. Ingress (external access):**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.rag-system.com
    secretName: rag-tls-secret
  rules:
  - host: api.rag-system.com
    http:
      paths:
      - path: /ask
        pathType: Prefix
        backend:
          service:
            name: rag-query-service
            port:
              number: 80
```

**5. ConfigMap:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-query-config
  namespace: rag-system
data:
  application.yml: |
    rag:
      vector-store:
        type: pgvector
        top-k: 3
    logging:
      level:
        com.example.rag: INFO
```

**6. Secret:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: azure-openai-secret
  namespace: rag-system
type: Opaque
data:
  api-key: <base64-encoded-key>
  endpoint: <base64-encoded-endpoint>
```

**7. HorizontalPodAutoscaler:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-query-service-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-query-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**8. Service Mesh (Istio):**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: rag-query-service
spec:
  hosts:
  - rag-query-service
  http:
  - match:
    - headers:
        X-Tenant-ID:
          exact: premium-tenant
    route:
    - destination:
        host: rag-query-service
        subset: premium
  - route:
    - destination:
        host: rag-query-service
        subset: standard
```

**Discussion:**
- Blue-green vs rolling vs canary deployments
- Resource quotas and limits
- Pod disruption budgets
- Init containers for migrations
- Sidecar patterns (logging, monitoring)

---

## Bonus Questions

### B1: Explain the trade-offs between OpenAI and open-source models
**Focus:** LLM ecosystem knowledge
**Comparison:**

| Aspect | OpenAI/Azure OpenAI | Open Source (Llama 2, Mistral) |
|--------|---------------------|--------------------------------|
| **Quality** | State-of-the-art (GPT-4) | Good, improving rapidly |
| **Cost** | Pay per token ($$$) | Infrastructure cost (GPU) |
| **Latency** | API overhead | Low (on-prem) |
| **Privacy** | Data sent to OpenAI | Complete data control |
| **Customization** | Limited fine-tuning | Full control, fine-tuning |
| **Compliance** | Vendor lock-in | Easier compliance |
| **Deployment** | SaaS, easy | Complex infrastructure |

**When to Use Open Source:**
- Strict data residency requirements
- High volume (cost optimization)
- Need for model customization
- Offline/edge deployment

**When to Use OpenAI:**
- Best quality needed
- Fast prototyping
- Low volume
- No ML expertise in-house

### B2: What is the future of RAG and Agentic AI?
**Focus:** Industry trends awareness
**Trends:**

**RAG Evolution:**
- **Hybrid search:** Combine vector + keyword (BM25) for better recall
- **Graph RAG:** Knowledge graphs + vector search
- **Multi-modal RAG:** Images, tables, charts as context
- **Adaptive retrieval:** Dynamic K based on query complexity
- **Self-RAG:** Model decides when to retrieve

**Agentic AI Evolution:**
- **Multi-agent systems:** Specialized agents collaborating
- **Long-running agents:** Persistent state, complex workflows
- **Human-in-the-loop:** Agent proposes, human approves
- **Tool learning:** Agents learn new tools dynamically
- **Autonomous code generation:** Agents write and deploy code

**Example Multi-Agent System:**
- **Planner Agent:** Decomposes task
- **Researcher Agent:** Gathers information (RAG)
- **Coder Agent:** Writes code
- **Reviewer Agent:** Checks quality
- **Executor Agent:** Runs and validates

---

## Interview Process Recommendations

### Technical Discussion (2-3 hours)

**Part 1: Codebase Walkthrough (30 min)**
- Candidate explains RAG workflow
- Dives into one component deeply (e.g., vector similarity)
- Discusses design decisions and trade-offs

**Part 2: Architecture Design (45 min)**
- Multi-tenant RAG system design
- Scalability and cost considerations
- Cloud architecture (AWS/Azure)

**Part 3: Hands-On Coding (45 min)**
- Implement a feature (e.g., document deduplication, streaming)
- Extend codebase with new functionality
- Write tests for implementation

**Part 4: Agentic AI Scenario (30 min)**
- Design an agentic workflow for a business problem
- Discuss tool selection and orchestration
- Error handling and reliability

### Evaluation Criteria

**Technical Depth (40%):**
- Java and Spring Boot expertise
- Understanding of LLMs, embeddings, RAG
- System design skills

**Production Readiness (30%):**
- Security awareness
- Scalability considerations
- Monitoring and observability

**Architecture Thinking (20%):**
- Multi-tenancy design
- Cloud-native patterns
- Trade-off analysis

**Communication (10%):**
- Clear explanations
- Thoughtful questioning
- Collaboration mindset

---

## Preparation Tips for Candidate

1. **Deep dive into Spring AI documentation**
2. **Understand RAG workflow end-to-end**
3. **Practice designing multi-tenant systems**
4. **Review Agentic AI concepts and frameworks**
5. **Prepare to write code during interview**
6. **Study vector databases (pgvector, Pinecone, Weaviate)**
7. **Understand LLM fundamentals (embeddings, tokens, prompts)**
8. **Review AWS services for AI (Bedrock, OpenSearch)**
9. **Practice system design for scale**
10. **Be ready to discuss production challenges**

---

**Good luck with the interview!**
