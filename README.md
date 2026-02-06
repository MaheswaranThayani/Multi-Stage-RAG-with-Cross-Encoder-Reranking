Multi-Stage RAG: Hybrid Retrieval & Cross-Encoder Re-ranking
This repository contains a production-grade Retrieval-Augmented Generation (RAG) pipeline designed to handle complex document QA with high precision. Moving beyond simple vector search, this implementation utilizes a multi-stage architecture to ensure the most relevant context is provided to the LLM, significantly reducing hallucinations.

🚀 Key Technical Features
1. Multi-Stage Retrieval Pipeline
Stage 1: Hybrid Search (Ensemble): Combines Semantic Search (ChromaDB + HuggingFace Embeddings) with Keyword Search (BM25) to capture both conceptual meaning and exact term matching.

Stage 2: Neural Re-ranking: Implements a Cross-Encoder (ms-marco-MiniLM) to re-score the top-k retrieved documents, ensuring the most contextually relevant chunks are prioritized for the LLM.

Stage 3: Contextual Generation: Leverages Flan-T5 with a strict anti-hallucination prompt to generate answers derived solely from the provided documentation.

2. Conversational Intelligence
Query Condensation: Includes a standalone question generator that utilizes chat history to rephrase follow-up queries, making the retriever "context-aware" across multiple turns of conversation.

3. Engineering Best Practices
Resource Management: Optimized for performance using Python Context Managers and Streamlit’s @st.cache_resource to handle model loading and UI states efficiently.

Source Attribution: Provides transparent citations with page-level metadata extraction from PDF sources.

🛠️ Tech Stack
Framework: LangChain

LLM: Google Flan-T5 (Large)

Vector Database: ChromaDB

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Re-ranker: Cross-Encoder (MS-MARCO)

UI: Streamlit