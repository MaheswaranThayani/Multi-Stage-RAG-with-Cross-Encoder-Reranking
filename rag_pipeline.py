# """
# RAG Pipeline using ChromaDB with persistent SQLite-backed storage.

# This module provides a comprehensive Retrieval-Augmented Generation pipeline with:
# - Document chunking with configurable parameters
# - Embedding generation using sentence-transformers
# - ChromaDB ingestion with persistent SQLite storage
# - Semantic search functionality
# - Keyword search functionality
# - Hybrid search combining both approaches
# """

# import os
# import re
# import sqlite3
# import tempfile
# from typing import List, Dict, Tuple, Optional, Any
# import numpy as np
# from dataclasses import dataclass
# import logging

# # Text processing and embeddings
# from sentence_transformers import SentenceTransformer
# import torch
# from sentence_transformers import util

# # Document processing
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ChromaDB
# import chromadb
# from chromadb.config import Settings
# from chromadb.utils import embedding_functions

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class ChunkConfig:
#     """Configuration for document chunking."""
#     chunk_size: int = 1000
#     chunk_overlap: int = 200
#     separators: List[str] = None
#     length_function: callable = len
    
#     def __post_init__(self):
#         if self.separators is None:
#             self.separators = ["\n\n", "\n", " ", ""]


# @dataclass
# class EmbeddingConfig:
#     """Configuration for embedding generation."""
#     model_name: str = "all-MiniLM-L6-v2"
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     batch_size: int = 32
#     normalize_embeddings: bool = True


# @dataclass
# class ChromaConfig:
#     """Configuration for ChromaDB."""
#     persist_directory: str = "./chroma_db"
#     collection_name: str = "documents"
#     distance_metric: str = "cosine"
#     use_sqlite: bool = True


# class DocumentChunker:
#     """Handles document chunking with various strategies."""
    
#     def __init__(self, config: ChunkConfig):
#         self.config = config
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=config.chunk_size,
#             chunk_overlap=config.chunk_overlap,
#             separators=config.separators,
#             length_function=config.length_function
#         )
    
#     def chunk_text(self, text: str) -> List[Dict[str, Any]]:
#         """
#         Split text into chunks with metadata.
        
#         Args:
#             text: Input text to chunk
            
#         Returns:
#             List of dictionaries containing chunk text and metadata
#         """
#         if not text.strip():
#             return []
        
#         chunks = self.splitter.split_text(text)
        
#         chunked_docs = []
#         for i, chunk in enumerate(chunks):
#             chunked_docs.append({
#                 "id": f"chunk_{i}",
#                 "text": chunk,
#                 "chunk_index": i,
#                 "word_count": len(chunk.split()),
#                 "char_count": len(chunk)
#             })
        
#         return chunked_docs
    
#     def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Chunk multiple documents with metadata preservation.
        
#         Args:
#             documents: List of documents with 'text' and optional metadata
            
#         Returns:
#             List of chunked documents
#         """
#         all_chunks = []
        
#         for doc_idx, document in enumerate(documents):
#             text = document.get("text", "")
#             base_metadata = document.get("metadata", {})
            
#             chunks = self.chunk_text(text)
            
#             for chunk in chunks:
#                 chunk_metadata = {
#                     **base_metadata,
#                     "document_index": doc_idx,
#                     "chunk_index": chunk["chunk_index"],
#                     "word_count": chunk["word_count"],
#                     "char_count": chunk["char_count"]
#                 }
                
#                 all_chunks.append({
#                     "id": f"doc_{doc_idx}_chunk_{chunk['chunk_index']}",
#                     "text": chunk["text"],
#                     "metadata": chunk_metadata
#                 })
        
#         return all_chunks


# class EmbeddingGenerator:
#     """Handles embedding generation using sentence-transformers."""
    
#     def __init__(self, config: EmbeddingConfig):
#         self.config = config
#         self.model = SentenceTransformer(config.model_name, device=config.device)
#         logger.info(f"Loaded embedding model: {config.model_name} on {config.device}")
    
#     def generate_embeddings(self, texts: List[str]) -> np.ndarray:
#         """
#         Generate embeddings for a list of texts.
        
#         Args:
#             texts: List of texts to embed
            
#         Returns:
#             Numpy array of embeddings
#         """
#         if not texts:
#             return np.array([])
        
#         embeddings = self.model.encode(
#             texts,
#             batch_size=self.config.batch_size,
#             normalize_embeddings=self.config.normalize_embeddings,
#             show_progress_bar=True
#         )
        
#         return embeddings
    
#     def generate_single_embedding(self, text: str) -> np.ndarray:
#         """
#         Generate embedding for a single text.
        
#         Args:
#             text: Text to embed
            
#         Returns:
#             Embedding vector
#         """
#         return self.model.encode(
#             text,
#             normalize_embeddings=self.config.normalize_embeddings
#         )


# class ChromaDBManager:
#     """Manages ChromaDB operations with persistent SQLite storage."""
    
#     def __init__(self, config: ChromaConfig):
#         self.config = config
        
#         # Ensure persist directory exists
#         os.makedirs(config.persist_directory, exist_ok=True)
        
#         # Initialize ChromaDB client with SQLite backend
#         if config.use_sqlite:
#             self.client = chromadb.PersistentClient(
#                 path=config.persist_directory,
#                 settings=Settings(
#                     anonymized_telemetry=False,
#                     allow_reset=True
#                 )
#             )
#         else:
#             self.client = chromadb.Client()
        
#         # Get or create collection
#         try:
#             self.collection = self.client.get_collection(name=config.collection_name)
#             logger.info(f"Connected to existing collection: {config.collection_name}")
#         except:
#             self.collection = self.client.create_collection(
#                 name=config.collection_name,
#                 metadata={"hnsw:space": config.distance_metric}
#             )
#             logger.info(f"Created new collection: {config.collection_name}")
    
#     def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
#         """
#         Add documents to ChromaDB collection.
        
#         Args:
#             documents: List of documents with 'id', 'text', and 'metadata'
#             embeddings: Embedding vectors for the documents
#         """
#         if not documents or embeddings.size == 0:
#             return
        
#         ids = [doc["id"] for doc in documents]
#         texts = [doc["text"] for doc in documents]
#         metadatas = [doc["metadata"] for doc in documents]
        
#         # Convert embeddings to list format if needed
#         if isinstance(embeddings, np.ndarray):
#             embeddings = embeddings.tolist()
        
#         self.collection.add(
#             ids=ids,
#             documents=texts,
#             metadatas=metadatas,
#             embeddings=embeddings
#         )
        
#         logger.info(f"Added {len(documents)} documents to ChromaDB")
    
#     def semantic_search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
#         """
#         Perform semantic search using embeddings.
        
#         Args:
#             query_embedding: Embedding of the query
#             n_results: Number of results to return
            
#         Returns:
#             Dictionary containing search results
#         """
#         if isinstance(query_embedding, np.ndarray):
#             query_embedding = query_embedding.tolist()
        
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_results,
#             include=["documents", "metadatas", "distances"]
#         )
        
#         return results
    
#     def keyword_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
#         """
#         Perform keyword search using ChromaDB's where clause.
        
#         Args:
#             query: Query string
#             n_results: Number of results to return
            
#         Returns:
#             List of matching documents
#         """
#         # Get all documents and filter them based on keyword matching
#         all_docs = self.collection.get(include=["documents", "metadatas"])
        
#         if not all_docs["documents"]:
#             return []
        
#         # Simple keyword matching - can be enhanced with more sophisticated methods
#         query_terms = query.lower().split()
#         scored_docs = []
        
#         for i, doc in enumerate(all_docs["documents"]):
#             doc_lower = doc.lower()
#             score = 0
            
#             # Count term frequency
#             for term in query_terms:
#                 term_count = doc_lower.count(term)
#                 if term_count > 0:
#                     score += term_count
            
#             if score > 0:
#                 scored_docs.append({
#                     "document": doc,
#                     "metadata": all_docs["metadatas"][i],
#                     "score": score,
#                     "id": all_docs["ids"][i]
#                 })
        
#         # Sort by score and return top results
#         scored_docs.sort(key=lambda x: x["score"], reverse=True)
#         return scored_docs[:n_results]
    
#     def hybrid_search(self, query: str, query_embedding: np.ndarray, 
#                      semantic_weight: float = 0.7, n_results: int = 5) -> List[Dict[str, Any]]:
#         """
#         Perform hybrid search combining semantic and keyword search.
        
#         Args:
#             query: Query string for keyword search
#             query_embedding: Query embedding for semantic search
#             semantic_weight: Weight for semantic search (0-1)
#             n_results: Number of results to return
            
#         Returns:
#             List of hybrid search results
#         """
#         # Get semantic search results
#         semantic_results = self.semantic_search(query_embedding, n_results * 2)
        
#         # Get keyword search results
#         keyword_results = self.keyword_search(query, n_results * 2)
        
#         # Combine and re-score results
#         combined_scores = {}
        
#         # Process semantic results
#         for i, (doc, metadata, distance) in enumerate(zip(
#             semantic_results["documents"][0],
#             semantic_results["metadatas"][0],
#             semantic_results["distances"][0]
#         )):
#             doc_id = semantic_results["ids"][0][i]
#             # Convert distance to similarity score (lower distance = higher similarity)
#             semantic_score = 1 - distance
#             combined_scores[doc_id] = {
#                 "document": doc,
#                 "metadata": metadata,
#                 "semantic_score": semantic_score,
#                 "keyword_score": 0.0,
#                 "combined_score": semantic_score * semantic_weight
#             }
        
#         # Process keyword results and combine
#         for result in keyword_results:
#             doc_id = result["id"]
#             keyword_score = min(result["score"] / 10.0, 1.0)  # Normalize keyword score
            
#             if doc_id in combined_scores:
#                 combined_scores[doc_id]["keyword_score"] = keyword_score
#                 combined_scores[doc_id]["combined_score"] = (
#                     combined_scores[doc_id]["semantic_score"] * semantic_weight +
#                     keyword_score * (1 - semantic_weight)
#                 )
#             else:
#                 combined_scores[doc_id] = {
#                     "document": result["document"],
#                     "metadata": result["metadata"],
#                     "semantic_score": 0.0,
#                     "keyword_score": keyword_score,
#                     "combined_score": keyword_score * (1 - semantic_weight)
#                 }
        
#         # Sort by combined score and return top results
#         sorted_results = sorted(
#             combined_scores.values(),
#             key=lambda x: x["combined_score"],
#             reverse=True
#         )
        
#         return sorted_results[:n_results]
    
#     def get_collection_stats(self) -> Dict[str, Any]:
#         """Get statistics about the collection."""
#         count = self.collection.count()
#         return {
#             "document_count": count,
#             "collection_name": self.config.collection_name,
#             "persist_directory": self.config.persist_directory
#         }
    
#     def reset_collection(self):
#         """Reset the collection (remove all documents)."""
#         self.client.delete_collection(name=self.config.collection_name)
#         self.collection = self.client.create_collection(
#             name=self.config.collection_name,
#             metadata={"hnsw:space": self.config.distance_metric}
#         )
#         logger.info(f"Reset collection: {self.config.collection_name}")


# class RAGPipeline:
#     """Main RAG Pipeline class that orchestrates all components."""
    
#     def __init__(self, 
#                  chunk_config: Optional[ChunkConfig] = None,
#                  embedding_config: Optional[EmbeddingConfig] = None,
#                  chroma_config: Optional[ChromaConfig] = None):
#         """
#         Initialize the RAG pipeline.
        
#         Args:
#             chunk_config: Configuration for document chunking
#             embedding_config: Configuration for embedding generation
#             chroma_config: Configuration for ChromaDB
#         """
#         self.chunk_config = chunk_config or ChunkConfig()
#         self.embedding_config = embedding_config or EmbeddingConfig()
#         self.chroma_config = chroma_config or ChromaConfig()
        
#         # Initialize components
#         self.chunker = DocumentChunker(self.chunk_config)
#         self.embedder = EmbeddingGenerator(self.embedding_config)
#         self.chroma_manager = ChromaDBManager(self.chroma_config)
        
#         logger.info("RAG Pipeline initialized successfully")
    
#     def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Ingest documents into the RAG pipeline.
        
#         Args:
#             documents: List of documents with 'text' and optional 'metadata'
            
#         Returns:
#             Dictionary with ingestion statistics
#         """
#         if not documents:
#             return {"error": "No documents provided"}
        
#         # Chunk documents
#         chunked_docs = self.chunker.chunk_documents(documents)
        
#         if not chunked_docs:
#             return {"error": "No chunks created from documents"}
        
#         # Generate embeddings
#         texts = [doc["text"] for doc in chunked_docs]
#         embeddings = self.embedder.generate_embeddings(texts)
        
#         # Add to ChromaDB
#         self.chroma_manager.add_documents(chunked_docs, embeddings)
        
#         stats = self.chroma_manager.get_collection_stats()
        
#         return {
#             "documents_processed": len(documents),
#             "chunks_created": len(chunked_docs),
#             "embeddings_generated": len(embeddings),
#             "total_documents_in_db": stats["document_count"]
#         }
    
#     def search(self, 
#                query: str, 
#                search_type: str = "hybrid",
#                n_results: int = 5,
#                semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
#         """
#         Search for relevant documents.
        
#         Args:
#             query: Search query
#             search_type: Type of search ('semantic', 'keyword', 'hybrid')
#             n_results: Number of results to return
#             semantic_weight: Weight for semantic search in hybrid mode
            
#         Returns:
#             List of search results
#         """
#         if not query.strip():
#             return []
        
#         # Generate query embedding
#         query_embedding = self.embedder.generate_single_embedding(query)
        
#         # Perform search based on type
#         if search_type == "semantic":
#             results = self.chroma_manager.semantic_search(query_embedding, n_results)
#             # Convert to consistent format
#             formatted_results = []
#             for i, (doc, metadata, distance) in enumerate(zip(
#                 results["documents"][0],
#                 results["metadatas"][0],
#                 results["distances"][0]
#             )):
#                 formatted_results.append({
#                     "document": doc,
#                     "metadata": metadata,
#                     "score": 1 - distance,  # Convert distance to similarity
#                     "search_type": "semantic"
#                 })
#             return formatted_results
        
#         elif search_type == "keyword":
#             results = self.chroma_manager.keyword_search(query, n_results)
#             for result in results:
#                 result["search_type"] = "keyword"
#             return results
        
#         elif search_type == "hybrid":
#             results = self.chroma_manager.hybrid_search(
#                 query, query_embedding, semantic_weight, n_results
#             )
#             for result in results:
#                 result["search_type"] = "hybrid"
#             return results
        
#         else:
#             raise ValueError(f"Invalid search type: {search_type}")
    
#     def get_stats(self) -> Dict[str, Any]:
#         """Get pipeline statistics."""
#         stats = self.chroma_manager.get_collection_stats()
#         return {
#             **stats,
#             "chunk_config": {
#                 "chunk_size": self.chunk_config.chunk_size,
#                 "chunk_overlap": self.chunk_config.chunk_overlap
#             },
#             "embedding_model": self.embedding_config.model_name,
#             "embedding_device": self.embedding_config.device
#         }
    
#     def reset(self):
#         """Reset the entire pipeline."""
#         self.chroma_manager.reset_collection()
#         logger.info("RAG Pipeline reset successfully")


# # Utility functions for integration with existing applications
# def create_simple_pipeline(persist_directory: str = "./chroma_db") -> RAGPipeline:
#     """
#     Create a simple RAG pipeline with default configurations.
    
#     Args:
#         persist_directory: Directory for persistent storage
        
#     Returns:
#         Configured RAGPipeline instance
#     """
#     chroma_config = ChromaConfig(persist_directory=persist_directory)
#     return RAGPipeline(chroma_config=chroma_config)


# def extract_text_from_pdf(pdf_path: str) -> str:
#     """
#     Extract text from a PDF file.
    
#     Args:
#         pdf_path: Path to the PDF file
        
#     Returns:
#         Extracted text
#     """
#     try:
#         from PyPDF2 import PdfReader
        
#         with open(pdf_path, 'rb') as file:
#             pdf_reader = PdfReader(file)
#             text = ""
            
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
            
#             return text
#     except Exception as e:
#         logger.error(f"Error extracting text from PDF: {e}")
#         return ""


# def prepare_documents_from_text(text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
#     """
#     Prepare documents for ingestion from raw text.
    
#     Args:
#         text: Raw text content
#         metadata: Optional metadata to attach
        
#     Returns:
#         List of document dictionaries
#     """
#     if not text.strip():
#         return []
    
#     return [{
#         "text": text,
#         "metadata": metadata or {}
#     }]


# # Example usage
# if __name__ == "__main__":
#     # Example of how to use the RAG pipeline
    
#     # 1. Create pipeline
#     pipeline = create_simple_pipeline("./example_chroma_db")
    
#     # 2. Prepare documents (example with text)
#     sample_text = """
#     Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines
#     that can perform tasks that typically require human intelligence. Machine learning is a subset of AI
#     that enables systems to learn and improve from experience without being explicitly programmed.
    
#     Deep learning is a type of machine learning that uses neural networks with multiple layers to model
#     and understand complex patterns in data. Natural language processing (NLP) is another important area
#     of AI that focuses on enabling computers to understand, interpret, and generate human language.
#     """
    
#     documents = prepare_documents_from_text(
#         sample_text, 
#         {"source": "example", "topic": "AI"}
#     )
    
#     # 3. Ingest documents
#     result = pipeline.ingest_documents(documents)
#     print("Ingestion result:", result)
    
#     # 4. Search examples
#     queries = [
#         "What is artificial intelligence?",
#         "machine learning algorithms",
#         "deep learning neural networks"
#     ]
    
#     for query in queries:
#         print(f"\nQuery: {query}")
        
#         # Semantic search
#         semantic_results = pipeline.search(query, search_type="semantic", n_results=2)
#         print("Semantic results:", len(semantic_results))
        
#         # Keyword search
#         keyword_results = pipeline.search(query, search_type="keyword", n_results=2)
#         print("Keyword results:", len(keyword_results))
        
#         # Hybrid search
#         hybrid_results = pipeline.search(query, search_type="hybrid", n_results=2)
#         print("Hybrid results:", len(hybrid_results))
    
#     # 5. Get pipeline stats
#     stats = pipeline.get_stats()
#     print("\nPipeline stats:", stats)
