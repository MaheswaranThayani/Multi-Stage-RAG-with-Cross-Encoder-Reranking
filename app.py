# import streamlit as st
# import os
# import re
# import html
# from typing import List
# from PyPDF2 import PdfReader

# # LangChain & Search imports
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever

# # 1. THE CORE: Basic building blocks like Document and BaseRetriever
# from langchain_core.documents import Document

# # 2. THE COMMUNITY: Third-party tools like BM25 and Chroma
# from sentence_transformers import CrossEncoder

# # --- CONFIG & INITIALIZATION ---
# st.set_page_config(page_title="Advanced History Bot", page_icon="🏛️", layout="wide")

# # CSS for better UI and source highlighting
# st.markdown("""
#     <style>
#     .source-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #ff4b4b; }
#     .page-label { font-weight: bold; color: #ff4b4b; margin-bottom: 5px; display: block; }
#     </style>
# """, unsafe_allow_html=True)

# # --- HELPER FUNCTIONS ---

# def load_pdf_with_metadata(file_path: str) -> List[Document]:
#     """Extracts text from PDF and preserves page numbers in metadata."""
#     docs = []
#     if os.path.exists(file_path):
#         reader = PdfReader(file_path)
#         for i, page in enumerate(reader.pages):
#             text = page.extract_text()
#             if text:
#                 # Store page number (i+1) in metadata
#                 docs.append(Document(page_content=text, metadata={"page": i + 1, "source": file_path}))
#     return docs

# @st.cache_resource
# def load_models():
#     """Load LLM, Embeddings, and Re-ranker once."""
#     from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    
#     # LLM (Flan-T5 for local QA)
#     model_id = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#     llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    
#     # Embeddings for Vector Search
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
#     # Cross-Encoder for Re-ranking
#     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
#     return llm_pipeline, embeddings, reranker

# # --- DATA PROCESSING ---

# if "initialized" not in st.session_state:
#     with st.spinner("Initializing Knowledge Base (Hybrid Search + Re-ranker)..."):
#         llm, embeddings, reranker = load_models()
        
#         # 1. Load Data
#         pdf_path = os.path.join("Data", "Sri_lanka_history.pdf")
#         raw_docs = load_pdf_with_metadata(pdf_path)
        
#         if raw_docs:
#             # 2. Setup Vector Store (Dense)
#             vectorstore = Chroma.from_documents(raw_docs, embeddings)
#             vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
#             # 3. Setup BM25 (Sparse/Keyword)
#             bm25_retriever = BM25Retriever.from_documents(raw_docs)
#             bm25_retriever.k = 10
            
#             # 4. Hybrid Ensemble
#             ensemble_retriever = EnsembleRetriever(
#                 retrievers=[bm25_retriever, vector_retriever], 
#                 weights=[0.5, 0.5]
#             )
            
#             st.session_state.llm = llm
#             st.session_state.reranker = reranker
#             st.session_state.retriever = ensemble_retriever
#             st.session_state.initialized = True
#         else:
#             st.error("History PDF not found in /Data folder.")

# # --- CHAT UI ---

# st.title("🏛️ Sri Lankan History AI")
# st.caption("Hybrid Search + BM25 + Cross-Encoder Re-ranking")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "sources" in msg:
#             with st.expander("View Sources"):
#                 for s in msg["sources"]:
#                     st.markdown(f"<div class='source-box'><span class='page-label'>Page {s['page']}</span>{s['text']}</div>", unsafe_allow_html=True)

# query = st.chat_input("Ask about Sri Lankan history...")

# if query and "retriever" in st.session_state:
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     with st.chat_message("assistant"):
#         with st.status("Searching & Re-ranking...", expanded=False) as status:
#             # Step 1: Hybrid Retrieval
#             initial_docs = st.session_state.retriever.invoke(query)
            
#             # Step 2: Re-ranking
#             pairs = [[query, doc.page_content] for doc in initial_docs]
#             scores = st.session_state.reranker.predict(pairs)
#             reranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            
#             # Keep top 3 most relevant chunks
#             final_docs = [doc for doc, score in reranked[:3]]
#             status.update(label="Generating Answer...", state="running")
            
#             # Step 3: Generation
#             context = "\n\n".join([f"[Source Page {d.metadata['page']}]: {d.page_content}" for d in final_docs])
#             prompt = f"Answer the question using the history context provided. If not in context, say you don't know.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
            
#             response = st.session_state.llm(prompt, max_new_tokens=300)
#             answer = response[0]['generated_text']
#             status.update(label="Done!", state="complete")

#         st.markdown(answer)
        
#         # Display Citations
#         sources_metadata = [{"page": d.metadata['page'], "text": d.page_content[:300] + "..."} for d in final_docs]
#         with st.expander("View Sources"):
#             for s in sources_metadata:
#                 st.markdown(f"<div class='source-box'><span class='page-label'>Page {s['page']}</span>{s['text']}</div>", unsafe_allow_html=True)

#     # Save to history
#     st.session_state.messages.append({
#         "role": "assistant", 
#         "content": answer, 
#         "sources": sources_metadata
#     })
import streamlit as st
import os
from PyPDF2 import PdfReader

# --- STICKING WITH YOUR WORKING IMPORTS ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# Using classic for the compression/rerank as well
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# --- CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Sri Lankan History AI", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_models():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    
    # LLM (Flan-T5 for local QA)
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    return llm_pipeline, embeddings, reranker

# --- DATA PROCESSING ---
if "initialized" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        llm, embeddings, reranker = load_models()
        
        pdf_path = os.path.join("Data", "Sri_lanka_history.pdf")
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            raw_docs = [Document(page_content=p.extract_text(), metadata={"page": i+1}) 
                        for i, p in enumerate(reader.pages) if p.extract_text()]
            
            # Hybrid Search Setup
            vectorstore = Chroma.from_documents(raw_docs, embeddings)
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            bm25_retriever = BM25Retriever.from_documents(raw_docs)
            bm25_retriever.k = 10
            
            # 1. Hybrid Ensemble
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
            )
            
            st.session_state.llm = llm
            st.session_state.reranker = reranker
            st.session_state.retriever = ensemble_retriever
            st.session_state.messages = [] 
            st.session_state.initialized = True
        else:
            st.error("PDF not found.")

# --- CHAT UI ---
st.title("pdf Question Answering Bot")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about Sri Lankan history...")

if query and "retriever" in st.session_state:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("Analyzing Question & History...", expanded=False) as status:
            
            # 1. STANDALONE QUESTION GENERATOR (New & Fixed)
            # This step converts "Who is the ruler of this?" -> "Who is the ruler of Anuradhapura?"
            if len(st.session_state.messages) > 1:
                # We take the last few turns of history
                history_turns = st.session_state.messages[-3:-1] 
                chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_turns])
                
                condense_prompt = (
                    f"Given the following conversation and a follow-up question, "
                    f"rephrase the follow-up question to be a STANDALONE question "
                    f"that can be searched in a document. Do NOT answer it.\n\n"
                    f"Chat History:\n{chat_history_text}\n\n"
                    f"Follow-up Question: {query}\n"
                    f"Standalone Question:"
                )
                # Generate the optimized search query
                search_query_response = st.session_state.llm(condense_prompt, max_new_tokens=50)
                search_query = search_query_response[0]['generated_text'].strip()
            else:
                search_query = query
            
            # 2. HYBRID RETRIEVAL (Using the refined query)
            status.update(label=f"Searching for: {search_query}")
            initial_docs = st.session_state.retriever.invoke(search_query)
            
            # 3. RERANK (Cross-Encoder)
            pairs = [[search_query, doc.page_content] for doc in initial_docs]
            scores = st.session_state.reranker.predict(pairs)
            reranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in reranked[:3]]
            
            # 4. ANTI-HALLUCINATION PROMPT (Strictly context-based)
            context_text = "\n\n".join([f"Page {d.metadata['page']}: {d.page_content}" for d in final_docs])
            answer_prompt = (
                f"You are a Sri Lankan history expert. Answer the question ONLY using the context provided. "
                f"If the answer is not in the context, say 'I don't know'. Do not use outside knowledge.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {search_query}\n"
                f"Answer:"
            )
            
            response = st.session_state.llm(answer_prompt, max_new_tokens=300)
            answer = response[0]['generated_text']
            status.update(label="Complete!", state="complete")

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
