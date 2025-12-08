import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import importlib.util
from typing import Any, List, Tuple
from PyPDF2 import PdfReader
import re
import html
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# Core imports (always needed)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    USE_MODERN_LANGCHAIN = True
except ImportError:
    # Fallback to older version
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
    from langchain.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
    USE_MODERN_LANGCHAIN = False

# Try to import RetrievalQA without raising module-level errors on Streamlit Cloud
lc_chains_spec = importlib.util.find_spec("langchain.chains")
if lc_chains_spec:
    try:
        from langchain.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
        HAS_RETRIEVALQA = True
    except ImportError:
        HAS_RETRIEVALQA = False
else:
    HAS_RETRIEVALQA = False

# Force manual QA path to ensure compatibility with local HuggingFace pipeline
HAS_RETRIEVALQA = False

# Legacy alias kept for compatibility with older references/lints
class HuggingFaceLLM:  # pragma: no cover - kept for tooling compatibility
    ...

def highlight_relevant_sentence(source_text: str, answer_text: str) -> str:
    escaped_source = html.escape(source_text)
    sentences = re.split(r'(?<=[.!?])\s+', escaped_source)
    if not sentences:
        return escaped_source

    answer_lower = answer_text.lower()
    best_idx = -1
    best_score = 0

    for idx, sentence in enumerate(sentences):
        words = set(re.findall(r"\w+", sentence.lower()))
        score = sum(1 for w in words if w and w in answer_lower)
        if score > best_score:
            best_score = score
            best_idx = idx

    highlighted_sentences = []
    for idx, sentence in enumerate(sentences):
        if idx == best_idx and best_score > 0:
            highlighted_sentences.append(f'<span class="source-highlight">{sentence}</span>')
        else:
            highlighted_sentences.append(sentence)

    return " ".join(highlighted_sentences)


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into smaller chunks with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def calculate_semantic_similarity(question: str, context: str) -> Tuple[float, str]:
    """Calculate semantic similarity between question and context using Sentence-BERT.
    Returns a tuple of (max_similarity, best_chunk)"""
    if not hasattr(st.session_state, 'semantic_model'):
        st.session_state.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    model = st.session_state.semantic_model
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Split context into smaller chunks
    chunks = split_into_chunks(context)
    if not chunks:
        return 0.0, ""
    
    # Encode all chunks at once for better performance
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding.unsqueeze(0), chunk_embeddings)[0]
    
    # Find the chunk with highest similarity
    max_idx = similarities.argmax().item()
    best_similarity = float(similarities[max_idx])
    best_chunk = chunks[max_idx]
    
    return best_similarity, best_chunk


def generate_anti_hallucination_prompt(context: str, query: str, similarity_score: float) -> str:
    """Generate a prompt that reduces hallucinations."""
    return f"""You are an AI assistant that provides helpful and accurate information based on the provided context. 
Your responses must adhere to these strict rules:

1. ONLY use information from the provided context to answer the question.
2. If the answer cannot be found in the context, respond with: "I don't have enough information to answer this question based on the provided document."
3. Do not make up or assume any information that is not explicitly stated in the context.
4. If the question is ambiguous or unclear, ask for clarification.
5. The context relevance score is {similarity_score:.2f}/1.00. Lower scores indicate the context may not be very relevant.

Context relevance: {similarity_score:.2f}/1.00
----------------
Context: {context}
----------------
Question: {query}

Answer the question truthfully and concisely based on the context above. If the answer cannot be found in the context, say so:"""


def question_has_overlap_with_context(question: str, context: str) -> bool:
    """Heuristic: check if any important word from the question appears in the
    retrieved context. This helps avoid answering about Sri Lanka when the
    user asks about India, etc.

    We ignore very short/common words and only keep keywords of length >= 4
    that are not typical stopwords.
    """
    if not question or not context:
        return False

    context_lower = context.lower()
    context_compact = context_lower.replace(" ", "")  # for matching without spaces
    words = re.findall(r"\w+", question.lower())
    stopwords = {
        "what", "when", "where", "who", "whom", "which", "why", "how",
        "is", "are", "was", "were", "be", "been", "being",
        "the", "a", "an", "of", "in", "on", "at", "for", "to",
        "and", "or", "if", "then", "else", "do", "does", "did",
        "from", "about", "this", "that", "these", "those", "with",
    }

    keywords = [w for w in words if len(w) >= 4 and w not in stopwords]
    if not keywords:
        return False

    # Be conservative: require that *all* important keywords appear in the
    # retrieved context. For example, for "independent day of India" we
    # require that something like "india" also appears, so we don't answer
    # from Sri Lanka paragraphs.
    #
    # To handle minor formatting differences such as missing spaces
    # ("srilanka" vs "sri lanka"), we also compare against a space-free
    # version of the context.
    for k in keywords:
        k_lower = k.lower()
        k_compact = k_lower.replace(" ", "")
        if k_lower not in context_lower and k_compact not in context_compact:
            return False
    return True


def context_has_sentence_with_all_keywords(question: str, context: str) -> bool:
    """Stricter check: is there at least one sentence in the context that
    contains all important keywords from the question?

    This helps cases like 'independence day of India' where 'India' might
    appear somewhere in the PDF, but never in the same sentence as the
    other important words.
    """
    if not question or not context:
        return False

    sentences = re.split(r"(?<=[.!?])\s+", context)
    words = re.findall(r"\w+", question.lower())
    stopwords = {
        "what", "when", "where", "who", "whom", "which", "why", "how",
        "is", "are", "was", "were", "be", "been", "being",
        "the", "a", "an", "of", "in", "on", "at", "for", "to",
        "and", "or", "if", "then", "else", "do", "does", "did",
        "from", "about", "this", "that", "these", "those", "with",
    }
    keywords = [w for w in words if len(w) >= 4 and w not in stopwords]
    if not keywords:
        return False

    for s in sentences:
        s_lower = s.lower()
        if all(k.lower() in s_lower for k in keywords):
            return True
    return False


# Cached local HuggingFace pipeline (avoids remote API limits)
@st.cache_resource(show_spinner=False)
def load_local_hf_pipeline(model_name: str = "google/flan-t5-small"):
    """Load a local HuggingFace text2text pipeline to stay within PDF context."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .source-panel {
        height: 260px;           /* fixed vertical size */
        width: 100%;             /* span the width of the expander */
        overflow-y: auto;        /* vertical scroll inside the box */
        overflow-x: auto;        /* horizontal scroll inside the box */
        padding: 0.75rem;
        border-radius: 6px;
        background-color: #ffffff;   /* solid white for maximum contrast */
        border: 1px solid #d0d0d0;
        font-family: "Source Code Pro", monospace;
        font-size: 0.9rem;
        color: #111111;              /* very dark text */
        white-space: pre-wrap;       /* allow wrapping but keep line breaks */
        box-sizing: border-box;
    }
    .source-highlight {
        background-color: #c8f3b4;   /* slightly stronger green but still soft */
        color: #000000;              /* ensure highlighted text is dark */
    }
    .upload-sticky {
        position: sticky;
        top: 0;
        z-index: 5;
        padding-bottom: 1rem;
        background-color: inherit;   /* blend with main background */
    }
    .chat-footer-note {
        position: fixed;
        bottom: 0.25rem;
        left: 0;
        right: 0;
        text-align: center;
        color: gray;
        font-size: 0.8rem;
        pointer-events: none;  /* don't block clicks on chat input */
    }
    /* Scrollable Ask Question section (vertical layout: section 1 = upload, section 2 = ask) */
    .ask-section-scroll {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📄 PDF Question Answering Bot</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("""
    **PDF Q&A Bot** is a Retrieval-Augmented Generation (RAG) application that allows you to:
    - Upload PDF documents
    - Ask questions about the content
    - Get accurate answers with source references
    
    Built with LangChain, Streamlit, FAISS, and HuggingFace.
    """)

# Main two-column layout: left = upload, right = Q&A
col_upload, col_chat = st.columns(2)

with col_upload:
    # SECTION 1: Upload PDF document (left side)
    st.markdown("<div class='upload-sticky'>", unsafe_allow_html=True)

    st.subheader("📤 Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start asking questions"
    )

    if uploaded_file:
        # Process PDF button
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("📑 Processing PDF... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Extract text from PDF
                    pdf_reader = PdfReader(tmp_path)
                    text = ""
                    total_pages = len(pdf_reader.pages)

                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    if not text.strip():
                        st.error("❌ Could not extract text from PDF. The file might be scanned or encrypted.")
                        st.session_state.pdf_processed = False
                    else:
                        # Split text into chunks
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        chunks = splitter.split_text(text)

                        # Create embeddings and vector store (HuggingFace only)
                        # Lazy import to avoid loading if not needed
                        try:
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                        except ImportError:
                            from langchain.embeddings import HuggingFaceEmbeddings
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                        # Configure local HuggingFace pipeline (no external API calls)
                        llm = load_local_hf_pipeline("google/flan-t5-small")

                        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                        # Create QA chain (only if an LLM is configured)
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                        if llm is not None:
                            if HAS_RETRIEVALQA:
                                # Use RetrievalQA if available
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    retriever=retriever,
                                    return_source_documents=True
                                )
                            else:
                                # Simple approach: store retriever and llm separately
                                qa_chain = {
                                    "retriever": retriever,
                                    "llm": llm
                                }
                        else:
                            qa_chain = None

                        # Store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = qa_chain
                        st.session_state.pdf_processed = True

                        # Clean up temp file
                        os.unlink(tmp_path)

                        st.info(f"📊 Pages: {total_pages} | Chunks: {len(chunks)}")
                        st.session_state.messages = []  # Clear previous chat

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.pdf_processed = False

        # Display PDF processing status
        if st.session_state.pdf_processed:
            st.success("✅ PDF is ready for questions!")
        else:
            st.info("👆 Click 'Process PDF' after uploading to enable Q&A")

    # When the PDF has been processed, tint the primary button (Process PDF) green
    if st.session_state.pdf_processed:
        st.markdown(
            """
            <style>
            .stButton > button[kind="primary"] {
                background-color: #16a34a !important;  /* green */
                border-color: #16a34a !important;
                color: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_chat:
    # SECTION 2: Ask Questions (right side)
    st.subheader("💬 Ask Questions")
    st.markdown("<div class='ask-section-scroll'>", unsafe_allow_html=True)

    if st.session_state.pdf_processed:
        if st.session_state.qa_chain is None:
            st.warning("PDF was processed, but the language model is not ready yet. Please reprocess the PDF or restart the app.")
            st.stop()
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Display chat history (all past messages)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    # Show only the relevant PDF chunks used to answer the question,
                    # with the most relevant sentence highlighted in light green.
                    with st.expander("📌 Source Documents"):
                        answer_text = message["content"]
                        combined_html_parts = []
                        for source in message["sources"]:
                            highlighted_html = highlight_relevant_sentence(source, answer_text)
                            combined_html_parts.append(highlighted_html)
                        combined_html = "<br><br>".join(combined_html_parts)
                        st.markdown(
                            f"<div class='source-panel'>{combined_html}</div>",
                            unsafe_allow_html=True,
                        )

        # Question input at the bottom of the ask section
        query = st.chat_input("Ask a question about the PDF:")

        if query:
            # Basic validation: ignore extremely short/noisy queries such as
            # single letters or random characters, which lead to meaningless
            # retrieval results.
            cleaned = query.strip()
            # Count alphanumeric characters only
            alnum_only = "".join(ch for ch in cleaned if ch.isalnum())
            if len(alnum_only) < 3 or len(cleaned.split()) < 1:
                warning_msg = "Please enter a more specific question based on the PDF content."
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": warning_msg,
                })
                st.rerun()

            try:
                # Retrieve components
                qa_data = st.session_state.qa_chain
                retriever = qa_data["retriever"]
                llm = qa_data["llm"]

                # Retrieve relevant documents (support modern retriever API)
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(query)
                else:
                    raise AttributeError(
                        "Retriever object does not support document lookup."
                    )

                # Build full context from retrieved chunks
                context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

                # Relevance gate: only answer when
                #  - there is non-empty context, AND
                #  - all important keywords appear somewhere in the context, AND
                #  - at least one sentence in the context contains all
                #    important keywords together.
                is_relevant = (
                    bool(context.strip())
                    and question_has_overlap_with_context(query, context)
                    and context_has_sentence_with_all_keywords(query, context)
                )

                if not is_relevant:
                    answer = "There is no proper answer for this question in your PDF. It may be about a topic that is not covered."
                    sources_text = []
                else:
                    # Generate answer using local HuggingFace pipeline
                    prompt = f"""Answer the following question using ONLY the context below. If the answer is not in the context, reply with "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

                    llm_output = llm(
                        prompt,
                        max_new_tokens=256,
                        do_sample=False,
                    )
                    answer = llm_output[0]["generated_text"].strip() if llm_output else "I couldn't generate an answer."
                    sources_text = [doc.page_content for doc in docs] if docs else []

                # Update chat history (user question + assistant answer)
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text,
                })

            except Exception as e:
                import traceback
                error_msg = f"❌ Error generating answer: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                # Optionally log traceback for debugging in the UI
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": traceback.format_exc(),
                })

            # Rerun so that the updated history is rendered above and the input
            # bar remains visually fixed at the bottom, similar to ChatGPT.
            st.rerun()
    else:
        st.info("📤 Please upload and process a PDF first to start asking questions.")

    st.markdown("</div>", unsafe_allow_html=True)

# Fixed footer text rendered near the bottom of the page, visually under the
# chat input bar.
st.markdown(
    """
    <div class='chat-footer-note'>
      Built with ❤️ using Streamlit, LangChain, FAISS, and HuggingFace
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

