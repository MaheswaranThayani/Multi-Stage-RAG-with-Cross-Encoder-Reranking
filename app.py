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
import textdistance
from langchain_community.retrievers import BM25Retriever


import nltk
from nltk.corpus import stopwords

CHROMA_DB_DIR = "./chroma_db"   # permanent storage location
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# #hybrid retriever
# def hybrid_retrieve(query, semantic_retriever, bm25_retriever):
#     # --- Semantic retriever (Chroma / VectorStore) ---
#     if hasattr(semantic_retriever, "invoke"):
#         semantic_docs = semantic_retriever.invoke(query)
#     else:
#         semantic_docs = semantic_retriever.get_relevant_documents(query)

#     # --- Keyword retriever (BM25) ---
#     if hasattr(bm25_retriever, "invoke"):
#         keyword_docs = bm25_retriever.invoke(query)
#     else:
#         keyword_docs = bm25_retriever.get_relevant_documents(query)

#     seen = set()
#     docs = []

#     for d in semantic_docs + keyword_docs:
#         if d.page_content not in seen:
#             seen.add(d.page_content)
#             docs.append(d)

#     return docs



# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load English stopwords set
EN_STOPWORDS = set(stopwords.words('english'))


# Core imports 
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    USE_MODERN_LANGCHAIN = True
except ImportError:
    # Fallback to older version
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
    from langchain.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
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


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Semantic chunking: split text into paragraph-based chunks, merging paragraphs to fit a target size."""
    import re
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para.split())
        if current_len + para_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # For overlap, start new chunk with last N words from previous chunk
            if overlap > 0 and current_chunk:
                overlap_words = ' '.join(' '.join(current_chunk).split()[-overlap:])
                current_chunk = [overlap_words] if overlap_words else []
                current_len = len(overlap_words.split())
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(para)
        current_len += para_len
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def is_followup_question(query: str) -> bool:
    """Detect if query is a vague follow-up that needs context."""
    vague_patterns = [
        r"^what\s+(are|is|were|was)\s+(they|it|those|these|that|this)\??$",
        r"^(tell|explain|describe|list)\s+(me\s+)?(more|them|it|those|these)\??$",
        r"^(and|also|what about)\s+(the\s+)?(others?|rest|more)\??$",
        r"^(can you|please)\s+(explain|list|tell)\s+(them|it|more)\??$",
        r"^(how|why|when|where)\??$",
        r"^(yes|no|okay|go on|continue)\??$",
    ]
    q = query.strip().lower()
    for pattern in vague_patterns:
        if re.match(pattern, q):
            return True
    words = q.split()
    if len(words) <= 4 and any(w in ["they", "it", "them", "those", "these", "that", "this"] for w in words):
        return True
    return False


def expand_with_context(query: str, messages: List[dict]) -> str:
    """Expand vague follow-up with previous Q&A context."""
    if not messages or not is_followup_question(query):
        return query
    
    last_qa = []
    for msg in reversed(messages[-4:]):
        if msg["role"] == "user":
            last_qa.insert(0, f"Q: {msg['content']}")
        elif msg["role"] == "assistant" and "sources" in msg:
            last_qa.insert(0, f"A: {msg['content']}")
    
    if last_qa:
        context_str = " ".join(last_qa)
        return f"{context_str} {query}"
    return query




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


def correct_spelling(query: str, vocabulary: set) -> str:
    """Correct spelling mistakes in the query using vocabulary from the document."""
    if not query or not vocabulary:
        return query
    
    words = query.split()
    corrected_words = []
    
    for word in words:
        # Remove punctuation from word for matching
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if not clean_word or len(clean_word) < 3:
            # Keep short words and punctuation as-is
            corrected_words.append(word)
            continue
        
        # Find the best match in vocabulary
        best_match = None
        best_score = float('inf')
        
        for vocab_word in vocabulary:
            if len(vocab_word) < 3:
                continue
                
            # Calculate Levenshtein distance
            distance = textdistance.levenshtein(clean_word, vocab_word.lower())
            
            # Normalize by length to get similarity score
            normalized_distance = distance / max(len(clean_word), len(vocab_word))
            
            # Accept matches with small differences (allowing for 1-2 character differences)
            if normalized_distance < best_score and normalized_distance <= 0.3:  # 30% difference threshold
                best_score = normalized_distance
                best_match = vocab_word
        
        if best_match and best_score <= 0.3:
            # Preserve original case pattern if possible
            if word.isupper():
                corrected_word = best_match.upper()
            elif word[0].isupper():
                corrected_word = best_match.capitalize()
            else:
                corrected_word = best_match
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)


def extract_vocabulary_from_text(text: str) -> set:
    """Extract unique words from document text to build vocabulary."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return set(words)


# def generate_anti_hallucination_prompt(context: str, query: str, similarity_score: float) -> str:
#     """Generate a prompt that reduces hallucinations."""
#     return f"""You are an AI assistant that provides helpful and accurate information based on the provided context. 
# Your responses must adhere to these strict rules:

# 1. ONLY use information from the provided context to answer the question.
# 2. If the answer cannot be found in the context, respond with: "I don't have enough information to answer this question based on the provided document."
# 3. Do not make up or assume any information that is not explicitly stated in the context.
# 4. If the question is ambiguous or unclear, ask for clarification.
# 5. The context relevance score is {similarity_score:.2f}/1.00. Lower scores indicate the context may not be very relevant.

# Context relevance: {similarity_score:.2f}/1.00
# ----------------
# Context: {context}
# ----------------
# Question: {query}

# Answer the question truthfully and concisely based on the context above. If the answer cannot be found in the context, say so:"""


# def question_has_overlap_with_context(question: str, context: str, vocabulary: set = None) -> bool:
#     """Heuristic: check if any important word from the question appears in the
#     retrieved context. This helps avoid answering about Sri Lanka when the
#     user asks about India, etc.

#     We ignore very short/common words and only keep keywords of length >= 4
#     that are not typical stopwords. Now includes fuzzy matching for spelling errors.
#     """
#     if not question or not context:
#         return False

#     context_lower = context.lower()
#     context_compact = context_lower.replace(" ", "")  # for matching without spaces
#     words = re.findall(r"\w+", question.lower())
#     stopwords = EN_STOPWORDS


#     keywords = [w for w in words if len(w) >= 4 and w not in stopwords]
#     if not keywords:
#         return False

#     # Be conservative: require that *all* important keywords appear in the
#     # retrieved context. For example, for "independent day of India" we
#     # require that something like "india" also appears, so we don't answer
#     # from Sri Lanka paragraphs.
#     #
#     # To handle minor formatting differences such as missing spaces
#     # ("srilanka" vs "sri lanka"), we also compare against a space-free
#     # version of the context.
#     for k in keywords:
#         k_lower = k.lower()
#         k_compact = k_lower.replace(" ", "")
        
#         # First try exact match
#         if k_lower in context_lower or k_compact in context_compact:
#             continue
            
#         # If vocabulary is provided, try fuzzy matching
#         if vocabulary:
#             found_match = False
#             for vocab_word in vocabulary:
#                 if len(vocab_word) < 4:
#                     continue
                    
#                 # Calculate normalized Levenshtein distance
#                 distance = textdistance.levenshtein(k_lower, vocab_word.lower())
#                 normalized_distance = distance / max(len(k_lower), len(vocab_word))
                
#                 # Accept if very close match (20% difference threshold)
#                 if normalized_distance <= 0.2:
#                     # Check if the corrected word exists in context
#                     corrected_word = vocab_word.lower()
#                     corrected_compact = corrected_word.replace(" ", "")
#                     if corrected_word in context_lower or corrected_compact in context_compact:
#                         found_match = True
#                         break
            
#             if not found_match:
#                 return False
#         else:
#             return False
    
#     return True


def compute_semantic_similarity(text1: str, text2: str, embeddings_model) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    try:
        emb1 = embeddings_model.embed_query(text1)
        emb2 = embeddings_model.embed_query(text2)
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception:
        return 0.0


def compute_max_chunk_similarity(query: str, docs: list, embeddings_model) -> float:
    """Compute max similarity between query and individual document chunks."""
    if not docs or not embeddings_model:
        return 0.0
    try:
        query_emb = np.array(embeddings_model.embed_query(query))
        max_sim = 0.0
        for doc in docs:
            chunk_emb = np.array(embeddings_model.embed_query(doc.page_content))
            sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
            max_sim = max(max_sim, float(sim))
        return max_sim
    except Exception:
        return 0.0


def validate_answer_grounding(answer: str, context: str, embeddings_model=None) -> Tuple[bool, float]:
    """Validate that the answer is grounded in the context.
    Uses PURE semantic similarity - no keyword matching.
    Returns (is_grounded, score).
    """
    if not answer or not context:
        return False, 0.0
    
    # Short answers are usually extracted directly - trust them
    if len(answer.split()) <= 15:
        return True, 1.0
    
    # Pure semantic check
    if embeddings_model:
        try:
            semantic_score = compute_semantic_similarity(answer, context, embeddings_model)
            is_grounded = semantic_score >= 0.4
            return is_grounded, semantic_score
        except Exception:
            pass
    
    # Fallback: if no embeddings, trust the answer
    return True, 1.0


# Cached local HuggingFace pipeline (avoids remote API limits)
@st.cache_resource(show_spinner=False)
def load_local_hf_pipeline(model_name: str = "google/flan-t5-large"):
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
if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    # Load and process Sri_lanka_history.pdf at startup
    pdf_path = os.path.join("Data", "Sri_lanka_history.pdf")
    if os.path.exists(pdf_path):
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        # Split text into semantic chunks (sentence-based)
        chunks = split_into_chunks(text, chunk_size=1000, overlap=200)
        # Create embeddings and vector store
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        from langchain_community.vectorstores import Chroma
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        vector_store.persist()
        # Configure local HuggingFace pipeline (no external API calls)
        llm = load_local_hf_pipeline("google/flan-t5-large")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = {"retriever": retriever, "llm": llm}
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.embeddings_model = embeddings
        st.session_state.vocabulary = set()
    else:
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.embeddings_model = None
        st.session_state.vocabulary = set()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vocabulary" not in st.session_state:
    st.session_state.vocabulary = set()

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
    
    Built with LangChain, Streamlit, CromaDB, and HuggingFace.
    """)

# Single column layout for chat
col_chat = st.container()

with col_chat:
    # st.subheader("💬 Chat with the AI")
    # SECTION 2: Ask Questions (right side)
    st.subheader("💬 Ask Questions")
    st.markdown("<div class='ask-section-scroll'>", unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Display chat history (all past messages)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
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
    query = st.chat_input("Ask a question:")

    if query:
        st.session_state.user_question = query
        st.session_state.messages.append({"role": "user", "content": query})

        cleaned = query.strip()
        alnum_only = "".join(ch for ch in cleaned if ch.isalnum())
        if len(alnum_only) < 3 or len(cleaned.split()) < 1:
            warning_msg = "Please enter a more specific question."
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            st.rerun()

        # Retrieval + Answer (skip if no retriever/LLM)
        try:
            qa_data = st.session_state.qa_chain if "qa_chain" in st.session_state else None
            if not qa_data:
                st.session_state.messages.append({"role": "assistant", "content": "No knowledge base is available for answering questions."})
                st.rerun()
            retriever = qa_data["retriever"]
            llm = qa_data["llm"]

            expanded_query = expand_with_context(query, st.session_state.messages)
            search_query = expanded_query if expanded_query != query else query

            # Aggregate top-k relevant chunks for context and source display
            k = 3  # Number of chunks to aggregate
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(search_query)
            elif hasattr(retriever, "invoke"):
                docs = retriever.invoke(search_query)
            else:
                docs = []

            # Take top-k chunks (if available)
            docs = docs[:k] if docs else []

            # For list-type questions, concatenate top-k chunks into a single context block
            list_question = any(word in search_query.lower() for word in ["what are", "which", "list", "who are", "name the", "give the", "mention the"])
            if list_question:
                combined_context = "\n".join([d.page_content for d in docs])
                # Remove duplicate lines
                lines = combined_context.splitlines()
                seen = set()
                deduped_lines = []
                for line in lines:
                    line_strip = line.strip()
                    if line_strip and line_strip not in seen:
                        deduped_lines.append(line_strip)
                        seen.add(line_strip)
                context = "\n".join(deduped_lines)
            else:
                # Remove duplicate lines from context as before
                raw_context = "\n\n".join([d.page_content for d in docs]) if docs else ""
                lines = raw_context.splitlines()
                seen = set()
                deduped_lines = []
                for line in lines:
                    line_strip = line.strip()
                    if line_strip and line_strip not in seen:
                        deduped_lines.append(line_strip)
                        seen.add(line_strip)
                context = "\n".join(deduped_lines)

            semantic_score = 0.0
            if docs and "embeddings_model" in st.session_state:
                semantic_score = compute_max_chunk_similarity(search_query, docs, st.session_state.embeddings_model)


            # Lower threshold for vague/about questions
            vague_about = any(word in search_query.lower() for word in ["about ", "describe ", "tell me about", "give an overview", "summary of", "history of"])
            SEMANTIC_THRESHOLD = 0.2 if vague_about else 0.35
            is_relevant = bool(context.strip()) and semantic_score >= SEMANTIC_THRESHOLD

            if not is_relevant:
                if not context.strip():
                    answer = "No relevant content found. Please try rephrasing your question."
                else:
                    answer = f"The question doesn't seem related to the content (score: {semantic_score:.2f}). Try asking something more specific to the knowledge base."
                sources_text = []
            else:
                question_for_llm = search_query if search_query != query else query
                # Build conversation history for context
                chat_history = ""
                recent_msgs = st.session_state.messages[-6:]  # Last 3 Q&A pairs
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        chat_history += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        chat_history += f"Assistant: {msg['content']}\n"

                # System prompt for consistent behavior
#                 if list_question:
#                     system_prompt = """You are a precise document assistant. Your rules:
# 1. Answer ONLY using information from the provided context.
# 2. If the answer is a list, enumerate all items from the context. Do not just state the number, but list the items explicitly.
# 3. If the answer cannot be found in the context, say so.
# """
#                 else:
                    system_prompt = """You are a precise document assistant. Your rules:
1. Answer ONLY using information from the provided context
2. Be specific and complete - list all relevant details
3. If information is not in context, say "I don't have that information"
4. Never make up facts or use external knowledge
5. Do not repeat any Sentence in the context verbatim in your answer.

"""

                prompt = f"""{system_prompt}

{f"Previous conversation:{chr(10)}{chat_history}" if chat_history else ""}

Context from document:
{context}

Current question: {question_for_llm}

Answer:"""

                llm_output = llm(
                    prompt,
                    max_new_tokens=300,
                    do_sample=False,
                )
                raw_answer = llm_output[0]["generated_text"].strip() if llm_output else ""
                
                # Step 3: Post-generation validation - check answer is grounded
                is_grounded, grounding_score = validate_answer_grounding(
                    raw_answer, context, st.session_state.embeddings_model
                )
                
                # Debug grounding in sidebar
                with st.sidebar:
                    st.write(f"📎 Grounding Score: {grounding_score:.3f}")
                    st.write(f"✅ Grounded: {is_grounded}")
                
                if not raw_answer or not is_grounded:
                    answer = "I couldn't find a reliable answer in the PDF for this question."
                else:
                    answer = raw_answer
                    # Post-processing: If answer is just a number or single word and context contains a list, extract and append the list
                    if list_question and (re.fullmatch(r"\d+", answer) or len(answer.split()) <= 2):
                        lines = context.splitlines()
                        list_lines = []
                        found_intro = False
                        for i, line in enumerate(lines):
                            if re.search(r"colonized in|stages|are|include|consist of|following", line, re.IGNORECASE):
                                found_intro = True
                                # Optionally include the intro line
                                list_lines.append(line.strip())
                                # Add next lines if they look like list items
                                for next_line in lines[i+1:i+6]:
                                    if re.match(r"\s*[-•\d.]+|[A-Z][a-z]+", next_line):
                                        list_lines.append(next_line.strip())
                                    else:
                                        break
                                break
                        # Fallback: look for lines with years or country names
                        if not list_lines:
                            for line in lines:
                                if re.search(r"Portuguese|Dutch|British|\d{4}", line, re.IGNORECASE):
                                    list_lines.append(line.strip())
                        if list_lines:
                            answer = answer + "\n" + "\n".join(list_lines)
                sources_text = [doc.page_content for doc in docs] if docs else []

            # Update chat history (user question + assistant answer)
            # st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_text,
            })

        except Exception as e:
            error_msg = "❌ An error occurred while processing your question. Please try again."
            # st.session_state.messages.append({
            #     "role": "user",
            #     "content": query,
            # })
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

        # Rerun so that the updated history is rendered above and the input
        # bar remains visually fixed at the bottom, similar to ChatGPT.
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Fixed footer text rendered near the bottom of the page, visually under the
# chat input bar.
st.markdown(
    """
    <div class='chat-footer-note'>
      Built with ❤️ using Streamlit, LangChain, CromaDB, and HuggingFace
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
