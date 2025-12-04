# 📄 PDF Question Answering Bot

A Retrieval-Augmented Generation (RAG) based PDF Question Answering application built using Python, LangChain, Streamlit, FAISS, and OpenAI/HuggingFace. This application allows users to upload PDF documents and ask questions, with the system retrieving relevant chunks using embeddings and generating accurate answers using Large Language Models (LLMs).

## Features

- **PDF Upload**: Easy drag-and-drop interface for uploading PDF documents
- **Intelligent Text Extraction**: Extracts text from PDF pages using PyPDF2
- **Smart Chunking**: Uses RecursiveCharacterTextSplitter for optimal text segmentation
- **Vector Embeddings**: Supports both OpenAI and HuggingFace embeddings
- **Vector Store**: FAISS for efficient similarity search
- **Interactive Q&A**: Chat-like interface for asking questions
- **Source Citations**: Shows source documents for each answer
- **Chat History**: Maintains conversation history during the session
- **Modern UI**: Clean and intuitive Streamlit interface

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Framework | Streamlit |
| LLM Framework | LangChain |
| Text Extraction | PyPDF2 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI Embeddings / HuggingFace Sentence Transformers |
| Vector Store | FAISS |
| Language Model | OpenAI GPT (with HuggingFace fallback option) |

##  Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional - can use HuggingFace for embeddings)
- pip package manager

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd New_Tech_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the example environment file:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

**Note**: You can get your OpenAI API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## 🎯 Usage

### Step 1: Run the Application

```bash
streamlit run app.py
```

### Step 2: Upload a PDF

1. Click on "Upload a PDF Document" or drag and drop a PDF file
2. Click the "🔄 Process PDF" button to extract and process the text
3. Wait for the processing to complete (you'll see page count and chunk count)

### Step 3: Ask Questions

1. Type your question in the chat input at the bottom
2. Press Enter or click send
3. View the answer and source documents

### Step 4: View Sources

- Click on "View Source Documents" to see the exact chunks from the PDF that were used to generate the answer
- This helps verify the accuracy and transparency of the responses

## 📁 Project Structure

```
New_Tech_project/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env.example          # Example environment variables template
├── .gitignore           # Git ignore rules
└── .env                  # Environment variables (not in git)
```

## Configuration Options

### Using HuggingFace
1. The app uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for embeddings
2. modify the code to use HuggingFace Hub models

## How It Works

1. **PDF Upload**: User uploads a PDF document through Streamlit interface
2. **Text Extraction**: PyPDF2 extracts text from all pages of the PDF
3. **Text Chunking**: RecursiveCharacterTextSplitter divides text into overlapping chunks (1000 chars with 200 char overlap)
4. **Embedding Generation**: Each chunk is converted to a vector using embeddings (OpenAI or HuggingFace)
5. **Vector Storage**: FAISS stores all chunk embeddings for efficient similarity search
6. **Question Processing**: User's question is embedded using the same embedding model
7. **Retrieval**: FAISS finds the top-k most similar chunks to the question
8. **Answer Generation**: LangChain's RetrievalQA chain uses the retrieved chunks as context and generates an answer using the LLM
9. **Response Display**: Answer and source documents are displayed to the user

## 📊 Features Breakdown

| Feature | Description |
|---------|-------------|
| **PDF Processing** | Extracts text from PDFs with page-by-page processing |
| **Smart Chunking** | 1000-character chunks with 200-character overlap for context preservation |
| **Vector Search** | FAISS-based similarity search (retrieves top 3 chunks) |
| **Source Tracking** | Shows exact source chunks used for each answer |
| **Chat Interface** | Conversational UI with message history |
| **Error Handling** | Graceful error handling with user-friendly messages |

## 🎨 UI Features

- **Two-Column Layout**: Upload area and chat area side by side
- **Status Indicators**: Visual feedback for PDF processing status
- **Chat History**: Persistent conversation during session
- **Source Expanders**: Collapsible sections for viewing sources
- **Clear Chat**: Button to reset conversation history

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and private
- The `.gitignore` file is configured to exclude sensitive files

## 🐛 Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Ensure you've created a `.env` file with `OPENAI_API_KEY=your_key`

### Issue: "Could not extract text from PDF"
**Solution**: The PDF might be scanned (image-based). Consider using OCR tools to convert scanned PDFs to text.

### Issue: "FAISS installation error"
**Solution**: If on Apple Silicon, you might need `faiss-cpu` instead. For GPU support, use `faiss-gpu`.

### Issue: Slow processing
**Solution**: Large PDFs take time. For better performance, consider:
- Using GPU for embeddings (if available)
- Reducing chunk size
- Processing PDFs in batches

## 📈 Future Enhancements

- [ ] Support for multiple PDF uploads
- [ ] Export chat history to PDF/text
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Multiple LLM provider support (Anthropic, Cohere, etc.)
- [ ] Conversation memory across sessions
- [ ] Advanced retrieval strategies (reranking, hybrid search)
- [ ] PDF annotation and highlighting
- [ ] User authentication and document management

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

Developed as a portfolio project demonstrating RAG (Retrieval-Augmented Generation) capabilities.

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://www.langchain.com/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)
- LLM by [OpenAI](https://openai.com/)

---

⭐ If you find this project helpful, please consider giving it a star!

