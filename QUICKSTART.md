<!--
IMPORTANT: This file has already been committed but you probably want it ignored.
To stop showing it in the repo, add DEPLOYMENT.md and QUICKSTART.md to .gitignore
and run:
  git rm --cached DEPLOYMENT.md QUICKSTART.md
  git add .gitignore
  git commit -m "Stop tracking DEPLOYMENT.md and QUICKSTART.md"
  git push

If you need to remove them from history after they've been pushed, use git filter-repo or BFG.
-->

# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

1. Create a `.env` file in the project root
2. Add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

**Get your API key from:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser automatically at `http://localhost:8501`

## Step 4: Use the App

1. **Upload PDF**: Click "Upload a PDF Document" and select your PDF file
2. **Process**: Click the "🔄 Process PDF" button
3. **Ask Questions**: Type your question in the chat and press Enter
4. **View Sources**: Click "📌 View Source Documents" to see where the answer came from

## 💡 Tips

- For best results, use PDFs with text (not scanned images)
- The app processes the entire PDF, so large files may take longer
- You can ask multiple questions after processing one PDF
- Use "🗑️ Clear Chat History" to start a new conversation

## 🐛 Troubleshooting

**Error: "OpenAI API key not found"**
- Make sure you created a `.env` file (not `.env.example`)
- Check that the file is in the project root directory
- Verify the key starts with `sk-`

**Error: "Could not extract text from PDF"**
- Your PDF might be scanned (image-based)
- Try a PDF that was created from text documents

**Slow processing?**
- Large PDFs take more time
- First-time HuggingFace model download takes time (only once)
- Check your internet connection for API calls

---

Happy Q&A! 📚✨

