# 🚀 Deployment Guide for Streamlit Community Cloud

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `pdf-qa-bot` (or any name you prefer)
   - **Description**: "PDF Question Answering Bot using RAG with LangChain and Streamlit"
   - **Visibility**: Choose **Public** (required for free Streamlit Cloud)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pdf-qa-bot.git

# Rename branch to main (Streamlit Cloud prefers 'main' branch)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

**If you don't have GitHub credentials set up:**
- GitHub may ask for your username and password
- For password, use a **Personal Access Token** (not your GitHub password)
- Create one at: https://github.com/settings/tokens
- Select scope: `repo`

## Step 3: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: Select your `pdf-qa-bot` repository
   - **Branch**: Select `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique URL (e.g., `your-name-pdf-qa-bot`)
5. Click **"Deploy"**

## Step 4: Configure Environment Variables

After deployment, you need to add your OpenAI API key:

1. In Streamlit Cloud, go to your app's settings (three dots → Settings)
2. Click on **"Secrets"** in the left sidebar
3. Add your secrets in this format:

```
OPENAI_API_KEY=your_actual_api_key_here
```

4. Click **"Save"**
5. The app will automatically redeploy

## ✅ Your App is Live!

Once deployed, your app will be available at:
`https://your-app-url.streamlit.app`

## 🔒 Important Notes

- **Never commit your `.env` file** - it's already in `.gitignore`
- Use **Streamlit Secrets** for API keys in production
- The `.env` file is for local development only
- Public repositories can be viewed by anyone, but secrets are secure

## 🐛 Troubleshooting

**Issue: "App failed to deploy"**
- Check that `requirements.txt` has all dependencies
- Verify `app.py` is the correct main file path
- Check the logs in Streamlit Cloud dashboard

**Issue: "OpenAI API key not found"**
- Make sure you added the secret in Streamlit Cloud
- Restart the app after adding secrets
- Check the secret name matches exactly: `OPENAI_API_KEY`

**Issue: "Module not found"**
- Update `requirements.txt` with missing packages
- Push the updated `requirements.txt` to GitHub
- Streamlit Cloud will reinstall dependencies

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Repository Setup](https://docs.github.com/en/repositories/creating-and-managing-repositories)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

