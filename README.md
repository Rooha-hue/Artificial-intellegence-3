# Artificial-intellegence-3
# 📄 ATS Resume Checker

A Streamlit app that analyzes an uploaded resume and returns an estimated ATS score, a category breakdown, and prioritized suggestions for improvement. AI analysis is powered by Google Gemini Flash.

## Features
- Upload PDF, DOCX or TXT resumes
- Optional job description for targeted keyword matching
- Overall score (0-100) calculated from five weighted categories
- Prioritized improvements with rewrite examples, plus missing keywords

> The score is an AI-based estimate. Real ATS software does not publish a standard score, so use it as guidance, not a guarantee.

## Run locally
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from https://aistudio.google.com/apikey
4. Provide the key (any one of these):
   - Paste it in the app sidebar, or
   - Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), or
   - Create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key or `secrets.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Files
- `app.py` - the Streamlit app
- `requirements.txt` - Python dependencies
- `README.md` - this file

## Notes
- The default model is `gemini-3.8-flash`. You can change it in the sidebar if Google releases a newer Flash model.
- Scanned (image-only) resumes can't be read, and an ATS couldn't read them either. Use a text-based PDF or DOCX.
