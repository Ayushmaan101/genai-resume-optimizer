# 🤖 GenAI-Powered Resume Optimizer

This project is a sophisticated RAG (Retrieval-Augmented Generation) system designed to function as an AI resume assistant. It leverages a powerful language model and a semantic search pipeline to automate the complex task of tailoring a resume to a specific job description.



## ✨ Features

* **AI-Powered Content Generation:** Automatically rewrites profile summaries and suggests tailored, action-oriented bullet points based on the job description.
* **Retrieval-Augmented Generation (RAG):** Uses a FAISS vector store built from the job description to ground all suggestions in facts, preventing hallucinations and ensuring relevance.
* **Deep Semantic Analysis:** Identifies critical skill gaps between the resume and the role, providing a justified match score.
* **Built-in Explainability:** Generates a step-by-step reasoning trace for every AI suggestion to ensure transparency and user trust.
* **Interactive UI:** A user-friendly interface built with Streamlit that supports PDF/DOCX uploads and allows users to download the final analysis report.

## 🛠️ Tech Stack

* **AI Framework & Pipeline:** LangChain
* **Generative LLM:** Groq Llama 3 / Google Gemini (via API)
* **Embedding & Retrieval:** Hugging Face Sentence-Transformers (Local), FAISS
* **UI & Data Parsing:** Streamlit, PyMuPDF, python-docx
* **Core Language & Environment:** Python, venv

## 🚀 Setup and Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/genai-resume-optimizer.git
    cd genai-resume-optimizer
    ```
2.  Create and activate the virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the root directory and add your API key(s):
    ```
    # For Groq
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    
    # Or for Google Gemini
    # GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```
5.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```