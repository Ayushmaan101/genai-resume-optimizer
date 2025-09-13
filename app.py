import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
import docx
# --- LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- Page & API Configuration ---
st.set_page_config(
    page_title="GenAI Resume Optimizer",
    page_icon="📄",
    layout="centered",
)
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🚨 Google API Key not found! Please add it to your .env file.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"🚨 An error occurred during API configuration: {e}")
    st.stop()

# --- Text Extraction Function ---
def get_text_from_files(uploaded_file):
    text = ""
    file_name = uploaded_file.name
    if file_name.endswith('.pdf'):
        try:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return None
    elif file_name.endswith('.docx'):
        try:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return None
    return text

# --- RAG Core Functions ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # This is where we update the prompt for explainability
    prompt_template = """
    You are an expert technical recruiter and resume writer with 20 years of experience. Your task is to analyze a resume against a job description with full explainability.
    Analyze the provided resume against the job description context and perform the following actions, structuring your response using Markdown:

    1.  **Match Score:** Provide an estimated percentage match score. After the score, briefly justify *why* you chose that score, citing the main reasons for alignment or misalignment.
    2.  **Missing Keywords & Gap Analysis:** Detail which key skills from the job description are missing or not emphasized enough in the resume.
    3.  **Profile Summary Rewrite:** Rewrite the resume's "Profile Summary" section to be more aligned with the job description. After the summary, add a "**Justification:**" section explaining why the new version is more impactful and how it targets the job.
    4.  **Bullet Point Suggestions:** Suggest 3-5 specific, action-oriented bullet points. For each suggestion, add a "**Justification:**" explaining how that bullet point directly addresses a key requirement from the job description.

    Context:\n {context}?\n
    Resume:\n{question}\n

    ---
    ### Your Expert Analysis Report
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_report(resume_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(resume_text)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": resume_text},
        return_only_outputs=True
    )
    return response["output_text"]

# --- Main Application UI ---
st.title("🤖 GenAI-Powered Resume Optimizer")
st.write("Upload your resume, paste a job description, and get an AI-powered analysis and rewrite.")
st.write("---")

if 'report_text' not in st.session_state:
    st.session_state.report_text = None

st.subheader("1. Paste the Job Description")
job_description = st.text_area(
    "Paste the full job description here",
    height=250,
    label_visibility="collapsed"
)

st.subheader("2. Upload Your Resume")
resume_file = st.file_uploader(
    "Upload your resume in PDF or DOCX format",
    type=['pdf', 'docx'],
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Process Documents 📁", use_container_width=True):
        if job_description:
            with st.spinner("Processing documents..."):
                raw_text = job_description
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents processed and indexed!")
        else:
            st.warning("Please provide the job description.")

with col2:
    if st.button("Tailor My Resume ✨", type="primary", use_container_width=True):
        if resume_file:
            with st.spinner("Analyzing and Tailoring... 🧠"):
                resume_text = get_text_from_files(resume_file)
                if resume_text:
                    st.session_state.report_text = generate_report(resume_text)
        else:
            st.warning("Please upload a resume to analyze.")

# --- Display Report and Download Button ---
if st.session_state.report_text:
    st.markdown("---")
    st.subheader("📊 Your Resume Analysis Report")
    st.markdown(st.session_state.report_text)
    
    st.download_button(
        label="⬇️ Download Report",
        data=st.session_state.report_text.encode('utf-8'),
        file_name='resume_analysis_report.md',
        mime='text/markdown',
        use_container_width=True
    )