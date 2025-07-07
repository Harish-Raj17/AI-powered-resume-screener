import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ------------------ Helper Functions ------------------ #

def extract_text_from_pdf(file):
    """Extracts text from uploaded PDF file using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """Cleans and preprocesses text."""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])

def rank_resumes(jd_text, resume_texts):
    """Computes cosine similarity between JD and each resume."""
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return similarities

# ------------------ Streamlit UI ------------------ #

st.set_page_config(page_title="Resume Screening Tool", layout="wide")
st.title("📄 AI-Powered Resume Screening Tool")

# Upload job description
st.subheader("Step 1: Paste Job Description")
jd_input = st.text_area("Enter or paste the job description here:", height=200)

# Upload resumes
st.subheader("Step 2: Upload Resumes (PDF only)")
uploaded_files = st.file_uploader("Upload one or more resume PDFs", type="pdf", accept_multiple_files=True)

# On button click
if st.button("🚀 Rank Resumes"):

    if not jd_input:
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        jd_cleaned = clean_text(jd_input)
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            cleaned_text = clean_text(raw_text)
            resume_texts.append(cleaned_text)
            resume_names.append(file.name)

        # Compute match scores
        scores = rank_resumes(jd_cleaned, resume_texts)
        ranked = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

        st.subheader("📊 Ranked Resumes Based on JD Match")
        for name, score in ranked:
            st.write(f"**{name}** — Match Score: `{score * 100:.2f}%`")
