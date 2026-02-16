import streamlit as st
import PyPDF2
import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------ NLTK Downloads ------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ------------------ Load BERT Model ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Helper Functions ------------------

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

def compute_tfidf_similarity(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def compute_bert_similarity(jd, resume):
    embeddings = model.encode([jd, resume])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(similarity * 100, 2)

def extract_skills(text):
    skill_keywords = [
        "python", "machine learning", "deep learning", "nlp",
        "sql", "data analysis", "pandas", "numpy",
        "scikit-learn", "tensorflow", "keras"
    ]
    found_skills = [skill for skill in skill_keywords if skill in text.lower()]
    return found_skills

# ------------------ Streamlit UI ------------------

st.title("📄 Resume Screening AI")

jd = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF only)",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Analyze Candidates"):

    if jd and uploaded_files:

        results = []

        processed_jd = preprocess_text(jd)

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            processed_resume = preprocess_text(resume_text)

            tfidf_score = compute_tfidf_similarity(processed_jd, processed_resume)
            bert_score = compute_bert_similarity(jd, resume_text)

            # Weighted Final Score
            final_score = round((0.4 * tfidf_score) + (0.6 * bert_score), 2)

            skills_found = extract_skills(resume_text)
            required_skills = extract_skills(jd)
            missing_skills = list(set(required_skills) - set(skills_found))

            results.append({
                "Candidate": file.name,
                "TF-IDF Score (%)": tfidf_score,
                "BERT Score (%)": bert_score,
                "Final Score (%)": final_score,
                "Skills Found": ", ".join(skills_found),
                "Missing Skills": ", ".join(missing_skills)
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Final Score (%)", ascending=False)

        st.subheader("📊 Candidate Ranking")
        st.dataframe(df)

        st.bar_chart(
        df.set_index("Candidate")[["TF-IDF Score (%)", "BERT Score (%)", "Final Score (%)"]]
)

        

    else:
        st.warning("Please enter Job Description and upload resumes.")
