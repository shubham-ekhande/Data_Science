import PyPDF2
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('stopwords')

# Load BERT model once
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)


def tfidf_similarity(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]


def bert_similarity(jd, resume):
    embeddings = bert_model.encode([jd, resume])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def extract_skills(text, skills_list):
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills
