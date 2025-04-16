# auto_indexing.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
from gensim.models import Word2Vec
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup
nltk.download('stopwords')
nltk.download('punkt')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_pdf_with_pages(file_path):
    reader = PdfReader(file_path)
    text_by_page = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            text_by_page.append((page_number, text))
    return text_by_page

def extract_tfidf_keywords_with_pages(documents, top_n=100):
    stop_words = stopwords.words('indonesian')
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)
    full_texts = [" ".join([text for _, text in doc]) for doc in documents]
    X = vectorizer.fit_transform(full_texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()
    keywords_with_pages = []
    for doc_idx, doc_scores in enumerate(tfidf_scores):
        top_indices = doc_scores.argsort()[-top_n:][::-1]
        top_keywords = [feature_array[i] for i in top_indices]
        keyword_pages = {}
        for page_number, page_text in documents[doc_idx]:
            for keyword in top_keywords:
                if keyword in page_text:
                    if keyword not in keyword_pages:
                        keyword_pages[keyword] = []
                    keyword_pages[keyword].append(page_number)
        keywords_with_pages.append(keyword_pages)
    return keywords_with_pages

def extract_rake_keywords_with_pages(documents, top_n=100):
    rake_extractor = Rake(stopwords=stopwords.words('indonesian'))
    keywords_with_pages = []
    for doc in documents:
        all_pages_keywords = {}
        for page_number, page_text in doc:
            rake_extractor.extract_keywords_from_text(page_text)
            ranked_phrases = rake_extractor.get_ranked_phrases()
            top_keywords = ranked_phrases[:top_n]
            for keyword in top_keywords:
                if keyword not in all_pages_keywords:
                    all_pages_keywords[keyword] = []
                all_pages_keywords[keyword].append(page_number)
        keywords_with_pages.append(all_pages_keywords)
    return keywords_with_pages

def preprocess_text(text, stop_words):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

def train_word2vec(documents, vector_size=150, window=10, min_count=5):
    stop_words = stopwords.words('indonesian')
    tokenized_documents = []
    for doc in documents:
        full_text = " ".join([text for _, text in doc])
        tokens = preprocess_text(full_text, stop_words)
        tokenized_documents.append(tokens)
    model = Word2Vec(sentences=tokenized_documents, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def find_similar_words_with_pages(model, documents, words_to_check, topn=10):
    results = {}
    for word in words_to_check:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=topn)
            word_data = []
            for sim_word, similarity in similar_words:
                pages_found = []
                for doc in documents:
                    for page_number, page_text in doc:
                        tokens = preprocess_text(page_text, stopwords.words('indonesian'))
                        if sim_word in tokens:
                            pages_found.append(page_number)
                pages_found = list(set(pages_found))
                word_data.append((sim_word, similarity, pages_found))
            results[word] = word_data
        else:
            results[word] = "Tidak ditemukan dalam model."
    return results

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            documents = [read_pdf_with_pages(filepath)]
            tfidf_result = extract_tfidf_keywords_with_pages(documents)
            rake_result = extract_rake_keywords_with_pages(documents)
            w2v_model = train_word2vec(documents)
            w2v_result = find_similar_words_with_pages(w2v_model, documents, ["bluetooth", "gerbang", "otomatis"], topn=5)
            results = {
                "tfidf": tfidf_result[0],
                "rake": rake_result[0],
                "word2vec": w2v_result
            }
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)