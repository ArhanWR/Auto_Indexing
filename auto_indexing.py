from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
from gensim.models import KeyedVectors
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io

# Setup
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf'}

# Load stopwords & stemmer
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# Load pre-trained fastText model
print("Loading pre-trained Word2Vec model...")
w2v_model = KeyedVectors.load_word2vec_format('cc.id.300.vec', binary=False)
print("Model loaded.")

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_pdf_with_pages(file_path):
    reader = PdfReader(file_path)
    return [(i + 1, page.extract_text()) for i, page in enumerate(reader.pages) if page.extract_text()]

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(t) for t in tokens if t not in stop_words]

def extract_tfidf_keywords(documents, top_n=50):
    texts = [" ".join([text for _, text in documents])]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    feature_array = vectorizer.get_feature_names_out()
    top_indices = X.toarray()[0].argsort()[-top_n:][::-1]
    top_keywords = [feature_array[i] for i in top_indices]

    page_map = {}
    for page_number, page_text in documents:
        lowered = page_text.lower()
        for kw in top_keywords:
            if kw in lowered:
                page_map.setdefault(kw, set()).add(page_number)
    return {k: sorted(v) for k, v in page_map.items()}

def extract_rake_keywords(documents, top_n=50):
    rake = Rake(stopwords=stop_words)
    page_map = {}
    for page_number, page_text in documents:
        rake.extract_keywords_from_text(page_text)
        top_keywords = rake.get_ranked_phrases()[:top_n]
        for kw in top_keywords:
            page_map.setdefault(kw, set()).add(page_number)
    return {k: sorted(v) for k, v in page_map.items()}

def find_similar_words(words_to_check, documents, topn=10):
    result = {}
    page_texts = {p: preprocess_text(t) for p, t in documents}
    for word in words_to_check:
        if word in w2v_model:
            sims = w2v_model.most_similar(word, topn=topn)
            word_data = []
            for sim_word, similarity in sims:
                pages_found = [p for p, tokens in page_texts.items() if sim_word in tokens]
                word_data.append((sim_word, similarity, pages_found))
            result[word] = word_data
        else:
            result[word] = "Tidak ditemukan dalam model."
    return result

# Create Index PDF
def create_index_pdf(tfidf, rake, w2v, output_path):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4
    y = height - 50

    def draw_line(line):
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line)
        y -= 14

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "=== HASIL INDEXING OTOMATIS ===")
    y -= 28

    c.setFont("Helvetica", 12)
    draw_line("1. Metode TF-IDF:")
    for kw, pages in tfidf.items():
        draw_line(f"- {kw} (hal: {', '.join(map(str, pages))})")
    y -= 14

    draw_line("2. Metode RAKE:")
    for kw, pages in rake.items():
        draw_line(f"- {kw} (hal: {', '.join(map(str, pages))})")
    y -= 14

    draw_line("3. Metode Word2Vec:")
    for kw, entries in w2v.items():
        draw_line(f"- Kata kunci: {kw}")
        if isinstance(entries, str):
            draw_line(f"   {entries}")
        else:
            for sim, sim_val, pages in entries:
                draw_line(f"   > {sim} ({sim_val:.2f}, hal: {', '.join(map(str, pages))})")
    c.save()

    with open(output_path, 'wb') as f:
        f.write(packet.getvalue())

# Merge original and index PDF
def merge_pdfs(original_pdf, index_pdf, output_pdf):
    merger = fitz.open(original_pdf)
    index_doc = fitz.open(index_pdf)
    merger.insert_pdf(index_doc)
    merger.save(output_pdf)
    merger.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    results, download_link = {}, None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            documents = read_pdf_with_pages(filepath)
            tfidf_result = extract_tfidf_keywords(documents)
            rake_result = extract_rake_keywords(documents)
            w2v_result = find_similar_words(["dokumen", "metode", "frekuensi"], documents)

            results = {
                "tfidf": tfidf_result,
                "rake": rake_result,
                "word2vec": w2v_result
            }

            index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
            final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
            create_index_pdf(tfidf_result, rake_result, w2v_result, index_pdf)
            merge_pdfs(filepath, index_pdf, final_pdf)
            download_link = f"/download/{os.path.basename(final_pdf)}"

    return render_template('index.html', results=results, download_link=download_link)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)