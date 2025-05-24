from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
from gensim.models import KeyedVectors  # Word2Vec
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
from collections import Counter
from textwrap import wrap

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
stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
stemmer = StemmerFactory().create_stemmer()

# Load pre-trained fastText model
model_path = 'cc.id.300.gensim.model'
if os.path.exists(model_path):
    print("Loading saved model...")
    w2v_model = KeyedVectors.load(model_path)
else:
    print("Loading .vec model")
    w2v_model = KeyedVectors.load_word2vec_format('cc.id.300.vec')
    w2v_model.save(model_path)
    print("Saved model")

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

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)     
    return text.lower()   

def compute_similarity(phrase, title):
    phrase_tokens = [w for w in phrase.lower().split() if w in w2v_model]
    title_tokens = [w for w in title.lower().split() if w in w2v_model]

    if not phrase_tokens or not title_tokens:
        return 0.0

    try:
        return w2v_model.n_similarity(phrase_tokens, title_tokens)
    except:
        return 0.0

def extract_rake_keywords(documents, title="", min_length=1, max_length=3):
    rake = Rake(stopwords=stop_words)
    phrase_counter = Counter()
    page_map = {}
    similarity_scores = {}

    for page_number, page_text in documents:
        cleaned_text = clean_text(page_text)
        rake.extract_keywords_from_text(cleaned_text)

        all_keywords = rake.get_ranked_phrases()
        filtered_keywords = []
        for kw in all_keywords:
            words = kw.split()
            if (
                min_length <= len(words) <= max_length and
                all(w.isalpha() and len(w) >= 3 for w in words)
            ):
                filtered_keywords.append(kw)

        for kw in filtered_keywords:
            phrase_counter[kw] += cleaned_text.lower().count(kw.lower())
            page_map.setdefault(kw, set()).add(page_number)

    scored_phrases = []
    for phrase, freq in phrase_counter.items():
        sim_score = compute_similarity(phrase, title) if title else 0.0
        if 0.3 <= sim_score <= 1.0:
            scored_phrases.append((phrase, freq, sim_score))
            
    # Urutkan berdasarkan frekuensi tertinggi, ambil top 100
    sorted_phrases = sorted(scored_phrases, key=lambda x: x[1], reverse=True)[:100]

    result = {
        phrase: {
            "frequency": freq,
            "similarity": round(sim_score, 4),
            "pages": sorted(page_map.get(phrase, []))
        }
        for phrase, freq, sim_score in sorted_phrases
    }

    return result

# Create Index PDF
def create_index_pdf(rake, output_path):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4
    y = height - 50

    def draw_line(line, max_width=90):
        nonlocal y
        wrapped_lines = wrap(line, width=max_width)
        for wrapped_line in wrapped_lines:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(40, y, wrapped_line)
            y -= 14

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, "=== HASIL INDEXING OTOMATIS ===")
    y -= 28

    c.setFont("Helvetica", 12)
    draw_line("2. Metode RAKE:")
    for kw, data in rake.items():
        pages = ', '.join(map(str, data["pages"]))
        freq = data["frequency"]
        sim = data["similarity"]
        draw_line(f"- {kw} (Halaman: {pages}, Frek: {freq}, Similaritas: {sim:.2f})")

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
        manual_title = request.form.get('manual_title', '')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            documents = read_pdf_with_pages(filepath)
            title = manual_title
            title_tokens = preprocess_text(title)
            rake_result = extract_rake_keywords(documents, title=title)

            results = {
                "title": title,
                "rake": rake_result
            }

            index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
            final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
            create_index_pdf(rake_result, index_pdf)
            merge_pdfs(filepath, index_pdf, final_pdf)
            download_link = f"/download/{os.path.basename(final_pdf)}"

    return render_template('index.html', results=results, download_link=download_link)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)