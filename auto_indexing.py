from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
from gensim.models import KeyedVectors  # menggunakan KeyedVectors, bukan Word2Vec
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

# Load pre-trained Word2Vec model once at startup
def load_pretrained_word2vec(model_path):
    return KeyedVectors.load_word2vec_format(model_path)

pretrained_w2v_path = 'cc.id.300.vec'  # pastikan file ini ada di folder yang sama
w2v_model = load_pretrained_word2vec(pretrained_w2v_path)

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

def find_similar_words_with_pages(model, documents, words_to_check, topn=10):
    stop_words = stopwords.words('indonesian')
    results = {}
    for word in words_to_check:
        if word in model.key_to_index:
            similar_words = model.most_similar(word, topn=topn)
            word_data = []
            for sim_word, similarity in similar_words:
                pages_found = []
                for doc in documents:
                    for page_number, page_text in doc:
                        tokens = preprocess_text(page_text, stop_words)
                        if sim_word in tokens:
                            pages_found.append(page_number)
                pages_found = list(set(pages_found))
                word_data.append((sim_word, similarity, pages_found))
            results[word] = word_data
        else:
            results[word] = "Tidak ditemukan dalam model."
    return results

def create_index_pdf(tfidf_dict, rake_dict, word2vec_dict, output_path):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4

    margin_left = 40
    margin_top = 800
    line_height = 14
    bottom_margin = 50
    y = margin_top

    def draw_line(line):
        nonlocal y
        if y < bottom_margin:
            c.showPage()
            y = margin_top
            c.setFont("Helvetica", 12)
        c.drawString(margin_left, y, line)
        y -= line_height

    title = "=== HASIL INDEXING OTOMATIS ==="
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y, title)
    y -= line_height * 2

    c.setFont("Helvetica", 12)

    draw_line("1. Metode TF-IDF:")
    y -= line_height
    for keyword, pages in tfidf_dict.items():
        draw_line(f"- {keyword} (halaman: {', '.join(map(str, pages))})")
    y -= line_height * 2

    draw_line("2. Metode RAKE:")
    y -= line_height
    for keyword, pages in rake_dict.items():
        draw_line(f"- {keyword} (halaman: {', '.join(map(str, pages))})")
    y -= line_height * 2

    draw_line("3. Metode Word2Vec:")
    y -= line_height
    for keyword, results in word2vec_dict.items():
        draw_line(f"- Kata kunci: {keyword}")
        if isinstance(results, str):
            draw_line(f"   {results}")
        else:
            for sim_word, similarity, pages in results:
                draw_line(f"   > {sim_word} (similarity: {similarity:.2f}, halaman: {', '.join(map(str, pages))})")
        y -= line_height

    c.save()

    with open(output_path, 'wb') as f:
        f.write(packet.getvalue())

def merge_pdfs(original_pdf, index_pdf, output_pdf):
    merger = fitz.open(original_pdf)
    index_doc = fitz.open(index_pdf)
    merger.insert_pdf(index_doc)
    merger.save(output_pdf)
    merger.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    download_link = None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            documents = [read_pdf_with_pages(filepath)]

            tfidf_result = extract_tfidf_keywords_with_pages(documents)
            rake_result = extract_rake_keywords_with_pages(documents)
            w2v_result = find_similar_words_with_pages(w2v_model, documents, ["dokumen", "metode", "frekuensi"], topn=10)

            results = {
                "tfidf": tfidf_result[0],
                "rake": rake_result[0],
                "word2vec": w2v_result
            }

            index_pdf_path = os.path.join(RESULT_FOLDER, 'indexing.pdf')
            final_pdf_path = os.path.join(RESULT_FOLDER, f"final_{filename}")

            create_index_pdf(tfidf_result[0], rake_result[0], w2v_result, index_pdf_path)
            merge_pdfs(filepath, index_pdf_path, final_pdf_path)

            download_link = f"/download/{os.path.basename(final_pdf_path)}"

    return render_template('index.html', results=results, download_link=download_link)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
