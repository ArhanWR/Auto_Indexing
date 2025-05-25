from flask import Flask, render_template, request, send_file, session, redirect
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
from gensim.models import KeyedVectors  # Word2Vec
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import io
from collections import defaultdict
from collections import Counter
from textwrap import wrap

# Setup
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'arhan-windu-rizki-putra-budianto-0908'
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
        sim_score = float(compute_similarity(phrase, title)) if title else 0.0
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
def wrap_text(text, max_width, canvas_obj, font_name="Helvetica", font_size=10):
    """Wrap text manually based on width in points"""
    words = text.split(', ')
    lines = []
    current_line = ""

    for word in words:
        trial_line = f"{current_line}, {word}".strip(', ')
        width = canvas_obj.stringWidth(trial_line, font_name, font_size)
        if width > max_width:
            if current_line:
                lines.append(current_line.strip(', '))
            current_line = word
        else:
            current_line = trial_line

    if current_line:
        lines.append(current_line.strip(', '))
    return lines

def create_index_pdf(rake, output_path):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4
    margin_x = 25 * mm
    col_gap = 10 * mm
    col_width = (width - 2 * margin_x - col_gap) / 2
    y_start = height - 40 * mm
    line_height = 12
    min_y = 20 * mm

    index_data = defaultdict(list)
    for kw, data in rake.items():
        if not data.get("pages"):  # skip keyword tanpa halaman
            continue
        first_letter = kw[0].upper()
        pages = sorted(set(data["pages"]))
        index_data[first_letter].append((kw, pages))

    sorted_letters = sorted(index_data.keys())
    for letter in sorted_letters:
        index_data[letter].sort()

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 30 * mm, "HASIL INDEKS")
    y = y_start
    x = margin_x
    col = 0
    c.setFont("Helvetica", 10)

    for letter in sorted_letters:
        if y < min_y:
            if col == 0:
                col = 1
                x = margin_x + col_width + col_gap
                y = y_start
            else:
                c.showPage()
                x = margin_x
                col = 0
                y = y_start
                c.setFont("Helvetica", 10)

        y -= line_height * 1.5
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, letter)
        underline_y = y - 2
        text_width = c.stringWidth(letter, "Helvetica-Bold", 12)
        c.line(x, underline_y, x + text_width, underline_y)
        y -= line_height
        c.setFont("Helvetica", 10)

        for keyword, pages in index_data[letter]:
            page_str = ', '.join(map(str, pages))
            line = f"{keyword}, {page_str}"
            wrapped_lines = wrap_text(line, col_width, c)
            for line_item in wrapped_lines:
                if y < min_y:
                    if col == 0:
                        col = 1
                        x = margin_x + col_width + col_gap
                        y = y_start
                    else:
                        c.showPage()
                        x = margin_x
                        col = 0
                        y = y_start
                        c.setFont("Helvetica", 10)
                c.drawString(x, y, line_item)
                y -= line_height

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

def convert_floats_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_floats_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_floats_to_native(v) for v in obj)
    elif str(type(obj)).startswith("<class 'numpy.float"):
        return float(obj)
    return obj

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
            # Convert semua float32 jadi float biasa agar bisa disimpan
            rake_result_clean = convert_floats_to_native(rake_result)

            # Simpan ke session
            session['rake_result'] = rake_result_clean
            session['uploaded_pdf_path'] = filepath

            # Dapatkan hasil frasa manual dari session (jika ada)
            manual_result = session.get('frasa_manual_result', [])
            manual_as_dict = {
                item['frasa']: {
                    'pages': item['pages'],
                    'frequency': len(item['pages']) if item['pages'] else 0,
                    'similarity': 0.0
                }
                for item in manual_result
            }

            combined_full_result = {**rake_result, **manual_as_dict}

            # Buat PDF index lengkap
            index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
            final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
            create_index_pdf(combined_full_result, index_pdf)
            merge_pdfs(filepath, index_pdf, final_pdf)
            download_link = f"/download/{os.path.basename(final_pdf)}"

            results = {
                "title": title,
                "rake": rake_result,
                "manual": manual_as_dict,
                "combined": combined_full_result
            }
            session['results'] = results
            session['download_link'] = download_link

    return render_template(
        'index.html',
        results=session.get('results', {}),
        download_link=session.get('download_link'),
        frasa_manual_result=session.get('frasa_manual_result')
    )

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    selected_keywords = request.form.getlist('selected_keywords')
    selected_manual_phrases = request.form.getlist('selected_manual_phrases')
    if not selected_keywords and not selected_manual_phrases:
        return "Tidak ada frasa kunci yang dipilih.", 400

    rake_result = session.get('rake_result', {})
    manual_result = session.get('frasa_manual_result', [])
    original_pdf_path = session.get('uploaded_pdf_path')
    if not original_pdf_path:
        return "Data tidak ditemukan di sesi.", 400

    # Ambil data hasil yang dipilih dari indexing otomatis
    filtered_rake = {
        kw: data for kw, data in rake_result.items() if kw in selected_keywords
    }

    # Ambil data hasil frasa manual yang dipilih
    filtered_manual = {}
    for item in manual_result:
        if item['frasa'] in selected_manual_phrases:
            filtered_manual[item['frasa']] = {
                'pages': item['pages'],
                'frequency': len(item['pages']) if item['pages'] else 0,
                'similarity': 0.0  # atau None jika tidak tersedia
            }

    # Gabungkan hasil terpilih dari RAKE dan manual
    combined_result = {**filtered_rake, **filtered_manual}
    index_pdf = os.path.join(RESULT_FOLDER, 'selected_indexing.pdf')
    final_pdf = os.path.join(RESULT_FOLDER, 'final_selected.pdf')

    create_index_pdf(combined_result, index_pdf)
    merge_pdfs(original_pdf_path, index_pdf, final_pdf)
    return send_file(final_pdf, as_attachment=True)

@app.route('/cari_frasa', methods=['POST'])
def cari_frasa():
    frasa = request.form.get('frasa_manual', '').strip().lower()
    filepath = session.get('uploaded_pdf_path')
    rake_result = session.get('rake_result', {})

    if not frasa or not filepath or not rake_result:
        print("Gagal: frasa / filepath / rake_result tidak lengkap")
        return redirect('/')

    documents = read_pdf_with_pages(filepath)
    frasa_pages = []
    for page_number, text in documents:
        if frasa in text.lower():
            frasa_pages.append(page_number)

    # Ambil hasil sebelumnya
    results = session.get('frasa_manual_result', [])
    results.append({
        'frasa': frasa,
        'pages': sorted(list(set(frasa_pages))) if frasa_pages else []
    })
    session['frasa_manual_result'] = results

    # Gabungkan hasil
    manual_as_dict = {
        item['frasa']: {
            'pages': item['pages'],
            'frequency': len(item['pages']) if item['pages'] else 0,
            'similarity': 0.0
        }
        for item in results
    }
    combined_full_result = {**rake_result, **manual_as_dict}

    # Buat ulang PDF index lengkap
    filename = os.path.basename(filepath)
    index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
    final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
    create_index_pdf(combined_full_result, index_pdf)
    merge_pdfs(filepath, index_pdf, final_pdf)
    session['download_link'] = f"/download/{os.path.basename(final_pdf)}"
    return redirect('/')

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)