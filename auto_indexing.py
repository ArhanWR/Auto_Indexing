from flask import Flask, render_template, request, send_file, session, redirect, url_for
from flask_session import Session
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
app.config['SESSION_TYPE'] = 'filesystem'  # Bisa juga 'redis', 'mongodb', dsb
app.config['SESSION_PERMANENT'] = False
Session(app)

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

def extract_rake_keywords(documents, title="", min_length=1, max_length=3, sort_by="frequency"):
    rake = Rake(stopwords=stop_words)     # bisa diganti rake_score atau frequency, ini ^
    phrase_counter = Counter()
    page_map = {}
    phrase_scores = {}

    for page_number, page_text in documents:
        cleaned_text = clean_text(page_text)
        rake.extract_keywords_from_text(cleaned_text)

        ranked_with_scores = rake.get_ranked_phrases_with_scores()
        for score, kw in ranked_with_scores:
            words = kw.split()
            if (
                min_length <= len(words) <= max_length and
                all(w.isalpha() and len(w) >= 3 for w in words)
            ):
                phrase_counter[kw] += cleaned_text.lower().count(kw.lower())
                page_map.setdefault(kw, set()).add(page_number)
                phrase_scores[kw] = max(phrase_scores.get(kw, 0), score)

    scored_phrases = []
    for phrase, freq in phrase_counter.items():
        sim_score = float(compute_similarity(phrase, title)) if title else 0.0
        if 0.3 <= sim_score <= 1.0:
            scored_phrases.append((phrase, freq, phrase_scores.get(phrase, 0), sim_score))

    if sort_by == "rake_score":
        sorted_phrases = sorted(scored_phrases, key=lambda x: x[2], reverse=True)[:100]
    else:  # default to frequency
        sorted_phrases = sorted(scored_phrases, key=lambda x: x[1], reverse=True)[:100]

    result = {
        phrase: {
            "frequency": freq,
            "rake_score": round(rake_score, 4),
            "similarity": round(sim_score, 4),
            "pages": sorted(page_map.get(phrase, []))
        }
        for phrase, freq, rake_score, sim_score in sorted_phrases
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
            session.pop('frasa_manual_result', None)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            documents = read_pdf_with_pages(filepath)
            title = manual_title
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
        return redirect('/')

    results = session.get('frasa_manual_result', [])
    # Cek duplikasi
    existing_phrases = [item['frasa'] for item in results]
    if frasa in existing_phrases:
        return redirect('/')  # Frasa sudah ada, tidak ditambahkan lagi

    documents = read_pdf_with_pages(filepath)
    frasa_pages = []
    for page_number, text in documents:
        if frasa in text.lower():
            frasa_pages.append(page_number)

    results.append({
        'frasa': frasa,
        'pages': sorted(set(frasa_pages)) if frasa_pages else []
    })
    session['frasa_manual_result'] = results

    # Gabung ulang & update PDF
    manual_as_dict = {
        item['frasa']: {
            'pages': item['pages'],
            'frequency': len(item['pages']) if item['pages'] else 0,
            'similarity': 0.0
        }
        for item in results
    }
    combined_full_result = {**rake_result, **manual_as_dict}
    filename = os.path.basename(filepath)
    index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
    final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
    create_index_pdf(combined_full_result, index_pdf)
    merge_pdfs(filepath, index_pdf, final_pdf)
    session['download_link'] = f"/download/{os.path.basename(final_pdf)}"
    return redirect('/')

@app.route('/hapus_frasa/<frasa>', methods=['POST'])
def hapus_frasa(frasa):
    results = session.get('frasa_manual_result', [])
    filepath = session.get('uploaded_pdf_path')
    rake_result = session.get('rake_result', {})

    # Hapus frasa yang cocok
    results = [item for item in results if item['frasa'] != frasa]
    session['frasa_manual_result'] = results
    # Perbarui PDF
    manual_as_dict = {
        item['frasa']: {
            'pages': item['pages'],
            'frequency': len(item['pages']) if item['pages'] else 0,
            'similarity': 0.0
        }
        for item in results
    }
    combined_full_result = {**rake_result, **manual_as_dict}
    filename = os.path.basename(filepath)
    index_pdf = os.path.join(RESULT_FOLDER, 'indexing.pdf')
    final_pdf = os.path.join(RESULT_FOLDER, f"final_{filename}")
    create_index_pdf(combined_full_result, index_pdf)
    merge_pdfs(filepath, index_pdf, final_pdf)
    session['download_link'] = f"/download/{os.path.basename(final_pdf)}"

    return redirect('/')

# EVALUASI PRECISION RECALL DAN COSINE SIMILARITY
def get_phrase_vector(phrase, model):
    words = phrase.split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def average_cosine_similarity(set1, set2, model):
    total_sim = 0
    count = 0
    for p1 in set1:
        vec1 = get_phrase_vector(p1, model)
        best_sim = 0
        for p2 in set2:
            vec2 = get_phrase_vector(p2, model)
            sim = cosine_similarity([vec1], [vec2])[0][0]
            if sim > best_sim:
                best_sim = sim
        total_sim += best_sim
        count += 1
    return total_sim / count if count > 0 else 0

@app.route('/evaluasi', methods=['POST'])
def evaluasi():
    if 'ground_truth' not in request.files:
        return redirect(url_for('index'))

    gt_file = request.files['ground_truth']
    if gt_file and allowed_file(gt_file.filename):
        gt_filename = secure_filename(gt_file.filename)
        gt_path = os.path.join(UPLOAD_FOLDER, gt_filename)
        gt_file.save(gt_path)

        # Ambil hasil frasa dari session
        combined_result = session.get('results', {}).get('combined', {})

        def preprocess_phrases(phrases):
            roman_numerals = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", 
                              "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"}
            cleaned = set()

            for phrase in phrases:
                phrase = phrase.strip()

                # Hanya ambil bagian luar tanda kurung (abaikan isi dalam kurung)
                phrase = re.sub(r"\(.*?\)", "", phrase)

                # Bersihkan simbol dan whitespace ganda
                phrase = re.sub(r"[^\w\s]", "", phrase)  # hapus simbol kecuali spasi
                phrase = re.sub(r"\s+", " ", phrase).lower().strip()

                if (
                    any(c.isalpha() for c in phrase)
                    and len(phrase) > 1
                    and phrase not in roman_numerals
                    and not re.fullmatch(r"[a-zA-Z]", phrase)
                ):
                    cleaned.add(phrase)

            return cleaned

        # Bersihkan hasil frasa sistem dan ground truth
        generated_keywords = preprocess_phrases(combined_result.keys())

        def extract_keywords_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            keywords = set()
            for page in doc:
                text = page.get_text()
                lines = text.split("\n")
                for line in lines:
                    parts = re.split(r"\s*[·,;:\-–—_]\s*", line.strip())
                    for part in parts:
                        phrase = " ".join(w for w in part.strip().split() if not w.isdigit())
                        phrase = phrase.lower().strip()
                        if any(c.isalpha() for c in phrase) and len(phrase) > 1:
                            keywords.add(phrase)
            return preprocess_phrases(keywords)

        ground_truth_keywords = extract_keywords_from_pdf(gt_path)

        # Evaluasi
        true_positives = generated_keywords & ground_truth_keywords
        false_positives = generated_keywords - ground_truth_keywords
        false_negatives = ground_truth_keywords - generated_keywords

        precision = len(true_positives) / len(generated_keywords) if generated_keywords else 0
        recall = len(true_positives) / len(ground_truth_keywords) if ground_truth_keywords else 0
        cosine_sim = average_cosine_similarity(generated_keywords, ground_truth_keywords, w2v_model)

        evaluation_result = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "cosine_similarity": round(cosine_sim, 4),
            "tp": sorted(true_positives),
            "fp": sorted(false_positives),
            "fn": sorted(false_negatives)
        }

        return render_template("index.html", evaluation_result=evaluation_result)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)