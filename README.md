# Auto_Indexing
Sistem Automatic Indexing Dokumen/Buku Digital

Cara menjalankan:
1. Pastikan sudah menginstall dependensi: <br>
``` pip install flask nltk gensim PyPDF2 rake-nltk Sastrawi scikit-learn PyMuPDF reportlab Flask-Session ```<br>
(jika terjadi error, pastikan sudah menginstall python (python digunakan versi 3.11), lalu menginstall pip pythonnya dengan ``` python -m ensurepip --upgrade ```)<br><br>
2. Download pre-trained FastText model<br>
Karena file model .vec terlalu besar untuk disimpan di GitHub, kamu perlu mengunduhnya manual:<br>
Link: `https://fasttext.cc/docs/en/crawl-vectors.html `<br>
Teks Bahasa Indonesia: `cc.id.300.vec.gz`<br>
Setelah diunduh, ekstrak file .gz sehingga menghasilkan file:
`cc.id.300.vec`<br>
Tempatkan file cc.id.300.vec di folder proyek utama.<br><br>
3. Jalankan aplikasinya: <br>
``` python auto_indexing.py ```<br><br>
4. Buka browser dan akses: http://127.0.0.1:5000