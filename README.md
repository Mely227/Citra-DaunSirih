# Klasifikasi Penyakit Daun Sirih dengan SVM + GLCM

Web app (Streamlit) untuk UAS Pengolahan Citra Digital.

## Menjalankan
```
pip install -r requirements.txt
streamlit run app.py          # jalankan dari DALAM folder ini
```

## Struktur proyek
```
app.py              router halaman + sidebar
core/               LOGIKA (tanpa Streamlit)
  config.py         konstanta & parameter default
  dataset.py        memuat/memindai dataset, ekstrak ZIP
  features.py       pra-pengolahan + ekstraksi fitur GLCM/HSV
  model.py          training, tuning (GridSearchCV), evaluasi, prediksi
  plots.py          confusion matrix & heatmap tuning
  imageio.py        baca gambar
ui/                 TAMPILAN (tanpa logika SVM)
  style.css         tema bergaya Bootstrap
  components.py     komponen HTML (hero, kartu, badge, progress bar)
  state.py          session_state
  pages/            satu file per halaman
.streamlit/config.toml   tema + batas upload 2 GB
```

## Dataset
Pilih "Folder di komputer" (disarankan) atau "Upload ZIP". Satu folder = satu kelas.
Folder bernama `images`, `train`, `val`, `test` diabaikan sebagai nama kelas, jadi
`Hijau_Sehat/images/*.jpg` dan `train/Hijau_Sehat/*.jpg` sama-sama terbaca sebagai kelas `Hijau_Sehat`.
Bila semua gambar menumpuk di satu folder, aplikasi mencoba membaca label dari file .xlsx/.csv di dataset.

## Catatan percobaan untuk laporan
- Default hanya fitur tekstur GLCM (sesuai judul). Centang "fitur warna HSV" untuk percobaan pembanding.
- Hasil tuning (tabel + heatmap) bisa diunduh dari halaman Training & Tuning.
