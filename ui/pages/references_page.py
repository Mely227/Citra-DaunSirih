"""Halaman Referensi."""
from ui import components as C

REFS = [
    ("Jurnal 1 (SINTA 2)", "N. Neneng, K. Adi, R. Isnanto, “Support Vector Machine Untuk Klasifikasi Citra Jenis Daging "
     "Berdasarkan Tekstur Menggunakan Ekstraksi Ciri GLCM,” Jurnal Sistem Informasi Bisnis."),
    ("Jurnal 2 (Nasional)", "Tim Peneliti, “Pengenalan Wajah dengan Algoritma Support Vector Machine dan Sobel Edge Detection "
     "Berbasis Computer Vision,” Jurnal Informatika."),
    ("Jurnal 3 (Internasional)", "“Implementation of an EfficientNet-B4 Model Architecture with a Convolutional Block Attention "
     "Module (CBAM) for Betel Leaf Disease Classification.”"),
    ("Jurnal 4 (Internasional)", "“Betel Leaf Disease Classification Under Limited Data Using ConvNeXt-Tiny and UCB-Guided "
     "Adaptive Augmentation.”"),
    ("Jurnal 5 (IEEE Access, Q1)", "S. Swamy et al., “EffNet-SVM: A Hybrid Model for Diabetic Retinopathy Classification "
     "Using Retinal Fundus Images,” IEEE Access."),
    ("Dataset", "A. Bauravindah, “Betel Leaf Disease Classification Dataset,” Kaggle."),
]


def render():
    C.show(C.page_header("Referensi", "Jurnal acuan dan sumber dataset proyek."))
    for title, text in REFS:
        C.show(C.card(title, "<p>%s</p>" % C.esc(text)))
    C.show(C.alert("info", "Lengkapi tahun, volume, halaman, dan DOI setiap referensi di laporan sesuai sumber aslinya."))
