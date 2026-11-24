"""Konfigurasi & konstanta aplikasi (tanpa dependensi Streamlit)."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_sirih_svm.joblib")

APP_TITLE = "Klasifikasi Penyakit Daun Sirih"
TARGET_ACCURACY = 0.85

IMG_EXT = (".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Nama folder yang BUKAN nama kelas (pembagi data / folder generik).
# Contoh: Hijau_Sehat/images/*.jpg  -> kelas "Hijau_Sehat"
#         train/Hijau_Sehat/*.jpg   -> kelas "Hijau_Sehat"
IGNORED_FOLDER_NAMES = {
    "train", "training", "val", "valid", "validation", "test", "testing",
    "images", "image", "imgs", "img", "photos", "pictures", "data", "dataset",
}

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# Default sengaja TANPA fitur warna agar sesuai judul: "berdasarkan ekstraksi ciri tekstur".
DEFAULT_CFG = {
    "size": 128,
    "levels": 32,
    "distances": [1, 2, 3],
    "angles": [0, 45, 90, 135],
    "avg_angle": True,
    "use_color": False,
}
