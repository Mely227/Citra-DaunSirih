import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

print("==================================================")
print("  EKSPERIMEN KLASIFIKASI DAUN SIRIH DENGAN SVM")
print("==================================================")

# ==========================================
# 1. PEMUATAN DIREKTORI DATASET LOKAL
# ==========================================
base_dir = 'dataset_sirih' 

if not os.path.exists(base_dir):
    print(f"Error: Folder '{base_dir}' tidak ditemukan di project VS Code Anda!")
else:
    # Penyesuaian jika ada subfolder bersarang
    subfolders = os.listdir(base_dir)
    if len(subfolders) == 1 and os.path.isdir(os.path.join(base_dir, subfolders[0])):
        base_dir = os.path.join(base_dir, subfolders[0])

    print(f"[INFO] Direktori aktif dataset: {base_dir}")

    # ==========================================
    # 2. BUKTI VISUAL SEBELUM KLASIFIKASI (SAMPEL GAMBAR)
    # ==========================================
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"[INFO] Kelas daun yang ditemukan: {classes}")

    print("[INFO] Menampilkan jendela sampel citra daun sirih (Bukti Sebelum Klasifikasi)...")
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(classes):
        class_path = os.path.join(base_dir, class_name)
        img_paths = glob.glob(os.path.join(class_path, '**/*.jpg'), recursive=True) + glob.glob(os.path.join(class_path, '**/*.png'), recursive=True)
        if img_paths:
            img_sample = plt.imread(img_paths[0])
            plt.subplot(1, len(classes), i + 1)
            plt.imshow(img_sample)
            plt.title(f"Kelas: {class_name}")
            plt.axis('off')
    plt.suptitle("Bukti Sebelum Klasifikasi: Sampel Citra Mentah Daun Sirih")
    plt.show(block=False)  # Menampilkan gambar sampel tanpa menghentikan program

    # ==========================================
    # 3. EKSTRAKSI FITUR (FEATURE EXTRACTION)
    # ==========================================
    print("[INFO] Memuat model ekstraktor fitur (EfficientNetB0)...")
    feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    def load_and_extract_features(data_dir, max_images_per_class=50):
        features = []
        labels = []
        
        for class_name in classes:
            class_path = os.path.join(data_dir, class_name)
            img_paths = []
            for ext in ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.JPG', '**/*.PNG', '**/*.jfif'):
                img_paths.extend(glob.glob(os.path.join(class_path, ext), recursive=True))
                
            print(f"       -> Kelas '{class_name}': Ditemukan {len(img_paths)} gambar. Memproses sampel...")
            
            for img_path in img_paths[:max_images_per_class]: 
                try:
                    img = image.load_img(img_path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    
                    # Ekstraksi fitur citra menjadi vektor numerik
                    feat = feature_extractor.predict(x, verbose=0)
                    features.append(feat.flatten())
                    labels.append(class_name)
                except Exception as e:
                    pass
                    
        return np.array(features), np.array(labels)

    # Menjalankan proses ekstraksi fitur
    X, y = load_and_extract_features(base_dir, max_images_per_class=50)
    print(f"\n[BUKTI DATA SEBELUM KLASIFIKASI MASUK SVM]")
    print(f" - Bentuk Vektor Fitur (X): {X.shape}")
    print(f" - Jumlah Total Label (y): {len(y)}")

    # ==========================================
    # 4. TRAIN-TEST SPLIT & PELATIHAN SVM
    # ==========================================
    if X.shape[0] > 0:
        # Membagi data menjadi 80% Latih dan 20% Uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("\n[INFO] Melatih model Support Vector Machine (SVM dengan Kernel RBF)...")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_model.fit(X_train, y_train)
        print("[INFO] Pelatihan model SVM selesai!")

        # ==========================================
        # 5. EVALUASI & BUKTI SESUDAH KLASIFIKASI
        # ==========================================
        y_pred = svm_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n[BUKTI SESUDAH KLASIFIKASI]")
        print(f" - Akurasi Model SVM pada Data Uji: {acc * 100:.2f}%\n")
        
        print("Laporan Klasifikasi Lengkap:")
        print(classification_report(y_test, y_pred))

        # Visualisasi Confusion Matrix (Pop-up Jendela Grafik)
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Prediksi Model')
        plt.ylabel('Label Aktual')
        plt.title('Bukti Sesudah Klasifikasi: Confusion Matrix Model SVM')
        plt.show()
    else:
        print("\n[ERROR] Data fitur (X) kosong. Periksa kembali struktur folder dataset Anda.")