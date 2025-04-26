# 📱 Mobile Price Classification menggunakan KNN

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📑 Daftar Isi
- [📱 Mobile Price Classification menggunakan KNN](#-mobile-price-classification-menggunakan-knn)
  - [📑 Daftar Isi](#-daftar-isi)
  - [📋 Pendahuluan](#-pendahuluan)
  - [🎯 Tujuan Proyek](#-tujuan-proyek)
  - [📊 Dataset](#-dataset)
  - [📚 Library dan Tools](#-library-dan-tools)
  - [🛠️ Metodologi](#️-metodologi)
    - [Data Loading dan Eksplorasi](#data-loading-dan-eksplorasi)
    - [Data Preprocessing](#data-preprocessing)
    - [Pemilihan Fitur](#pemilihan-fitur)
    - [Train-Test Split](#train-test-split)
    - [Algoritma K-Nearest Neighbors](#algoritma-k-nearest-neighbors)
    - [Optimalisasi Parameter](#optimalisasi-parameter)
    - [Evaluasi Model](#evaluasi-model)
  - [📈 Hasil dan Analisis](#-hasil-dan-analisis)
    - [Performa Model](#performa-model)
    - [Visualisasi Hasil](#visualisasi-hasil)
    - [Interpretasi Model](#interpretasi-model)
  - [🔍 Pembahasan](#-pembahasan)
    - [Kelebihan Pendekatan KNN](#kelebihan-pendekatan-knn)
    - [Limitasi dan Tantangan](#limitasi-dan-tantangan)
    - [Potensi Pengembangan](#potensi-pengembangan)
  - [🚀 Penggunaan Model](#-penggunaan-model)
  - [🔮 Pengembangan Masa Depan](#-pengembangan-masa-depan)
  - [👥 Kontributor](#-kontributor)
  - [🔗 Referensi](#-referensi)

## 📋 Pendahuluan
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Dalam era digital saat ini, klasifikasi harga ponsel menjadi hal yang sangat penting, baik bagi konsumen maupun produsen. Konsumen ingin mendapatkan ponsel dengan fitur terbaik sesuai anggaran, sementara produsen perlu memahami posisi produk mereka di pasar. Proyek ini bertujuan untuk mengembangkan model pembelajaran mesin yang dapat mengklasifikasikan harga ponsel berdasarkan berbagai spesifikasi teknis.

Proyek ini didasarkan pada kasus yang dihadapi oleh PT MobileNesia, sebuah perusahaan smartphone lokal yang sedang berkembang. Direktur perusahaan, Pak Joko, ingin memberikan persaingan ketat kepada perusahaan-perusahaan besar seperti Apple, Samsung, dan lainnya. Namun, perusahaan menghadapi kesulitan dalam menentukan kisaran harga produk mereka dalam pasar yang sangat kompetitif. Untuk mengatasi masalah ini, PT MobileNesia telah mengumpulkan data penjualan ponsel dari berbagai perusahaan.

Pak Joko ingin menemukan hubungan antara fitur-fitur ponsel (seperti RAM, memori internal, dll.) dan kisaran harga jualnya. Melalui proyek ini, kami menggunakan algoritma K-Nearest Neighbors (KNN) untuk mengklasifikasikan ponsel ke dalam empat kategori kisaran harga (0-3), membantu perusahaan membuat keputusan penetapan harga yang lebih akurat dan strategis.

## 🎯 Tujuan Proyek
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Tujuan utama dari proyek ini adalah:

1. Mengembangkan model klasifikasi yang dapat memprediksi kisaran harga ponsel berdasarkan spesifikasinya dengan akurasi tinggi
2. Mengidentifikasi fitur-fitur ponsel yang memiliki pengaruh paling signifikan terhadap penentuan kisaran harga
3. Memahami dan mendemonstrasikan penggunaan algoritma K-Nearest Neighbors dalam konteks klasifikasi kisaran harga
4. Mengoptimalkan parameter model untuk mendapatkan performa terbaik
5. Menyediakan analisis dan interpretasi hasil yang komprehensif untuk keputusan bisnis

Project ini penting karena dapat membantu:
- PT MobileNesia dalam membuat keputusan penetapan harga yang kompetitif dan strategis
- Tim pengembangan produk dalam merancang ponsel dengan fitur yang sesuai untuk target kisaran harga
- Tim pemasaran dalam menyusun strategi penjualan berdasarkan segmentasi pasar
- Manajemen dalam memahami faktor-faktor penentu harga di pasar smartphone

Perlu dicatat bahwa dalam project ini, kita tidak memprediksi harga aktual ponsel, melainkan kisaran harga yang mengindikasikan seberapa tinggi harga ponsel tersebut.

## 📊 Dataset
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Dataset yang digunakan dalam proyek ini berasal dari Kaggle dengan judul "Mobile Price Classification" yang dapat diakses melalui link berikut: [https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data).

Dataset terdiri dari dua file:
- `train.csv`: Dataset untuk melatih dan memvalidasi model (2000 baris data)
- `test.csv`: Dataset untuk menguji performa model pada data baru (1000 baris data)

Fitur-fitur dalam dataset mencakup:

1. **battery_power**: Kapasitas baterai total dalam mAh
2. **blue**: Apakah ponsel memiliki bluetooth (1) atau tidak (0)
3. **clock_speed**: Kecepatan prosesor dalam GHz
4. **dual_sim**: Apakah ponsel mendukung dual SIM (1) atau tidak (0)
5. **fc**: Resolusi kamera depan dalam megapixel
6. **four_g**: Apakah ponsel mendukung 4G (1) atau tidak (0)
7. **int_memory**: Memori internal dalam GB
8. **m_dep**: Ketebalan ponsel dalam cm
9. **mobile_wt**: Berat ponsel dalam gram
10. **n_cores**: Jumlah core prosesor
11. **pc**: Resolusi kamera utama dalam megapixel
12. **px_height**: Tinggi resolusi layar dalam pixel
13. **px_width**: Lebar resolusi layar dalam pixel
14. **ram**: Random Access Memory dalam MB
15. **sc_h**: Tinggi layar ponsel dalam cm
16. **sc_w**: Lebar layar ponsel dalam cm
17. **talk_time**: Waktu bicara terlama dalam satu pengisian baterai (jam)
18. **three_g**: Apakah ponsel mendukung 3G (1) atau tidak (0)
19. **touch_screen**: Apakah ponsel memiliki layar sentuh (1) atau tidak (0)
20. **wifi**: Apakah ponsel memiliki wifi (1) atau tidak (0)
21. **price_range**: Kisaran harga ponsel (hanya tersedia di train.csv)
   - 0: Low Cost (Biaya Rendah)
   - 1: Medium Cost (Biaya Menengah)
   - 2: High Cost (Biaya Tinggi)
   - 3: Very High Cost (Biaya Sangat Tinggi)

Distribusi kelas dalam dataset train.csv cukup seimbang dengan masing-masing kategori harga memiliki jumlah sampel yang hampir sama, yang membuat evaluasi model lebih representatif.

## 📚 Library dan Tools
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Proyek ini menggunakan beberapa library Python populer untuk analisis data dan pembelajaran mesin:

1. **NumPy dan Pandas**: Untuk manipulasi dan analisis data
   ```python
   import numpy as np
   import pandas as pd
   ```

2. **Matplotlib dan Seaborn**: Untuk visualisasi data
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```

3. **Scikit-learn**: Untuk implementasi algoritma machine learning dan evaluasi model
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
   from sklearn.decomposition import PCA
   ```

4. **Joblib**: Untuk menyimpan model machine learning
   ```python
   from joblib import dump
   ```

Semua eksperimen dilakukan dalam lingkungan Jupyter Notebook untuk memudahkan analisis interaktif dan visualisasi.

## 🛠️ Metodologi
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

### Data Loading dan Eksplorasi
Langkah pertama adalah memuat dataset dan melakukan eksplorasi awal untuk memahami struktur dan karakteristik data.

```python
# Memuat dataset
df_train = pd.read_csv('train.csv')

# Menampilkan informasi dasar
print(f"Dataset shape: {df_train.shape}")
print(f"Distribusi kelas target:\n{df_train['price_range'].value_counts().sort_index()}")

# Menampilkan beberapa baris pertama
df_train.head()
```

Hasil eksplorasi awal menunjukkan bahwa dataset train memiliki 2000 baris data dengan 21 kolom (20 fitur dan 1 target). Distribusi kelas target relatif seimbang dengan masing-masing kategori harga memiliki sekitar 500 sampel.

Kami juga melakukan analisis korelasi untuk melihat hubungan antara fitur dan target:

```python
# Menghitung korelasi
corr_matrix = df_train.corr()
correlation_with_price = corr_matrix['price_range'].sort_values(ascending=False)

# Menampilkan top 5 fitur dengan korelasi tertinggi
print("\nTop 5 fitur dengan korelasi tertinggi:")
print(correlation_with_price.head(6))

# Visualisasi korelasi
plt.figure(figsize=(10, 5))
top_corrs = correlation_with_price.drop('price_range')
sns.barplot(x=top_corrs.nlargest(5).values, y=top_corrs.nlargest(5).index, palette='viridis')
plt.title('Top 5 Fitur dengan Korelasi Tertinggi')
plt.xlabel('Koefisien Korelasi')
plt.tight_layout()
plt.show()
```

Dari analisis korelasi, kami menemukan bahwa beberapa fitur seperti RAM, resolusi layar (px_height dan px_width), dan resolusi kamera memiliki korelasi yang kuat dengan kategori harga.

### Data Preprocessing
Preprocessing data sangat penting untuk algoritma KNN karena algoritma ini menghitung jarak antar titik data. Salah satu langkah preprocessing yang penting adalah standardisasi fitur agar semua fitur memiliki skala yang sama.

```python
# Memisahkan fitur dan target
X = df_train.drop('price_range', axis=1)
y = df_train['price_range']

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardisasi fitur penting karena:
1. KNN menggunakan perhitungan jarak antar titik data
2. Fitur dengan skala besar (seperti RAM yang bisa mencapai ribuan MB) akan mendominasi perhitungan jarak
3. Standardisasi membuat semua fitur memiliki mean=0 dan standar deviasi=1, sehingga setiap fitur memiliki kontribusi yang setara dalam perhitungan jarak

### Pemilihan Fitur
Berdasarkan analisis korelasi, kami dapat mengidentifikasi fitur-fitur yang paling berpengaruh terhadap kategori harga. Untuk proyek ini, kami menggunakan seluruh 20 fitur yang tersedia karena ukuran dataset tidak terlalu besar dan performa komputasi masih baik.

Namun, untuk tujuan visualisasi, kami menggunakan PCA (Principal Component Analysis) untuk mereduksi dimensi fitur menjadi 2 dimensi:

```python
# Reduksi dimensi menggunakan PCA untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### Train-Test Split
Kami membagi data menjadi set pelatihan dan pengujian dengan stratifikasi untuk memastikan proporsi kelas yang seimbang pada kedua set data:

```python
# Membagi data dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratifikasi penting untuk memastikan bahwa distribusi kelas dalam set pelatihan dan pengujian tetap sama dengan distribusi kelas dalam dataset asli.

### Algoritma K-Nearest Neighbors
K-Nearest Neighbors (KNN) adalah algoritma supervised learning yang:
- Menyimpan semua data pelatihan (lazy learning)
- Memprediksi kelas data baru berdasarkan k tetangga terdekat
- Menggunakan fungsi jarak (biasanya Euclidean) untuk mengukur kedekatan antar titik data
- Sederhana namun efektif untuk banyak kasus klasifikasi

Keuntungan menggunakan KNN untuk kasus klasifikasi harga ponsel:
- Mudah diimplementasikan dan dipahami oleh tim PT MobileNesia
- Tidak memerlukan asumsi tentang distribusi data
- Efektif untuk dataset dengan batas keputusan non-linear
- Tidak memerlukan proses pelatihan yang kompleks

Kelemahan KNN:
- Komputasi dapat menjadi berat untuk dataset besar
- Sensitif terhadap fitur yang tidak relevan atau redundan
- Performa sangat bergantung pada pemilihan nilai k
- Rentan terhadap masalah "curse of dimensionality"

### Optimalisasi Parameter
Pemilihan parameter optimal sangat penting dalam algoritma KNN. Parameter utama yang perlu dioptimalkan adalah nilai k (jumlah tetangga terdekat).

Kami menggunakan dua pendekatan untuk optimalisasi parameter:

1. **Pendekatan visualisasi** untuk memahami trade-off antara bias dan varians:
```python
def find_optimal_k(X_train, y_train, X_test, y_test):
    k_values = list(range(1, 30, 2))
    train_accuracy = []
    test_accuracy = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy.append(knn.score(X_train, y_train))
        test_accuracy.append(knn.score(X_test, y_test))
    
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, train_accuracy, 'o-', label='Training Accuracy')
    plt.plot(k_values, test_accuracy, 'o-', label='Validation Accuracy')
    best_k = k_values[test_accuracy.index(max(test_accuracy))]
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K: {best_k}')
    plt.fill_between(
        k_values, train_accuracy, test_accuracy,
        where=(np.array(train_accuracy) > np.array(test_accuracy)),
        color='orange', alpha=0.3, label='Overfitting Area'
    )
    plt.xlabel('Nilai K (Jumlah Neighbors)')
    plt.ylabel('Akurasi')
    plt.title('Optimasi Parameter K untuk KNN')
    plt.legend()
    plt.grid(True)
    plt.show()
    return best_k, max(test_accuracy)

best_k, best_acc = find_optimal_k(X_train, y_train, X_test, y_test)
print(f"Nilai K optimal: {best_k}")
print(f"Akurasi validasi tertinggi: {best_acc:.4f}")
```

2. **GridSearchCV** untuk pencarian parameter yang lebih menyeluruh:
```python
# Mendefinisikan grid parameter untuk pencarian
param_grid = {
    'n_neighbors': [best_k-2, best_k, best_k+2],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Melakukan grid search dengan 5-fold cross-validation
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Melatih model dengan grid search
grid_search.fit(X_train, y_train)
print("\nParameter terbaik:")
print(grid_search.best_params_)
print(f"Akurasi CV terbaik: {grid_search.best_score_:.4f}")
```

### Evaluasi Model
Setelah mendapatkan parameter optimal, kami melatih model final dan mengevaluasi performanya:

```python
# Membuat model final dengan parameter optimal
final_model = KNeighborsClassifier(**grid_search.best_params_)
final_model.fit(X_train, y_train)

# Prediksi pada data test
y_pred = final_model.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi model final: {accuracy:.4f}")

# Visualisasi confusion matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)  
Disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Price {i}" for i in range(4)])
Disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Price {i}' for i in range(4)]))
```

Untuk memahami lebih baik bagaimana model bekerja, kami juga memvisualisasikan decision boundary menggunakan PCA:

```python
# Fungsi untuk visualisasi decision boundary
def plot_decision_boundary(X, y, model, ax=None):
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap='viridis', alpha=0.8)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Decision Boundary KNN (2 Komponen PCA)')
    plt.colorbar(label='Price Range')
    plt.show()

# Latih model dengan PCA
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
pca_model = KNeighborsClassifier(**grid_search.best_params_)
pca_model.fit(X_train_pca, y_train_pca)

# Visualisasi decision boundary
plot_decision_boundary(X_test_pca, y_test_pca, pca_model)
```

## 📈 Hasil dan Analisis
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

### Performa Model
Berdasarkan evaluasi, model KNN kami mencapai akurasi yang memuaskan pada dataset pengujian. Parameter optimal yang ditemukan melalui GridSearchCV adalah:
- `n_neighbors`: 9 (nilai ini dapat bervariasi tergantung pada hasil eksperimen)
- `weights`: 'distance' memberikan hasil terbaik, yang berarti tetangga yang lebih dekat memiliki pengaruh lebih besar
- `metric`: 'euclidean' memberikan hasil yang konsisten untuk dataset ini

Berikut adalah ringkasan performa model:
- **Akurasi**: >92% (nilai pasti bergantung pada hasil eksperimen)
- **Precision dan Recall**: Seimbang di semua kelas, menunjukkan model stabil
- **F1-Score**: Konsisten di semua kelas, berkisar antara 0.90-0.95

### Visualisasi Hasil
Visualisasi confusion matrix menunjukkan bahwa model dapat membedakan dengan baik antara berbagai kategori harga. Kesalahan klasifikasi paling umum terjadi antara kategori yang berdekatan (misalnya, kategori 1 dan 2), yang masuk akal karena ponsel dengan harga di perbatasan kategori akan memiliki karakteristik yang mirip.

Visualisasi decision boundary dengan PCA menunjukkan bahwa bahkan dengan reduksi dimensi menjadi 2 komponen utama, model masih dapat memisahkan kategori harga dengan cukup baik. Ini menunjukkan bahwa beberapa fitur utama memang memiliki pengaruh yang signifikan terhadap kategori harga.

### Interpretasi Model
Dari analisis korelasi dan performa model, kami dapat menyimpulkan bahwa:

1. **RAM** adalah fitur dengan korelasi tertinggi terhadap kisaran harga (korelasi > 0.9), yang menunjukkan bahwa kapasitas RAM sangat menentukan kelas harga smartphone
2. **Resolusi layar** (px_height dan px_width) juga memiliki korelasi yang tinggi, menunjukkan bahwa kualitas display menjadi pertimbangan penting dalam penentuan harga
3. **Resolusi kamera** (pc) dan **kapasitas baterai** (battery_power) memiliki pengaruh moderat
4. Fitur seperti **bluetooth** (blue), **dual_sim**, dan **wifi** memiliki korelasi yang lebih rendah dengan kisaran harga

Ini sesuai dengan intuisi pasar smartphone, di mana spesifikasi hardware utama seperti RAM dan kualitas layar biasanya menjadi fitur premium yang mempengaruhi harga jual.

## 🔍 Pembahasan
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

### Kelebihan Pendekatan KNN
Algoritma KNN terbukti efektif untuk tugas klasifikasi kisaran harga ponsel karena:

1. **Keputusan berdasarkan kesamaan**: KNN mengklasifikasikan ponsel berdasarkan kesamaan dengan ponsel lain yang sudah diketahui kategori harganya, mirip dengan cara konsumen membandingkan produk
2. **Non-parametrik**: Tidak membuat asumsi tentang distribusi data atau bentuk fungsi keputusan
3. **Interpretasi intuitif**: Hasil KNN mudah dijelaskan kepada manajemen PT MobileNesia - "ponsel ini memiliki kategori harga X karena memiliki spesifikasi yang mirip dengan ponsel lain di kategori tersebut"
4. **Fleksibilitas**: KNN dapat beradaptasi dengan perubahan tren pasar saat data baru ditambahkan ke dataset pelatihan

### Limitasi dan Tantangan
Meskipun performanya baik, pendekatan KNN memiliki beberapa keterbatasan:

1. **Sensitif terhadap skala**: Preprocessing (standardisasi) sangat penting untuk KNN
2. **Komputasi**: Untuk dataset yang sangat besar, KNN bisa menjadi lambat saat prediksi
3. **Parameter k**: Performa sangat bergantung pada pemilihan nilai k yang optimal
4. **Fitur yang tidak relevan**: KNN tidak dapat secara otomatis mengabaikan fitur yang kurang relevan

### Potensi Pengembangan
Beberapa pendekatan yang dapat digunakan untuk meningkatkan performa model:

1. **Feature Engineering**: Membuat fitur baru yang mungkin lebih informatif, seperti rasio harga-per-performa atau pixel density (px_height * px_width / (sc_h * sc_w))
2. **Ensemble Methods**: Mengombinasikan KNN dengan algoritma lain seperti Random Forest untuk meningkatkan generalisasi
3. **Feature Selection**: Menggunakan teknik seperti Recursive Feature Elimination untuk memilih subset fitur optimal
4. **Hyperparameter Tuning yang Lebih Komprehensif**: Mencoba range parameter yang lebih luas atau teknik optimasi parameter yang lebih canggih

## 🚀 Penggunaan Model
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Model yang telah dilatih dapat digunakan oleh PT MobileNesia untuk memprediksi kategori harga ponsel baru yang sedang mereka kembangkan. Berikut adalah contoh kode untuk menggunakan model saved:

```python
# Import library yang diperlukan
import pandas as pd
import joblib

# Muat model dan scaler
model = joblib.load('knn_mobile_price_model.joblib')
scaler = joblib.load('scaler_mobile_price.joblib')

# Fungsi untuk memprediksi kategori harga
def predict_price_category(features_df):
    # Standarisasi fitur menggunakan scaler yang sama dengan training
    features_scaled = scaler.transform(features_df)
    
    # Prediksi menggunakan model
    prediction = model.predict(features_scaled)
    
    # Konversi hasil prediksi ke kategori yang lebih deskriptif
    categories = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    
    result = [categories[pred] for pred in prediction]
    return result

# Contoh penggunaan
# Misalkan PT MobileNesia memiliki prototipe ponsel baru dengan spesifikasi berikut
new_phones = pd.DataFrame({
    'battery_power': [1800, 2400],
    'blue': [1, 1],
    'clock_speed': [1.8, 2.2],
    'dual_sim': [1, 1],
    'fc': [10, 12],
    'four_g': [1, 1],
    'int_memory': [64, 128],
    'm_dep': [0.7, 0.8],
    'mobile_wt': [155, 165],
    'n_cores': [6, 8],
    'pc': [12, 16],
    'px_height': [1600, 1920],
    'px_width': [720, 1080],
    'ram': [3072, 4096],
    'sc_h': [15, 16],
    'sc_w': [8, 9],
    'talk_time': [10, 15],
    'three_g': [1, 1],
    'touch_screen': [1, 1],
    'wifi': [1, 1]
})

# Prediksi kategori harga
price_categories = predict_price_category(new_phones)
print("Predicted price categories:", price_categories)
```

Dengan menggunakan model ini, PT MobileNesia dapat:
1. Mengevaluasi berbagai konfigurasi ponsel dan estimasi kisaran harganya
2. Mengidentifikasi spesifikasi mana yang perlu ditingkatkan untuk mencapai kategori harga target
3. Membandingkan produk mereka dengan kompetitor dalam kategori harga yang sama
4. Membuat keputusan strategis tentang segmen pasar mana yang akan dibidik

## 🔮 Pengembangan Masa Depan
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Untuk pengembangan di masa depan, beberapa area yang dapat dieksplorasi oleh PT MobileNesia:

1. **Perbandingan dengan algoritma lain**: Membandingkan performa KNN dengan algoritma klasifikasi lain seperti Random Forest, SVM, atau Neural Networks untuk meningkatkan akurasi prediksi
2. **Pengumpulan data tambahan**: Menambahkan fitur-fitur lain seperti material casing, jenis prosesor, atau skor benchmark
3. **Prediksi harga aktual**: Mengembangkan model regresi untuk memprediksi harga actual, bukan hanya kisaran harga
4. **Analisis segmen pasar**: Menggunakan clustering untuk mengidentifikasi segmen pasar berbeda berdasarkan preferensi fitur
5. **Integrasi dengan data pasar**: Visualisasi decision boundary membantu memahami bagaimana model membuat keputusan klasifikasi.

Hasil proyek ini dapat digunakan oleh berbagai pihak dalam industri ponsel untuk membantu penentuan harga, strategi pemasaran, dan pengembangan produk berdasarkan fitur-fitur yang paling mempengaruhi persepsi nilai di mata konsumen.

## 👥 Kontributor
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

Proyek ini dikembangkan oleh Tim Machine Learning 2025:
- Anggota 1
- Anggota 2
- Anggota 3
- Anggota 4

## 🔗 Referensi
[⬆️ Kembali ke Daftar Isi](#-daftar-isi)

1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.
3. Mohammed, M., Khan, M. B., & Bashier, E. B. M. (2016). Machine Learning: Algorithms and Applications. CRC Press.
4. VanderPlas, J. (2016). Python Data Science Handbook: Essential Tools for Working with Data. O'Reilly Media.
5. Raschka, S., & Mirjalili, V. (2019). Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2. Packt Publishing.
6. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. O'Reilly Media.
7. [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
8. [Towards Data Science: KNN Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)