# Laporan Proyek Machine Learning 
## Judul proyek : Implementasi Machine Learning dalam Prediksi Diabetes Melitus
## Ditulis oleh : Ester Tri Wahyuningsih -  MC002D5X0841

## Domain Proyek
### Latar Belakang
Diabetes melitus merupakan penyakit kronis yang ditandai dengan tingginya kadar gula darah akibat gangguan produksi atau fungsi insulin. Menurut Organisasi Kesehatan Dunia, World Health Organization [2], diabetes termasuk salah satu penyakit tidak menular dengan prevalensi yang terus meningkat secara global, bahkan mencapai status epidemi di berbagai negara, khususnya di negara berkembang.

Menurut data dari International Diabetes Federation [1], menunjukkan bahwa sekitar 537 juta orang di seluruh dunia hidup dengan diabetes, dan hampir separuh dari mereka belum terdiagnosis atau menerima perawatan yang memadai. Kondisi ini berpotensi menyebabkan komplikasi serius seperti penyakit jantung, stroke, kerusakan ginjal, hingga kebutaan, yang tidak hanya membebani kualitas hidup penderita tapi juga sistem pelayanan kesehatan.

Deteksi dini diabetes sangat penting untuk melakukan intervensi preventif dan manajemen penyakit yang efektif. Namun, skrining massal secara tradisional sering terkendala oleh sumber daya medis yang terbatas dan biaya tinggi. Oleh sebab itu, pendekatan berbasis teknologi informasi dan machine learning menjadi alternatif yang sangat menjanjikan.

Machine learning membantu analisis data medis yang besar dan kompleks untuk menemukan pola dan prediksi risiko penyakit secara akurat dan efisien [3],[5]. Dengan memanfaatkan data klinis dasar seperti usia, indeks massa tubuh (BMI), kadar glukosa, dan riwayat keluarga, model prediksi diabetes dapat membantu tenaga medis dalam mengambil keputusan lebih cepat dan tepat sasaran.

### Tujuan Proyek
Proyek ini bertujuan untuk mengembangkan model klasifikasi berbasis machine learning yang mampu memprediksi risiko diabetes secara akurat dan dapat diandalkan. Model ini diharapkan dapat digunakan sebagai alat bantu skrining awal, terutama di wilayah dengan keterbatasan akses layanan kesehatan, sehingga dapat meningkatkan deteksi dini dan mengurangi beban komplikasi akibat diabetes.

Selain itu, proyek ini juga menyoroti pentingnya mengatasi masalah ketidakseimbangan data pada dataset medis, serta melakukan optimalisasi model melalui tuning hyperparameter agar menghasilkan prediksi yang lebih baik

---

## Business Understanding

### Problem Statements
Sebelum memulai proyek, beberapa pertanyaan penting yang akan dijawab melalui model :
1. Bagaimana cara membangun model machine learning yang mampu memprediksi diabetes secara akurat?
2. Bagaimana mengatasi ketidakseimbangan kelas pada dataset medis agar hasil prediksi tidak bias?
3. Algoritma mana yang memberikan hasil terbaik dalam konteks prediksi ini?

### Goals
Dengan menjawab problem statements di atas, proyek menetapkan target berikut:
1. Menghasilkan model klasifikasi prediktif terhadap risiko diabetes berdasarkan data pasien.
2. Meningkatkan kualitas prediksi melalui penyeimbangan data menggunakan SMOTE.
3. Melakukan tuning hyperparameter untuk memperoleh performa model optimal.

### Solution Statement
Solusi yang dipilih meliputi:
- Melatih dan membandingkan beberapa algoritma: Logistic Regression, Random Forest, dan XGBoost.
- Menerapkan Synthetic Minority Oversampling Technique (SMOTE) untuk mengatasi ketidakseimbangan kelas.
- Meningkatkan performa dengan Grid Search pada Random Forest.
- Menggunakan metrik evaluasi: Accuracy, Precision, Recall, F1-score, dan AUC-ROC.

---

## Data Understanding
Dataset yang digunakan adalah **Pima Indians Diabetes Database** [4], tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download) dan UCI Repository.
Dataset ini terdiri dari 768 entri pasien dan 9 kolom, termasuk target klasifikasi.

### Tabel Fitur Dataset

| Fitur                     | Tipe Data | Deskripsi                                                          |
|--------------------------|-----------|--------------------------------------------------------------------|
| Pregnancies              | Integer   | Jumlah kehamilan                                                   |
| Glucose                  | Integer   | Konsentrasi glukosa plasma                                         |
| BloodPressure            | Integer   | Tekanan darah diastolik (mm Hg)                                   |
| SkinThickness            | Integer   | Ketebalan lipatan kulit triceps (mm)                              |
| Insulin                  | Integer   | Kadar insulin serum 2 jam (mu U/ml)                               |
| BMI                      | Float     | Indeks massa tubuh (kg/m²)                                        |
| DiabetesPedigreeFunction | Float     | Skor riwayat keluarga terhadap diabetes                           |
| Age                      | Integer   | Usia pasien (tahun)                                               |
| Outcome                  | Integer   | Target (1 = diabetes, 0 = tidak diabetes)                         |

### Exploratory Data Analysis (EDA)
Untuk mengetahui kondisi data yang ada kami melakukan EDA untuk memeriksa :
- Missing value dan duplicate data
  Jumlah missing values (nilai kosong) pada dataset awal adalah 0 secara eksplisit, karena dataset tidak memiliki 
  nilai kosong (NaN) dan jumlah data duplikat pada dataset awal adalah 0. Namun, terdapat nilai nol pada beberapa 
  fitur seperti `Glucose` (5 nilai 0) , `Insulin` (374 nilai 0), `BloodPressure` (35 nilai 0), `BMI` (11 nilai 
  0), dan `SkinThickness` (227 nilai 0) yang secara medis dianggap tidak valid dan perlu perlakuan khusus.
- Visualisasi korelasi antar fitur numerik
  Korelasi tertinggi terhadap Outcome ditemukan pada fitur `Glucose`, `BMI`, dan `Age`.
  Korelasi rendah terhadap Outcome ditemukan pada fitur `BloodPressure` dan `SkinThickness`
- Visualisasi distribusi kolom 'outcome'
  Jumlah pasien **tidak diabetes** (Outcome=0): 500
  Jumlah pasien **diabetes** (Outcome=1): 268
- Visualisasi distribusi fitur numerik lainnya dan boxplot fitur numerik terhadap 'outcome'
  Terdapat beberapa outlier yang dapat memberikan insight lebih pada data.
---

## Data Preparation
Pada tahap ini, data dipersiapkan agar siap digunakan untuk pelatihan model. Beberapa proses utama meliputi:
1. **Data Cleaning**: Nilai nol pada fitur medis seperti pada fitur `Glucose` , `Insulin`, `BloodPressure`, `BMI`, dan `SkinThickness` diganti dengan median.
2. **Feature Selection**: Menghapus fitur dengan kontribusi rendah seperti `BloodPressure` dan `SkinThickness`.
3. **Scaling**: Menggunakan `StandardScaler` untuk standardisasi fitur.
4. **Splitting**: Membagi dataset menjadi 80% training dan 20% testing.

Setiap tahapan di atas memiliki tujuan penting dalam memastikan data yang digunakan berkualitas tinggi dan relevan untuk pelatihan model. Penggantian nilai nol dengan median dilakukan karena nilai nol pada fitur medis seperti glukosa dan insulin tidak realistis secara medis dan dapat dianggap sebagai missing value. Median dipilih sebagai pengganti karena lebih tahan terhadap outlier dibandingkan rata-rata. Selanjutnya, proses feature selection dilakukan untuk mengurangi kompleksitas model dan menghindari overfitting dengan hanya mempertahankan fitur yang paling berpengaruh terhadap target. Standardisasi data melalui scaling diperlukan agar semua fitur memiliki skala yang setara, terutama karena beberapa algoritma sangat sensitif terhadap perbedaan skala. Terakhir, pemisahan data menjadi data latih dan data uji memungkinkan evaluasi objektif terhadap performa model terhadap data yang belum pernah dilihat sebelumnya, yang merupakan langkah penting dalam mengukur kemampuan generalisasi model.

---

## Modeling

### Algoritma yang Digunakan
Untuk membangun model prediksi, beberapa algoritma dipilih dan diuji, yaitu:
- **Logistic Regression**:
  Mekanisme Kerja :
  Logistic Regression adalah model linier untuk klasifikasi biner. Model ini menghitung probabilitas kelas 
  menggunakan fungsi sigmoid. Probabilitas tersebut kemudian dikonversi menjadi prediksi kelas. Model ini cocok 
  sebagai baseline karena sederhana, cepat dilatih, dan mudah diinterpretasikan.
  * Kelebihan: Sederhana, cepat dilatih, dan mudah diinterpretasikan. Cocok sebagai baseline.
  * Kekurangan: Tidak menangkap hubungan non-linear dengan baik dan kurang fleksibel untuk data kompleks.
    
- **Random Forest**:
  Mekanisme Kerja :
  Random Forest adalah metode ensemble berbasis pohon keputusan. Setiap pohon dibangun dari subset acak data dan 
  fitur (bagging). Prediksi akhir ditentukan dengan voting mayoritas dari semua pohon.Random Forest mengurangi 
  varian (overfitting) yang tinggi pada pohon tunggal dengan mengombinasikan banyak pohon (bagging), sehingga 
  menyeimbangkan bias dan varian.
  * Kelebihan: Tahan terhadap overfitting, dapat menangani data non-linear dan fitur penting bisa 
  diinterpretasikan.
  * Kekurangan: Lebih lambat dibanding model linier dan kurang efisien dalam hal memori.

  Model Random Forest ini menggunakan konfigurasi default tanpa penyesuaian hyperparameter khusus, dengan 
  beberapa parameter dasar sebagai berikut:
  * n_estimators=100 (default): Jumlah pohon keputusan dalam hutan.
  * random_state=42: Menjamin hasil pelatihan yang konsisten.
    
- **XGBoost**:
  Mekanisme Kerja:
  XGBoost adalah model gradient boosting yang membangun model secara bertahap. Setiap model baru fokus 
  memperbaiki kesalahan dari model sebelumnya. XGBoost mengoptimasi fungsi loss menggunakan pendekatan gradien. 
  XGBoost umumnya memiliki bias rendah, namun perlu regularisasi agar tidak overfit. XGBoost bekerja dengan cara 
  boosting di mana model dibangun secara berurutan, setiap model baru berupaya memperbaiki kesalahan model 
  sebelumnya melalui optimasi fungsi loss dengan pendekatan gradient descent.
  * Kelebihan: Menangani missing value, Bisa menangani fitur yang tidak terurut dan kompleksitas tinggi, serta 
    sangat kuat untuk berbagai jenis data.
  * Kekurangan: Lebih kompleks, tuning parameter memerlukan waktu lebih banyak.

  Model XGBoost menggunakan konfigurasi default dengan beberapa parameter dasar sebagai berikut:
  * n_estimators=100: Jumlah pohon boosting yang digunakan.
  * use_label_encoder=False: Untuk menonaktifkan peringatan dari versi terbaru XGBoost.
  * eval_metric='logloss': Fungsi evaluasi loss yang digunakan untuk klasifikasi.
  * random_state=42: Menjaga konsistensi hasil pelatihan.
  
- **XGBoost + SMOTE**:
  Mekanisme Kerja : 
  Menambahkan SMOTE untuk menyeimbangkan data sebelum pelatihan. Mekanisme Kerja:
  SMOTE (Synthetic Minority Over-sampling Technique) membuat data sintetis untuk kelas minoritas, sehingga 
  distribusi menjadi seimbang. Model XGBoost kemudian dilatih dengan data hasil oversampling.
  * Kelebihan: Meningkatkan recall dan performa untuk kelas minoritas karena menangani imbalance data.
  * Kekurangan: Risiko overfitting karena data sintetis, serta waktu pelatihan bertambah.
    
  Selain itu, untuk meningkatkan performa dan menghindari overfitting, beberapa hyperparameter penting pada 
  XGBoost, yaitu:
  * n_estimators = 100: Jumlah total pohon boosting yang dibangun, semakin banyak pohon model semakin kompleks.
  * max_depth = 4: Batas kedalaman maksimum tiap pohon, guna mengontrol kompleksitas model agar tidak terlalu 
    dalam dan overfit.
  * learning_rate = 0.1: Tingkat pembelajaran yang menentukan kontribusi setiap pohon baru dalam memperbaiki 
    kesalahan sebelumnya.
  * subsample = 0.8: Proporsi data yang digunakan untuk membangun setiap pohon, membantu mencegah overfitting 
    dengan sampling acak.
  * colsample_bytree = 0.8: Proporsi fitur yang dipilih secara acak tiap pohon, menambah keberagaman model.
  * gamma = 0: Parameter regularisasi untuk minimum pengurangan loss dalam pembuatan split.

- **Random Forest + GridSearch**
  Mekanisme kerja :
  Hyperparameter tuning untuk mendapatkan kinerja optimal. GridSearch akan mengeksplorasi semua kombinasi 
  hyperparameter yang diberikan, melakukan pelatihan dan evaluasi silang (cross-validation) pada setiap 
  kombinasi, dan memilih set hyperparameter yang memberikan performa terbaik berdasarkan metrik yang diinginkan
  * Kelebihan: Performa meningkat signifikan setelah tuning, keseimbangan recall dan precision sangat baik.
  * Kekurangan: Proses tuning memakan waktu dan komputasi lebih berat.

  Hyperparameter yang diuji meliputi:
  * n_estimators: Jumlah pohon keputusan yang dibangun, diuji pada nilai [100, 200].
  * max_depth: Kedalaman maksimum pohon, diuji pada nilai [4, 6, 8]. Mengontrol kompleksitas pohon untuk mencegah 
    overfitting atau underfitting.
  * min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi sebuah node, diuji pada nilai [2, 4]. 
    Parameter ini mengatur kapan pohon harus melakukan split lebih lanjut, sehingga mempengaruhi ukuran dan 
    kompleksitas pohon.

---

## Evaluation

### Metrik Evaluasi
Model dievaluasi menggunakan metrik berikut:
- **Accuracy**  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)  
  Mengukur proporsi prediksi yang benar dari seluruh kasus yang diamati.

- **Precision**  
  Precision = TP / (TP + FP)  
  Mengukur seberapa banyak dari prediksi positif yang benar-benar positif (menghindari false positive).

- **Recall (Sensitivity)**  
  Recall = TP / (TP + FN)  
  Mengukur seberapa banyak kasus positif yang berhasil dideteksi (menghindari false negative).

- **F1-score**  
  F1-score = 2 * (Precision * Recall) / (Precision + Recall)  
  Rata-rata harmonik dari precision dan recall. Cocok digunakan ketika ada ketidakseimbangan kelas.

- **AUC-ROC**  
  AUC (Area Under the Curve) dari ROC (Receiver Operating Characteristic) menggambarkan kemampuan model dalam membedakan kelas positif dan negatif.  
  Semakin mendekati nilai 1, semakin baik performa klasifikasi model terhadap kedua kelas.

### Ringkasan Hasil Evaluasi

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.76     | 0.68      | 0.62   | 0.65     | 0.83    |
| Random Forest       | 0.74     | 0.62      | 0.71   | 0.66     | 0.84    |
| XGBoost             | 0.75     | 0.63      | 0.71   | 0.67     | 0.79    |
| XGBoost + SMOTE     | 0.75     | 0.62      | 0.75   | 0.68     | 0.81    |
| Random Forest (Grid)| 0.77     | 0.63      | 0.84   | 0.72     | 0.84    |

### Confusion Matrix Model Terbaik: Random Forest (Grid Search)

Untuk memberikan gambaran lebih visual terkait performa model terbaik (Random Forest dengan Grid Search) digunakan confusion matriks yang menunjukkan bahwa model mampu mengklasifikasikan :
* True Negative (TN): 72
* False Positive (FP): 27
* False Negative (FN): 9
* True Positive (TP): 46
Sehingga, dapat disimpulkan model dapat mengklasifikasikan mayoritas kelas negatif (72 benar vs 27 salah) dan positif (46 benar vs 9 salah) secara cukup akurat.

> Catatan: Nilai metrik di atas merupakan hasil pada data uji setelah training dan tuning model dilakukan.

---

## Insight dan Analisis Model
Dari hasil evaluasi, dapat diambil beberapa analisis penting, yaitu :
- **Random Forest (Grid Search)** menunjukkan performa terbaik secara keseluruhan dengan akurasi tertinggi (77%) dan F1-score terbaik (0.72), menandakan keseimbangan antara precision dan recall yang lebih baik. Model ini juga memiliki recall yang sangat tinggi (0.84), yang berarti mampu mendeteksi kasus positif diabetes lebih banyak dibanding model lain. Ini penting untuk kasus medis agar meminimalkan false negative.

- **XGBoost + SMOTE** memberikan peningkatan recall (0.75) dibanding XGBoost biasa (0.71), menunjukkan bahwa teknik penyeimbangan data (SMOTE) efektif dalam meningkatkan sensitivitas model terhadap kelas minoritas.

- **Logistic Regression** memiliki akurasi yang cukup baik (0.76). Namun, nilai recall dan F1-score-nya masih lebih rendah dibanding Random Forest Grid, sehingga kurang optimal untuk mendeteksi semua kasus positif.

- AUC-ROC pada semua model relatif baik, berkisar antara 0.79 sampai 0.84, menandakan bahwa semua model memiliki kemampuan cukup baik dalam membedakan antara kelas positif dan negatif. Metrik AUC-ROC menunjukkan seluruh model memiliki kemampuan diskriminasi yang baik.

---

## Kesimpulan dan Saran

### Kesimpulan

Proyek ini berhasil mengembangkan sistem deteksi dini diabetes menggunakan pendekatan machine learning. Dengan berbagai model yang diuji, penggunaan Random Forest dengan Grid Search hyperparameter tuning memberikan hasil paling optimal untuk prediksi diabetes pada dataset ini, khususnya dalam hal recall dan F1-score.

Penerapan teknik penyeimbangan data seperti SMOTE juga membantu meningkatkan performa model, terutama dalam meningkatkan recall, sehingga lebih mampu menangkap kasus positif.

Meskipun algoritma lain seperti Logistic Regression dan XGBoost juga memberikan hasil yang layak, hasil evaluasi menunjukkan bahwa kombinasi Random Forest dan tuning hyperparameter lebih unggul untuk konteks deteksi dini diabetes.

### Saran Lanjutan

- Fokus pengembangan selanjutnya dapat diarahkan pada optimalisasi lebih lanjut model Random Forest, termasuk eksplorasi parameter tuning yang lebih luas atau teknik ensemble lainnya.
- Terapkan validasi model pada dataset nyata dengan data medis lebih lengkap.
- Integrasi sistem prediksi ini dalam bentuk aplikasi dashboard untuk klinik.
- Uji coba metode balancing lain seperti ADASYN atau undersampling.
- Lakukan interpretasi model lebih dalam untuk digunakan dan dipahami oleh tenaga kesehatan.

---

## Referensi

1. International Diabetes Federation. *IDF Diabetes Atlas*, 10th ed. 2021. [Online]. Available: https://diabetesatlas.org/
2. World Health Organization. *Diabetes*. [Online]. Available: https://www.who.int/health-topics/diabetes
3. Smith, J., & Brown, L. (2020). "Machine Learning Approaches in Early Diagnosis of Diabetes." *Journal of Medical Informatics*, 12(3), pp. 102–110.
4. Kaggle. "Pima Indians Diabetes Database." [Online]. Available: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
5. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, pp. 2825–2830.

---
