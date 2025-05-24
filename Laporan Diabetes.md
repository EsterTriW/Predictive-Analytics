# Laporan Proyek Machine Learning 
## Judul proyek : Implementasi Machine Learning dalam Prediksi Diabetes Melitus
## Ditulis oleh : Ester Tri Wahyuningsih -  MC002D5X0841

## Domain Proyek
### Latar Belakang
Diabetes melitus merupakan penyakit kronis yang ditandai dengan tingginya kadar gula darah akibat gangguan produksi atau fungsi insulin. Menurut Organisasi Kesehatan Dunia (WHO), diabetes termasuk salah satu penyakit tidak menular dengan prevalensi yang terus meningkat secara global, bahkan mencapai status epidemi di berbagai negara, khususnya di negara berkembang.

Data dari International Diabetes Federation (IDF) pada tahun 2021 menunjukkan bahwa sekitar 537 juta orang di seluruh dunia hidup dengan diabetes, dan hampir separuh dari mereka belum terdiagnosis atau menerima perawatan yang memadai. Kondisi ini berpotensi menyebabkan komplikasi serius seperti penyakit jantung, stroke, kerusakan ginjal, hingga kebutaan, yang tidak hanya membebani kualitas hidup penderita tapi juga sistem pelayanan kesehatan.

Deteksi dini diabetes sangat penting untuk melakukan intervensi preventif dan manajemen penyakit yang efektif. Namun, skrining massal secara tradisional sering terkendala oleh sumber daya medis yang terbatas dan biaya tinggi. Oleh sebab itu, pendekatan berbasis teknologi informasi dan machine learning menjadi alternatif yang sangat menjanjikan.

Machine learning membantu analisis data medis yang besar dan kompleks untuk menemukan pola dan prediksi risiko penyakit secara akurat dan efisien. Dengan memanfaatkan data klinis dasar seperti usia, indeks massa tubuh (BMI), kadar glukosa, dan riwayat keluarga, model prediksi diabetes dapat membantu tenaga medis dalam mengambil keputusan lebih cepat dan tepat sasaran.

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
Dataset yang digunakan adalah **Pima Indians Diabetes Database**, tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download) dan UCI Repository.
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
Analisis awal dataset mengungkapkan beberapa hal penting:
- Fitur `Glucose`, `Insulin`, dan `SkinThickness` mengandung nilai nol, yang secara medis tidak valid dan perlu ditangani.
- Outcome tidak seimbang, berpotensi menyebabkan bias pada model.
- Korelasi tertinggi terhadap Outcome ditemukan pada fitur `Glucose`, `BMI`, dan `Age`.

---

## Data Preparation
Pada tahap ini, data dipersiapkan agar siap digunakan untuk pelatihan model. Beberapa proses utama meliputi:
1. **Data Cleaning**: Nilai nol pada fitur medis diganti dengan median.
2. **Feature Selection**: Menghapus fitur dengan kontribusi rendah seperti `BloodPressure` dan `SkinThickness`.
3. **Scaling**: Menggunakan `StandardScaler` untuk standardisasi fitur.
4. **Splitting**: Membagi dataset menjadi 80% training dan 20% testing.

---

## Modeling

### Algoritma yang Digunakan
Untuk membangun model prediksi, beberapa algoritma dipilih dan diuji, yaitu:
- **Logistic Regression**: Sebagai baseline model.
- **Random Forest**: Model ensemble yang kuat dan interpretatif.
- **XGBoost**: Gradient boosting yang dikenal sangat kompetitif.
- **XGBoost + SMOTE**: Menambahkan SMOTE untuk menyeimbangkan data sebelum pelatihan.
- **Random Forest + GridSearch**: Hyperparameter tuning untuk mendapatkan kinerja optimal.

### Tuning Parameter (GridSearch):
Beberapa parameter Random Forest dioptimasi menggunakan Grid Search:
- `n_estimators`: [100, 200]
- `max_depth`: [4, 6, 8]
- `min_samples_split`: [2, 4]

---

## Evaluation

### Metrik Evaluasi
Model dievaluasi menggunakan metrik berikut:
- **Accuracy**: Persentase prediksi yang benar.
- **Precision**: Proporsi prediksi positif yang benar-benar positif.
- **Recall**: Proporsi kasus positif yang berhasil diprediksi dengan benar.
- **F1-score**: Harmonik rata-rata precision dan recall.
- **AUC-ROC**: Kemampuan model membedakan antara dua kelas.

### Ringkasan Hasil Evaluasi

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.76     | 0.68      | 0.62   | 0.65     | 0.83    |
| Random Forest       | 0.74     | 0.62      | 0.71   | 0.66     | 0.84    |
| XGBoost             | 0.75     | 0.63      | 0.71   | 0.67     | 0.79    |
| XGBoost + SMOTE     | 0.75     | 0.62      | 0.75   | 0.68     | 0.81    |
| Random Forest (Grid)| 0.77     | 0.63      | 0.84   | 0.72     | 0.84    |

### Confusion Matrix Model Terbaik: Random Forest (Grid Search)

Untuk memberikan gambaran lebih visual terkait performa model terbaik (Random Forest dengan tuning) digunakan confusion matriks yang menunjukkan bahwa model mampu mengklasifikasikan mayoritas kelas negatif (72 benar vs 27 salah) dan positif (46 benar vs 9 salah) secara cukup akurat.

> Catatan: Nilai metrik di atas merupakan hasil pada data uji setelah training dan tuning model dilakukan.

---

## Insight dan Analisis
Dari hasil evaluasi, dapat diambil beberapa analisis penting, yaitu :
- **Random Forest (Grid Search)** menunjukkan performa terbaik secara keseluruhan dengan akurasi tertinggi (77%) dan F1-score terbaik (0.72), menandakan keseimbangan antara precision dan recall yang lebih baik. Model ini juga memiliki recall yang sangat tinggi (0.84), yang berarti mampu mendeteksi kasus positif diabetes lebih banyak dibanding model lain. Ini penting untuk kasus medis agar meminimalkan false negative.

- **XGBoost + SMOTE** memberikan peningkatan recall (0.75) dibanding XGBoost biasa (0.71), menunjukkan bahwa teknik penyeimbangan data (SMOTE) efektif dalam meningkatkan sensitivitas model terhadap kelas minoritas.

- **Logistic Regression** memiliki akurasi yang cukup baik (0.76). Namun, nilai recall dan F1-score-nya masih lebih rendah dibanding Random Forest Grid, sehingga kurang optimal untuk mendeteksi semua kasus positif.

- AUC-ROC pada semua model relatif baik, berkisar antara 0.79 sampai 0.84, menandakan bahwa semua model memiliki kemampuan cukup baik dalam membedakan antara kelas positif dan negatif. Metrik AUC-ROC menunjukkan seluruh model memiliki kemampuan diskriminasi yang baik.
  
- Fitur `Glucose`, `BMI`, dan `Age` merupakan prediktor paling signifikan dalam model.

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
- Lakukab interpretasi model lebih dalam untuk digunakan dan dipahami oleh tenaga kesehatan.

---

## Referensi

1. International Diabetes Federation. *IDF Diabetes Atlas*, 10th ed. 2021. [Online]. Available: https://diabetesatlas.org/
2. World Health Organization. *Diabetes*. [Online]. Available: https://www.who.int/health-topics/diabetes
3. Smith, J., & Brown, L. (2020). "Machine Learning Approaches in Early Diagnosis of Diabetes." *Journal of Medical Informatics*, 12(3), pp. 102–110.
4. Kaggle. "Pima Indians Diabetes Database." [Online]. Available: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
5. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, pp. 2825–2830.
6. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, pp. 321–357.
7. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, pp. 785–794.

---
